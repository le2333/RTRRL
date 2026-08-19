"""What a checkpoint has to restore before a fork means anything.

R3.4 branches three update rules from one moment of one run and reads the
difference between them as a result about the rules. That reading is only
valid if the moment is the same moment: a branch that restored the weights but
began the eligibility traces, the recurrent carry, the environment or the PRNG
afresh would differ from its sibling for reasons that have nothing to do with
which rule it was given.

So these tests are about identity rather than about serialization. The round
trip is checked leaf by leaf against a trained state, the continuation is
checked against the run that would have continued, and the twin fork -- two
branches under the same rule -- is checked for being the *same run*, not merely
a similar one. The last is what makes the whole design falsifiable: if two
unchanged branches diverge, no comparison between changed ones means anything.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from memorax.runtime.checkpoint import (
    CheckpointDirectory,
    CheckpointError,
    dumps,
    loads,
    read,
    write,
)
from tests.support.builders import D_RTRRL, assemble_rtrrl, rtrrl_parameters
from tests.support.numerics import flattened

STEPS = 6
BRANCH = 5


def trained(steps=STEPS, optimizer=None, seed=0):
    """A run carried far enough that every trace and carry holds something."""

    built = assemble_rtrrl(optimizer=optimizer)
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    state = built.program.init(init_key)
    key, chunk_key = jax.random.split(key)
    state, _ = built.program.train(chunk_key, state, steps)
    return built, state, key


def fresh(optimizer=None, seed=0):
    """The template a resuming run builds: its own graph, its own fresh state."""

    built = assemble_rtrrl(optimizer=optimizer)
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    return built, built.program.init(init_key), key


def test_a_checkpoint_restores_every_quantity_the_run_was_carrying():
    """Not the parameters. Everything.

    The list is the one R3.4 names: learner parameters, the rule's own state,
    the eligibility traces, the recurrent carry and its differentiation state,
    the environment's state, the normalization statistics and the counters. It
    is checked by flattening the whole tree rather than by naming fields, so a
    quantity added to the state later is covered the day it is added instead of
    the day someone remembers to extend this list.
    """

    built, state, key = trained()
    payload = dumps(env_steps=STEPS, state=state, key=key)

    _, template, template_key = fresh()
    restored = loads(payload, state=template, key=template_key)

    assert restored.env_steps == STEPS
    assert restored.replaced == ()
    before, after = flattened(state), flattened(restored.state)
    assert set(before) == set(after)
    moved = [
        path for path, leaf in before.items() if not np.array_equal(leaf, after[path])
    ]
    assert not moved, f"{moved} did not come back"

    # And the template really was different, or the comparison above is between
    # a fresh state and a fresh state and asserts nothing.
    untrained = flattened(template)
    assert any(
        not np.array_equal(leaf, untrained[path]) for path, leaf in before.items()
    )


def test_a_restored_run_is_the_run_that_saved_it_continuing():
    """The stored key is the next chunk's key, so the continuation is the same.

    This is what makes a same-rule branch a continuation rather than a restart
    from equal numbers: not only the state but the position in the PRNG stream
    is carried, so the transitions the branch sees are the transitions the
    parent would have seen.
    """

    built, state, key = trained()
    payload = dumps(env_steps=STEPS, state=state, key=key)

    chunk_key, _ = jax.random.split(key)
    carried_on, expected = built.program.train(chunk_key, state, BRANCH)

    _, template, template_key = fresh()
    restored = loads(payload, state=template, key=template_key)
    branch_key, _ = jax.random.split(restored.key)
    branched, produced = built.program.train(branch_key, restored.state, BRANCH)

    apart = [
        path
        for path, leaf in flattened(carried_on).items()
        if not np.array_equal(leaf, flattened(branched)[path])
    ]
    assert not apart, f"the branch diverged from the continuation at {apart}"
    np.testing.assert_array_equal(
        np.asarray(expected.interaction.reward), np.asarray(produced.interaction.reward)
    )


def test_two_unchanged_branches_agree_until_one_is_deliberately_perturbed():
    """The determinism the fork mechanism is required to have, and its control.

    Both halves matter. Twins that agree could be twins of a run that ignores
    its restored state entirely, so the perturbed pair is what shows the branch
    is reading what it was given: one weight moved by a millionth, and the two
    branches part.
    """

    built, state, key = trained()
    payload = dumps(env_steps=STEPS, state=state, key=key)
    _, template, template_key = fresh()

    def branch(nudge=0.0):
        restored = loads(payload, state=template, key=template_key)
        core = restored.state.core
        if nudge:
            critic = core.critic
            params = jax.tree.map(lambda leaf: leaf + nudge, critic.params)
            core = core.replace(critic=critic.replace(params=params))
        branch_key, _ = jax.random.split(restored.key)
        return built.program.train(
            branch_key, restored.state.replace(core=core), BRANCH
        )[0]

    twin, other_twin, perturbed = branch(), branch(), branch(nudge=1e-6)

    left, right = flattened(twin), flattened(other_twin)
    disagreed = [path for path in left if not np.array_equal(left[path], right[path])]
    assert not disagreed, f"two unchanged branches disagreed at {disagreed}"

    moved = flattened(perturbed)
    assert any(
        not np.array_equal(left[path], moved[path]) for path in left
    ), "the perturbation did not reach the branch: it is not reading its checkpoint"


def test_a_checkpoint_of_another_graph_is_refused_by_the_leaf_that_does_not_fit():
    """Shape is checked because nothing else checks it.

    Restoring an array reads the stored one, so a graph whose torso got wider
    would take the narrow one and train on it. The failure names the path, so
    the answer to "which checkpoint is this" does not require reading it back
    in a notebook.
    """

    _, state, key = trained()
    payload = dumps(env_steps=STEPS, state=state, key=key)

    narrower = assemble_rtrrl(
        optimizer={"torso.backbone.lru.hidden_dim": 4},
    )
    template = narrower.program.init(jax.random.key(0))

    with pytest.raises(CheckpointError, match="this run wants"):
        loads(payload, state=template, key=jax.random.key(0))


def test_a_branch_onto_another_rule_declares_the_state_it_cannot_take():
    """Adam carries moments; the D-RTRRL arms carry nothing.

    That is the one honest mismatch a fork has, and it is declared per fork
    rather than inferred -- inferring it is what would let a checkpoint from an
    unrelated run pass as a deliberate branch. Undeclared it is an error;
    declared, the branch takes everything else and says what it did not take.
    """

    _, state, key = trained()
    payload = dumps(env_steps=STEPS, state=state, key=key)

    arm = assemble_rtrrl(optimizer=D_RTRRL)
    template = arm.program.init(jax.random.key(0))

    with pytest.raises(CheckpointError):
        loads(payload, state=template, key=jax.random.key(0))

    restored = loads(
        payload, state=template, key=jax.random.key(0), replacing=("core.rule",)
    )
    assert restored.replaced == ("core.rule",)

    # Everything outside the rule came from the parent, which is the whole
    # point: the arms are compared from identical parameters and traces.
    parent, branch = flattened(state.core.torso), flattened(restored.state.core.torso)
    assert all(np.array_equal(leaf, branch[path]) for path, leaf in parent.items())
    # And the rule's own state is the branch's own, not the parent's moments.
    assert not jax.tree.leaves(restored.state.core.rule["torso"])


def test_a_path_neither_side_has_cannot_be_declared_replaceable():
    _, state, key = trained()
    payload = dumps(env_steps=STEPS, state=state, key=key)
    _, template, template_key = fresh()

    with pytest.raises(CheckpointError, match="neither run has anything at"):
        loads(payload, state=template, key=template_key, replacing=("core.moments",))


def test_a_document_that_is_not_a_checkpoint_is_not_read_as_one():
    _, template, key = fresh()

    with pytest.raises(CheckpointError, match="not a checkpoint"):
        loads(b"\x80", state=template, key=key)


def test_a_checkpoint_survives_the_file_it_was_written_to(tmp_path):
    _, state, key = trained()
    path = write(tmp_path / "step.msgpack", env_steps=STEPS, state=state, key=key)

    _, template, template_key = fresh()
    restored = read(path, state=template, key=template_key)

    assert restored.env_steps == STEPS
    stored, wanted = flattened(restored.state), flattened(state)
    assert all(np.array_equal(leaf, stored[path]) for path, leaf in wanted.items())


def test_a_directory_keeps_the_last_checkpoints_and_names_them_by_step(tmp_path):
    """Retention is counted in checkpoints because a fork is placed by boundary.

    Every one is kept by default: which boundary a fork needs is decided after
    the run is over, from a collapse the run had not had yet.
    """

    _, state, key = trained(steps=1)
    directory = CheckpointDirectory(tmp_path / "checkpoints", keep=2)
    for boundary in (10, 20, 30):
        directory.save(env_steps=boundary, state=state, key=key)

    kept = sorted(path.name for path in (tmp_path / "checkpoints").iterdir())
    assert kept == ["step-000000000020.msgpack", "step-000000000030.msgpack"]
    assert [path.name for path in directory.written] == kept


def test_a_directory_that_keeps_nothing_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cannot be forked"):
        CheckpointDirectory(tmp_path, keep=0)


def test_the_parameters_the_fixtures_share_are_the_ones_the_arms_are_written_over():
    """The fixture's own contract: both suites build the same tiny RTRRL."""

    assert rtrrl_parameters()["torso.optimizer.kind"] == "adam"
    assert rtrrl_parameters(optimizer=D_RTRRL)["torso.optimizer.kind"] == "d_rtrrl"
