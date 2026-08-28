"""RTRRL-SSM-RFLO as a whole: what it selects, what it steps, what it refuses.

``tests/test_dense_ssm_rflo.py`` holds the cell, its two traces and its norm
ball against the equations and against autodiff. It does not say the algorithm
is wired to them. These do: every claim below is about the graph an entry
builds and the state one transition leaves behind.

Two are this torso's alone. ``C`` holds a third of the recurrence's parameters
and carries no trace, so whether it learns is a question about the graph. And
``A`` is the second parameter in this repository with a domain it can be
stepped out of, so the projection ``rtrrl_ctrnn_rflo`` introduced for ``tau``
has to reach both copies here as well -- which is what made the helper that
applies it shared rather than private.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from entries._ensemble import GroupError, swept_parameters
from memorax.algorithms import rtrrl_ssm_rflo as ssm_rflo
from memorax.algorithms.rtrrl_aaai import Recurrence
from memorax.algorithms.rtrrl_ssm_rflo import RTRRLSsmRflo
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.networks.sequence_models.dense_ssm import (
    DENSE_SSM_DIFFERENTIATION_FAMILY,
    TRACED,
    DenseSSMCell,
    DenseSSMRflo,
)
from memorax.parameters import expand, flatten
from tests.support.builders import graph_of
from tests.support.environments import TinyContinuousEnv
from tests.support.parameters import kinds

ENVS = 3
OBSERVATION = 2  # TinyContinuousEnv's, and its action is as wide again
ACTION = 2
HIDDEN = 4
CELL = "components_0"
MATRICES = ("A", "B", "C")
BOUND = 0.9

# Every decay live, so a claim about one of them is not a claim about zero.
LIVE = {
    "gamma": 0.9,
    "lambda_pi": 0.8,
    "lambda_v": 0.7,
    "lambda_rnn": 0.6,
    "eta_pi": 0.5,
    "eta_f": 0.5,
    "entropy_rate": 1e-3,
}


def parameters(**overrides):
    return expand(
        ssm_rflo.PARAMETERS,
        {
            "torso.hidden_dim": HIDDEN,
            "torso.spectral_bound": BOUND,
            "torso.layer_norm": False,
            "torso.differentiation.kind": "rflo",
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 1e-3,
            "torso.grad_clip": 1.0,
            "torso.follow": 0.25,
            "actor.optimizer.kind": "adam",
            "actor.optimizer.adam.lr": 1e-3,
            "critic.optimizer.kind": "adam",
            "critic.optimizer.adam.lr": 1e-3,
            "actor.head.kind": "state_std",
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": True,
            **LIVE,
            **overrides,
        },
    )


def environment(identifier, **options):
    del identifier, options
    env = TinyContinuousEnv()
    return env, env.default_params


def build(**overrides) -> RTRRLSsmRflo:
    """The graph an entry builds, on an environment small enough to read."""

    built = assemble(
        RTRRLSsmRflo,
        BuildRequest(
            parameters=parameters(**overrides),
            environment=EnvironmentSpec(
                id="tiny", backend=None, observed=None, episode_length=8
            ),
            num_envs=ENVS,
        ),
        environment_factory=environment,
    )
    return graph_of(built)


def run(agent, rounds, key=1):
    state = agent.init(jax.random.key(0))
    return agent.train(jax.random.key(key), state, rounds * agent.cfg.num_envs)


def apart(a, b) -> float:
    return max(
        float(jnp.max(jnp.abs(x - y)))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


def cell_of(agent) -> DenseSSMCell:
    return agent.core.torso._network.components[0].cell


def torso_params(state):
    return state.core.torso.params[CELL]["cell"]


def row_norm(matrix) -> float:
    return float(jnp.abs(matrix).sum(-1).max())


# ------------------------------------------------------------- what it selects
def test_the_torso_is_a_dense_ssm_and_holds_exactly_its_three_matrices():
    """``A``, ``B`` and ``C``, and nothing before or behind them.

    ``B``'s width says the layout: ``[observation, previous action, previous
    reward, bias]``, one column each, with the state entering through ``A``
    alone. A projection in front of the cell would show up here as a second
    component, and the affine-free normalization behind it holds nothing to
    credit -- so the whole torso is the recurrence and its readout.
    """

    state = build().init(jax.random.key(0))
    tree = state.core.torso.params

    assert set(tree) == {CELL}
    assert set(tree[CELL]["cell"]) == set(MATRICES)
    features = OBSERVATION + ACTION + 1
    assert tree[CELL]["cell"]["A"].shape == (HIDDEN, HIDDEN)
    assert tree[CELL]["cell"]["B"].shape == (HIDDEN, features + 1)
    assert tree[CELL]["cell"]["C"].shape == (HIDDEN, HIDDEN)


def test_the_selected_differentiation_is_rflo_and_carries_two_traces():
    """Two and not three, which is the equations rather than a saving.

    ``dh/dC`` is identically zero, so a third trace would be a zero the scan
    carries and the phantom contracts to nothing. That it is absent *here*, in
    the state the algorithm carries between transitions, is the claim.
    """

    agent = build()
    assert isinstance(agent.core.torso._differentiation, DenseSSMRflo)

    after, _ = run(agent, 3)
    sensitivity = after.core.torso.recurrence.differentiation_state
    assert set(sensitivity) == set(TRACED)
    assert "C" not in sensitivity
    for name in TRACED:
        assert sensitivity[name].shape[0] == ENVS
        assert float(jnp.abs(sensitivity[name]).max()) > 0.0


def test_this_entry_may_not_select_anything_but_rflo():
    """The entry's name is a claim about which online gradient produced a run.

    It matters more here than anywhere else in the repository, because this
    entry exists to be compared against ``rtrrl``'s diagonal backbones -- and
    the whole content of that comparison is which gradient each side spent.
    """

    assert set(kinds(ssm_rflo.PARAMETERS, "torso.differentiation")) == {"rflo"}
    assert set(DENSE_SSM_DIFFERENTIATION_FAMILY.branches) == {"rflo", "tbptt"}

    with pytest.raises(KeyError):
        build(**{"torso.differentiation.kind": "tbptt"})


def test_the_normalization_behind_the_cell_is_a_choice_that_holds_no_parameters():
    """Selecting it changes what the heads read and not what is credited."""

    plain = build().init(jax.random.key(0))
    normalized = build(**{"torso.layer_norm": True}).init(jax.random.key(0))

    assert jax.tree.structure(plain.core.torso.params) == jax.tree.structure(
        normalized.core.torso.params
    )
    assert apart(torso_params(plain), torso_params(normalized)) == 0.0
    moved = run(build(**{"torso.layer_norm": True}), 2)[0]
    assert apart(torso_params(moved), torso_params(normalized)) > 0.0


# --------------------------------------------------------------- the update rule
TORSO_LR = 1e-2
HEAD_LR = 5e-3

SGD = {
    "torso.optimizer.kind": "sgd",
    "torso.optimizer.sgd.lr": TORSO_LR,
    "actor.optimizer.kind": "sgd",
    "actor.optimizer.sgd.lr": HEAD_LR,
    "critic.optimizer.kind": "sgd",
    "critic.optimizer.sgd.lr": HEAD_LR,
}


def test_every_block_may_be_stepped_by_plain_sgd():
    """This entry reads RTRRL's optimizer family, so it offers what that offers.

    Reusing the family is not the same as reaching it -- an entry that declared
    its own would look identical at the call site and quietly offer less. So
    the rule is asserted where it is selected: declared on all three blocks,
    built at the rate each one named, and stepping the cell this entry exists
    for. The rule's arithmetic is driven in ``test_rtrrl_sgd_rule``.
    """

    for block in ("torso", "actor", "critic"):
        assert "sgd" in kinds(ssm_rflo.PARAMETERS, f"{block}.optimizer")
        assert f"{block}.optimizer.sgd.lr" in flatten(ssm_rflo.PARAMETERS)

    agent = build(**SGD)
    before = agent.init(jax.random.key(0))
    after, metrics = run(agent, 8)

    for block, rate in (("torso", TORSO_LR), ("actor", HEAD_LR), ("critic", HEAD_LR)):
        taken = getattr(metrics.update, block)
        assert float(jnp.max(jnp.abs(taken.step_size - rate))) == 0.0
        assert (
            apart(getattr(before.core, block).params, getattr(after.core, block).params)
            > 0.0
        ), f"{block} never moved"
    assert all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(after.core)
    ), "the run went non-finite"


def test_the_configuration_the_launch_document_names_builds():
    """``hidden_dim: 32``, which is the width the other RTRRL entries run."""

    agent = build(**{"torso.hidden_dim": 32})
    state = agent.init(jax.random.key(0))
    assert torso_params(state)["A"].shape == (32, 32)


# --------------------------------------------------------- what one step is
def test_one_transition_is_the_state_space_step():
    """The transition inside a real step, recomputed from the parameters alone.

    The reading copy is what acts and what the torso is walked with, so this is
    that copy rather than the stepped one.
    """

    agent = build(**{"torso.follow": 1.0})
    state, _ = run(agent, 4)
    torso = agent.core.torso
    timestep = state.timestep.to_sequence()
    weights = state.core.torso.slow_params[CELL]["cell"]

    advanced, output = torso.apply(
        state.core.torso.slow_params, timestep, state.core.torso.recurrence
    )

    carry = state.core.torso.recurrence.carry[0]
    row = jnp.concatenate([torso._input(timestep)[:, 0], jnp.ones((ENVS, 1))], -1)
    wanted = carry @ weights["A"].T + row @ weights["B"].T
    assert apart(advanced.carry[0], wanted) < 1e-5
    assert apart(output[:, 0], jnp.tanh(wanted @ weights["C"].T)) < 1e-5


def test_the_sensitivity_the_algorithm_carries_is_the_cell_s_own_recurrence():
    """The torso advances the trace the differentiation component defines."""

    agent = build()
    state, _ = run(agent, 4)
    torso = agent.core.torso
    timestep = state.timestep.to_sequence()

    advanced, _ = torso.apply(
        state.core.torso.slow_params, timestep, state.core.torso.recurrence
    )
    standalone = DenseSSMRflo(torso._network.components[0])
    _, _, wanted = standalone(
        state.core.torso.slow_params[CELL],
        torso._input(timestep),
        timestep.done,
        state.core.torso.recurrence.carry[0],
        state.core.torso.recurrence.differentiation_state,
    )
    assert apart(advanced.differentiation_state, wanted) == 0.0


def test_the_step_is_taken_along_the_trace_as_it_stood():
    """Update order: the first transition of a traced-only run moves nothing."""

    agent = build(entropy_rate=0.0)
    start = agent.init(jax.random.key(0))
    after, _ = agent.train(jax.random.key(1), start, agent.cfg.num_envs)

    for name in ("torso", "actor", "critic"):
        block = getattr(after.core, name)
        assert apart(block.params, getattr(start.core, name).params) == 0.0
    assert float(jnp.abs(after.core.torso.traces[CELL]["cell"]["B"]).max()) > 0.0


def test_the_td_error_is_the_one_step_return_against_the_carried_value():
    agent = build()
    state, metrics = run(agent, 6)

    value = metrics.forward.critic.value
    reward = metrics.interaction.reward
    terminal = metrics.interaction.terminal
    wanted = reward[:-1] + 0.9 * value[1:] * (1 - terminal[:-1]) - value[:-1]

    assert apart(metrics.update.td_error[1:], wanted) < 1e-5
    assert float(jnp.abs(metrics.update.td_error).max()) > 1e-6


@pytest.mark.parametrize(
    "decay,owner",
    (("lambda_rnn", "torso"), ("lambda_pi", "actor"), ("lambda_v", "critic")),
)
def test_each_decay_moves_one_block_s_trace_and_no_other(decay, owner):
    """Which trace forgets at which rate, one block at a time."""

    reference, _ = run(build(entropy_rate=0.0), 2)
    changed, _ = run(build(entropy_rate=0.0, **{decay: 0.1}), 2)

    for name in ("torso", "actor", "critic"):
        moved = apart(
            getattr(reference.core, name).traces, getattr(changed.core, name).traces
        )
        if name == owner:
            assert moved > 0.0, f"{decay} did not reach {name}"
        else:
            assert moved == 0.0, f"{decay} reached {name}"


def test_a_zero_decay_leaves_each_trace_this_step_s_emphasised_derivative():
    """The eligibility recurrence, read off the algorithm's own reports."""

    agent = build(entropy_rate=0.0, lambda_pi=0.0, lambda_v=0.0, lambda_rnn=0.0)
    _, metrics = run(agent, 5)
    emphasis = metrics.update.emphasis

    torso = metrics.update.torso
    assert (
        apart(torso.trace_norm["recurrence"], emphasis * torso.grad_norm["recurrence"])
        < 1e-5
    )
    assert float(jnp.abs(torso.grad_norm["recurrence"]).max()) > 1e-6

    for name in ("actor", "critic"):
        block = getattr(metrics.update, name)
        assert apart(block.trace_norm, emphasis * block.grad_norm) < 1e-5
        assert float(jnp.abs(block.grad_norm).max()) > 1e-6


def test_an_ending_restarts_the_emphasis_the_trace_and_the_sensitivity():
    """Everything the episode owned is cleared together, and only then."""

    agent = build(entropy_rate=0.0)
    _, metrics = run(agent, 12)

    emphasis = metrics.update.emphasis
    ended = metrics.interaction.done
    assert bool(jnp.any(ended)), "nothing ended, so this asserts nothing"

    wanted = jnp.where(ended[:-1], 1.0, 0.9 * emphasis[:-1])
    assert apart(emphasis[1:], wanted) < 1e-6
    assert float(jnp.min(emphasis)) < 1.0, "nothing decayed, so nothing restarted"


def test_a_stream_that_ended_walks_the_torso_from_an_empty_carry_and_trace():
    """The cell's reset reaching the algorithm, on the stream that ended."""

    agent = build()
    state, _ = run(agent, 5)
    torso = agent.core.torso
    ended = state.timestep.replace(done=jnp.arange(ENVS) == 0).to_sequence()

    advanced, _ = torso.apply(
        state.core.torso.slow_params, ended, state.core.torso.recurrence
    )
    fresh, _ = torso.apply(
        state.core.torso.slow_params,
        ended,
        Recurrence(
            carry=torso._network.initialize_carry(jax.random.key(0), (ENVS, None)),
            differentiation_state=torso._differentiation.initialize(
                jax.random.key(0), (ENVS, None)
            ),
        ),
    )

    def first(tree):
        return jax.tree.map(lambda leaf: leaf[:1], tree)

    def rest(tree):
        return jax.tree.map(lambda leaf: leaf[1:], tree)

    assert (
        apart(first(advanced.differentiation_state), first(fresh.differentiation_state))
        == 0.0
    )
    assert apart(first(advanced.carry[0]), first(fresh.carry[0])) == 0.0
    assert (
        apart(rest(advanced.differentiation_state), rest(fresh.differentiation_state))
        > 0.0
    )
    assert apart(rest(advanced.carry[0]), rest(fresh.carry[0])) > 0.0


def test_every_matrix_of_the_cell_learns_including_the_untraced_one():
    """Three matrices move, and one of them has no trace to move along.

    ``C`` receives its whole gradient from the instantaneous path
    ``y_t = tanh(C h_t)``. Nothing else in this suite would notice a graph that
    credited only the traced two: the run would train, the metrics would be
    finite, and a third of the torso would be frozen at its initialisation.
    """

    start = build().init(jax.random.key(0))
    after, _ = run(build(), 6)

    for name in MATRICES:
        assert apart(torso_params(after)[name], torso_params(start)[name]) > 0.0


# ------------------------------------------------------------ the bounded A
def test_a_is_projected_back_onto_its_norm_ball_after_a_step():
    """A step that would carry ``A`` out of the ball leaves it on the boundary.

    The bound is lowered under a learning rate large enough to reach it, and
    the same run at the launch document's bound is there to say what would
    otherwise have happened. Above one the recurrence diverges over an episode,
    so this is a bound on the parameter's domain rather than a preference.
    """

    settings = {"torso.optimizer.adam.lr": 1.0}
    held, _ = run(build(**settings, **{"torso.spectral_bound": 0.1}), 4)
    loose, _ = run(build(**settings, **{"torso.spectral_bound": BOUND}), 4)

    assert row_norm(torso_params(held)["A"]) <= 0.1 + 1e-6
    assert row_norm(torso_params(loose)["A"]) > 0.1, (
        "the run at the launch document's bound stayed inside the tight one "
        "anyway, so the tight one says nothing"
    )


def test_the_reading_copy_is_inside_the_ball_too():
    """Both copies are points of the set, so acting never walks a divergent A.

    With a partial follow the reading copy is a convex combination of two
    points that are already inside -- and a ball is convex, so the combination
    is inside as well. What has to be checked is that it starts inside: the
    drawn parameters are projected before either copy is taken from them.
    """

    agent = build(
        **{
            "torso.spectral_bound": 0.2,
            "torso.optimizer.adam.lr": 1.0,
            "torso.follow": 0.25,
        }
    )
    state, _ = run(agent, 4)

    assert row_norm(state.core.torso.slow_params[CELL]["cell"]["A"]) <= 0.2 + 1e-6


def test_the_ball_is_the_kernel_s_to_name_and_the_projection_is_shared():
    """The bound belongs to the component whose parameter it bounds.

    ``kernel_constraint`` is ``rtrrl_aaai``'s because two algorithms now need
    the same traversal of the same tree shape: the CTRNN's ``tau`` floor and
    this matrix's norm ball are two sets, one mechanism. The LSTM torso names
    none and gets no projection, which is what makes this the kernel's
    statement rather than a step every core has to implement.
    """

    from memorax.algorithms.rtrrl_aaai import kernel_constraint

    agent = build(**{"torso.spectral_bound": 0.3})
    projected = cell_of(agent).constrain({"A": jnp.ones((HIDDEN, HIDDEN))})
    assert float(np.abs(np.asarray(projected["A"])).sum(-1).max()) <= 0.3 + 1e-6

    assert agent.core.torso._constraint is not None
    assert kernel_constraint(agent.core.torso._network) is not None


# ------------------------------------------------------- what a sweep may vary
STRUCTURAL = (
    "torso.hidden_dim",
    "torso.spectral_bound",
    "torso.layer_norm",
    "torso.optimizer.kind",
    "meta_rl",
)
SWEEPABLE = (
    "gamma",
    "lambda_rnn",
    "lambda_pi",
    "lambda_v",
    "eta_f",
    "eta_pi",
    "entropy_rate",
    "torso.optimizer.adam.lr",
    "torso.grad_clip",
    "torso.follow",
)


@pytest.mark.parametrize("name", STRUCTURAL)
def test_a_structural_leaf_is_declared_static(name):
    assert flatten(ssm_rflo.PARAMETERS)[name].static, f"{name} may be swept"


@pytest.mark.parametrize("name", SWEEPABLE)
def test_a_leaf_the_graph_reads_arithmetically_may_be_swept(name):
    assert not flatten(ssm_rflo.PARAMETERS)[name].static, f"{name} may not be swept"


def members(**varying) -> Any:
    """Two run documents differing in exactly what is passed."""

    base = parameters()
    return cast(
        "Any",
        [
            (
                SimpleNamespace(
                    identity=SimpleNamespace(run_id=name),
                    algorithm=SimpleNamespace(parameters=values),
                ),
                None,
            )
            for name, values in (("a", dict(base)), ("b", dict(base) | dict(varying)))
        ],
    )


def test_a_group_may_not_vary_the_shape_of_the_recurrence():
    with pytest.raises(GroupError, match="torso.hidden_dim"):
        swept_parameters(members(**{"torso.hidden_dim": 8}), ssm_rflo.PARAMETERS)


def test_a_group_may_not_vary_the_ball_a_is_held_inside():
    with pytest.raises(GroupError, match="torso.spectral_bound"):
        swept_parameters(members(**{"torso.spectral_bound": 0.5}), ssm_rflo.PARAMETERS)


def test_a_group_may_vary_a_decay():
    swept = swept_parameters(members(lambda_rnn=0.1), ssm_rflo.PARAMETERS)
    assert set(swept) == {"lambda_rnn"}


# --------------------------------------------------------- one graph, N members
def test_a_member_of_a_vmapped_round_is_the_run_it_would_have_been_alone():
    """The ensemble entry's arithmetic, against the same seeds run one at a time.

    Not bit-for-bit, and it cannot be: ``jax.vmap`` rewrites the computation
    into batched operations and XLA reduces them in a different order. What must
    hold is that a member is a function of its seed, so the comparison is to a
    budget small enough that a member picking up its neighbour's parameters
    could not fit inside it.
    """

    agent = build()
    seeds = (0, 1, 2)
    keys = jnp.stack([jax.random.key(seed) for seed in seeds])
    steps = 4 * agent.cfg.num_envs

    def alone(key):
        return agent.train(jax.random.key(9), agent.init(key), steps)[0]

    grouped = jax.vmap(alone)(keys)
    for index, seed in enumerate(seeds):
        one = alone(jax.random.key(seed))
        member = jax.tree.map(lambda leaf: leaf[index], grouped)  # noqa: B023
        assert apart(member.core.torso.params, one.core.torso.params) < 1e-5
        assert apart(member.core.actor.params, one.core.actor.params) < 1e-5
        assert apart(member.core.critic.params, one.core.critic.params) < 1e-5

    first = jax.tree.map(lambda leaf: leaf[0], grouped)
    second = jax.tree.map(lambda leaf: leaf[1], grouped)
    assert apart(first.core.torso.params, second.core.torso.params) > 1e-3


# ------------------------------------------------------------- the declarations
def test_the_algorithm_declares_its_own_surface_beside_the_shared_one():
    """Its parameters are its own; its metrics and its schema are RTRRL's."""

    from memorax.algorithms import rtrrl_aaai, rtrrl_ctrnn_rflo, rtrrl_lstm_rflo

    assert ssm_rflo.METRICS == rtrrl_aaai.METRICS
    assert ssm_rflo.OBSERVATIONS == rtrrl_aaai.OBSERVATIONS
    for other in (rtrrl_aaai, rtrrl_ctrnn_rflo, rtrrl_lstm_rflo):
        assert ssm_rflo.PARAMETERS != other.PARAMETERS

    declared = flatten(ssm_rflo.PARAMETERS)
    assert "torso.hidden_dim" in declared
    assert "torso.spectral_bound" in declared
    # The one this entry is a control for. `rtrrl` reaches its state-space
    # backbones through a `backbone` branch, and a run document that named one
    # would be selecting a diagonal recurrence and `exact_rtrl` with it.
    assert "torso.backbone.kind" not in declared
    assert "torso.backbone.kind" in flatten(rtrrl_aaai.PARAMETERS)


def test_the_two_entries_declare_one_schema_and_one_set_of_metrics():
    from entries import rtrrl_ssm_rflo, rtrrl_ssm_rflo_ensemble

    assert rtrrl_ssm_rflo.PARAMETERS == rtrrl_ssm_rflo_ensemble.PARAMETERS
    assert rtrrl_ssm_rflo.METRICS == rtrrl_ssm_rflo_ensemble.METRICS
    assert rtrrl_ssm_rflo_ensemble.GROUPED is True
    assert getattr(rtrrl_ssm_rflo, "GROUPED", False) is False


# ---------------------------------------------------- the launch documents
EXPERIMENTS = Path(__file__).resolve().parents[5] / "experiments"
ALONE = EXPERIMENTS / "rtrrl issue67 ssm rflo.yaml"
GROUPED = EXPERIMENTS / "rtrrl issue67 ssm rflo ensemble.yaml"


def document(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def paths(space, prefix=""):
    """A search space as the flat leaf names an entry reads."""

    found = {}
    for name, value in space.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            found |= paths(value, prefix=f"{key}.")
        else:
            found[key] = value
    return found


def test_the_two_launch_documents_differ_only_in_the_entry_and_what_it_runs():
    alone, grouped = document(ALONE), document(GROUPED)

    assert alone["entry"] == "rtrrl_ssm_rflo"
    assert grouped["entry"] == "rtrrl_ssm_rflo_ensemble"
    assert alone["space"] == grouped["space"]
    assert grouped["environment"]["seeds"] == [0, 1, 2]

    differing = {
        key for key in set(alone) | set(grouped) if alone.get(key) != grouped.get(key)
    }
    assert differing == {"entry", "name", "description", "environment"}


def test_the_launch_document_matches_the_arm_it_is_a_control_for():
    """Every leaf resolves, it builds, and it runs the width the others run.

    The width is the comparison: this arm and the CTRNN and LSTM ones are read
    against each other and against `rtrrl`'s diagonal backbones, and a torso
    of a different size would be answering a different question.
    """

    declared = flatten(ssm_rflo.PARAMETERS)
    space = paths(document(ALONE)["space"])

    unknown = sorted(name for name in space if name not in declared)
    assert not unknown, f"the document names {unknown}, which the entry does not"

    chosen = {name: values[0] for name, values in space.items()}
    assert chosen["torso.hidden_dim"] == 32
    assert chosen["torso.differentiation.kind"] == "rflo"
    assert chosen["torso.spectral_bound"] < 1.0

    agent = build(**chosen)
    state, _ = run(agent, 2)
    assert torso_params(state)["A"].shape == (32, 32)
    assert row_norm(torso_params(state)["A"]) <= chosen["torso.spectral_bound"] + 1e-6
