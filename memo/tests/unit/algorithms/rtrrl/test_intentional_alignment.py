"""Which transition an intentional step spends, and what it says when it fails.

Two claims, and the second exists because the first was once wrong.

**Alignment.** RTRRL runs one forward per transition and carries the previous
value, so ``delta_t = r + gamma*V(s_t) - V(s_{t-1})`` belongs to the transition
that *ended* on this step. The derivative that transition is credited through
joined the trace on the *previous* step, so the trace holding it as its newest
term is the carried one. Reading the current trace instead put the derivative of
the bootstrap state at the head of the trace -- a state the TD error carries
with a ``+gamma`` rather than a ``-1`` -- and a value step then raised the error
it had sized itself to remove. See :func:`rtrrl._trace_for`, and issue 87 for
what that did to masked HalfCheetah.

**Finiteness.** A run that diverges anyway should say where. The watch names the
component and the update step of the first non-finite value, and the cases below
hold it to that.

``tests/unit/components/test_intentional_update.py`` holds the optimizer itself
against the paper's equations, and nothing here re-derives them. What is tested
here is which of the algorithm's trees reaches that optimizer.
"""

from __future__ import annotations

from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import expand
from memorax.rl.intentional import IntentionalUpdate
from memorax.rl.traces import CARRIED
from memorax.rl.updates import Adam, ObBound, ObGDStep
from tests.support.builders import graph_of
from tests.support.environments import TinyContinuousEnv

# The published actor-critic's two intended reductions: a policy step aiming at
# roughly five percent of the log-probability, a value step at half the error.
ETA_ACTOR = 0.05
ETA_CRITIC = 0.5

# `expand` fills anything unset from the low end of its search domain, which
# would leave every scale at zero and the torso never stepping at all.
LIVE = {
    "gamma": 0.9,
    "eta_f": 1.0,
    "eta_pi": 1.0,
    "lambda_pi": 0.9,
    "lambda_v": 0.9,
    "lambda_rnn": 0.9,
    "entropy_rate": 1e-5,
}


def iu(prefix, eta):
    """One intentional selection, spelled the way a run document spells it."""

    return {
        f"{prefix}.eta": eta,
        f"{prefix}.clip": 20.0,
        f"{prefix}.beta_rms": 0.999,
        f"{prefix}.beta_clip": 0.9998,
        f"{prefix}.beta_advantage": 0.9998,
        f"{prefix}.beta_momentum": 0.0,
        f"{prefix}.denominator_floor": 1e-8,
        f"{prefix}.eps": 1e-8,
    }


def torso_branch(position):
    """The torso's selection at whichever position the credit is combined."""

    if position == "input_iu":
        return {
            "torso.optimizer.kind": "input_iu",
            **iu("torso.optimizer.input_iu", ETA_CRITIC),
        }
    return {
        "torso.optimizer.kind": "output_iu",
        **iu("torso.optimizer.output_iu.actor", ETA_ACTOR),
        **iu("torso.optimizer.output_iu.critic", ETA_CRITIC),
    }


def parameters(position, *, head, **overrides):
    return expand(
        rtrrl.PARAMETERS,
        {
            **LIVE,
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 4,
            "torso.backbone.lru.hidden_dim": 2,
            "torso.backbone.lru.differentiation.kind": "exact_rtrl",
            # A rule that sizes its own step refuses a second, outer bound.
            "torso.grad_clip": 0.0,
            "torso.follow": 1.0,
            **torso_branch(position),
            "actor.optimizer.kind": "iu",
            **iu("actor.optimizer.iu", ETA_ACTOR),
            "critic.optimizer.kind": "iu",
            **iu("critic.optimizer.iu", ETA_CRITIC),
            "actor.head.kind": head,
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": False,
            **overrides,
        },
    )


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


def assembled(position="input_iu", *, num_envs=1, head="bounded", **overrides):
    return assemble(
        rtrrl.RTRRL,
        BuildRequest(
            parameters=parameters(position, head=head, **overrides),
            environment=EnvironmentSpec(
                id="tiny", backend=None, observed=None, episode_length=8
            ),
            num_envs=num_envs,
            record=rtrrl.OBSERVATIONS.trajectory_fields,
        ),
        environment_factory=tiny_environment,
    )


def finite(tree) -> bool:
    """Is every number in a tree a number, over every leaf it has."""

    return all(
        bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in jax.tree.leaves(tree)
    )


def leaves(tree):
    return {
        jax.tree_util.keystr(path): np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


# --------------------------------------------------- which transition it spends
@pytest.mark.parametrize(
    "step",
    (
        IntentionalUpdate(eta=ETA_CRITIC),
        Adam(lr=1e-3),
        ObGDStep(bound=ObBound(kappa=2.0), lr=1e-3),
    ),
    ids=("intentional", "adam", "obgd"),
)
def test_every_rule_reads_the_trace_at_rtrrl_s_own_index(step):
    """One index for the whole algorithm, because there is one TD error.

    The recurrences differ -- the intentional one carries no followed-trace
    emphasis -- and the reading does not. RTRRL's error is one transition behind
    its forward pass, so the trace holding that transition's derivative as its
    newest term is the carried one, whichever rule is about to read it.
    """

    trace = rtrrl._trace_for(step, decay=0.81)

    assert trace.reads == CARRIED
    assert trace.emphasized is not isinstance(step, IntentionalUpdate)


@pytest.mark.parametrize("position", ("input_iu", "output_iu"))
def test_the_first_transition_of_an_intentional_run_moves_nothing(position):
    """There is no derivative for the first error to be spent along yet.

    The first update reads a trace nothing has joined, so every block's step is
    exactly zero. That is the alignment above stated where a run can see it: a
    run that moved on its first transition would be spending the derivative of
    the state it bootstrapped *from*, against an error that state enters with
    the opposite sign.
    """

    built = assembled(position)
    graph = graph_of(built)
    state = built.program.init(jax.random.key(0))
    stepped, _ = graph.train_step(state, jax.random.key(1))

    for block in ("torso", "actor", "critic"):
        before = leaves(getattr(state.core, block).params)
        after = leaves(getattr(stepped.core, block).params)
        moved = [
            path
            for path, leaf in after.items()
            if not np.array_equal(leaf, before[path])
        ]
        assert not moved, f"{block} moved on its first transition: {moved}"


@pytest.mark.parametrize("position", ("input_iu", "output_iu"))
def test_an_intentional_run_does_not_amplify_the_error_it_sized_itself_against(
    position,
):
    """The property the misalignment broke, driven through a whole graph.

    An intentional value step is derived to remove a fixed fraction of the TD
    error. Handed the bootstrap state's derivative instead, it *added* about
    ``gamma * eta`` of that error back on every transition, and the value, the
    error and the critic's parameters grew geometrically -- by roughly a third
    per update, which reaches float32's ceiling in a few hundred steps.

    The bound is loose on purpose. It is not a claim about how well this tiny
    environment is learned; it is the difference between a run that is learning
    and a run whose sign is wrong, and nothing lands between the two.
    """

    built = assembled(position, num_envs=2)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 256)

    assert finite(stepped.core), "the run left the finite numbers"
    error = float(jnp.max(jnp.abs(metrics.update.td_error)))
    assert error < 1e3, f"the TD error reached {error:.3g}"
    value = float(jnp.max(jnp.abs(metrics.forward.critic.value)))
    assert value < 1e3, f"the value reached {value:.3g}"


# ------------------------------------------------- a bounded continuous action
@pytest.mark.parametrize("position", ("input_iu", "output_iu"))
def test_a_bounded_gaussian_run_keeps_every_intentional_quantity_finite(position):
    """Both assemblies, over many updates, on a continuous bounded action.

    The configuration issue 87 was raised against, at the size a unit test can
    hold: the bounded policy head, an intentional rule on all three blocks, and
    enough transitions for every running statistic to have moved off the zero it
    was allocated at. What has to stay finite is not only the parameters --
    optimiser state, the actions the environment was handed, and every reading a
    run would be scored on.
    """

    built = assembled(position, num_envs=2)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 256)

    assert finite(stepped.core.torso.params), "the torso's parameters"
    assert finite(stepped.core.actor.params), "the actor's parameters"
    assert finite(stepped.core.critic.params), "the critic's parameters"
    assert finite(stepped.core.rule), "the rules' state"
    assert finite(stepped.core.torso.traces), "the torso's traces"
    assert finite(metrics.interaction.action), "the actions"
    assert finite(metrics.update), "the update readings"
    assert finite(metrics.forward), "the forward readings"

    # And every rule really engaged, so the run was not 256 quiet steps.
    for block in ("actor", "critic"):
        size = getattr(metrics.update, block).step_size
        assert float(jnp.max(jnp.abs(size))) > 0.0, f"the {block} never stepped"


def test_a_bounded_head_draws_outside_the_interval_it_locates_in():
    """A bounded head bounds its location, not what it samples.

    ``BoundedGaussian`` squashes the mean into ``loc_bounds`` and the scale into
    a softplus of a squashed range; the *sample* is a normal draw around that
    mean and is under no bound at all. That is the head as published, and it is
    as intended under an intentional rule as under any other -- the whole
    quantity the policy gradient is taken of is the log-probability of what was
    drawn, and a draw the distribution could not have made is not one it can be
    credited for.

    What must not happen is the environment integrating the unbounded number,
    and for the environment issue 87 was raised on it does not: the bound is in
    the adapter, and
    ``tests/test_environments.py::test_the_adapter_bounds_the_action_it_was_given``
    is that half of the statement. It is deliberately not asserted here --
    ``TinyContinuousEnv`` is a bare gymnax environment with no adapter in front
    of it, so a run against it really does hand the raw draw over, and a test
    pretending otherwise would be testing the stand-in.
    """

    built = assembled("input_iu", num_envs=4)
    graph = graph_of(built)
    state = built.program.init(jax.random.key(0))

    hidden = jnp.zeros((1, 1, 4), dtype=jnp.float32)
    timestep = graph.environment.blank_timestep(
        jnp.zeros((4, 2), dtype=jnp.float32)
    ).to_sequence()
    distribution = graph.core.actor.apply(state.core.actor.params, hidden, timestep)

    assert float(jnp.max(jnp.abs(distribution.mode()))) <= 1.0, "the mean left [-1, 1]"
    drawn = distribution.sample(seed=jax.random.key(3), sample_shape=(512,))
    assert float(jnp.max(jnp.abs(drawn))) > 1.0, "nothing was drawn outside [-1, 1]"


# --------------------------------------------------------- what the watch says
def test_the_watch_records_the_first_step_and_keeps_it():
    """A first is a first: set once, and never moved again.

    Which is the whole reason the step is carried rather than recovered from a
    per-step indicator afterwards. A reading taken every thousand updates still
    reports the exact step, because the maximum over any window containing it is
    it.
    """

    watch = rtrrl.FiniteWatch.clean()
    assert all(float(value) == 0.0 for value in watch.first.values())

    clean = {name: jnp.zeros((2,)) for name in rtrrl.WATCHED}
    watch = watch.advance(clean, step=1)
    assert all(float(value) == 0.0 for value in watch.first.values())

    failed = {**clean, "update": jnp.asarray([1.0, jnp.nan])}
    watch = watch.advance(failed, step=7)
    assert float(watch.first["update"]) == 7.0
    assert float(watch.first["params"]) == 0.0

    # Still non-finite three steps later, and still reported as the seventh.
    watch = watch.advance(failed, step=10)
    assert float(watch.first["update"]) == 7.0

    # A component that fails later gets its own step and not the first one's.
    watch = watch.advance({**clean, "params": jnp.asarray([jnp.inf])}, step=11)
    assert float(watch.first["params"]) == 11.0
    assert float(watch.first["update"]) == 7.0


def test_the_watch_names_the_component_that_failed_and_not_its_neighbours():
    """One poisoned component at a time, and the reading that says which."""

    for name in rtrrl.WATCHED:
        components = {other: jnp.zeros((1,)) for other in rtrrl.WATCHED}
        components[name] = jnp.asarray([jnp.nan])
        marked = rtrrl.FiniteWatch.clean().advance(components, step=3)

        assert float(marked.first[name]) == 3.0, name
        assert [
            other for other in rtrrl.WATCHED if float(marked.first[other]) != 0.0
        ] == [name]


def test_a_finite_run_reports_a_watch_of_zeros_under_every_name():
    """What a healthy run files, so that a non-zero is unambiguous."""

    built = assembled("output_iu", num_envs=2)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 64)

    assert set(stepped.core.finiteness.first) == set(rtrrl.WATCHED)
    assert all(
        float(value) == 0.0 for value in stepped.core.finiteness.first.values()
    ), "a finite run recorded a failure"
    assert set(metrics.update.finiteness) == set(rtrrl.WATCHED)
    assert (
        float(jnp.max(jnp.stack(list(metrics.update.finiteness.values())))) == 0.0
    ), "a finite run filed a failure"


def test_the_watch_does_not_reach_the_update_it_is_watching():
    """A diagnostic, and a diagnostic does not move parameters.

    Established by poisoning it. The carried watch is filled with the very
    values it exists to report -- a non-finite number under every name -- and
    the run then has to be the same run, leaf for leaf. Nothing downstream can
    read a quantity that garbage does not change, which is what "without
    changing normal-run numerics" means when it is checked rather than
    asserted; a watch some arm of the update divided by, or added, or gated on,
    could not survive this.
    """

    built = assembled("input_iu", num_envs=2)
    state = built.program.init(jax.random.key(0))
    poisoned = state.replace(
        core=state.core.replace(
            finiteness=rtrrl.FiniteWatch(
                first={
                    name: jnp.asarray(jnp.nan, jnp.float32) for name in rtrrl.WATCHED
                }
            )
        )
    )

    clean_run, _ = built.program.train(jax.random.key(1), state, 64)
    poisoned_run, _ = built.program.train(jax.random.key(1), poisoned, 64)

    for block in ("torso", "actor", "critic"):
        expected = leaves(getattr(clean_run.core, block).params)
        actual = leaves(getattr(poisoned_run.core, block).params)
        for path, leaf in actual.items():
            np.testing.assert_array_equal(
                leaf,
                expected[path],
                err_msg=f"{block}{path} moved when only the watch was changed",
            )
    for path, leaf in leaves(poisoned_run.core.rule).items():
        np.testing.assert_array_equal(
            leaf,
            leaves(clean_run.core.rule)[path],
            err_msg=f"the rule state at {path} moved when only the watch was changed",
        )


def test_every_watched_name_is_a_reading_a_run_can_be_told_about():
    """The declaration and the carried state name the same nine components."""

    assert set(rtrrl.WATCHED) == {item.name for item in fields(rtrrl.FinitenessReports)}
    for name in rtrrl.WATCHED:
        assert f"update.finiteness.{name}" in rtrrl.TRAINING_METRICS
