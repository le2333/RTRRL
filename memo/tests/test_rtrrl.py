"""What RTRRL does that StreamAC does not, checked without a reference.

The parity tests next door say this kernel computes what the published one
computes. They cannot say what it *is*, because two implementations of the same
mistake agree. These do: each one names a property the algorithm is supposed to
have and drives the kernel until it either holds or does not.

Horizons matter here in a way they did not for StreamAC. Once the shared torso
moves, every later hidden state moves with it and both heads see a different
input, so an "X does not touch Y" claim is only true up to the step at which
sharing carries it across. Each test below runs for the shortest horizon at
which its claim has content, and the coupling that appears after that is the
algorithm working rather than a leak -- which is itself checked.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.algorithms.rtrrl_aaai import RTRRL, RTRRLConfig
from memorax.networks import heads
from memorax.networks.components import LayerNorm
from memorax.networks.sequence import Sequence
from memorax.networks.sequence_models import LRUCell, LRUConfig, Memoroid
from memorax.networks.sequence_models.lru import LRUStructuredRTRL
from memorax.rl.updates import Adam
from tests.support.environments import TinyContinuousEnv

ENVS = 3
OBSERVATION = 2  # TinyContinuousEnv's, and its action is as wide again
ACTION = 2
READOUT = 4
HIDDEN = 2
CELL = "components_0"


def torso_network(*, meta_rl: bool = True) -> Sequence:
    """The deployed torso's shape: the cell, and one affine-free LayerNorm.

    Nothing in front of the cell, so its input width is whatever
    ``Torso._input`` hands it -- which is why this has to be told whether the
    previous action and reward are riding along.
    """

    features = OBSERVATION + (ACTION + 1 if meta_rl else 0)
    return Sequence(
        components=(
            Memoroid(
                cell=LRUCell(
                    config=LRUConfig(
                        features=features, hidden_dim=HIDDEN, output_dim=READOUT
                    )
                )
            ),
            LayerNorm(use_scale=False, use_bias=False),
        )
    )


def eligibility(kind: str) -> rtrrl.EligibilityReadout:
    """The readout an experiment naming one of the four conditions gets."""

    direction, magnitude = rtrrl.ELIGIBILITY_SOURCES[kind]
    return rtrrl.EligibilityReadout(direction=direction, magnitude=magnitude)


def build(
    head: str = "state_std", reports=None, reads: str = "trace", **overrides
) -> RTRRL:
    """The kernel, on the smallest sequence that still has a recurrence in it."""

    env = TinyContinuousEnv()
    action_dim = int(env.action_space(env.default_params).shape[0])
    torso = torso_network(meta_rl=overrides.get("meta_rl", True))
    settings = {
        "num_envs": ENVS,
        "gamma": 0.9,
        "lambda_pi": 0.8,
        "lambda_v": 0.7,
        "lambda_rnn": 0.6,
        "torso_optimizer": Adam(lr=1e-3),
        "actor_optimizer": Adam(lr=1e-3),
        "critic_optimizer": Adam(lr=1e-3),
        "eta_pi": 0.5,
        "eta_f": 0.5,
        "entropy_rate": 1e-3,
        "torso_follow": 0.25,
    }
    # The published actor reads both the location and the log scale off the
    # hidden state. ``heads.Gaussian`` keeps the scale as a free parameter, and
    # then its entropy does not depend on the hidden state at all -- so the
    # immediate objective's whole path to the shared torso is identically zero.
    # That is a fact about the head, and one test below is about exactly it.
    actor = (
        heads.StateStdGaussian(action_dim=action_dim)
        if head == "state_std"
        else heads.Gaussian(action_dim=action_dim)
    )
    return RTRRL(
        RTRRLConfig(**{**settings, **overrides}),
        env,
        env.default_params,
        torso,
        LRUStructuredRTRL(torso.core),
        actor,
        heads.VNetwork(),
        reports=rtrrl.Reports() if reports is None else reports,
        eligibility=eligibility(reads),
    )


def apart(a, b) -> float:
    return max(
        float(jnp.max(jnp.abs(x - y)))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


def biggest(tree) -> float:
    return max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree.leaves(tree))


def run(agent: RTRRL, rounds: int, key: int = 1):
    state = agent.init(jax.random.key(0))
    return agent.train(jax.random.key(key), state, rounds * agent.cfg.num_envs)[0]


def moved(before, after) -> dict:
    """How far each block's parameters travelled between two states."""

    return {
        name: apart(getattr(before.core, name).params, getattr(after.core, name).params)
        for name in ("torso", "actor", "critic")
    }


def traced_only(**overrides):
    """Two rounds with no immediate objective.

    Step one then moves nothing at all, so step two is the first update and it
    comes entirely through the trace. That is the only horizon at which "this
    dial belongs to that group" is a statement about the dial rather than about
    how far the sharing has spread.
    """

    return run(build(entropy_rate=0.0, **overrides), 2)


# ------------------------------------------- the trace is used before advancing


def test_the_first_step_moves_nothing_when_nothing_is_immediate():
    """The trace an update would step along has not been written yet."""

    agent = build(entropy_rate=0.0)
    start = agent.init(jax.random.key(0))
    after = agent.train(jax.random.key(1), start, agent.cfg.num_envs)[0]
    assert max(moved(start, after).values()) == 0.0


def test_the_immediate_objective_reaches_the_actor_and_the_torso_and_not_the_critic():
    """Entropy is the actor's, and it arrives on the step it arises.

    The critic has no immediate objective at all, so on a step where the trace
    is still empty it cannot move.
    """

    agent = build()
    start = agent.init(jax.random.key(0))
    after = agent.train(jax.random.key(1), start, agent.cfg.num_envs)[0]
    travelled = moved(start, after)
    assert travelled["actor"] > 0
    assert travelled["torso"] > 0
    assert travelled["critic"] == 0.0


def test_a_state_independent_scale_cuts_the_immediate_path_to_the_torso():
    """Not a defect: a property of the head, and the reason for the other one.

    ``heads.Gaussian`` holds its log scale as a free parameter, so the entropy
    of what it produces does not depend on the hidden state, so the cotangent
    the immediate objective would send up is identically zero.
    """

    agent = build(head="global_std")
    start = agent.init(jax.random.key(0))
    after = agent.train(jax.random.key(1), start, agent.cfg.num_envs)[0]
    travelled = moved(start, after)
    assert travelled["actor"] > 0
    assert travelled["torso"] == 0.0


# --------------------------------------------------- both heads steer the torso


def test_silencing_the_policy_changes_the_shared_torso():
    """If it did not, the torso would not be shared with the actor."""

    base = run(build(), 6)
    quiet = run(build(eta_pi=0.0, entropy_rate=0.0), 6)
    assert apart(base.core.torso.params, quiet.core.torso.params) > 0


def test_the_torso_still_hears_what_is_not_traced():
    """With ``eta_f`` at zero the trace contributes nothing and entropy remains."""

    quiet = run(build(eta_pi=0.0, entropy_rate=0.0), 6)
    silent = run(build(eta_pi=0.0, entropy_rate=0.0, eta_f=0.0), 6)
    assert apart(quiet.core.torso.params, silent.core.torso.params) > 0


# ------------------------------------- the groups, on the step they act and after


@pytest.mark.parametrize(
    "dial,overrides,expected",
    [
        ("eta_f", ({"eta_f": 1.0}, {"eta_f": 0.1}), ("torso",)),
        (
            "the torso's clip",
            ({"torso_grad_clip": 1e-8}, {"torso_grad_clip": 0.0}),
            ("torso",),
        ),
        (
            "the head rates",
            (
                {"actor_optimizer": Adam(lr=1e-3), "critic_optimizer": Adam(lr=1e-3)},
                {"actor_optimizer": Adam(lr=1e-5), "critic_optimizer": Adam(lr=1e-5)},
            ),
            ("actor", "critic"),
        ),
        (
            "the actor rate alone",
            ({"actor_optimizer": Adam(lr=1e-3)}, {"actor_optimizer": Adam(lr=1e-5)}),
            ("actor",),
        ),
        (
            "the critic rate alone",
            ({"critic_optimizer": Adam(lr=1e-3)}, {"critic_optimizer": Adam(lr=1e-5)}),
            ("critic",),
        ),
    ],
)
def test_a_dial_moves_its_own_group_and_no_other(dial, overrides, expected):
    loud, soft = traced_only(**overrides[0]), traced_only(**overrides[1])
    travelled = moved(loud, soft)
    for name, distance in travelled.items():
        if name in expected:
            assert distance > 0, f"{dial} did not move {name}"
        else:
            assert distance == 0.0, f"{dial} reached {name}"


@pytest.mark.parametrize(
    "overrides,crosses_to",
    [
        (
            (
                {"actor_optimizer": Adam(lr=1e-3), "critic_optimizer": Adam(lr=1e-3)},
                {"actor_optimizer": Adam(lr=1e-5), "critic_optimizer": Adam(lr=1e-5)},
            ),
            ("torso",),
        ),
        (
            ({"torso_grad_clip": 1e-8}, {"torso_grad_clip": 0.0}),
            ("actor", "critic"),
        ),
    ],
)
def test_a_dial_stops_being_its_own_group_s_a_few_steps_later(overrides, crosses_to):
    """Which is what sharing is.

    A head rate that only ever touched the heads would mean the torso was not
    shared with them, and a torso clip that never reached a head would mean the
    heads did not read the torso.
    """

    loud, soft = run(build(**overrides[0]), 6), run(build(**overrides[1]), 6)
    for name in crosses_to:
        assert (
            apart(getattr(loud.core, name).params, getattr(soft.core, name).params) > 0
        )


# ------------------------------------------------------------ three decays


@pytest.mark.parametrize(
    "decay,owner,others",
    [
        ("lambda_rnn", "torso", ("actor", "critic")),
        ("lambda_pi", "actor", ("critic",)),
        ("lambda_v", "critic", ("actor",)),
    ],
)
def test_each_decay_moves_one_block_s_trace(decay, owner, others):
    """Three blocks, three rates. StreamAC has one, which proves nothing."""

    # Which decay is overridden is the parameter, so the name is only known here.
    slow: dict[str, Any] = {decay: 0.1}
    fast: dict[str, Any] = {decay: 0.9}
    low, high = run(build(**slow), 2), run(build(**fast), 2)
    assert apart(getattr(low.core, owner).traces, getattr(high.core, owner).traces) > 0
    for name in others:
        assert (
            apart(getattr(low.core, name).traces, getattr(high.core, name).traces)
            == 0.0
        )


# ------------------------------------ what a step reads of the trace it keeps


def test_an_empty_trace_moves_nothing_unless_the_step_reads_its_own_gradient():
    """The first transition, where the four conditions are furthest apart.

    Nothing has accumulated yet, so three of them have no direction to step
    along: the trace is zero, and a zero rescaled to any norm is still zero.
    Only the condition that reads the gradient instead of the trace has one.
    """

    def travelled(reads):
        agent = build(entropy_rate=0.0, reads=reads)
        start = agent.init(jax.random.key(0))
        after = agent.train(jax.random.key(1), start, agent.cfg.num_envs)[0]
        return max(moved(start, after).values())

    assert travelled("gradient") > 0
    for reads in ("trace", "direction", "gain"):
        assert travelled(reads) == 0.0, reads


def test_the_trace_a_step_leaves_behind_does_not_depend_on_what_it_read():
    """One recursion, four readings of it, which is what makes them comparable.

    One round is the horizon at which this has content. The traces are built
    from gradients taken at the same parameters there, because it is the update
    those gradients feed that the conditions differ in. From the second round
    the parameters have moved apart and everything downstream with them, which
    is the ablation working rather than the recursion leaking.
    """

    stored = {
        reads: run(build(entropy_rate=0.0, reads=reads), 1).core
        for reads in rtrrl.ELIGIBILITY_SOURCES
    }

    for reads, core in stored.items():
        for name in ("torso", "actor", "critic"):
            distance = apart(
                getattr(core, name).traces, getattr(stored["trace"], name).traces
            )
            assert distance == 0.0, f"{reads} stored a different {name} trace"


def test_no_two_conditions_step_the_same_way_once_a_trace_exists():
    """Four conditions and not two: the mixtures are their own runs."""

    ends = {
        reads: run(build(entropy_rate=0.0, reads=reads), 4)
        for reads in rtrrl.ELIGIBILITY_SOURCES
    }

    for one, other in combinations(ends, 2):
        assert (
            apart(ends[one].core.torso.params, ends[other].core.torso.params) > 0
        ), f"{one} and {other} left the torso in the same place"


# ----------------------------------------------------- the recurrence is exact


def test_a_differentiation_state_is_carried_and_is_not_zero():
    """Truncated credit carries none, so this is what tells the two apart."""

    state = run(build(), 6)
    differentiation_state = state.core.torso.recurrence.differentiation_state
    assert differentiation_state is not None
    assert biggest(differentiation_state) > 0


def test_every_recurrent_parameter_of_the_cell_learns():
    """A cell whose eigenvalue never moved would be a fixed filter."""

    fresh = build().init(jax.random.key(0))
    state = run(build(), 6)
    for name in ("nu_log", "theta_log", "gamma_log", "B_real", "B_imag"):
        assert (
            apart(
                state.core.torso.params[CELL]["cell"][name],
                fresh.core.torso.params[CELL]["cell"][name],
            )
            > 0
        ), name


# ------------------------------------------------------- the following copy


def test_the_reading_copy_lags_the_updated_one():
    state = run(build(torso_follow=0.25), 6)
    assert apart(state.core.torso.params, state.core.torso.slow_params) > 0


def test_following_all_the_way_makes_them_one_tree():
    state = run(build(torso_follow=1.0), 6)
    assert apart(state.core.torso.params, state.core.torso.slow_params) == 0.0


# ---------------------------------------------------------------- the ending


def test_an_ending_restores_the_emphasis_and_gamma_decays_it_otherwise():
    """Read off the carried ending, so it lands the step after the transition."""

    agent = build(gamma=0.5)
    state = agent.init(jax.random.key(0))
    seen = []
    for _ in range(7):
        state, metrics = agent.train(jax.random.key(1), state, agent.cfg.num_envs)
        seen.append(
            (bool(metrics.interaction.done[0][0]), float(state.core.emphasis[0]))
        )

    assert any(done for done, _ in seen), "no episode ended, so nothing was tested"
    for index, (_, emphasis) in enumerate(seen):
        if index and seen[index - 1][0]:
            assert emphasis == 1.0
        elif index:
            assert emphasis == pytest.approx(0.5 * seen[index - 1][1])


# ------------------------------------------------- evaluation leaves nothing


def test_evaluation_carries_a_state_of_its_own_and_leaves_training_alone():
    """The state an evaluation hands back is the evaluation's, not training's.

    A checkpoint is scored on a number of episodes, so the rollout has to
    survive between calls and therefore has to be handed back. What keeps that
    from becoming an update is where it came from: ``open_evaluation`` builds
    it on a fresh environment and sequence, so nothing that reaches the caller
    is the training state, and the training state is not touched by any of it.
    """

    agent = build()
    trained = run(agent, 6)
    kept = jax.tree.map(jnp.copy, trained.core)

    opened = agent.open_evaluation(jax.random.key(2), trained)
    advanced, evaluated = agent.evaluate(jax.random.key(3), opened, 4 * ENVS)

    assert evaluated.interaction.reward is not None
    assert evaluated.update is None, "nothing is updated during an evaluation"
    assert max(moved(trained, trained.replace(core=kept)).values()) == 0.0
    # Opened on its own environment, and advanced from there: neither the
    # rollout it starts nor the one it hands on is the training one.
    assert int(advanced.step) != int(trained.step)
    assert max(moved(advanced, advanced.replace(core=kept)).values()) == 0.0


# ------------------------------------------------------------------ meta-rl


def test_meta_rl_widens_the_first_layer_and_nothing_else():
    """The previous action and reward ride along with the observation.

    The first layer is the cell itself now that nothing precedes it, so what
    widens is the input matrix it reads the observation through.
    """

    def fan_in(agent):
        state = agent.init(jax.random.key(0))
        return state.core.torso.params[CELL]["cell"]["B_real"].shape[1]

    assert fan_in(build(meta_rl=True)) == fan_in(build(meta_rl=False)) + ACTION + 1


# --------------------------------------------------------- what a run declares


def test_a_reading_nobody_declared_is_absent():
    """Absent, not zero: an empty pytree is what a scan stacks nothing for."""

    agent = build(
        reports=rtrrl.Reports(
            log_prob=False,
            emphasis=False,
            torso=rtrrl.BlockReports(grad_norm=False, trace_norm=True),
            actor=rtrrl.HeadReports(step_size=False),
        )
    )
    state = agent.init(jax.random.key(0))
    _, metrics = agent.train(jax.random.key(1), state, 2 * ENVS)

    assert metrics.forward.actor.log_prob is None
    assert metrics.update.emphasis is None
    assert metrics.update.torso.grad_norm is None
    assert metrics.update.actor.step_size is None
    # and what was declared is still there
    assert metrics.forward.actor.entropy is not None
    assert metrics.update.torso.trace_norm is not None
    assert metrics.update.torso.step_size is not None
    assert metrics.update.critic.step_size is not None


def test_declaring_less_compiles_to_less():
    """The gate is a Python bool at trace time, so XLA drops what it guards."""

    def lowered(reports):
        agent = build(reports=reports)
        state = agent.init(jax.random.key(0))
        return (
            jax.jit(agent.train, static_argnums=2)
            .lower(jax.random.key(1), state, 2 * ENVS)
            .compile()
        )

    everything = lowered(rtrrl.Reports()).as_text()
    nothing = lowered(
        rtrrl.Reports(
            log_prob=False,
            entropy=False,
            value=False,
            td_error=False,
            emphasis=False,
            torso=rtrrl.BlockReports(False, False, False),
            actor=rtrrl.HeadReports(False, False, False),
            critic=rtrrl.HeadReports(False, False, False),
        )
    ).as_text()

    assert everything is not None and nothing is not None
    assert len(nothing.splitlines()) < len(everything.splitlines())
