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

import jax
import jax.numpy as jnp
import pytest

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.algorithms.rtrrl_aaai import RTRRL, RTRRLConfig
from memorax.networks import heads
from memorax.networks.components import FFN, LayerNorm, Tanh
from memorax.networks.sequence import Sequence
from memorax.networks.sequence_models import LRUCell, LRUConfig, Memoroid
from memorax.rl.updates import Adam
from tests.support.environments import TinyContinuousEnv

ENVS = 3
FEATURES = 4
HIDDEN = 2
CELL = "components_3"


def torso_network() -> Sequence:
    return Sequence(
        components=(
            FFN(features=FEATURES),
            LayerNorm(),
            Tanh(),
            Memoroid(
                cell=LRUCell(config=LRUConfig(features=FEATURES, hidden_dim=HIDDEN))
            ),
        )
    )


def build(head: str = "state_std", reports=None, **overrides) -> RTRRL:
    """The kernel, on the smallest sequence that still has a recurrence in it."""

    env = TinyContinuousEnv()
    action_dim = int(env.action_space(env.default_params).shape[0])
    torso = torso_network()
    settings = {
        "num_envs": ENVS,
        "gamma": 0.9,
        "lambda_pi": 0.8,
        "lambda_v": 0.7,
        "lambda_rnn": 0.6,
        "torso_optimizer": Adam(lr=1e-3),
        "heads_optimizer": Adam(lr=1e-3),
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
        actor,
        heads.VNetwork(),
        reports=rtrrl.Reports() if reports is None else reports,
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
            "the head rate",
            (
                {"heads_optimizer": Adam(lr=1e-3)},
                {"heads_optimizer": Adam(lr=1e-5)},
            ),
            ("actor", "critic"),
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
                {"heads_optimizer": Adam(lr=1e-3)},
                {"heads_optimizer": Adam(lr=1e-5)},
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

    low, high = run(build(**{decay: 0.1}), 2), run(build(**{decay: 0.9}), 2)
    assert apart(getattr(low.core, owner).traces, getattr(high.core, owner).traces) > 0
    for name in others:
        assert (
            apart(getattr(low.core, name).traces, getattr(high.core, name).traces)
            == 0.0
        )


# ----------------------------------------------------- the recurrence is exact


def test_a_sensitivity_is_carried_and_is_not_zero():
    """Truncated credit carries none, so this is what tells the two apart."""

    state = run(build(), 6)
    sensitivity = state.core.torso.recurrence.sensitivity
    assert sensitivity is not None
    assert biggest(sensitivity) > 0


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


def test_evaluation_hands_back_readings_and_no_state():
    """The two checks that used to live here are unrepresentable now.

    "No parameter moved" and "it began on a fresh sequence" needed an evaluation
    state to compare; there is none to hand out. That is the guarantee, and the
    return type is what carries it.
    """

    trained = run(build(), 6)
    kept = jax.tree.map(jnp.copy, trained.core)
    evaluated = build().evaluate(jax.random.key(2), trained, 4 * ENVS)

    assert not isinstance(evaluated, tuple)
    assert evaluated.interaction.reward is not None
    assert evaluated.update is None, "nothing is updated during an evaluation"
    assert max(moved(trained, trained.replace(core=kept)).values()) == 0.0


# ------------------------------------------------------------------ meta-rl


def test_meta_rl_widens_the_first_layer_and_nothing_else():
    """The previous action and reward ride along with the observation."""

    def fan_in(agent):
        state = agent.init(jax.random.key(0))
        return state.core.torso.params["components_0"]["Dense_0"]["kernel"].shape[0]

    action_dim = 2
    assert fan_in(build(meta_rl=True)) == fan_in(build(meta_rl=False)) + action_dim + 1


# --------------------------------------------------------- what a run declares


def test_a_reading_nobody_declared_is_absent():
    """Absent, not zero: an empty pytree is what a scan stacks nothing for."""

    agent = build(
        reports=rtrrl.Reports(
            log_prob=False,
            emphasis=False,
            torso=rtrrl.BlockReports(grad_norm=False, trace_norm=True),
            heads_step=rtrrl.GroupReports(step_size=False),
        )
    )
    state = agent.init(jax.random.key(0))
    _, metrics = agent.train(jax.random.key(1), state, 2 * ENVS)

    assert metrics.forward.actor.log_prob is None
    assert metrics.update.emphasis is None
    assert metrics.update.torso.grad_norm is None
    assert metrics.update.heads_step.step_size is None
    # and what was declared is still there
    assert metrics.forward.actor.entropy is not None
    assert metrics.update.torso.trace_norm is not None
    assert metrics.update.torso_step.step_size is not None


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

    everything = lowered(rtrrl.Reports())
    nothing = lowered(
        rtrrl.Reports(
            log_prob=False,
            entropy=False,
            value=False,
            td_error=False,
            emphasis=False,
            torso=rtrrl.BlockReports(False, False),
            actor=rtrrl.BlockReports(False, False),
            critic=rtrrl.BlockReports(False, False),
            torso_step=rtrrl.GroupReports(False),
            heads_step=rtrrl.GroupReports(False),
        )
    )
    assert len(nothing.as_text().splitlines()) < len(everything.as_text().splitlines())
