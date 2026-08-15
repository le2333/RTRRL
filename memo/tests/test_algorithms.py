"""The online kernel runs end to end and answers the contract.

These are wiring tests, not numerical ones. They exist because the kernel is
assembled from a config, a set of modules, and a choice of update rule, and a
mistake in that assembly shows up as a shape error or a silent NaN rather than
as an import failure.

RTRRL has a different shared-torso graph and lives in its own semantic and
assembly suites. This file exercises StreamAC's independent online blocks and
its optimizer variants only.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from memorax.algorithms.stream_ac import OBSERVATIONS, StreamAC, StreamACConfig
from memorax.networks import (
    FFN,
    Readout,
    Sequence,
    Tanh,
    backbone,
    heads,
)
from memorax.networks.differentiation import TruncatedBPTT
from memorax.networks.sequence_models.rtu import RTUStructuredRTRL
from memorax.rl import NormalizationConfig, make_bounded_rule, make_optax_rule
from memorax.rl.updates import (
    AdaptiveObBound,
    AdaptiveObBoundFixed,
    ObBound,
    Sgd,
)
from memorax.runtime.rollout import complete_episodes
from tests.support.environments import TinyContinuousEnv


def stream_ac_program(
    observation_normalization=None,
    reward_normalization=None,
    chosen="rtu",
    record=(),
    adaptive=False,
    fixed=False,
    unbounded=False,
    rate=None,
    differentiation="exact_rtrl",
    **overrides,
):
    env = TinyContinuousEnv()

    def network(head):
        sequence = Sequence(
            components=(
                FFN(features=3),
                Tanh(),
                *backbone(chosen, features=3, hidden_dim=2, output_dim=3),
                Readout(module=head),
            )
        )
        selected = (
            RTUStructuredRTRL(sequence.core)
            if chosen == "rtu" and differentiation == "exact_rtrl"
            else TruncatedBPTT(sequence.core)
        )
        return sequence, selected

    bound = None if unbounded else ObBound(kappa=2.0)
    if adaptive:
        maker = AdaptiveObBoundFixed if fixed else AdaptiveObBound
        bound = maker(kappa=2.0, beta2=0.95, eps=1e-6)
    config = StreamACConfig(
        num_envs=1,
        gamma=0.89,
        trace_lambda=0.71,
        actor_bound=bound,
        actor_base=Sgd(lr=0.15 if rate is None else rate),
        critic_bound=bound,
        critic_base=Sgd(lr=0.12 if rate is None else rate),
        entropy_coefficient=0.02,
        **overrides,
    )
    actor_network, actor_differentiation = network(heads.Gaussian(action_dim=2))
    critic_network, critic_differentiation = network(heads.VNetwork())
    agent = StreamAC(
        config,
        env,
        env.default_params,
        actor_network,
        critic_network,
        actor_differentiation,
        critic_differentiation,
        observation_normalization=observation_normalization,
        reward_normalization=reward_normalization,
        record=record,
    )
    return agent.init, agent.train, agent.evaluate


def _readings(found):
    """An evaluation's readings, whether or not it also handed back a state."""

    return found[1] if isinstance(found, tuple) else found


def finite(tree):
    return all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree))


# Every variant, exercised by everything below rather than by training alone.
# Evaluation rebuilds the carry from scratch and a variant that changes what is
# carried can be trainable and unevaluable at once: the truncated credit was,
# and it reached a Batch queue to say so.
PROGRAMS = [
    pytest.param(stream_ac_program, id="stream_ac"),
    pytest.param(
        lambda: stream_ac_program(adaptive=True),
        id="stream_ac_adaptive",
    ),
    pytest.param(
        lambda: stream_ac_program(adaptive=True, fixed=True),
        id="stream_ac_adaptive_fixed",
    ),
    pytest.param(
        lambda: stream_ac_program(differentiation="tbptt"),
        id="stream_ac_truncated",
    ),
    # ``optimizer_bound=none`` is a branch the surface offers, and the rule it
    # builds carries an optimiser state rather than a second moment. At a rate
    # of its own: keeping a single update from crossing the TD target is what
    # the bound does, and without one the rates the bounded variants use here
    # walk this environment's growing reward straight to a non-finite value.
    pytest.param(
        lambda: stream_ac_program(unbounded=True, rate=1e-4),
        id="stream_ac_unbounded",
    ),
    # A torso with nothing to carry, under the credit that expects to carry
    # something: there is no sensitivity, and the kernel should not need one.
    pytest.param(
        lambda: stream_ac_program(chosen="mlp"),
        id="stream_ac_memoryless",
    ),
]


@pytest.mark.parametrize("build", PROGRAMS)
def test_epoch_moves_parameters_and_stays_finite(build):
    init, train, _ = build()
    state = jax.jit(init)(jax.random.key(0))
    trained, metrics = jax.jit(train, static_argnums=2)(jax.random.key(1), state, 8)

    assert finite(metrics), "an epoch produced a non-finite observable"
    assert finite(trained)
    assert int(trained.step) == 8

    before = jax.tree.leaves(state)
    after = jax.tree.leaves(trained)
    assert any(
        not jnp.allclose(old, new)
        for old, new in zip(before, after)
        if jnp.issubdtype(jnp.asarray(old).dtype, jnp.floating)
    ), "training left every parameter untouched"


@pytest.mark.parametrize("build", PROGRAMS)
def test_evaluation_runs_without_training(build):
    init, _, evaluate = build()
    state = jax.jit(init)(jax.random.key(0))
    summary = _readings(
        jax.jit(evaluate, static_argnums=2)(jax.random.key(2), state, 4)
    )

    assert finite(summary)
    assert summary.interaction.info is not None


def test_evaluation_reports_the_reward_the_environment_gave():
    """Reward normalisation is a training device, not a change of task.

    The evaluation summary is what episode returns and the recorded score are
    built from, so a scaled reward there silently rescales the score and makes
    it incomparable to anything recorded before.
    """

    plain = stream_ac_program()
    normalized = stream_ac_program(
        reward_normalization=NormalizationConfig(center=False, discount=0.9)
    )

    # The networks here read observations only, so scaling the reward cannot
    # move the policy: the two rollouts differ in what they report, not in
    # what they do.
    rewards = []
    for init, _, evaluate in (plain, normalized):
        state = jax.jit(init)(jax.random.key(0))
        summary = _readings(
            jax.jit(evaluate, static_argnums=2)(jax.random.key(2), state, 8)
        )
        rewards.append(summary.interaction.reward)

    assert jnp.allclose(rewards[0], rewards[1])


def test_the_two_rules_agree_on_their_contract():
    """Either rule can step the same tree with the same call."""

    traces = {"w": jnp.ones((2, 3))}
    direct = {"w": jnp.full((2, 3), 0.1)}
    delta = jnp.array([0.5, -0.25])
    rules = {
        "adam": make_optax_rule(optax.scale(0.1)),
        "obgd": make_bounded_rule(bound=ObBound(kappa=2.0), base=Sgd(lr=0.1)),
    }
    for name, rule in rules.items():
        state = rule.init(params={"w": jnp.ones((3,))}, traces=traces)
        output = rule.apply(
            traces, direct, state, delta=delta, step=1, params={"w": jnp.ones((3,))}
        )
        assert output.updates["w"].shape == (3,), f"{name} dropped the env axis"
        assert finite(output.updates)


def test_the_bound_never_lets_obgd_step_past_its_learning_rate():
    rule = make_bounded_rule(bound=ObBound(kappa=2.0), base=Sgd(lr=0.5))
    traces = {"w": jnp.full((1, 8), 4.0)}
    state = rule.init(params={"w": jnp.zeros((8,))}, traces=traces)
    output = rule.apply(
        traces,
        None,
        state,
        delta=jnp.array([9.0]),
        step=1,
        params={"w": jnp.zeros((8,))},
    )
    assert float(jnp.max(output.metrics["step_size"])) <= 0.5


def test_the_algorithm_restarts_the_environment_when_an_episode_ends():
    """Nothing below the algorithm resets, so a run that did not would stall.

    The environment hands back the state its episode ended in and keeps handing
    back an ended one; only the act phase can begin the next. Two whole episodes
    of the horizon's length is what restarting looks like, and one long dead one
    is what not restarting looks like.
    """

    horizon = TinyContinuousEnv().default_params.horizon
    init, train, _ = stream_ac_program()
    state = jax.jit(init)(jax.random.key(0))
    _, metrics = jax.jit(train, static_argnums=2)(
        jax.random.key(1), state, 2 * horizon + 2
    )

    episodes = list(
        complete_episodes(
            metrics,
            phase="train",
            start_env_steps=0,
            num_envs=1,
            observations=OBSERVATIONS,
        )
    )
    assert [len(episode.rewards) for episode in episodes] == [horizon, horizon]
