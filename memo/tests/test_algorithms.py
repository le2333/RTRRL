"""Both online kernels run end to end and answer the same contract.

These are wiring tests, not numerical ones. They exist because the kernels are
assembled from a config, a set of modules, and a choice of update rule, and a
mistake in that assembly shows up as a shape error or a silent NaN rather than
as an import failure.

Each builder below hands back the three functions the driver takes, which is
all either kernel has to have in common: one is a class with methods and the
other a record of closures.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pytest
from conftest import TinyContinuousEnv

from memorax.algorithms.rtrrl import RTRRL, RTRRLConfig
from memorax.algorithms.stream_ac_rtrl import StreamACRTRL, StreamACRTRLConfig
from memorax.networks import (
    RNN,
    FeatureExtractor,
    LRUCell,
    LRUConfig,
    Memoroid,
    Network,
    RTUCell,
    RTUConfig,
    heads,
)
from memorax.rl import NormalizationConfig, make_obgd_rule, make_optax_rule


def rtrrl_program(*, record_trajectory=False, **overrides):
    env = TinyContinuousEnv()
    config = RTRRLConfig(
        num_envs=1,
        gamma=0.91,
        lambda_pi=0.73,
        lambda_v=0.67,
        lambda_rnn=0.61,
        td_lr=2e-4,
        rnn_lr=3e-5,
        eta_pi=0.4,
        eta_f=0.6,
        entropy_rate=1e-4,
        update_period=0.2,
        **overrides,
    )
    agent = RTRRL(
        config,
        env,
        env.default_params,
        FeatureExtractor(
            observation_extractor=nn.Sequential((nn.Dense(3), nn.tanh)),
            action_extractor=nn.Sequential((nn.Dense(3), nn.tanh)),
            reward_extractor=nn.Sequential((nn.Dense(3), nn.tanh)),
        ),
        Memoroid(
            cell=LRUCell(config=LRUConfig(features=9, hidden_dim=2, output_dim=3))
        ),
        heads.Gaussian(action_dim=2),
        heads.VNetwork(),
        record_trajectory=record_trajectory,
    )
    return agent.init, agent.train, agent.evaluate


def stream_ac_program(normalization=None, **overrides):
    env = TinyContinuousEnv()

    def network(head):
        return Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=nn.Sequential((nn.Dense(3), nn.tanh))
            ),
            torso=RNN(cell=RTUCell(config=RTUConfig(features=3, hidden_dim=2))),
            head=head,
        )

    config = StreamACRTRLConfig(
        num_envs=1,
        gamma=0.89,
        trace_lambda=0.71,
        actor_lr=0.15,
        critic_lr=0.12,
        entropy_coefficient=0.02,
        beta2=0.95,
        eps=1e-6,
        **overrides,
    )
    agent = StreamACRTRL(
        config,
        env,
        env.default_params,
        network(heads.Gaussian(action_dim=2)),
        network(heads.VNetwork()),
        normalization=normalization,
    )
    return agent.init, agent.train, agent.evaluate


def finite(tree):
    return all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(rtrrl_program, id="rtrrl"),
        pytest.param(stream_ac_program, id="stream_ac_rtrl"),
        pytest.param(
            lambda: rtrrl_program(update_rule="obgd", kappa=2.0),
            id="rtrrl_obgd",
        ),
        pytest.param(
            lambda: stream_ac_program(bounded_rule="adaptive_obgd"),
            id="stream_ac_rtrl_adaptive",
        ),
        pytest.param(
            lambda: stream_ac_program(bounded_rule="adaptive_obgd_fixed"),
            id="stream_ac_rtrl_adaptive_fixed",
        ),
        pytest.param(
            lambda: stream_ac_program(credit="tbptt"),
            id="stream_ac_truncated",
        ),
        pytest.param(
            lambda: rtrrl_program(record_trajectory=True),
            id="rtrrl_trajectory",
        ),
    ],
)
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


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(rtrrl_program, id="rtrrl"),
        pytest.param(stream_ac_program, id="stream_ac_rtrl"),
    ],
)
def test_evaluation_runs_without_training(build):
    init, _, evaluate = build()
    state = jax.jit(init)(jax.random.key(0))
    _, summary = jax.jit(evaluate, static_argnums=2)(jax.random.key(2), state, 4)

    assert finite(summary)
    assert summary.info is not None


def test_evaluation_reports_the_reward_the_environment_gave():
    """Reward normalisation is a training device, not a change of task.

    The evaluation summary is what episode returns and the recorded score are
    built from, so a scaled reward there silently rescales the score and makes
    it incomparable to anything recorded before.
    """

    normalization = NormalizationConfig(normalize_reward=True)
    plain = stream_ac_program()
    normalized = stream_ac_program(normalization=normalization)

    # The networks here read observations only, so scaling the reward cannot
    # move the policy: the two rollouts differ in what they report, not in
    # what they do.
    rewards = []
    for init, _, evaluate in (plain, normalized):
        state = jax.jit(init)(jax.random.key(0))
        _, summary = jax.jit(evaluate, static_argnums=2)(jax.random.key(2), state, 8)
        rewards.append(summary.reward)

    assert jnp.allclose(rewards[0], rewards[1])


def test_closing_both_gates_leaves_the_shared_torso_still():
    """With neither head steering it, nothing reaches the recurrent core.

    This is the ablation's floor. If the torso still moved here, some other
    path would be feeding it and the gates would not mean what they say.
    """

    init, train, _ = rtrrl_program(actor_to_recurrent=False, critic_to_recurrent=False)
    state = jax.jit(init)(jax.random.key(0))
    trained, _ = jax.jit(train, static_argnums=2)(jax.random.key(1), state, 8)

    for name in ("feature_extractor", "torso"):
        for before, after in zip(
            jax.tree.leaves(state.params[name]),
            jax.tree.leaves(trained.params[name]),
        ):
            assert jnp.allclose(before, after), f"{name} moved with both gates shut"

    assert any(
        not jnp.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(state.params["actor"]),
            jax.tree.leaves(trained.params["actor"]),
        )
    ), "the actor should still learn from a frozen representation"


def test_one_gate_still_reaches_the_shared_torso():
    for gates in ({"critic_to_recurrent": False}, {"actor_to_recurrent": False}):
        init, train, _ = rtrrl_program(**gates)
        state = jax.jit(init)(jax.random.key(0))
        trained, _ = jax.jit(train, static_argnums=2)(jax.random.key(1), state, 8)
        assert any(
            not jnp.allclose(before, after)
            for before, after in zip(
                jax.tree.leaves(state.params["torso"]),
                jax.tree.leaves(trained.params["torso"]),
            )
        ), f"the torso stopped learning with {gates}"


def test_the_two_rules_agree_on_their_contract():
    """Either rule can step the same tree with the same call."""

    traces = {"w": jnp.ones((2, 3))}
    direct = {"w": jnp.full((2, 3), 0.1)}
    delta = jnp.array([0.5, -0.25])
    rules = {
        "adam": make_optax_rule(optax.scale(0.1)),
        "obgd": make_obgd_rule(learning_rate=0.1, kappa=2.0),
    }
    for name, rule in rules.items():
        state = rule.init(params={"w": jnp.ones((3,))}, traces=traces)
        output = rule.apply(
            traces, direct, state, delta=delta, step=1, params={"w": jnp.ones((3,))}
        )
        assert output.updates["w"].shape == (3,), f"{name} dropped the env axis"
        assert finite(output.updates)


def test_the_bound_never_lets_obgd_step_past_its_learning_rate():
    rule = make_obgd_rule(learning_rate=0.5, kappa=2.0)
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
