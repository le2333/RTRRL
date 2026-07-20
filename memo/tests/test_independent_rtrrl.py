import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from base.experiment import (  # noqa: E402
    build_independent_rtrrl_agent,
    build_rtrrl_agent,
)


class TinyContinuousEnv:
    def observation_space(self, params):
        return spaces.Box(-10.0, 10.0, (2,), jnp.float32)

    def action_space(self, params):
        return spaces.Box(-1.0, 1.0, (1,), jnp.float32)

    def reset(self, key, params):
        del key, params
        return jnp.zeros((2,), jnp.float32), jnp.asarray(0, jnp.int32)

    def step(self, key, state, action, params):
        del key, params
        next_state = state + 1
        obs = jnp.asarray([next_state, action[0]], jnp.float32)
        reward = jnp.asarray(0.25, jnp.float32)
        done = next_state >= 2
        return obs, next_state, reward, done, {}


def config(**overrides):
    values = dict(
        profile="memo_experimental",
        num_envs=1,
        hidden_dim=2,
        encoder_dim=2,
        meta_rl=True,
        use_encoder=False,
        lru_output_dim=2,
        backbone="lru",
        bound_actor=False,
        pred_obs=False,
        pred_coeff=1.0,
        gamma=0.95,
        lambda_pi=0.97,
        lambda_v=0.9,
        lambda_rnn=0.945,
        td_lr=3e-5,
        rnn_lr=2e-6,
        eta_pi=0.38,
        eta_f=0.5,
        entropy_rate=3e-5,
        update_period=0.1,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        rnn_grad_clip=1.0,
        act_clip=1.0,
        freeze_gamma=False,
        update_trace_before_td=True,
        logprob_reduction="mean",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def assert_tree_allclose(left, right, **kwargs):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
    ):
        np.testing.assert_allclose(left_leaf, right_leaf, **kwargs)


def assert_tree_zeros(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        np.testing.assert_allclose(leaf, jnp.zeros_like(leaf), atol=0, rtol=0)


def assert_any_leaf_differs(left, right):
    pairs = zip(
        jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True
    )
    assert any(not np.allclose(a, b) for a, b in pairs)


def make_agent(**overrides):
    env = TinyContinuousEnv()
    return build_independent_rtrrl_agent(config(**overrides), env, None)


def test_init_has_independent_parameter_state_and_rng():
    agent = make_agent()
    state = agent.init(jax.random.key(0))

    assert agent.actor_feature_extractor is not agent.critic_feature_extractor
    assert agent.actor_torso is not agent.critic_torso
    assert agent.actor_optimizer is not agent.critic_optimizer
    assert_any_leaf_differs(
        state.actor_params["torso"], state.critic_params["torso"]
    )
    assert state.actor_carry is not state.critic_carry
    assert state.actor_sensitivity is not state.critic_sensitivity
    assert state.actor_traces is not state.critic_traces
    assert state.actor_opt_state is not state.critic_opt_state


def _objective_inputs(state):
    return state.timestep.to_sequence()


def test_actor_objective_has_exact_zero_route_to_critic_params():
    agent = make_agent()
    state = agent.init(jax.random.key(1))
    obs, done, action, reward = _objective_inputs(state)
    sampled_action = jnp.zeros((1, 1, 1), jnp.float32)

    def objective(actor_params, critic_params):
        del critic_params
        _, dist = agent._actor_forward(
            agent._grad_params(actor_params, state.actor_slow_torso),
            obs,
            action,
            reward,
            done,
            state.actor_carry,
            state.actor_sensitivity,
        )
        return dist.log_prob(sampled_action).sum()

    _, critic_grad = jax.grad(objective, argnums=(0, 1))(
        state.actor_params, state.critic_params
    )
    assert_tree_zeros(critic_grad)


def test_critic_objective_has_exact_zero_route_to_actor_params():
    agent = make_agent()
    state = agent.init(jax.random.key(2))
    obs, done, action, reward = _objective_inputs(state)

    def objective(actor_params, critic_params):
        del actor_params
        _, value = agent._critic_forward(
            agent._grad_params(critic_params, state.critic_slow_torso),
            obs,
            action,
            reward,
            done,
            state.critic_carry,
            state.critic_sensitivity,
        )
        return value.sum()

    actor_grad, _ = jax.grad(objective, argnums=(0, 1))(
        state.actor_params, state.critic_params
    )
    assert_tree_zeros(actor_grad)


def test_entropy_direct_gradient_updates_actor_only():
    agent = make_agent(update_trace_before_td=False, entropy_rate=1e-2)
    state = agent.init(jax.random.key(3))
    next_state, _ = agent._update_step(state, jax.random.key(4))

    assert_any_leaf_differs(next_state.actor_params, state.actor_params)
    assert_tree_allclose(next_state.critic_params, state.critic_params, atol=0, rtol=0)


def test_bootstrap_does_not_commit_critic_carry():
    agent = make_agent()
    state = agent.init(jax.random.key(5))
    obs, done, action, reward = state.timestep.to_sequence()
    critic_gp = agent._grad_params(state.critic_params, state.critic_slow_torso)
    (expected_carry, expected_sensitivity), _ = agent._critic_forward(
        critic_gp,
        obs,
        action,
        reward,
        done,
        state.critic_carry,
        state.critic_sensitivity,
    )
    next_state, _ = agent._update_step(state, jax.random.key(6))
    assert_tree_allclose(next_state.critic_carry, expected_carry)
    assert_tree_allclose(next_state.critic_sensitivity, expected_sensitivity)


def test_terminal_input_resets_both_recurrent_paths():
    agent = make_agent()
    state = agent.init(jax.random.key(7))
    actor_carry = jax.tree.map(jnp.ones_like, state.actor_carry)
    critic_carry = jax.tree.map(jnp.ones_like, state.critic_carry)
    actor_sens = jax.tree.map(jnp.ones_like, state.actor_sensitivity)
    critic_sens = jax.tree.map(jnp.ones_like, state.critic_sensitivity)
    dirty = state.replace(
        actor_carry=actor_carry,
        critic_carry=critic_carry,
        actor_sensitivity=actor_sens,
        critic_sensitivity=critic_sens,
    )
    clean_next, _ = agent._update_step(state, jax.random.key(8))
    dirty_next, _ = agent._update_step(dirty, jax.random.key(8))
    assert_tree_allclose(dirty_next.actor_carry, clean_next.actor_carry)
    assert_tree_allclose(dirty_next.critic_carry, clean_next.critic_carry)
    assert_tree_allclose(
        dirty_next.actor_sensitivity, clean_next.actor_sensitivity
    )
    assert_tree_allclose(
        dirty_next.critic_sensitivity, clean_next.critic_sensitivity
    )


def test_slow_targets_are_updated_from_their_own_fast_torso():
    agent = make_agent(update_period=0.25)
    state = agent.init(jax.random.key(9))
    next_state, _ = agent._update_step(state, jax.random.key(10))
    expected_actor = jax.tree.map(
        lambda fast, slow: 0.25 * fast + 0.75 * slow,
        next_state.actor_params["torso"],
        state.actor_slow_torso,
    )
    expected_critic = jax.tree.map(
        lambda fast, slow: 0.25 * fast + 0.75 * slow,
        next_state.critic_params["torso"],
        state.critic_slow_torso,
    )
    assert_tree_allclose(next_state.actor_slow_torso, expected_actor)
    assert_tree_allclose(next_state.critic_slow_torso, expected_critic)


def test_short_scan_preserves_state_treedef():
    agent = make_agent()
    state = agent.init(jax.random.key(11))
    trained = jax.jit(agent.train, static_argnums=(2,))(
        jax.random.key(12), state, 2
    )
    assert jax.tree_util.tree_structure(trained) == jax.tree_util.tree_structure(
        state
    )


def test_pred_obs_is_rejected_explicitly():
    env = TinyContinuousEnv()
    with np.testing.assert_raises_regex(ValueError, "does not support pred_obs"):
        build_independent_rtrrl_agent(config(pred_obs=True), env, None)


def test_shared_legacy_agent_still_initializes_and_updates():
    env = TinyContinuousEnv()
    agent = build_rtrrl_agent(config(), env, None)
    state = agent.init(jax.random.key(13))
    next_state, _ = agent._update_step(state, jax.random.key(14))
    assert next_state.update_step == state.update_step + 1
