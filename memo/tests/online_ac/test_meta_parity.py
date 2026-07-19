from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from golden import (
    GoldenSnapshot,
    _rtrrl_observables,
    assert_tree_allclose,
    load_golden,
)

from memorax.networks import heads
from memorax.utils import Timestep
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)


def _make_program(agent):
    from memorax.online_ac.meta import make_meta_program

    return make_meta_program(agent, agent.cfg)


def _tree_at(tree, index):
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _golden_section(snapshot, section):
    prefix = f"{section}/"
    return GoldenSnapshot(
        {
            path[len(prefix) :]: leaf
            for path, leaf in snapshot.leaves.items()
            if path.startswith(prefix)
        }
    )


def _legacy_gradients(agent, state, sampled_action):
    obs, done, action, reward = state.timestep.to_sequence()
    params = agent._grad_params(state.params, state.slow_torso)
    carry = jax.lax.stop_gradient(state.carry)
    sensitivity = jax.lax.stop_gradient(state.sensitivity)

    def traced(p):
        _, (dist, value, _) = agent._forward(
            p, obs, action, reward, done, carry, sensitivity
        )
        log_prob = remove_time_axis(dist.log_prob(add_time_axis(sampled_action)))
        value = remove_feature_axis(remove_time_axis(value))
        return agent.cfg.eta_pi * agent.cfg.logprob_scale * log_prob + value

    def direct(p):
        _, (dist, _, _) = agent._forward(
            p, obs, action, reward, done, carry, sensitivity
        )
        return (
            agent.cfg.entropy_rate
            * agent.cfg.logprob_scale
            * remove_time_axis(dist.entropy())
        )

    return jax.jacobian(traced)(params), jax.jacobian(direct)(params)


def _legacy_one_step_debug(agent, state, key, final_state):
    action_key, step_key = jax.random.split(key)
    obs, done, previous_action, reward = state.timestep.to_sequence()
    params = agent._grad_params(state.params, state.slow_torso)
    (acting_carry, acting_sensitivity), (dist, value_raw, _) = agent._forward(
        params,
        obs,
        previous_action,
        reward,
        done,
        state.carry,
        state.sensitivity,
    )
    sampled_action, _ = dist.sample_and_log_prob(seed=action_key)
    sampled_action = remove_time_axis(sampled_action)
    value = remove_feature_axis(remove_time_axis(value_raw))
    env_action = (
        jnp.clip(sampled_action, -agent.cfg.act_clip, agent.cfg.act_clip)
        if agent.cfg.act_clip
        else sampled_action
    )
    step_keys = jax.random.split(step_key, agent.cfg.num_envs)
    next_obs, _, next_reward, next_done, _ = jax.vmap(
        agent.env.step, in_axes=(0, 0, 0, None)
    )(step_keys, state.env_state, env_action, agent.env_params)
    next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
        obs=next_obs,
        action=env_action,
        reward=next_reward,
        done=next_done,
    ).to_sequence()

    def bootstrap(carry, sensitivity):
        bootstrap_state, (_, next_value_raw, _) = agent._forward(
            jax.lax.stop_gradient(params),
            next_obs_s,
            next_action_s,
            next_reward_s,
            next_done_s,
            jax.lax.stop_gradient(carry),
            jax.lax.stop_gradient(sensitivity),
        )
        return bootstrap_state, remove_feature_axis(remove_time_axis(next_value_raw))

    bootstrap_state, next_value = bootstrap(acting_carry, acting_sensitivity)
    wrong_pre_acting_bootstrap, _ = bootstrap(state.carry, state.sensitivity)
    td_error = next_reward + agent.cfg.gamma * (1 - next_done) * next_value - value
    _, entropy_grads = _legacy_gradients(agent, state, sampled_action)
    update_traces = (
        final_state.traces if agent.cfg.update_trace_before_td else state.traces
    )

    def apply_delta(trace, scale):
        delta = td_error[(slice(None),) + (None,) * (trace.ndim - 1)]
        return scale * delta * trace

    ascent_updates = {}
    for name in state.params:
        scale = agent.cfg.eta_f if name in ("feature_extractor", "torso") else 1.0
        combined = jax.tree.map(
            lambda trace, direct: apply_delta(trace, scale) + direct,
            update_traces[name],
            entropy_grads[name],
        )
        ascent_updates[name] = jax.tree.map(
            lambda update: jnp.mean(update, axis=0), combined
        )
    adam_updates, adam_state = agent.optimizer.update(
        ascent_updates, state.opt_state, state.params
    )
    return {
        "acting_carry": acting_carry,
        "acting_sensitivity": acting_sensitivity,
        "bootstrap_carry": bootstrap_state[0],
        "bootstrap_sensitivity": bootstrap_state[1],
        "wrong_pre_acting_bootstrap": wrong_pre_acting_bootstrap,
        "incoming_traces": state.traces,
        "carried_traces": final_state.traces,
        "update_traces": update_traces,
        "ascent_updates": ascent_updates,
        "adam_updates": adam_updates,
        "adam_state": adam_state,
        "prediction_target": jnp.concatenate(
            [next_obs, jnp.asarray(next_reward, jnp.float32)[..., None]],
            axis=-1,
        ),
    }


def _legacy_prediction_gradients(agent, state, prediction_target):
    obs, done, action, reward = state.timestep.to_sequence()
    params = agent._grad_params(state.params, state.slow_torso)
    carry = jax.lax.stop_gradient(state.carry)
    sensitivity = jax.lax.stop_gradient(state.sensitivity)

    def pred_loss(p):
        _, (_, _, prediction) = agent._forward(
            p, obs, action, reward, done, carry, sensitivity
        )
        prediction = remove_time_axis(prediction)
        error = prediction - jax.lax.stop_gradient(prediction_target)
        return -agent.cfg.pred_coeff * 0.5 * jnp.sum(jnp.square(error), axis=-1)

    return jax.jacobian(pred_loss)(params)


def test_meta_init_matches_legacy_rtrrl_leaf_for_leaf(rtrrl_agent_factory):
    legacy = rtrrl_agent_factory(fresh_trace=False)
    meta = _make_program(rtrrl_agent_factory(fresh_trace=False))
    key = jax.random.fold_in(jax.random.key(7), 0)

    expected = legacy.init(key)
    actual = meta.init_fn(key)
    golden, manifest = load_golden("rtrrl_lru")

    assert manifest["algorithm"] == "rtrrl_lru"
    assert_tree_allclose(
        actual,
        _golden_section(golden["trace_timing/incoming"], "init"),
        rtol=0.0,
        atol=0.0,
    )
    assert_tree_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_meta_init_does_not_mutate_captured_parts(rtrrl_agent_factory):
    parts = rtrrl_agent_factory(fresh_trace=False)
    meta = _make_program(parts)

    assert parts.optimizer is None
    meta.init_fn(jax.random.key(3))
    assert parts.optimizer is None


@pytest.mark.parametrize("fresh_trace", [False, True], ids=["incoming", "fresh"])
def test_meta_one_step_matches_legacy_state_and_action_views(
    rtrrl_agent_factory, fresh_trace
):
    legacy = rtrrl_agent_factory(fresh_trace=fresh_trace)
    parts = rtrrl_agent_factory(fresh_trace=fresh_trace)
    meta = _make_program(parts)
    base_key = jax.random.key(7)
    init_key = jax.random.fold_in(base_key, 0)
    train_key = jax.random.fold_in(base_key, 1)
    expected_initial = legacy.init(init_key)
    actual_initial = meta.init_fn(init_key)
    expected = legacy.train(train_key, expected_initial, num_steps=1)
    actual, metrics = meta.train_epoch_fn(train_key, actual_initial, num_steps=1)

    assert_tree_allclose(actual, expected)
    oracle_key = jax.random.split(train_key, 1)[0]
    observables = _rtrrl_observables(legacy, expected_initial, oracle_key)
    assert_tree_allclose(
        metrics.action_decision.sampled_action[0], observables["sampled_action"]
    )
    assert_tree_allclose(
        metrics.action_decision.logprob_action[0], observables["logprob_action"]
    )
    assert_tree_allclose(
        metrics.action_decision.env_action[0], observables["env_action"]
    )
    assert_tree_allclose(
        metrics.action_decision.bootstrap_feedback_action[0],
        observables["feedback_action"],
    )
    assert_tree_allclose(metrics.value[0], observables["value"])
    assert_tree_allclose(metrics.next_value[0], observables["next_value"])
    assert_tree_allclose(metrics.td_error[0], observables["td"])
    assert_tree_allclose(metrics.log_prob[0], observables["logprob"])
    expected_traced, expected_direct = _legacy_gradients(
        legacy, expected_initial, observables["sampled_action"]
    )
    debug = _legacy_one_step_debug(legacy, expected_initial, oracle_key, expected)
    assert_tree_allclose(_tree_at(metrics.differentiation_grads, 0), expected_traced)
    assert_tree_allclose(_tree_at(metrics.direct_grads, 0), expected_direct)
    assert_tree_allclose(_tree_at(metrics.acting_carry, 0), debug["acting_carry"])
    assert_tree_allclose(
        _tree_at(metrics.acting_sensitivity, 0),
        debug["acting_sensitivity"],
    )
    assert_tree_allclose(
        _tree_at(metrics.bootstrap_carry, 0),
        debug["bootstrap_carry"],
    )
    assert_tree_allclose(
        _tree_at(metrics.bootstrap_sensitivity, 0),
        debug["bootstrap_sensitivity"],
    )
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            _tree_at(metrics.bootstrap_carry, 0),
            debug["wrong_pre_acting_bootstrap"][0],
        )
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            _tree_at(metrics.bootstrap_sensitivity, 0),
            debug["wrong_pre_acting_bootstrap"][1],
        )
    for name in (
        "incoming_traces",
        "carried_traces",
        "update_traces",
        "ascent_updates",
        "adam_updates",
        "adam_state",
    ):
        assert_tree_allclose(_tree_at(getattr(metrics, name), 0), debug[name])
    assert_tree_allclose(_tree_at(metrics.fast_params, 0), actual.params)
    assert_tree_allclose(_tree_at(metrics.slow_torso, 0), actual.slow_torso)

    wrong_sign = jax.tree.map(lambda update: -update, debug["ascent_updates"])
    with pytest.raises(AssertionError):
        assert_tree_allclose(_tree_at(metrics.ascent_updates, 0), wrong_sign)

    if fresh_trace:
        wrong_domain = {
            **debug["ascent_updates"],
            "actor": jax.tree.map(
                lambda update: legacy.cfg.eta_f * update,
                debug["ascent_updates"]["actor"],
            ),
            "torso": jax.tree.map(
                lambda update: update / legacy.cfg.eta_f,
                debug["ascent_updates"]["torso"],
            ),
        }
        with pytest.raises(AssertionError):
            assert_tree_allclose(_tree_at(metrics.ascent_updates, 0), wrong_domain)
        wrong_trace_timing = debug["incoming_traces"]
    else:
        wrong_trace_timing = debug["carried_traces"]
    with pytest.raises(AssertionError):
        assert_tree_allclose(_tree_at(metrics.update_traces, 0), wrong_trace_timing)

    wrong_adam_sign, _ = legacy.optimizer.update(
        wrong_sign,
        expected_initial.opt_state,
        expected_initial.params,
    )
    with pytest.raises(AssertionError):
        assert_tree_allclose(_tree_at(metrics.adam_updates, 0), wrong_adam_sign)


def test_meta_three_steps_match_every_intermediate_state_and_terminal_zeroing(
    rtrrl_agent_factory,
):
    legacy = rtrrl_agent_factory(fresh_trace=False)
    parts = rtrrl_agent_factory(fresh_trace=False)
    meta = _make_program(parts)
    init_key = jax.random.fold_in(jax.random.key(7), 0)
    train_key = jax.random.fold_in(jax.random.key(7), 2)
    expected = legacy.init(init_key)
    actual = meta.init_fn(init_key)
    actual, metrics = meta.train_epoch_fn(train_key, actual, num_steps=3)

    for index, key in enumerate(jax.random.split(train_key, 3)):
        expected, _ = legacy._update_step(expected, key)
        assert_tree_allclose(
            _tree_at(metrics.state_after, index),
            expected,
        )
    assert_tree_allclose(actual, expected)
    np.testing.assert_array_equal(
        metrics.action_decision.persisted_feedback_action[-1],
        np.zeros((1, 2), np.float32),
    )
    np.testing.assert_array_equal(
        metrics.state_after.timestep.reward[-1], np.zeros((1,), np.float32)
    )
    assert bool(metrics.state_after.timestep.done[-1, 0])
    assert not np.allclose(
        metrics.action_decision.bootstrap_feedback_action[-1],
        metrics.action_decision.persisted_feedback_action[-1],
    )


def test_meta_clips_only_environment_and_feedback_action(rtrrl_agent_factory):
    cfg_agent = rtrrl_agent_factory(fresh_trace=False)
    cfg = replace(cfg_agent.cfg, act_clip=0.05)
    legacy = replace(cfg_agent, cfg=cfg)
    parts = replace(rtrrl_agent_factory(fresh_trace=False), cfg=cfg)
    meta = _make_program(parts)
    init_key = jax.random.fold_in(jax.random.key(7), 0)
    train_key = jax.random.fold_in(jax.random.key(7), 1)
    expected = legacy.init(init_key)
    actual = meta.init_fn(init_key)
    expected = legacy.train(train_key, expected, num_steps=1)
    actual, metrics = meta.train_epoch_fn(train_key, actual, num_steps=1)

    assert_tree_allclose(actual, expected)
    sampled = metrics.action_decision.sampled_action[0]
    env_action = metrics.action_decision.env_action[0]
    assert np.max(np.abs(env_action)) <= 0.05
    assert not np.allclose(sampled, env_action)
    assert_tree_allclose(metrics.action_decision.logprob_action[0], sampled)
    assert_tree_allclose(
        metrics.action_decision.bootstrap_feedback_action[0], env_action
    )


def test_meta_prediction_direct_gradient_matches_legacy(rtrrl_agent_factory):
    cfg_agent = rtrrl_agent_factory(fresh_trace=False)
    cfg = replace(cfg_agent.cfg, pred_obs=True, pred_coeff=0.7)
    prediction_head = heads.Regressor(out_dim=3)
    legacy = replace(cfg_agent, cfg=cfg, pred_head=prediction_head)
    parts = replace(
        rtrrl_agent_factory(fresh_trace=False),
        cfg=cfg,
        pred_head=prediction_head,
    )
    meta = _make_program(parts)
    init_key = jax.random.fold_in(jax.random.key(7), 0)
    train_key = jax.random.fold_in(jax.random.key(7), 1)
    expected_initial = legacy.init(init_key)
    actual_initial = meta.init_fn(init_key)
    expected = legacy.train(train_key, expected_initial, num_steps=1)
    actual, metrics = meta.train_epoch_fn(train_key, actual_initial, num_steps=1)

    assert_tree_allclose(actual, expected)
    oracle_key = jax.random.split(train_key, 1)[0]
    debug = _legacy_one_step_debug(legacy, expected_initial, oracle_key, expected)
    expected_prediction = _legacy_prediction_gradients(
        legacy,
        expected_initial,
        debug["prediction_target"],
    )
    actual_prediction = _tree_at(metrics.prediction_direct_grads, 0)
    assert_tree_allclose(actual_prediction, expected_prediction)
    assert any(
        np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(actual_prediction)
    )
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            actual_prediction,
            jax.tree.map(lambda gradient: -gradient, expected_prediction),
        )
