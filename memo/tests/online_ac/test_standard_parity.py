from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TinyContinuousEnv
from golden import GoldenSnapshot, assert_tree_allclose, load_golden

from memorax.utils import Timestep
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)


@dataclass(frozen=True)
class ClippedContinuousEnv(TinyContinuousEnv):
    clip: float = 0.05

    def step(self, key, state, action, params):
        return super().step(
            key,
            state,
            jnp.clip(action, -self.clip, self.clip),
            params,
        )


def _golden_section(snapshot, section):
    prefix = f"{section}/"
    return GoldenSnapshot(
        {
            path[len(prefix) :]: leaf
            for path, leaf in snapshot.leaves.items()
            if path.startswith(prefix)
        }
    )


def _tree_at(tree, index):
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _materialize_tree(tree):
    return jax.tree.map(
        lambda leaf: jax.block_until_ready(jnp.array(leaf)),
        tree,
    )


def _capture_legacy_step(agent, state, key):
    captured = []
    original_obgd = agent._obgd_update
    had_instance_override = "_obgd_update" in vars(agent)
    original_instance_override = vars(agent).get("_obgd_update")
    original_forward = agent._rtrl_forward
    had_forward_override = "_rtrl_forward" in vars(agent)
    original_forward_override = vars(agent).get("_rtrl_forward")
    forward_views = {}
    critic_call_count = 0

    def capture_forward(network, *args, **kwargs):
        nonlocal critic_call_count
        result = original_forward(network, *args, **kwargs)
        if network is agent.actor_network and "sampled_action" not in forward_views:
            dist = result[1][0]
            action_key = jax.random.split(key)[0]
            sampled_action, log_prob = dist.sample_and_log_prob(seed=action_key)
            forward_views.update(
                _materialize_tree(
                    {
                        "sampled_action": remove_time_axis(sampled_action),
                        "log_prob": remove_time_axis(log_prob),
                        "actor_carry": result[0][0],
                        "actor_sensitivity": result[0][1],
                    }
                )
            )
        elif network is agent.critic_network and critic_call_count < 2:
            name = "value" if critic_call_count == 0 else "next_value"
            forward_views[name] = _materialize_tree(
                remove_feature_axis(remove_time_axis(result[1][0]))
            )
            if critic_call_count < 2:
                (
                    params,
                    obs,
                    action,
                    reward,
                    done,
                    carry,
                    sensitivity,
                ) = args
                input_name = (
                    "critic_inputs" if critic_call_count == 0 else "bootstrap_inputs"
                )
                forward_views[input_name] = _materialize_tree(
                    {
                        "params": params,
                        "obs": obs,
                        "action": action,
                        "reward": reward,
                        "done": done,
                        "carry": carry,
                        "sensitivity": sensitivity,
                    }
                )
            critic_call_count += 1
        return result

    def capture_obgd(traces, v, td, learning_rate, kappa, current_step):
        inputs = _materialize_tree(
            {
                "traces": traces,
                "v": v,
                "td_error": td,
                "learning_rate": learning_rate,
                "kappa": kappa,
                "current_step": current_step,
            }
        )
        updates, new_v = original_obgd(
            traces,
            v,
            td,
            learning_rate,
            kappa,
            current_step,
        )
        outputs = _materialize_tree(
            {
                "updates": updates,
                "new_v": new_v,
            }
        )
        captured.append(
            {
                "position": len(captured),
                "inputs": inputs,
                "outputs": outputs,
            }
        )
        return updates, new_v

    agent._rtrl_forward = capture_forward
    agent._obgd_update = capture_obgd
    try:
        live_state, _ = agent._update_step(state, key)
        live_state = _materialize_tree(live_state)
    finally:
        if had_instance_override:
            agent._obgd_update = original_instance_override
        else:
            delattr(agent, "_obgd_update")
        if had_forward_override:
            agent._rtrl_forward = original_forward_override
        else:
            delattr(agent, "_rtrl_forward")

    assert len(captured) == 2
    captured[0]["domain"] = "critic"
    captured[1]["domain"] = "actor"
    _, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, agent.cfg.num_envs)
    next_obs, env_state, next_reward, next_done, info = jax.vmap(
        agent.env.step,
        in_axes=(0, 0, 0, None),
    )(
        step_keys,
        state.env_state,
        forward_views["sampled_action"],
        agent.env_params,
    )
    forward_views.update(
        _materialize_tree(
            {
                "next_obs": next_obs,
                "env_state": env_state,
                "next_reward": next_reward,
                "next_done": next_done,
                "info": info,
            }
        )
    )
    forward_views["reconstructed_td_error"] = _materialize_tree(
        forward_views["next_reward"]
        + agent.cfg.gamma
        * (1 - forward_views["next_done"])
        * forward_views["next_value"]
        - forward_views["value"]
    )
    return live_state, captured, forward_views


def _legacy_gradients(agent, state, sampled_action, td_error):
    obs, done, action, reward = state.timestep.to_sequence()
    actor_carry = jax.lax.stop_gradient(state.actor_carry)
    actor_sensitivity = jax.lax.stop_gradient(state.actor_sensitivity)
    critic_carry = jax.lax.stop_gradient(state.critic_carry)
    critic_sensitivity = jax.lax.stop_gradient(state.critic_sensitivity)

    def actor_direction(params):
        _, (dist, _) = agent._rtrl_forward(
            agent.actor_network,
            params,
            obs,
            action,
            reward,
            done,
            actor_carry,
            actor_sensitivity,
        )
        log_prob = remove_time_axis(dist.log_prob(add_time_axis(sampled_action)))
        entropy = remove_time_axis(dist.entropy())
        return log_prob + agent.cfg.entropy_coefficient * jnp.sign(td_error) * entropy

    def critic_direction(params):
        _, (value, _) = agent._rtrl_forward(
            agent.critic_network,
            params,
            obs,
            action,
            reward,
            done,
            critic_carry,
            critic_sensitivity,
        )
        return remove_feature_axis(remove_time_axis(value))

    return (
        jax.jacobian(actor_direction)(state.actor_params),
        jax.jacobian(critic_direction)(state.critic_params),
    )


def _legacy_actor_gradient_with_mean_entropy(
    agent,
    state,
    sampled_action,
    td_error,
):
    obs, done, action, reward = state.timestep.to_sequence()
    carry = jax.lax.stop_gradient(state.actor_carry)
    sensitivity = jax.lax.stop_gradient(state.actor_sensitivity)

    def actor_direction(params):
        _, (dist, _) = agent._rtrl_forward(
            agent.actor_network,
            params,
            obs,
            action,
            reward,
            done,
            carry,
            sensitivity,
        )
        log_prob = remove_time_axis(dist.log_prob(add_time_axis(sampled_action)))
        mean_entropy = remove_time_axis(dist.entropy()).mean()
        return (
            log_prob
            + agent.cfg.entropy_coefficient
            * jnp.sign(jax.lax.stop_gradient(td_error))
            * mean_entropy
        )

    return jax.jacobian(actor_direction)(state.actor_params)


def _legacy_one_step_debug(agent, state, key):
    action_key, step_key = jax.random.split(key)
    obs, done, previous_action, reward = state.timestep.to_sequence()
    (actor_carry, actor_sensitivity), (dist, _) = agent._rtrl_forward(
        agent.actor_network,
        state.actor_params,
        obs,
        previous_action,
        reward,
        done,
        state.actor_carry,
        state.actor_sensitivity,
    )
    sampled_action, log_prob = dist.sample_and_log_prob(seed=action_key)
    entropy = remove_time_axis(dist.entropy()).mean()
    sampled_action = remove_time_axis(sampled_action)
    log_prob = remove_time_axis(log_prob)
    (critic_carry, critic_sensitivity), (value_raw, _) = agent._rtrl_forward(
        agent.critic_network,
        state.critic_params,
        obs,
        previous_action,
        reward,
        done,
        state.critic_carry,
        state.critic_sensitivity,
    )
    value = remove_feature_axis(remove_time_axis(value_raw))
    step_keys = jax.random.split(step_key, agent.cfg.num_envs)
    next_obs, env_state, next_reward, next_done, info = jax.vmap(
        agent.env.step, in_axes=(0, 0, 0, None)
    )(step_keys, state.env_state, sampled_action, agent.env_params)
    next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
        obs=next_obs,
        action=sampled_action,
        reward=next_reward,
        done=next_done,
    ).to_sequence()

    def bootstrap(carry, sensitivity):
        bootstrap_state, (next_value_raw, _) = agent._rtrl_forward(
            agent.critic_network,
            jax.lax.stop_gradient(state.critic_params),
            next_obs_s,
            next_action_s,
            next_reward_s,
            next_done_s,
            jax.lax.stop_gradient(carry),
            jax.lax.stop_gradient(sensitivity),
        )
        return bootstrap_state, remove_feature_axis(remove_time_axis(next_value_raw))

    bootstrap_state, next_value = bootstrap(
        critic_carry,
        critic_sensitivity,
    )
    wrong_bootstrap_state, _ = bootstrap(
        state.critic_carry,
        state.critic_sensitivity,
    )
    td_error = next_reward + agent.cfg.gamma * (1 - next_done) * next_value - value
    actor_grads, critic_grads = _legacy_gradients(
        agent,
        state,
        sampled_action,
        td_error,
    )
    decay = agent.cfg.gamma * agent.cfg.trace_lambda

    def fresh_trace(old, gradient):
        reset = state.timestep.done[(slice(None),) + (None,) * (old.ndim - 1)]
        return decay * (1 - reset) * old + gradient

    actor_traces = jax.tree.map(
        fresh_trace,
        state.actor_traces,
        actor_grads,
    )
    critic_traces = jax.tree.map(
        fresh_trace,
        state.critic_traces,
        critic_grads,
    )
    step = state.update_step + 1
    critic_updates, critic_v = agent._obgd_update(
        critic_traces,
        state.critic_v,
        td_error,
        agent.cfg.critic_lr,
        agent.cfg.critic_kappa,
        step,
    )
    actor_updates, actor_v = agent._obgd_update(
        actor_traces,
        state.actor_v,
        td_error,
        agent.cfg.actor_lr,
        agent.cfg.actor_kappa,
        step,
    )

    def obgd_step_size(traces, v, learning_rate, kappa):
        if agent.cfg.adaptive:
            v_hat = jax.tree.map(
                lambda moment: moment / (1 - agent.cfg.beta2**step),
                v,
            )
            normalized = jax.tree.map(
                lambda trace, moment: (
                    jnp.abs(trace) / (jnp.sqrt(moment) + agent.cfg.eps)
                ),
                traces,
                v_hat,
            )
            leaves = jax.tree.leaves(normalized)
        else:
            leaves = jax.tree.leaves(traces)
        trace_sum = sum(
            jnp.sum(jnp.abs(leaf), axis=tuple(range(1, leaf.ndim))) for leaf in leaves
        )
        return learning_rate / jnp.maximum(
            1.0,
            jnp.maximum(jnp.abs(td_error), 1.0) * trace_sum * learning_rate * kappa,
        )

    return {
        "sampled_action": sampled_action,
        "log_prob": log_prob,
        "entropy": entropy,
        "value": value,
        "next_value": next_value,
        "td_error": td_error,
        "next_obs": next_obs,
        "env_state": env_state,
        "next_reward": next_reward,
        "next_done": next_done,
        "info": info,
        "actor_carry": actor_carry,
        "actor_sensitivity": actor_sensitivity,
        "critic_carry": critic_carry,
        "critic_sensitivity": critic_sensitivity,
        "bootstrap_carry": bootstrap_state[0],
        "bootstrap_sensitivity": bootstrap_state[1],
        "wrong_bootstrap_carry": wrong_bootstrap_state[0],
        "wrong_bootstrap_sensitivity": wrong_bootstrap_state[1],
        "actor_grads": actor_grads,
        "critic_grads": critic_grads,
        "actor_traces": actor_traces,
        "critic_traces": critic_traces,
        "actor_updates": actor_updates,
        "critic_updates": critic_updates,
        "actor_v": actor_v,
        "critic_v": critic_v,
        "actor_step_size": obgd_step_size(
            actor_traces,
            actor_v,
            agent.cfg.actor_lr,
            agent.cfg.actor_kappa,
        ),
        "critic_step_size": obgd_step_size(
            critic_traces,
            critic_v,
            agent.cfg.critic_lr,
            agent.cfg.critic_kappa,
        ),
    }


def test_standard_init_matches_legacy_and_golden_leaf_for_leaf(
    stream_ac_agent_factory,
):
    from memorax.online_ac.standard import make_standard_program

    legacy = stream_ac_agent_factory(adaptive=False)
    program = make_standard_program(
        stream_ac_agent_factory(adaptive=False),
        legacy.cfg,
    )
    key = jax.random.fold_in(jax.random.key(7), 0)

    expected = legacy.init(key)
    actual = program.init_fn(key)
    golden, manifest = load_golden("stream_ac_rtu")

    assert manifest["algorithm"] == "stream_ac_rtu"
    assert_tree_allclose(
        actual,
        _golden_section(golden["obgd/non_adaptive"], "init"),
        rtol=0.0,
        atol=0.0,
    )
    assert_tree_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert actual.actor_params is not actual.critic_params
    assert actual.actor_carry is not actual.critic_carry
    assert actual.actor_sensitivity is not actual.critic_sensitivity
    assert actual.actor_traces is not actual.critic_traces
    assert actual.actor_v is not actual.critic_v


def test_live_legacy_obgd_capture_matches_manual_debug_inputs(
    stream_ac_agent_factory,
):
    legacy = stream_ac_agent_factory(adaptive=False)
    base_key = jax.random.key(7)
    initial = legacy.init(jax.random.fold_in(base_key, 0))
    key = jax.random.split(jax.random.fold_in(base_key, 1), 1)[0]
    before = {name: id(value) for name, value in vars(legacy).items()}

    _, captures, live_views = _capture_legacy_step(legacy, initial, key)
    assert {name: id(value) for name, value in vars(legacy).items()} == before
    manual = _legacy_one_step_debug(legacy, initial, key)

    for field in (
        "sampled_action",
        "log_prob",
        "actor_carry",
        "actor_sensitivity",
        "value",
        "env_state",
        "next_reward",
        "next_done",
        "info",
    ):
        expected_field = (
            manual["td_error"] if field == "reconstructed_td_error" else manual[field]
        )
        try:
            assert_tree_allclose(
                live_views[field],
                expected_field,
                rtol=0.0,
                atol=0.0,
            )
        except AssertionError as error:
            raise AssertionError(f"live forward mismatch in {field}") from error
    obs, done, action, reward = initial.timestep.to_sequence()
    expected_critic_inputs = {
        "params": initial.critic_params,
        "obs": obs,
        "action": action,
        "reward": reward,
        "done": done,
        "carry": initial.critic_carry,
        "sensitivity": initial.critic_sensitivity,
    }
    for field, value in expected_critic_inputs.items():
        try:
            assert_tree_allclose(
                live_views["critic_inputs"][field],
                value,
                rtol=0.0,
                atol=0.0,
            )
        except AssertionError as error:
            raise AssertionError(f"live critic input mismatch in {field}") from error
    next_obs, next_done, next_action, next_reward = Timestep(
        obs=manual["next_obs"],
        action=manual["sampled_action"],
        reward=manual["next_reward"],
        done=manual["next_done"],
    ).to_sequence()
    expected_bootstrap_inputs = {
        "params": initial.critic_params,
        "obs": next_obs,
        "action": next_action,
        "reward": next_reward,
        "done": next_done,
        "carry": manual["critic_carry"],
        "sensitivity": manual["critic_sensitivity"],
    }
    for field, value in expected_bootstrap_inputs.items():
        try:
            assert_tree_allclose(
                live_views["bootstrap_inputs"][field],
                value,
                rtol=0.0,
                atol=0.0,
            )
        except AssertionError as error:
            raise AssertionError(f"live bootstrap input mismatch in {field}") from error
    assert_tree_allclose(
        live_views["next_value"],
        manual["next_value"],
        rtol=0.0,
        atol=0.0,
    )
    assert_tree_allclose(
        live_views["reconstructed_td_error"],
        captures[0]["inputs"]["td_error"],
        rtol=0.0,
        atol=0.0,
    )

    for capture, domain in zip(captures, ("critic", "actor"), strict=True):
        assert capture["domain"] == domain
        expected = _materialize_tree(
            {
                "traces": manual[f"{domain}_traces"],
                "v": getattr(initial, f"{domain}_v"),
                "td_error": manual["td_error"],
                "learning_rate": getattr(legacy.cfg, f"{domain}_lr"),
                "kappa": getattr(legacy.cfg, f"{domain}_kappa"),
                "current_step": initial.update_step + 1,
            }
        )
        for field, value in expected.items():
            assert_tree_allclose(
                capture["inputs"][field],
                value,
                rtol=0.0,
                atol=0.0,
            )
        assert_tree_allclose(
            capture["outputs"]["updates"],
            manual[f"{domain}_updates"],
            rtol=0.0,
            atol=0.0,
        )
        assert_tree_allclose(
            capture["outputs"]["new_v"],
            manual[f"{domain}_v"],
            rtol=0.0,
            atol=0.0,
        )


def test_standard_one_step_matches_legacy_every_exposed_leaf(
    stream_ac_agent_factory,
):
    from memorax.online_ac.standard import make_standard_program

    legacy = stream_ac_agent_factory(adaptive=False)
    debug_sink = []
    program = make_standard_program(
        stream_ac_agent_factory(adaptive=False),
        legacy.cfg,
        _debug_sink=debug_sink,
    )
    base_key = jax.random.key(7)
    init_key = jax.random.fold_in(base_key, 0)
    train_key = jax.random.fold_in(base_key, 1)
    expected_initial = legacy.init(init_key)
    actual_initial = program.init_fn(init_key)
    oracle_key = jax.random.split(train_key, 1)[0]
    agent_attributes = {name: id(value) for name, value in vars(legacy).items()}
    expected, captures, _ = _capture_legacy_step(
        legacy,
        expected_initial,
        oracle_key,
    )
    assert {name: id(value) for name, value in vars(legacy).items()} == agent_attributes
    actual, step_metrics = debug_sink[0].step(actual_initial, oracle_key)
    metrics = jax.tree.map(
        lambda leaf: jnp.expand_dims(leaf, axis=0),
        step_metrics,
    )

    debug = _legacy_one_step_debug(legacy, expected_initial, oracle_key)
    expected_capture_values = {
        "critic": {
            "traces": debug["critic_traces"],
            "v": expected_initial.critic_v,
            "learning_rate": legacy.cfg.critic_lr,
            "kappa": legacy.cfg.critic_kappa,
            "updates": debug["critic_updates"],
            "new_v": debug["critic_v"],
        },
        "actor": {
            "traces": debug["actor_traces"],
            "v": expected_initial.actor_v,
            "learning_rate": legacy.cfg.actor_lr,
            "kappa": legacy.cfg.actor_kappa,
            "updates": debug["actor_updates"],
            "new_v": debug["actor_v"],
        },
    }
    metric_capture_values = {
        "critic": {
            "traces": _tree_at(metrics.update_critic_traces, 0),
            "v": actual_initial.critic_v,
            "updates": _tree_at(metrics.critic_updates, 0),
            "new_v": _tree_at(metrics.critic_v, 0),
        },
        "actor": {
            "traces": _tree_at(metrics.update_actor_traces, 0),
            "v": actual_initial.actor_v,
            "updates": _tree_at(metrics.actor_updates, 0),
            "new_v": _tree_at(metrics.actor_v, 0),
        },
    }
    for position, capture in enumerate(captures):
        domain = capture["domain"]
        assert capture["position"] == position
        expected_domain = expected_capture_values[domain]
        metric_domain = metric_capture_values[domain]
        for field in ("traces", "v"):
            assert_tree_allclose(
                capture["inputs"][field],
                expected_domain[field],
                rtol=0.0,
                atol=0.0,
            )
            assert_tree_allclose(
                capture["inputs"][field],
                metric_domain[field],
                rtol=0.0,
                atol=0.0,
            )
        assert_tree_allclose(
            capture["inputs"]["td_error"],
            debug["td_error"],
            rtol=0.0,
            atol=0.0,
        )
        assert_tree_allclose(
            capture["inputs"]["td_error"],
            _tree_at(metrics.td_error, 0),
            rtol=0.0,
            atol=0.0,
        )
        for field in ("learning_rate", "kappa"):
            assert_tree_allclose(
                capture["inputs"][field],
                jnp.asarray(expected_domain[field]),
                rtol=0.0,
                atol=0.0,
            )
        assert_tree_allclose(
            capture["inputs"]["current_step"],
            jnp.asarray(expected_initial.update_step + 1),
            rtol=0.0,
            atol=0.0,
        )
        for field in ("updates", "new_v"):
            assert_tree_allclose(
                capture["outputs"][field],
                expected_domain[field],
                rtol=0.0,
                atol=0.0,
            )
            assert_tree_allclose(
                capture["outputs"][field],
                metric_domain[field],
                rtol=0.0,
                atol=0.0,
            )
    assert_tree_allclose(
        metrics.action_decision.sampled_action[0],
        debug["sampled_action"],
        rtol=0.0,
        atol=0.0,
    )
    for field in (
        "log_prob",
        "entropy",
        "value",
        "next_value",
        "td_error",
        "actor_carry",
        "actor_sensitivity",
        "critic_carry",
        "critic_sensitivity",
        "bootstrap_carry",
        "bootstrap_sensitivity",
        "actor_grads",
        "critic_grads",
        "actor_step_size",
        "critic_step_size",
        "actor_updates",
        "critic_updates",
        "actor_v",
        "critic_v",
        "info",
    ):
        try:
            assert_tree_allclose(
                _tree_at(getattr(metrics, field), 0),
                debug[field],
                rtol=0.0,
                atol=0.0,
            )
        except AssertionError as error:
            raise AssertionError(f"mismatch in {field}") from error
    for field, state_field in (
        ("incoming_actor_traces", "actor_traces"),
        ("incoming_critic_traces", "critic_traces"),
    ):
        assert_tree_allclose(
            _tree_at(getattr(metrics, field), 0),
            getattr(expected_initial, state_field),
            rtol=0.0,
            atol=0.0,
        )
    for field in ("actor_traces", "critic_traces"):
        assert_tree_allclose(
            _tree_at(getattr(metrics, field), 0),
            debug[field],
            rtol=0.0,
            atol=0.0,
        )
        assert_tree_allclose(
            _tree_at(getattr(metrics, f"update_{field}"), 0),
            debug[field],
            rtol=0.0,
            atol=0.0,
        )
    assert_tree_allclose(
        _tree_at(metrics.actor_params, 0),
        actual.actor_params,
        rtol=0.0,
        atol=0.0,
    )
    assert_tree_allclose(
        _tree_at(metrics.critic_params, 0),
        actual.critic_params,
        rtol=0.0,
        atol=0.0,
    )
    assert_tree_allclose(
        _tree_at(metrics.state_after, 0),
        actual,
        rtol=0.0,
        atol=0.0,
    )
    assert_tree_allclose(
        metrics.action_decision.logprob_action[0],
        debug["sampled_action"],
        rtol=0.0,
        atol=0.0,
    )
    assert_tree_allclose(
        metrics.action_decision.env_action[0],
        debug["sampled_action"],
        rtol=0.0,
        atol=0.0,
    )
    assert_tree_allclose(
        metrics.action_decision.bootstrap_feedback_action[0],
        debug["sampled_action"],
        rtol=0.0,
        atol=0.0,
    )
    assert any(
        np.any(np.asarray(moment) != 0)
        for moment in jax.tree.leaves(_tree_at(metrics.actor_v, 0))
    )
    assert_tree_allclose(
        _tree_at(metrics.actor_updates, 0),
        jax.tree.map(
            lambda after, before: after - before,
            expected.actor_params,
            expected_initial.actor_params,
        ),
    )
    assert_tree_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_standard_continuous_wrapper_clip_does_not_replace_feedback(
    stream_ac_agent_factory,
):
    from memorax.online_ac.standard import make_standard_program

    env = ClippedContinuousEnv()
    legacy = stream_ac_agent_factory(
        adaptive=True,
        continuous=True,
        env=env,
    )
    program = make_standard_program(
        stream_ac_agent_factory(
            adaptive=True,
            continuous=True,
            env=env,
        ),
        legacy.cfg,
    )
    base_key = jax.random.key(17)
    initial_key = jax.random.fold_in(base_key, 0)
    train_key = jax.random.fold_in(base_key, 1)
    expected_initial = legacy.init(initial_key)
    actual_initial = program.init_fn(initial_key)
    step_key = jax.random.split(train_key, 1)[0]

    expected, _ = legacy._update_step(expected_initial, step_key)
    actual, metrics = program.train_epoch_fn(train_key, actual_initial, num_steps=1)

    assert_tree_allclose(actual, expected)
    sampled = metrics.action_decision.sampled_action[0]
    assert np.max(np.abs(np.asarray(sampled))) > env.clip
    assert_tree_allclose(metrics.action_decision.logprob_action[0], sampled)
    assert_tree_allclose(metrics.action_decision.env_action[0], sampled)
    assert_tree_allclose(
        metrics.action_decision.bootstrap_feedback_action[0],
        sampled,
    )
    expected_observation = (
        actual_initial.timestep.obs
        + jnp.clip(sampled, -env.clip, env.clip)
        + jnp.array([0.15, 0.45], dtype=jnp.float32)
    )
    assert_tree_allclose(actual.timestep.obs, expected_observation)
    assert_tree_allclose(actual.timestep.action, sampled)


def test_standard_fresh_trace_resets_nonzero_incoming_at_initial_boundary(
    stream_ac_agent_factory,
):
    from memorax.online_ac.standard import make_standard_program

    legacy = stream_ac_agent_factory(adaptive=False)
    program = make_standard_program(
        stream_ac_agent_factory(adaptive=False),
        legacy.cfg,
    )
    initial_key = jax.random.fold_in(jax.random.key(23), 0)
    train_key = jax.random.fold_in(jax.random.key(23), 1)
    expected_initial = legacy.init(initial_key)
    actual_initial = program.init_fn(initial_key)
    actor_ones = jax.tree.map(jnp.ones_like, actual_initial.actor_traces)
    critic_ones = jax.tree.map(jnp.ones_like, actual_initial.critic_traces)
    expected_initial = expected_initial.replace(
        actor_traces=actor_ones,
        critic_traces=critic_ones,
    )
    actual_initial = actual_initial.replace(
        actor_traces=actor_ones,
        critic_traces=critic_ones,
    )
    step_key = jax.random.split(train_key, 1)[0]

    expected, _ = legacy._update_step(expected_initial, step_key)
    actual, metrics = program.train_epoch_fn(train_key, actual_initial, num_steps=1)

    assert_tree_allclose(actual, expected)
    actor_grads = _tree_at(metrics.actor_grads, 0)
    critic_grads = _tree_at(metrics.critic_grads, 0)
    assert_tree_allclose(_tree_at(metrics.actor_traces, 0), actor_grads)
    assert_tree_allclose(_tree_at(metrics.critic_traces, 0), critic_grads)
    decay = legacy.cfg.gamma * legacy.cfg.trace_lambda
    wrong_actor = jax.tree.map(
        lambda old, gradient: decay * old + gradient,
        actor_ones,
        actor_grads,
    )
    with pytest.raises(AssertionError):
        assert_tree_allclose(_tree_at(metrics.actor_traces, 0), wrong_actor)


def test_standard_two_envs_rejects_entropy_mean_obgd_mean_first_and_wrong_domain(
    stream_ac_agent_factory,
):
    from memorax.online_ac.standard import make_standard_program
    from memorax.online_ac.updates import make_whole_tree_obgd

    legacy = stream_ac_agent_factory(adaptive=False, num_envs=2)
    program = make_standard_program(
        stream_ac_agent_factory(adaptive=False, num_envs=2),
        legacy.cfg,
    )
    initial_key = jax.random.fold_in(jax.random.key(27), 0)
    train_key = jax.random.fold_in(jax.random.key(27), 1)
    expected_initial = legacy.init(initial_key)
    actual_initial = program.init_fn(initial_key)
    distinct_obs = expected_initial.timestep.obs.at[1].add(
        jnp.array([0.3, -0.2], dtype=jnp.float32)
    )
    expected_initial = expected_initial.replace(
        timestep=expected_initial.timestep.replace(obs=distinct_obs),
        env_state=expected_initial.env_state.replace(observation=distinct_obs),
    )
    actual_initial = actual_initial.replace(
        timestep=actual_initial.timestep.replace(obs=distinct_obs),
        env_state=actual_initial.env_state.replace(observation=distinct_obs),
    )
    step_key = jax.random.split(train_key, 1)[0]
    debug = _legacy_one_step_debug(legacy, expected_initial, step_key)

    _, metrics = program.train_epoch_fn(train_key, actual_initial, num_steps=2)
    actor_grads = _tree_at(metrics.actor_grads, 0)
    assert_tree_allclose(actor_grads, debug["actor_grads"])
    wrong_entropy = _legacy_actor_gradient_with_mean_entropy(
        legacy,
        expected_initial,
        debug["sampled_action"],
        debug["td_error"],
    )
    with pytest.raises(AssertionError):
        assert_tree_allclose(actor_grads, wrong_entropy)

    traces = _tree_at(metrics.update_actor_traces, 0)
    delta = _tree_at(metrics.td_error, 0)
    mean_delta = jnp.mean(delta)
    mean_traces = jax.tree.map(lambda trace: jnp.mean(trace, axis=0), traces)
    trace_sum = sum(jnp.sum(jnp.abs(leaf)) for leaf in jax.tree.leaves(mean_traces))
    mean_first_step = legacy.cfg.actor_lr / jnp.maximum(
        1.0,
        jnp.maximum(jnp.abs(mean_delta), 1.0)
        * trace_sum
        * legacy.cfg.actor_lr
        * legacy.cfg.actor_kappa,
    )
    mean_first_updates = jax.tree.map(
        lambda trace: mean_first_step * mean_delta * trace,
        mean_traces,
    )
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            _tree_at(metrics.actor_updates, 0),
            mean_first_updates,
        )

    wrong_domain_updates, _ = make_whole_tree_obgd(legacy.cfg)(
        traces,
        actual_initial.actor_v,
        delta=delta,
        learning_rate=legacy.cfg.critic_lr,
        kappa=legacy.cfg.critic_kappa,
        step=1,
    )
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            _tree_at(metrics.actor_updates, 0),
            wrong_domain_updates,
        )


def test_standard_three_steps_match_each_state_and_terminal_zeroing(
    stream_ac_agent_factory,
):
    from memorax.online_ac.standard import make_standard_program

    legacy = stream_ac_agent_factory(adaptive=False)
    program = make_standard_program(
        stream_ac_agent_factory(adaptive=False),
        legacy.cfg,
    )
    initial_key = jax.random.fold_in(jax.random.key(29), 0)
    train_key = jax.random.fold_in(jax.random.key(29), 2)
    expected = legacy.init(initial_key)
    actual = program.init_fn(initial_key)

    actual, metrics = program.train_epoch_fn(train_key, actual, num_steps=3)
    keys = jax.random.split(train_key, 3)
    for index, key in enumerate(keys):
        expected, _ = legacy._update_step(expected, key)
        assert_tree_allclose(_tree_at(metrics.state_after, index), expected)
    assert_tree_allclose(actual, expected)
    assert bool(metrics.state_after.timestep.done[-1, 0])
    np.testing.assert_array_equal(
        metrics.action_decision.persisted_feedback_action[-1],
        np.zeros((1,), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        metrics.state_after.timestep.reward[-1],
        np.zeros((1,), dtype=np.float32),
    )
    assert_tree_allclose(
        metrics.action_decision.bootstrap_feedback_action[-1],
        metrics.action_decision.sampled_action[-1],
    )


def test_standard_second_step_uses_post_acting_bootstrap_state(
    stream_ac_agent_factory,
):
    from memorax.online_ac.standard import make_standard_program

    legacy = stream_ac_agent_factory(adaptive=False)
    program = make_standard_program(
        stream_ac_agent_factory(adaptive=False),
        legacy.cfg,
    )
    initial_key = jax.random.fold_in(jax.random.key(31), 0)
    train_key = jax.random.fold_in(jax.random.key(31), 1)
    expected = legacy.init(initial_key)
    actual = program.init_fn(initial_key)
    keys = jax.random.split(train_key, 2)
    expected, _ = legacy._update_step(expected, keys[0])
    second_debug = _legacy_one_step_debug(legacy, expected, keys[1])

    _, metrics = program.train_epoch_fn(train_key, actual, num_steps=2)

    assert_tree_allclose(
        _tree_at(metrics.bootstrap_carry, 1),
        second_debug["bootstrap_carry"],
    )
    assert_tree_allclose(
        _tree_at(metrics.bootstrap_sensitivity, 1),
        second_debug["bootstrap_sensitivity"],
    )
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            _tree_at(metrics.bootstrap_carry, 1),
            second_debug["wrong_bootstrap_carry"],
        )
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            _tree_at(metrics.bootstrap_sensitivity, 1),
            second_debug["wrong_bootstrap_sensitivity"],
        )
