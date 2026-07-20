from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from golden import (
    GoldenSnapshot,
    _rtrrl_observables,
    assert_tree_allclose,
    load_golden,
)

from memorax.algorithms.rtrrl import RTRRL
from memorax.algorithms.rtrrl.compatibility import normalize_legacy_config
from memorax.algorithms.rtrrl.components import (
    MemoraxRecurrentAdapter,
    RecurrentComponent,
    select_memorax_components,
)
from memorax.networks import Memoroid, RNN, heads
from memorax.networks.sequence_models.lru import LRUCell
from memorax.networks.sequence_models.rtu import RTUCell
from memorax.utils import Timestep
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)


def _tree_at(tree, index):
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _golden_section(snapshot, section):
    prefix = f"{section}/"

    def canonical(path):
        return path.replace("/.", "/").removeprefix(".")

    return GoldenSnapshot(
        {
            canonical(path[len(prefix) :]): leaf
            for path, leaf in snapshot.leaves.items()
            if path.startswith(prefix)
        }
    )


def _legacy_init_counter_view(state):
    """Adapt only the pre-JIT Python-counter schema recorded by the oracle."""
    assert state.step.dtype == jnp.int32
    assert state.update_step.dtype == jnp.int32
    return state.replace(
        step=np.asarray(state.step, dtype=np.int64),
        update_step=np.asarray(state.update_step, dtype=np.int64),
    )


def _exact_gradients(agent, state, sampled_action):
    obs, done, action, reward = state.timestep.to_sequence()
    params = agent._grad_params(state.params, state.slow_torso)
    carry = jax.lax.stop_gradient(state.carry)
    sensitivity = jax.lax.stop_gradient(state.sensitivity)

    def traced(p):
        _, (dist, value, _) = agent._forward(
            p, obs, action, reward, done, carry, sensitivity
        )
        log_prob = remove_time_axis(
            dist.log_prob(add_time_axis(sampled_action))
        )
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


def _legacy_oracle_step(agent, state, key):
    """44a5483 legacy update math, independent of the composed scan."""
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
    next_obs, env_state, next_reward, next_done, _ = jax.vmap(
        agent.env.step, in_axes=(0, 0, 0, None)
    )(step_keys, state.env_state, env_action, agent.env_params)
    next_inputs = Timestep(
        obs=next_obs,
        action=env_action,
        reward=next_reward,
        done=next_done,
    ).to_sequence()
    next_obs_s, next_done_s, next_action_s, next_reward_s = next_inputs

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
        return bootstrap_state, remove_feature_axis(
            remove_time_axis(next_value_raw)
        )

    bootstrap_state, next_value = bootstrap(
        acting_carry, acting_sensitivity
    )
    wrong_bootstrap, _ = bootstrap(state.carry, state.sensitivity)
    td_error = (
        next_reward + agent.cfg.gamma * (1 - next_done) * next_value - value
    )
    traced_grads, direct_grads = _exact_gradients(
        agent, state, sampled_action
    )
    trace_decays = {
        "actor": agent.cfg.gamma * agent.cfg.lambda_pi,
        "critic": agent.cfg.gamma * agent.cfg.lambda_v,
        "feature_extractor": agent.cfg.gamma * agent.cfg.lambda_rnn,
        "torso": agent.cfg.gamma * agent.cfg.lambda_rnn,
        "pred": agent.cfg.gamma * agent.cfg.lambda_rnn,
    }

    def carry_trace(old, gradient, decay):
        trailing = old.ndim - 1
        continues = (1 - next_done)[
            (slice(None),) + (None,) * trailing
        ]
        emphasis = state.I[
            (slice(None),) + (None,) * (gradient.ndim - 1)
        ]
        return decay * continues * old + emphasis * gradient

    carried_traces = {
        name: jax.tree.map(
            lambda old, gradient: carry_trace(
                old, gradient, trace_decays[name]
            ),
            state.traces[name],
            traced_grads[name],
        )
        for name in state.traces
    }
    update_traces = (
        carried_traces
        if agent.cfg.update_trace_before_td
        else state.traces
    )

    def apply_delta(trace, scale):
        delta = td_error[(slice(None),) + (None,) * (trace.ndim - 1)]
        return scale * delta * trace

    ascent_updates = {}
    for name in state.params:
        scale = (
            agent.cfg.eta_f
            if name in ("feature_extractor", "torso")
            else 1.0
        )
        combined = jax.tree.map(
            lambda trace, direct: apply_delta(trace, scale) + direct,
            update_traces[name],
            direct_grads[name],
        )
        ascent_updates[name] = jax.tree.map(
            lambda update: jnp.mean(update, axis=0), combined
        )
    adam_updates, adam_state = agent.optimizer.update(
        ascent_updates, state.opt_state, state.params
    )
    fast_params = cast(
        Any, optax.apply_updates(state.params, adam_updates)
    )
    slow_torso = (
        fast_params["torso"]
        if agent.cfg.update_period == 1.0
        else optax.incremental_update(
            fast_params["torso"],
            state.slow_torso,
            agent.cfg.update_period,
        )
    )
    broadcast_dims = tuple(
        range(state.timestep.done.ndim, state.timestep.action.ndim)
    )
    persisted_action = jnp.where(
        jnp.expand_dims(next_done, axis=broadcast_dims),
        jnp.zeros_like(env_action),
        env_action,
    )
    next_reward_f = jnp.asarray(next_reward, dtype=jnp.float32)
    persisted_reward = jnp.where(
        next_done, jnp.zeros_like(next_reward_f), next_reward_f
    )
    oracle_state = state.replace(
        step=state.step + agent.cfg.num_envs,
        update_step=state.update_step + 1,
        timestep=Timestep(
            obs=next_obs,
            action=persisted_action,
            reward=persisted_reward,
            done=next_done,
        ),
        env_state=env_state,
        params=fast_params,
        slow_torso=slow_torso,
        traces=carried_traces,
        opt_state=adam_state,
        carry=acting_carry,
        sensitivity=acting_sensitivity,
        I=agent.cfg.gamma * state.I * (1 - next_done) + next_done,
    )
    return oracle_state, {
        "acting_carry": acting_carry,
        "acting_sensitivity": acting_sensitivity,
        "bootstrap_carry": bootstrap_state[0],
        "bootstrap_sensitivity": bootstrap_state[1],
        "wrong_bootstrap": wrong_bootstrap,
        "incoming_traces": state.traces,
        "carried_traces": carried_traces,
        "update_traces": update_traces,
        "ascent_updates": ascent_updates,
        "adam_updates": adam_updates,
        "adam_state": adam_state,
        "prediction_target": jnp.concatenate(
            [next_obs, jnp.asarray(next_reward, jnp.float32)[..., None]],
            axis=-1,
        ),
    }


def _exact_prediction_gradients(agent, state, target):
    obs, done, action, reward = state.timestep.to_sequence()
    params = agent._grad_params(state.params, state.slow_torso)

    def loss(p):
        _, (_, _, prediction) = agent._forward(
            p,
            obs,
            action,
            reward,
            done,
            jax.lax.stop_gradient(state.carry),
            jax.lax.stop_gradient(state.sensitivity),
        )
        error = remove_time_axis(prediction) - jax.lax.stop_gradient(target)
        return -agent.cfg.pred_coeff * 0.5 * jnp.sum(
            jnp.square(error), axis=-1
        )

    return jax.jacobian(loss)(params)


@pytest.mark.parametrize(
    ("overrides", "recurrent_type", "cell_type", "expected"),
    [
        (
            {},
            Memoroid,
            LRUCell,
            {
                "recurrent_component": "memorax_memoroid_lru",
                "feature_component": "encoder",
                "actor_component": "unbounded_gaussian",
                "action_clipping": "none",
                "input_gain": "trainable",
                "prediction_component": "none",
                "trace_timing": "fresh",
                "logprob_reduction": "sum",
                "topology": "shared",
                "normalize_observation": False,
                "normalize_reward": False,
            },
        ),
        (
            {
                "backbone": "rtu",
                "use_encoder": False,
                "bound_actor": True,
                "act_clip": 0.5,
                "freeze_gamma": True,
                "pred_obs": True,
                "update_trace_before_td": False,
                "logprob_reduction": "mean",
                "normalize_obs": True,
                "normalize_reward": True,
            },
            RNN,
            RTUCell,
            {
                "recurrent_component": "memorax_rnn_rtu",
                "feature_component": "raw",
                "actor_component": "bounded_gaussian",
                "action_clipping": "env",
                "input_gain": "frozen",
                "prediction_component": "observation_reward",
                "trace_timing": "incoming",
                "logprob_reduction": "mean",
                "topology": "shared",
                "normalize_observation": True,
                "normalize_reward": True,
            },
        ),
    ],
)
def test_memorax_component_selection_records_every_retained_shared_branch(
    monkeypatch, overrides, recurrent_type, cell_type, expected
):
    legacy = normalize_legacy_config(
        {"profile": "memo_experimental", **overrides}
    )
    monkeypatch.setattr(
        jax,
        "jit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("component selection reached JIT")
        ),
    )

    selected = select_memorax_components(
        legacy,
        observation_dim=3,
        action_dim=2,
        topology="shared",
    )

    assert isinstance(selected.recurrent, RecurrentComponent)
    assert isinstance(selected.recurrent, MemoraxRecurrentAdapter)
    assert isinstance(selected.recurrent.module, recurrent_type)
    assert isinstance(selected.recurrent.module.cell, cell_type)
    assert {
        key: getattr(selected.effective_config, key) for key in expected
    } == expected
    assert (selected.prediction_head is not None) == bool(
        overrides.get("pred_obs")
    )


def test_experimental_facade_exposes_the_one_composed_program(
    rtrrl_agent_factory,
):
    agent = rtrrl_agent_factory(fresh_trace=False)

    assert isinstance(agent, RTRRL)
    assert agent.profile == "memo_experimental"
    assert agent.as_legacy_program() is agent
    assert agent.program is agent._delegate.program
    assert agent.program_config.static_config is agent.cfg


@pytest.mark.parametrize("fresh_trace", [False, True], ids=["incoming", "fresh"])
def test_composed_program_selects_trace_timing_before_scan(
    rtrrl_agent_factory, fresh_trace
):
    agent = rtrrl_agent_factory(fresh_trace=fresh_trace)
    state = agent.program.init_fn(jax.random.key(7))

    final_state, metrics = agent.program.train_epoch_fn(
        jax.random.key(11), state, 1
    )

    selected = (
        _tree_at(metrics.carried_traces, 0)
        if fresh_trace
        else _tree_at(metrics.incoming_traces, 0)
    )
    assert_tree_allclose(_tree_at(metrics.update_traces, 0), selected)
    assert jax.tree.structure(final_state) == jax.tree.structure(state)


def test_composed_program_clips_only_environment_action(rtrrl_agent_factory):
    base = rtrrl_agent_factory(fresh_trace=False)
    agent = replace(base, cfg=replace(base.cfg, act_clip=0.05))
    state = agent.program.init_fn(jax.random.key(13))

    _, metrics = agent.program.train_epoch_fn(jax.random.key(17), state, 1)

    sampled = metrics.action_decision.sampled_action[0]
    env_action = metrics.action_decision.env_action[0]
    assert np.max(np.abs(env_action)) <= 0.05
    assert not np.allclose(sampled, env_action)
    assert_tree_allclose(metrics.action_decision.logprob_action[0], sampled)
    assert_tree_allclose(
        metrics.action_decision.bootstrap_feedback_action[0], env_action
    )


def test_composed_program_routes_prediction_head_direct_gradient(
    rtrrl_agent_factory,
):
    base = rtrrl_agent_factory(fresh_trace=False)
    agent = replace(
        base,
        cfg=replace(base.cfg, pred_obs=True, pred_coeff=0.7),
        pred_head=heads.Regressor(out_dim=3),
    )
    state = agent.program.init_fn(jax.random.key(19))

    _, metrics = agent.program.train_epoch_fn(jax.random.key(23), state, 1)

    prediction = metrics.prediction_direct_grads
    assert any(
        np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(prediction)
    )


def test_composed_init_matches_versioned_legacy_golden_leaf_for_leaf(
    rtrrl_agent_factory,
):
    agent = rtrrl_agent_factory(fresh_trace=False)
    key = jax.random.fold_in(jax.random.key(7), 0)

    actual = agent.program.init_fn(key)
    golden, manifest = load_golden("rtrrl_lru")

    assert manifest["algorithm"] == "rtrrl_lru"
    assert_tree_allclose(
        _legacy_init_counter_view(actual),
        _golden_section(golden["trace_timing/incoming"], "init"),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("fresh_trace", [False, True], ids=["incoming", "fresh"])
def test_composed_one_step_preserves_exact_update_domain_and_adam(
    rtrrl_agent_factory, fresh_trace
):
    agent = rtrrl_agent_factory(fresh_trace=fresh_trace)
    init_key = jax.random.fold_in(jax.random.key(7), 0)
    train_key = jax.random.fold_in(jax.random.key(7), 1)
    initial = agent.init(init_key)
    final, metrics = agent.program.train_epoch_fn(
        train_key, initial, num_steps=1
    )
    oracle_key = jax.random.split(train_key, 1)[0]
    observables = _rtrrl_observables(agent, initial, oracle_key)
    oracle_state, debug = _legacy_oracle_step(agent, initial, oracle_key)
    expected_traced, expected_direct = _exact_gradients(
        agent, initial, observables["sampled_action"]
    )

    assert_tree_allclose(final, oracle_state)
    assert_tree_allclose(
        metrics.action_decision.sampled_action[0],
        observables["sampled_action"],
    )
    assert_tree_allclose(
        metrics.action_decision.logprob_action[0],
        observables["logprob_action"],
    )
    assert_tree_allclose(
        metrics.action_decision.env_action[0],
        observables["env_action"],
    )
    assert_tree_allclose(
        metrics.action_decision.bootstrap_feedback_action[0],
        observables["feedback_action"],
    )
    assert_tree_allclose(metrics.value[0], observables["value"])
    assert_tree_allclose(metrics.next_value[0], observables["next_value"])
    assert_tree_allclose(metrics.td_error[0], observables["td"])
    assert_tree_allclose(metrics.log_prob[0], observables["logprob"])
    assert_tree_allclose(
        _tree_at(metrics.differentiation_grads, 0), expected_traced
    )
    assert_tree_allclose(_tree_at(metrics.direct_grads, 0), expected_direct)
    assert_tree_allclose(
        _tree_at(metrics.bootstrap_carry, 0), debug["bootstrap_carry"]
    )
    assert_tree_allclose(
        _tree_at(metrics.bootstrap_sensitivity, 0),
        debug["bootstrap_sensitivity"],
    )
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            _tree_at(metrics.bootstrap_carry, 0),
            debug["wrong_bootstrap"][0],
        )
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            _tree_at(metrics.bootstrap_sensitivity, 0),
            debug["wrong_bootstrap"][1],
        )
    for name in (
        "acting_carry",
        "acting_sensitivity",
        "incoming_traces",
        "carried_traces",
        "update_traces",
        "ascent_updates",
        "adam_updates",
        "adam_state",
    ):
        assert_tree_allclose(_tree_at(getattr(metrics, name), 0), debug[name])
    expected_fast_params = cast(
        Any,
        optax.apply_updates(initial.params, debug["adam_updates"]),
    )
    expected_slow_torso = (
        expected_fast_params["torso"]
        if agent.cfg.update_period == 1.0
        else optax.incremental_update(
            expected_fast_params["torso"],
            initial.slow_torso,
            agent.cfg.update_period,
        )
    )
    assert_tree_allclose(
        _tree_at(metrics.fast_params, 0), expected_fast_params
    )
    assert_tree_allclose(final.params, expected_fast_params)
    assert_tree_allclose(
        _tree_at(metrics.slow_torso, 0), expected_slow_torso
    )
    assert_tree_allclose(final.slow_torso, expected_slow_torso)

    wrong_sign = jax.tree.map(lambda update: -update, debug["ascent_updates"])
    wrong_adam_sign, _ = agent.optimizer.update(
        wrong_sign, initial.opt_state, initial.params
    )
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            _tree_at(metrics.adam_updates, 0), wrong_adam_sign
        )

    if fresh_trace:
        wrong_domain = {
            **debug["ascent_updates"],
            "actor": jax.tree.map(
                lambda update: agent.cfg.eta_f * update,
                debug["ascent_updates"]["actor"],
            ),
            "torso": jax.tree.map(
                lambda update: update / agent.cfg.eta_f,
                debug["ascent_updates"]["torso"],
            ),
        }
        with pytest.raises(AssertionError):
            assert_tree_allclose(
                _tree_at(metrics.ascent_updates, 0), wrong_domain
            )
        wrong_trace_timing = debug["incoming_traces"]
    else:
        wrong_trace_timing = debug["carried_traces"]
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            _tree_at(metrics.update_traces, 0), wrong_trace_timing
        )


def test_composed_terminal_step_preserves_exact_intermediate_states(
    rtrrl_agent_factory,
):
    agent = rtrrl_agent_factory(fresh_trace=False)
    init_key = jax.random.fold_in(jax.random.key(7), 0)
    train_key = jax.random.fold_in(jax.random.key(7), 2)
    initial = agent.init(init_key)
    actual, metrics = agent.program.train_epoch_fn(
        train_key, initial, num_steps=3
    )
    golden, _ = load_golden("rtrrl_lru")
    expected = initial
    expected_states = []

    for index, key in enumerate(jax.random.split(train_key, 3)):
        expected, _ = _legacy_oracle_step(agent, expected, key)
        expected_states.append(expected)
        assert_tree_allclose(_tree_at(metrics.state_after, index), expected)

    assert_tree_allclose(
        actual,
        _golden_section(golden["trace_timing/incoming"], "train"),
    )
    assert_tree_allclose(actual, expected_states[-1])
    mutated_first = expected_states[0].replace(
        step=expected_states[0].step + 1
    )
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            _tree_at(metrics.state_after, 0), mutated_first
        )
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


def test_composed_prediction_gradient_matches_exact_objective(
    rtrrl_agent_factory,
):
    base = rtrrl_agent_factory(fresh_trace=False)
    agent = replace(
        base,
        cfg=replace(base.cfg, pred_obs=True, pred_coeff=0.7),
        pred_head=heads.Regressor(out_dim=3),
    )
    init_key = jax.random.fold_in(jax.random.key(7), 0)
    train_key = jax.random.fold_in(jax.random.key(7), 1)
    initial = agent.init(init_key)
    final, metrics = agent.program.train_epoch_fn(
        train_key, initial, num_steps=1
    )
    _, debug = _legacy_oracle_step(
        agent, initial, jax.random.split(train_key, 1)[0]
    )
    expected = _exact_prediction_gradients(
        agent, initial, debug["prediction_target"]
    )
    actual = _tree_at(metrics.prediction_direct_grads, 0)

    assert_tree_allclose(actual, expected)
    with pytest.raises(AssertionError):
        assert_tree_allclose(
            actual, jax.tree.map(lambda gradient: -gradient, expected)
        )
