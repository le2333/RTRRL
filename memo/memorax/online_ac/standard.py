"""Concrete composable kernel for the standard StreamAC-RTRL program."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from memorax.networks import RTUCell
from memorax.utils import Timestep
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)

from .credit import make_exact_rtrl_credit
from .normalization import (
    environment_owns_normalization,
    make_normalizer,
    normalization_metrics,
)
from .objectives import make_stream_ac_objective
from .td import make_td0
from .traces import make_stream_ac_trace
from .types import ActionDecision, AgentProgram, EvalSummary
from .updates import make_whole_tree_obgd


def _delegate_rtu_init_forward(next_fun, args, kwargs, context):
    module = context.module
    if context.method_name == "__call__" and type(module) is RTUCell:
        carry, inputs = args
        sensitivity = module.initialize_sensitivity(jax.random.key(0), inputs.shape)
        if sensitivity is None:
            raise TypeError("RTUCell initialization requires local sensitivity")
        next_carry, output, _ = module.local_jacobian(carry, inputs, sensitivity)
        return next_carry, output
    return next_fun(*args, **kwargs)


@struct.dataclass(frozen=True)
class NetworkState:
    """Independent online state for one recurrent actor or critic network."""

    params: Any
    traces: Any
    v: Any
    carry: Any
    sensitivity: Any


@struct.dataclass(frozen=True)
class StandardState:
    """Legacy-compatible state of the concrete standard program."""

    step: Any
    update_step: Any
    timestep: Timestep
    env_state: Any
    actor_params: Any
    actor_traces: Any
    actor_v: Any
    actor_carry: Any
    actor_sensitivity: Any
    critic_params: Any
    critic_traces: Any
    critic_v: Any
    critic_carry: Any
    critic_sensitivity: Any
    normalizer_state: Any = None


@struct.dataclass(frozen=True)
class StandardStepMetrics:
    """Fixed-shape observables from one standard-program transition."""

    action_decision: ActionDecision | None = None
    log_prob: Any = None
    entropy: Any = None
    value: Any = None
    next_value: Any = None
    td_error: Any = None
    actor_carry: Any = None
    actor_sensitivity: Any = None
    critic_carry: Any = None
    critic_sensitivity: Any = None
    bootstrap_carry: Any = None
    bootstrap_sensitivity: Any = None
    actor_grads: Any = None
    critic_grads: Any = None
    incoming_actor_traces: Any = None
    incoming_critic_traces: Any = None
    actor_traces: Any = None
    critic_traces: Any = None
    update_actor_traces: Any = None
    update_critic_traces: Any = None
    actor_step_size: Any = None
    critic_step_size: Any = None
    actor_updates: Any = None
    critic_updates: Any = None
    actor_v: Any = None
    critic_v: Any = None
    actor_params: Any = None
    critic_params: Any = None
    info: Any = None
    raw_episode_return: Any = None
    normalization: Any = None
    state_after: Any = None


@dataclass(frozen=True)
class StandardDebugInterface:
    """Exact kernel hook used by parity tests, never by traced control flow."""

    step: Any


def make_standard_program(
    parts,
    static_config,
    normalization_config=None,
    *,
    reset_on_start=None,
    update_during_eval=None,
    _debug_sink=None,
) -> AgentProgram:
    """Build a concrete StreamAC-RTRL kernel with independent network state."""

    config = static_config
    normalizer = make_normalizer(normalization_config or config)
    normalization_config = normalizer.config
    if reset_on_start is not None:
        normalization_config = replace(
            normalization_config, reset_on_start=reset_on_start
        )
    if update_during_eval is not None:
        normalization_config = replace(
            normalization_config, update_during_eval=update_during_eval
        )
    normalizer = make_normalizer(normalization_config)
    normalization_enabled = (
        normalizer.config.normalize_observation or normalizer.config.normalize_reward
    )
    env = parts.env
    env_params = parts.env_params
    actor_network = parts.actor_network
    critic_network = parts.critic_network
    if normalization_enabled and environment_owns_normalization(env):
        raise ValueError(
            "normalization owner conflict: wrapper and program normalization are both enabled"
        )
    actor_credit = make_exact_rtrl_credit(actor_network.torso)
    critic_credit = make_exact_rtrl_credit(critic_network.torso)
    objective = make_stream_ac_objective(config)
    trace_kernel = make_stream_ac_trace(config)
    obgd = make_whole_tree_obgd(config)
    td0 = make_td0()

    def forward(
        network,
        credit,
        params,
        obs,
        action,
        reward,
        done,
        carry,
        sensitivity,
    ):
        parameter_tree = params["params"] if "params" in params else params
        features, _ = network.feature_extractor.apply(
            {"params": parameter_tree["feature_extractor"]},
            observation=obs,
            action=action,
            reward=reward,
            done=done,
        )
        next_carry, hidden, next_sensitivity = credit(
            parameter_tree["torso"],
            features,
            done,
            carry,
            sensitivity,
        )
        output = network.head.apply(
            {"params": parameter_tree["head"]},
            hidden,
            action=action,
            reward=reward,
            done=done,
        )
        return (next_carry, next_sensitivity), output

    def initialize_network(network, key, timestep):
        carry_shape = (config.num_envs, None)
        carry = network.initialize_carry(carry_shape)
        sensitivity = network.torso.initialize_sensitivity(key, carry_shape)
        obs, done, action, reward = timestep
        with nn.intercept_methods(_delegate_rtu_init_forward):
            params = network.init(
                {"params": key},
                observation=obs,
                action=action,
                reward=reward,
                done=done,
                initial_carry=carry,
            )
        traces = jax.tree.map(
            lambda param: jnp.zeros((config.num_envs, *param.shape)),
            params,
        )
        return NetworkState(
            params=params,
            traces=traces,
            v=jax.tree.map(jnp.zeros_like, traces),
            carry=carry,
            sensitivity=sensitivity,
        )

    def init_fn(key):
        env_key, actor_key, critic_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, config.num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            env_keys,
            env_params,
        )
        normalizer_state = None
        if normalization_enabled:
            obs, normalizer_state = normalizer.reset(obs)
        action_space = env.action_space(env_params)
        action = jnp.zeros(
            (config.num_envs, *action_space.shape),
            dtype=action_space.dtype,
        )
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=jnp.zeros((config.num_envs,), dtype=jnp.float32),
            done=jnp.ones((config.num_envs,), dtype=jnp.bool_),
        ).to_sequence()
        actor = initialize_network(actor_network, actor_key, timestep)
        critic = initialize_network(critic_network, critic_key, timestep)
        return StandardState(
            step=jnp.asarray(0, dtype=jnp.int32),
            update_step=jnp.asarray(0, dtype=jnp.int32),
            timestep=timestep.from_sequence(),
            env_state=env_state,
            actor_params=actor.params,
            actor_traces=actor.traces,
            actor_v=actor.v,
            actor_carry=actor.carry,
            actor_sensitivity=actor.sensitivity,
            critic_params=critic.params,
            critic_traces=critic.traces,
            critic_v=critic.v,
            critic_carry=critic.carry,
            critic_sensitivity=critic.sensitivity,
            normalizer_state=normalizer_state,
        )

    def obgd_step_size(traces, v, *, delta, learning_rate, kappa, step):
        if config.adaptive:
            v_hat = jax.tree.map(
                lambda moment: moment / (1.0 - config.beta2**step),
                v,
            )
            leaves = jax.tree.leaves(
                jax.tree.map(
                    lambda trace, moment: (
                        jnp.abs(trace) / (jnp.sqrt(moment) + config.eps)
                    ),
                    traces,
                    v_hat,
                )
            )
        else:
            leaves = jax.tree.leaves(traces)
        trace_sum = sum(
            jnp.sum(jnp.abs(leaf), axis=tuple(range(1, leaf.ndim))) for leaf in leaves
        )
        return learning_rate / jnp.maximum(
            1.0,
            jnp.maximum(jnp.abs(delta), 1.0) * trace_sum * learning_rate * kappa,
        )

    def step_fn(state, key):
        action_key, env_key = jax.random.split(key)
        obs, done, previous_action, reward = state.timestep.to_sequence()
        reset_before = state.timestep.done
        pre_actor_carry = jax.lax.stop_gradient(state.actor_carry)
        pre_actor_sensitivity = jax.lax.stop_gradient(state.actor_sensitivity)
        pre_critic_carry = jax.lax.stop_gradient(state.critic_carry)
        pre_critic_sensitivity = jax.lax.stop_gradient(state.critic_sensitivity)

        (actor_carry, actor_sensitivity), (dist, _) = forward(
            actor_network,
            actor_credit,
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

        (critic_carry, critic_sensitivity), (value_raw, _) = forward(
            critic_network,
            critic_credit,
            state.critic_params,
            obs,
            previous_action,
            reward,
            done,
            state.critic_carry,
            state.critic_sensitivity,
        )
        value = remove_feature_axis(remove_time_axis(value_raw))

        step_keys = jax.random.split(env_key, config.num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            env.step,
            in_axes=(0, 0, 0, None),
        )(step_keys, state.env_state, sampled_action, env_params)
        normalizer_state = state.normalizer_state
        raw_episode_return = None
        if normalization_enabled:
            normalized = normalizer.step(
                normalizer_state,
                observation=next_obs,
                reward=next_reward,
                done=next_done,
            )
            next_obs = normalized.observation
            next_reward = normalized.reward
            normalizer_state = normalized.state
            raw_episode_return = normalized.raw_episode_return
        next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
            obs=next_obs,
            action=sampled_action,
            reward=next_reward,
            done=next_done,
        ).to_sequence()
        (
            bootstrap_carry,
            bootstrap_sensitivity,
        ), (next_value_raw, _) = forward(
            critic_network,
            critic_credit,
            jax.lax.stop_gradient(state.critic_params),
            next_obs_s,
            next_action_s,
            next_reward_s,
            next_done_s,
            jax.lax.stop_gradient(critic_carry),
            jax.lax.stop_gradient(critic_sensitivity),
        )
        next_value = remove_feature_axis(remove_time_axis(next_value_raw))
        td_error = td0(
            reward=next_reward,
            value=value,
            next_value=next_value,
            bootstrap_discount=config.gamma * (1 - next_done),
        )

        def actor_direction(params):
            _, (diff_dist, _) = forward(
                actor_network,
                actor_credit,
                params,
                obs,
                previous_action,
                reward,
                done,
                pre_actor_carry,
                pre_actor_sensitivity,
            )
            directions = objective(
                log_prob=remove_time_axis(
                    diff_dist.log_prob(add_time_axis(sampled_action))
                ),
                value=jnp.zeros_like(td_error),
                entropy=remove_time_axis(diff_dist.entropy()),
                delta=td_error,
            )
            return directions.traced_by_domain["actor"]

        def critic_direction(params):
            _, (diff_value_raw, _) = forward(
                critic_network,
                critic_credit,
                params,
                obs,
                previous_action,
                reward,
                done,
                pre_critic_carry,
                pre_critic_sensitivity,
            )
            diff_value = remove_feature_axis(remove_time_axis(diff_value_raw))
            directions = objective(
                log_prob=jnp.zeros_like(diff_value),
                value=diff_value,
                entropy=jnp.zeros_like(diff_value),
                delta=td_error,
            )
            return directions.traced_by_domain["critic"]

        actor_grads = jax.jacobian(actor_direction)(state.actor_params)
        critic_grads = jax.jacobian(critic_direction)(state.critic_params)
        actor_trace_result = trace_kernel(
            state.actor_traces,
            actor_grads,
            reset_before=reset_before,
        )
        critic_trace_result = trace_kernel(
            state.critic_traces,
            critic_grads,
            reset_before=reset_before,
        )
        actor_traces = actor_trace_result.carried
        critic_traces = critic_trace_result.carried
        current_step = state.update_step + 1
        critic_updates, critic_v = obgd(
            critic_trace_result.update,
            state.critic_v,
            delta=td_error,
            learning_rate=config.critic_lr,
            kappa=config.critic_kappa,
            step=current_step,
        )
        actor_updates, actor_v = obgd(
            actor_trace_result.update,
            state.actor_v,
            delta=td_error,
            learning_rate=config.actor_lr,
            kappa=config.actor_kappa,
            step=current_step,
        )
        critic_step_size = obgd_step_size(
            critic_trace_result.update,
            critic_v,
            delta=td_error,
            learning_rate=config.critic_lr,
            kappa=config.critic_kappa,
            step=current_step,
        )
        actor_step_size = obgd_step_size(
            actor_trace_result.update,
            actor_v,
            delta=td_error,
            learning_rate=config.actor_lr,
            kappa=config.actor_kappa,
            step=current_step,
        )
        critic_params = jax.tree.map(
            lambda param, update: param + update,
            state.critic_params,
            critic_updates,
        )
        actor_params = jax.tree.map(
            lambda param, update: param + update,
            state.actor_params,
            actor_updates,
        )

        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        next_reward_f = jnp.asarray(next_reward, dtype=jnp.float32)
        persisted_action = jnp.where(
            jnp.expand_dims(next_done, axis=broadcast_dims),
            jnp.zeros_like(sampled_action),
            sampled_action,
        )
        persisted_reward = jnp.where(
            next_done,
            jnp.zeros_like(next_reward_f),
            next_reward_f,
        )
        action_decision = ActionDecision(
            sampled_action=sampled_action,
            logprob_action=sampled_action,
            env_action=sampled_action,
            bootstrap_feedback_action=sampled_action,
            persisted_feedback_action=persisted_action,
        )
        next_state = state.replace(
            step=state.step + config.num_envs,
            update_step=current_step,
            timestep=Timestep(
                obs=next_obs,
                action=persisted_action,
                reward=persisted_reward,
                done=next_done,
            ),
            env_state=env_state,
            actor_params=actor_params,
            actor_traces=actor_traces,
            actor_v=actor_v,
            actor_carry=actor_carry,
            actor_sensitivity=actor_sensitivity,
            critic_params=critic_params,
            critic_traces=critic_traces,
            critic_v=critic_v,
            critic_carry=critic_carry,
            critic_sensitivity=critic_sensitivity,
            normalizer_state=normalizer_state,
        )
        return next_state, StandardStepMetrics(
            action_decision=action_decision,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            next_value=next_value,
            td_error=td_error,
            actor_carry=actor_carry,
            actor_sensitivity=actor_sensitivity,
            critic_carry=critic_carry,
            critic_sensitivity=critic_sensitivity,
            bootstrap_carry=bootstrap_carry,
            bootstrap_sensitivity=bootstrap_sensitivity,
            actor_grads=actor_grads,
            critic_grads=critic_grads,
            incoming_actor_traces=state.actor_traces,
            incoming_critic_traces=state.critic_traces,
            actor_traces=actor_traces,
            critic_traces=critic_traces,
            update_actor_traces=actor_trace_result.update,
            update_critic_traces=critic_trace_result.update,
            actor_step_size=actor_step_size,
            critic_step_size=critic_step_size,
            actor_updates=actor_updates,
            critic_updates=critic_updates,
            actor_v=actor_v,
            critic_v=critic_v,
            actor_params=actor_params,
            critic_params=critic_params,
            info=info,
            raw_episode_return=raw_episode_return,
            normalization=normalization_metrics(
                normalizer_state, normalizer.config.eps
            ),
            state_after=next_state,
        )

    def train_epoch_fn(key, state, num_steps):
        keys = jax.random.split(key, num_steps // config.num_envs)
        return jax.lax.scan(step_fn, state, keys)

    def evaluate_fn(key, state, num_steps):
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, config.num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        normalizer_state = state.normalizer_state
        if normalization_enabled:
            if normalizer.config.reset_on_start:
                obs, normalizer_state = normalizer.reset(obs)
            else:
                obs, normalizer_state = normalizer.reset(
                    obs,
                    normalizer_state,
                    update=normalizer.config.update_during_eval,
                )
        action_space = env.action_space(env_params)
        action = jnp.zeros(
            (config.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=jnp.zeros((config.num_envs,), dtype=jnp.float32),
            done=jnp.ones((config.num_envs,), dtype=jnp.bool_),
        )
        carry_shape = (config.num_envs, None)
        eval_state = state.replace(
            timestep=timestep,
            env_state=env_state,
            actor_carry=actor_network.initialize_carry(carry_shape),
            critic_carry=critic_network.initialize_carry(carry_shape),
            actor_sensitivity=actor_network.torso.initialize_sensitivity(
                jax.random.key(0), carry_shape
            ),
            critic_sensitivity=critic_network.torso.initialize_sensitivity(
                jax.random.key(0), carry_shape
            ),
            normalizer_state=normalizer_state,
        )

        def eval_step(current, step_key):
            action_key, env_key = jax.random.split(step_key)
            del action_key
            obs_s, done_s, action_s, reward_s = current.timestep.to_sequence()
            (actor_carry, actor_sensitivity), (dist, _) = forward(
                actor_network,
                actor_credit,
                current.actor_params,
                obs_s,
                action_s,
                reward_s,
                done_s,
                current.actor_carry,
                current.actor_sensitivity,
            )
            chosen = (
                jnp.argmax(dist.logits, axis=-1)
                if hasattr(dist, "logits")
                else dist.mode()
            )
            chosen = remove_time_axis(chosen)
            step_keys = jax.random.split(env_key, config.num_envs)
            next_obs, next_env_state, next_reward, next_done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_keys, current.env_state, chosen, env_params)
            next_normalizer_state = current.normalizer_state
            if normalization_enabled:
                normalized = normalizer.step(
                    next_normalizer_state,
                    observation=next_obs,
                    reward=next_reward,
                    done=next_done,
                    update=normalizer.config.update_during_eval,
                )
                next_obs = normalized.observation
                next_reward = normalized.reward
                next_normalizer_state = normalized.state
            broadcast_dims = tuple(
                range(current.timestep.done.ndim, current.timestep.action.ndim)
            )
            next_reward = jnp.asarray(next_reward, dtype=jnp.float32)
            return current.replace(
                step=current.step + config.num_envs,
                timestep=Timestep(
                    obs=next_obs,
                    action=jnp.where(
                        jnp.expand_dims(next_done, axis=broadcast_dims),
                        jnp.zeros_like(chosen),
                        chosen,
                    ),
                    reward=jnp.where(
                        next_done, jnp.zeros_like(next_reward), next_reward
                    ),
                    done=next_done,
                ),
                env_state=next_env_state,
                actor_carry=actor_carry,
                actor_sensitivity=actor_sensitivity,
                normalizer_state=next_normalizer_state,
            ), EvalSummary(
                info=info,
                normalization=normalization_metrics(
                    next_normalizer_state, normalizer.config.eps
                ),
            )

        step_keys = jax.random.split(eval_key, num_steps)
        eval_state, summary = jax.lax.scan(eval_step, eval_state, step_keys)
        return eval_state, summary

    if _debug_sink is not None:
        _debug_sink.append(StandardDebugInterface(step=step_fn))
    return AgentProgram(
        init_fn=init_fn,
        train_epoch_fn=train_epoch_fn,
        evaluate_fn=evaluate_fn,
        state_schema=StandardState,
        metric_schema=StandardStepMetrics,
    )
