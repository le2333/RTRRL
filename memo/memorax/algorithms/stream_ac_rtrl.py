"""StreamAC-RTRL: an actor and a critic with separate recurrent networks.

Neither network sees the other's gradient, so there is no shared torso to
target and no emphasis to carry. Each keeps its own eligibility trace and each
steps under its own overshooting bound, which is what lets the pair learn from
a single transition at a time without a replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from memorax.networks import RTUCell
from memorax.rl import (
    ObjectiveDirections,
    environment_owns_normalization,
    make_exact_rtrl_credit,
    make_normalizer,
    make_obgd_rule,
    make_td0,
    normalization_metrics,
)
from memorax.utils import Timestep
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)

from .contract import ActionDecision, AgentProgram, EvalSummary, EvaluationConfig


@dataclass(frozen=True)
class StreamACRTRLConfig:
    """Everything the kernel reads that does not change during a run."""

    num_envs: int
    gamma: float
    trace_lambda: float
    actor_lr: float
    critic_lr: float
    actor_kappa: float = 3.0
    critic_kappa: float = 2.0
    entropy_coefficient: float = 0.01
    adaptive: bool = False
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass(frozen=True)
class StreamACRTRLParts:
    """The networks and environment a program is built around."""

    env: Any = None
    env_params: Any = None
    actor_network: Any = None
    critic_network: Any = None
    normalization: Any = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    record_trajectory: bool = False

    def replace(self, **updates):
        return replace(self, **updates)


@struct.dataclass
class TraceDirections:
    """The trace carried to the next step and the trace used now."""

    carried: Any
    update: Any


def _broadcast_env(values, leaf):
    return values[(slice(None),) + (None,) * (leaf.ndim - 1)]


def make_stream_ac_trace(config):
    """Build StreamAC's pre-forward reset, always-fresh trace recurrence."""

    decay = config.gamma * config.trace_lambda

    def stream_ac_trace(incoming, gradient, *, reset_before):
        carried = jax.tree.map(
            lambda old, grad: (
                decay * (1 - _broadcast_env(reset_before, old)) * old + grad
            ),
            incoming,
            gradient,
        )
        return TraceDirections(carried=carried, update=carried)

    return stream_ac_trace


def make_stream_ac_objective(config):
    """Route the actor and critic ascent directions.

    Entropy rides on the actor objective rather than arriving separately,
    signed by the TD error so it pushes toward exploration only where the
    critic was surprised.
    """

    def stream_ac_objective(*, log_prob, value, entropy, delta):
        actor = (
            log_prob
            + config.entropy_coefficient
            * jnp.sign(jax.lax.stop_gradient(delta))
            * entropy
        )
        return ObjectiveDirections(
            traced_by_domain={"actor": actor, "critic": value},
            direct_by_domain={
                "actor": jnp.zeros_like(actor),
                "critic": jnp.zeros_like(value),
            },
            metrics={"entropy": entropy},
        )

    return stream_ac_objective


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
class StreamACRTRLState:
    """Everything the kernel carries from one transition to the next."""

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
class StreamACRTRLStepMetrics:
    """Fixed-shape observables from one transition.

    Scalars and one step of trajectory only. The kernel runs under
    ``lax.scan``, so whole parameter and trace trees returned from here would
    be stacked once per step.
    """

    action_decision: ActionDecision | None = None
    log_prob: Any = None
    entropy: Any = None
    value: Any = None
    next_value: Any = None
    td_error: Any = None
    actor_step_size: Any = None
    critic_step_size: Any = None
    observation: Any = None
    reward: Any = None
    done: Any = None
    info: Any = None
    raw_episode_return: Any = None
    normalization: Any = None


def build_stream_ac_rtrl(
    config: StreamACRTRLConfig,
    parts: StreamACRTRLParts,
) -> AgentProgram:
    """Close over the networks, the environment, and every static choice."""

    normalizer = make_normalizer(parts.normalization or config)
    normalization_config = replace(
        normalizer.config,
        reset_on_start=parts.evaluation.reset_on_start,
        update_during_eval=parts.evaluation.update_during_eval,
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
    record_trajectory = parts.record_trajectory
    actor_credit = make_exact_rtrl_credit(actor_network.torso)
    critic_credit = make_exact_rtrl_credit(critic_network.torso)
    objective = make_stream_ac_objective(config)
    trace_kernel = make_stream_ac_trace(config)
    actor_rule = make_obgd_rule(
        learning_rate=config.actor_lr,
        kappa=config.actor_kappa,
        beta2=config.beta2,
        eps=config.eps,
        adaptive=config.adaptive,
    )
    critic_rule = make_obgd_rule(
        learning_rate=config.critic_lr,
        kappa=config.critic_kappa,
        beta2=config.beta2,
        eps=config.eps,
        adaptive=config.adaptive,
    )
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
        parameter_tree = params.get("params", params)
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
        return StreamACRTRLState(
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
            _bootstrap_carry,
            _bootstrap_sensitivity,
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
        critic_step = critic_rule.apply(
            critic_trace_result.update,
            None,
            state.critic_v,
            delta=td_error,
            step=current_step,
            params=state.critic_params,
        )
        actor_step = actor_rule.apply(
            actor_trace_result.update,
            None,
            state.actor_v,
            delta=td_error,
            step=current_step,
            params=state.actor_params,
        )
        critic_v = critic_step.state
        actor_v = actor_step.state
        critic_params = jax.tree.map(
            lambda param, update: param + update,
            state.critic_params,
            critic_step.updates,
        )
        actor_params = jax.tree.map(
            lambda param, update: param + update,
            state.actor_params,
            actor_step.updates,
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
        return next_state, StreamACRTRLStepMetrics(
            action_decision=action_decision,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            next_value=next_value,
            td_error=td_error,
            actor_step_size=actor_step.metrics["step_size"],
            critic_step_size=critic_step.metrics["step_size"],
            observation=next_obs if record_trajectory else None,
            reward=next_reward_f if record_trajectory else None,
            done=next_done if record_trajectory else None,
            info=info,
            raw_episode_return=raw_episode_return,
            normalization=normalization_metrics(
                normalizer_state, normalizer.config.eps
            ),
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
                observation=current.timestep.obs,
                next_observation=next_obs,
                action=chosen,
                reward=next_reward,
                done=next_done,
            )

        step_keys = jax.random.split(eval_key, num_steps)
        eval_state, summary = jax.lax.scan(eval_step, eval_state, step_keys)
        return eval_state, summary

    return AgentProgram(
        init_fn=init_fn,
        train_epoch_fn=train_epoch_fn,
        evaluate_fn=evaluate_fn,
        state_schema=StreamACRTRLState,
        metric_schema=StreamACRTRLStepMetrics,
    )
