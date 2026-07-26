"""Concrete composable kernel for the meta-recurrent RTRRL program."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from flax import core, struct

from memorax.rl import (
    environment_owns_normalization,
    make_exact_rtrl_credit,
    make_normalizer,
    make_td0,
    normalization_metrics,
)
from memorax.utils import Timestep
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)

from .objectives import make_rtrrl_objective
from .targets import make_slow_subtree_target
from .traces import make_rtrrl_trace
from .types import ActionDecision, AgentProgram, EvalSummary
from .updates import make_grouped_adam


@dataclass(frozen=True)
class MetaDebugInterface:
    """Exact kernel hooks used by parity tests, never by traced control flow."""

    forward: Any
    optimizer: Any
    step: Any

    @staticmethod
    def grad_params(params, slow_torso):
        return {**params, "torso": slow_torso}


@struct.dataclass(frozen=True)
class MetaState:
    step: Any
    update_step: Any
    timestep: Timestep
    env_state: Any
    params: Any
    slow_torso: Any
    traces: Any
    opt_state: Any
    carry: Any
    sensitivity: Any
    I: Any  # noqa: E741 - legacy RTRRL emphasis-state name
    normalizer_state: Any = None


@struct.dataclass(frozen=True)
class MetaStepMetrics:
    """Fixed-shape observables from one concrete meta-program transition."""

    action_decision: ActionDecision | None = None
    log_prob: Any = None
    value: Any = None
    next_value: Any = None
    td_error: Any = None
    entropy: Any = None
    acting_carry: Any = None
    acting_sensitivity: Any = None
    bootstrap_carry: Any = None
    bootstrap_sensitivity: Any = None
    differentiation_grads: Any = None
    direct_grads: Any = None
    incoming_traces: Any = None
    carried_traces: Any = None
    update_traces: Any = None
    ascent_updates: Any = None
    adam_updates: Any = None
    adam_state: Any = None
    prediction_direct_grads: Any = None
    fast_params: Any = None
    slow_torso: Any = None
    emphasis: Any = None
    diag_lambda_max: Any = None
    diag_gamma_max: Any = None
    diag_sens_norm: Any = None
    diag_carry_norm: Any = None
    diag_z_rnn: Any = None
    diag_z_actor: Any = None
    diag_z_critic: Any = None
    diag_grad_rnn: Any = None
    diag_grad_actor: Any = None
    diag_grad_critic: Any = None
    diag_upd_rnn: Any = None
    diag_p_torso: Any = None
    diag_p_actor: Any = None
    diag_p_critic: Any = None
    diag_value_abs: Any = None
    diag_td_abs: Any = None
    diag_actor_loc_abs: Any = None
    diag_actor_scale: Any = None
    diag_act_abs: Any = None
    info: Any = None
    raw_episode_return: Any = None
    normalization: Any = None
    state_after: Any = None


def _tree_norm(tree):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves))


def _find_leaf(tree, name):
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if any(getattr(key, "key", None) == name for key in path):
            return leaf
    return None


def make_meta_program(
    parts,
    static_config,
    normalization_config=None,
    *,
    reset_on_start=None,
    update_during_eval=None,
    _debug_sink=None,
) -> AgentProgram:
    """Build the concrete shared-torso RTRRL kernel.

    Modules and environment functions are captured at build time.  In
    particular, optimizer labels are derived once here rather than by mutating
    Python state from ``init_fn``.
    """

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
    feature_extractor = parts.feature_extractor
    torso = parts.torso
    actor_head = parts.actor_head
    critic_head = parts.critic_head
    pred_head = parts.pred_head
    activation = parts.activation
    if normalization_enabled and environment_owns_normalization(env):
        raise ValueError(
            "normalization owner conflict: wrapper and program normalization are both enabled"
        )
    credit = make_exact_rtrl_credit(torso)
    objective = make_rtrrl_objective(config)
    trace_kernel = make_rtrrl_trace(config)
    td0 = make_td0()
    target = make_slow_subtree_target(config)

    def forward(params, obs, action, reward, done, carry, sensitivity):
        x, _ = feature_extractor.apply(
            {"params": params["feature_extractor"]},
            observation=obs,
            action=action,
            reward=reward,
            done=done,
        )
        next_carry, h, next_sensitivity = credit(
            params["torso"], x, done, carry, sensitivity
        )
        h = activation(h)
        dist, _ = actor_head.apply(
            {"params": params["actor"]},
            h,
            action=action,
            reward=reward,
            done=done,
        )
        value, _ = critic_head.apply(
            {"params": params["critic"]},
            h,
            action=action,
            reward=reward,
            done=done,
        )
        prediction = None
        if pred_head is not None:
            prediction, _ = pred_head.apply(
                {"params": params["pred"]},
                h,
                action=action,
                reward=reward,
                done=done,
            )
        return (
            next_carry,
            next_sensitivity,
        ), (dist, value, prediction)

    def initialize_arrays(key):
        (
            env_key,
            feat_key,
            torso_key,
            actor_key,
            critic_key,
            pred_key,
            sens_key,
        ) = jax.random.split(key, 7)
        env_keys = jax.random.split(env_key, config.num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(env_keys, env_params)
        normalizer_state = None
        if normalization_enabled:
            obs, normalizer_state = normalizer.reset(obs)
        action_space = env.action_space(env_params)
        action = jnp.zeros(
            (config.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((config.num_envs,), dtype=jnp.float32)
        done = jnp.ones((config.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()
        ts_obs, ts_done, ts_action, ts_reward = timestep

        carry_shape = (config.num_envs, None)
        carry = torso.initialize_carry(jax.random.key(0), carry_shape)
        sensitivity = torso.initialize_sensitivity(sens_key, carry_shape)
        feat_vars = feature_extractor.init(
            {"params": feat_key},
            observation=ts_obs,
            action=ts_action,
            reward=ts_reward,
            done=ts_done,
        )
        x, _ = feature_extractor.apply(
            feat_vars,
            observation=ts_obs,
            action=ts_action,
            reward=ts_reward,
            done=ts_done,
        )
        torso_vars = torso.init({"params": torso_key}, x, ts_done, initial_carry=carry)
        _, h = torso.apply(torso_vars, x, ts_done, initial_carry=carry)
        h = activation(h)
        actor_vars = actor_head.init(
            {"params": actor_key},
            h,
            action=ts_action,
            reward=ts_reward,
            done=ts_done,
        )
        critic_vars = critic_head.init(
            {"params": critic_key},
            h,
            action=ts_action,
            reward=ts_reward,
            done=ts_done,
        )
        params = {
            "feature_extractor": feat_vars.get("params", core.FrozenDict()),
            "torso": torso_vars["params"],
            "actor": actor_vars["params"],
            "critic": critic_vars["params"],
        }
        if pred_head is not None:
            pred_vars = pred_head.init(
                {"params": pred_key},
                h,
                action=ts_action,
                reward=ts_reward,
                done=ts_done,
            )
            params["pred"] = pred_vars["params"]
        traces = jax.tree.map(
            lambda param: jnp.zeros((config.num_envs, *param.shape)), params
        )
        return {
            "timestep": timestep.from_sequence(),
            "env_state": env_state,
            "params": params,
            "traces": traces,
            "carry": carry,
            "sensitivity": sensitivity,
            "normalizer_state": normalizer_state,
        }

    abstract = jax.eval_shape(initialize_arrays, jax.random.key(0))
    optimizer = make_grouped_adam(config, abstract["params"])

    def init_fn(key):
        arrays = initialize_arrays(key)
        params = arrays["params"]
        return MetaState(
            step=jnp.asarray(0, dtype=jnp.int32),
            update_step=jnp.asarray(0, dtype=jnp.int32),
            timestep=arrays["timestep"],
            env_state=arrays["env_state"],
            params=params,
            slow_torso=params["torso"],
            traces=arrays["traces"],
            opt_state=optimizer.init(params),
            carry=arrays["carry"],
            sensitivity=arrays["sensitivity"],
            I=jnp.ones((config.num_envs,), dtype=jnp.float32),
            normalizer_state=arrays["normalizer_state"],
        )

    def step_fn(state, key):
        action_key, env_key = jax.random.split(key)
        obs, done, previous_action, reward = state.timestep.to_sequence()
        views = target.views(fast_params=state.params, slow_subtree=state.slow_torso)
        pre_carry = jax.lax.stop_gradient(state.carry)
        pre_sensitivity = jax.lax.stop_gradient(state.sensitivity)

        (carry, sensitivity), (dist, value_raw, _) = forward(
            views.acting,
            obs,
            previous_action,
            reward,
            done,
            state.carry,
            state.sensitivity,
        )
        sampled_action, log_prob = dist.sample_and_log_prob(seed=action_key)
        entropy = (config.logprob_scale * remove_time_axis(dist.entropy())).mean()
        sampled_action = remove_time_axis(sampled_action)
        log_prob = remove_time_axis(log_prob)
        value = remove_feature_axis(remove_time_axis(value_raw))
        env_action = (
            jnp.clip(sampled_action, -config.act_clip, config.act_clip)
            if config.act_clip
            else sampled_action
        )

        step_keys = jax.random.split(env_key, config.num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, env_action, env_params)
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
            action=env_action,
            reward=next_reward,
            done=next_done,
        ).to_sequence()
        (
            bootstrap_carry,
            bootstrap_sensitivity,
        ), (_, next_value_raw, _) = forward(
            jax.lax.stop_gradient(views.bootstrap),
            next_obs_s,
            next_action_s,
            next_reward_s,
            next_done_s,
            jax.lax.stop_gradient(carry),
            jax.lax.stop_gradient(sensitivity),
        )
        next_value = remove_feature_axis(remove_time_axis(next_value_raw))
        td_error = td0(
            reward=next_reward,
            value=value,
            next_value=next_value,
            bootstrap_discount=config.gamma * (1 - next_done),
        )
        prediction_target = jnp.concatenate(
            [next_obs, jnp.asarray(next_reward, jnp.float32)[..., None]],
            axis=-1,
        )

        def differentiation(params):
            _, (diff_dist, diff_value_raw, diff_prediction) = forward(
                params,
                obs,
                previous_action,
                reward,
                done,
                pre_carry,
                pre_sensitivity,
            )
            diff_log_prob = remove_time_axis(
                diff_dist.log_prob(add_time_axis(sampled_action))
            )
            diff_value = remove_feature_axis(remove_time_axis(diff_value_raw))
            diff_entropy = remove_time_axis(diff_dist.entropy())
            if diff_prediction is not None:
                diff_prediction = remove_time_axis(diff_prediction)
            directions = objective(
                log_prob=diff_log_prob,
                value=diff_value,
                entropy=diff_entropy,
                prediction=diff_prediction,
                prediction_target=prediction_target,
            )
            return (
                directions.traced_by_domain,
                directions.direct_by_domain,
            )

        traced_by_domain, direct_by_domain = jax.jacobian(differentiation)(
            views.differentiation
        )
        recurrent_keys = ("feature_extractor", "torso")
        trace_gradients = {
            "actor": traced_by_domain["actor"]["actor"],
            "critic": traced_by_domain["critic"]["critic"],
            "recurrent": {
                name: traced_by_domain["recurrent"][name] for name in recurrent_keys
            },
        }
        incoming_domains = {
            "actor": state.traces["actor"],
            "critic": state.traces["critic"],
            "recurrent": {name: state.traces[name] for name in recurrent_keys},
        }
        trace_result = trace_kernel(
            incoming_domains,
            trace_gradients,
            terminated_after=next_done,
            emphasis=state.I,
        )
        carried_traces = {
            "actor": trace_result.carried["actor"],
            "critic": trace_result.carried["critic"],
            **trace_result.carried["recurrent"],
        }
        update_traces = {
            "actor": trace_result.update["actor"],
            "critic": trace_result.update["critic"],
            **trace_result.update["recurrent"],
        }
        if pred_head is not None:
            decay = config.gamma * config.lambda_rnn
            carried_traces["pred"] = jax.tree.map(
                lambda old, grad: (
                    decay
                    * (1 - next_done)[(slice(None),) + (None,) * (old.ndim - 1)]
                    * old
                    + state.I[(slice(None),) + (None,) * (grad.ndim - 1)] * grad
                ),
                state.traces["pred"],
                traced_by_domain["prediction"]["pred"],
            )
            update_traces["pred"] = (
                carried_traces["pred"]
                if config.update_trace_before_td
                else state.traces["pred"]
            )

        direct_grads = {
            "actor": direct_by_domain["actor"]["actor"],
            "critic": direct_by_domain["critic"]["critic"],
            "feature_extractor": direct_by_domain["recurrent"]["feature_extractor"],
            "torso": direct_by_domain["recurrent"]["torso"],
        }
        if pred_head is not None:
            direct_grads["pred"] = direct_by_domain["prediction"]["pred"]

        def scale_trace(trace, scale):
            delta = td_error[(slice(None),) + (None,) * (trace.ndim - 1)]
            return scale * delta * trace

        ascent_updates = {}
        for name in state.params:
            scale = config.eta_f if name in recurrent_keys else 1.0
            combined = jax.tree.map(
                lambda trace, direct: (scale_trace(trace, scale) + direct),
                update_traces[name],
                direct_grads[name],
            )
            ascent_updates[name] = jax.tree.map(
                lambda update: jnp.mean(update, axis=0), combined
            )
        mapped = views.gradient_to_destination(ascent_updates)
        adam_updates, opt_state = optimizer.update(
            mapped.gradient, state.opt_state, mapped.destination
        )
        fast_params = optax.apply_updates(mapped.destination, adam_updates)
        finished = target.finish_update(
            fast_params=fast_params,
            previous_slow_subtree=state.slow_torso,
            sensitivity=sensitivity,
        )
        not_done = 1 - next_done
        next_I = config.gamma * state.I * not_done + next_done
        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        next_reward_f = jnp.asarray(next_reward, dtype=jnp.float32)
        persisted_action = jnp.where(
            jnp.expand_dims(next_done, axis=broadcast_dims),
            jnp.zeros_like(env_action),
            env_action,
        )
        persisted_reward = jnp.where(
            next_done, jnp.zeros_like(next_reward_f), next_reward_f
        )
        action_decision = ActionDecision(
            sampled_action=sampled_action,
            logprob_action=sampled_action,
            env_action=env_action,
            bootstrap_feedback_action=env_action,
            persisted_feedback_action=persisted_action,
        )
        next_state = state.replace(
            step=state.step + config.num_envs,
            update_step=state.update_step + 1,
            timestep=Timestep(
                obs=next_obs,
                action=persisted_action,
                reward=persisted_reward,
                done=next_done,
            ),
            env_state=env_state,
            params=finished.fast_params,
            slow_torso=finished.slow_subtree,
            traces=carried_traces,
            opt_state=opt_state,
            carry=carry,
            sensitivity=finished.sensitivity,
            I=next_I,
            normalizer_state=normalizer_state,
        )
        nu_log = _find_leaf(finished.fast_params["torso"], "nu_log")
        gamma_log = _find_leaf(finished.fast_params["torso"], "gamma_log")
        metrics = MetaStepMetrics(
            action_decision=action_decision,
            log_prob=log_prob,
            value=value,
            next_value=next_value,
            td_error=td_error,
            entropy=entropy,
            acting_carry=carry,
            acting_sensitivity=sensitivity,
            bootstrap_carry=bootstrap_carry,
            bootstrap_sensitivity=bootstrap_sensitivity,
            differentiation_grads={
                "actor": trace_gradients["actor"],
                "critic": trace_gradients["critic"],
                **trace_gradients["recurrent"],
                **(
                    {"pred": traced_by_domain["prediction"]["pred"]}
                    if pred_head is not None
                    else {}
                ),
            },
            direct_grads=direct_grads,
            incoming_traces=state.traces,
            carried_traces=carried_traces,
            update_traces=update_traces,
            ascent_updates=ascent_updates,
            adam_updates=adam_updates,
            adam_state=opt_state,
            prediction_direct_grads=direct_by_domain["prediction"],
            fast_params=finished.fast_params,
            slow_torso=finished.slow_subtree,
            emphasis=state.I.mean(),
            diag_lambda_max=(
                jnp.max(jnp.exp(-jnp.exp(nu_log))) if nu_log is not None else jnp.nan
            ),
            diag_gamma_max=(
                jnp.max(jnp.exp(gamma_log)) if gamma_log is not None else jnp.nan
            ),
            diag_sens_norm=_tree_norm(sensitivity),
            diag_carry_norm=_tree_norm(carry),
            diag_z_rnn=_tree_norm(
                {name: carried_traces[name] for name in recurrent_keys}
            ),
            diag_z_actor=_tree_norm(carried_traces["actor"]),
            diag_z_critic=_tree_norm(carried_traces["critic"]),
            diag_grad_rnn=_tree_norm(trace_gradients["recurrent"]),
            diag_grad_actor=_tree_norm(trace_gradients["actor"]),
            diag_grad_critic=_tree_norm(trace_gradients["critic"]),
            diag_upd_rnn=_tree_norm(
                {name: cast(Any, adam_updates)[name] for name in recurrent_keys}
            ),
            diag_p_torso=_tree_norm(finished.fast_params["torso"]),
            diag_p_actor=_tree_norm(finished.fast_params["actor"]),
            diag_p_critic=_tree_norm(finished.fast_params["critic"]),
            diag_value_abs=jnp.abs(value).mean(),
            diag_td_abs=jnp.abs(td_error).mean(),
            diag_actor_loc_abs=jnp.abs(dist.loc).mean(),
            diag_actor_scale=dist.scale_diag.mean(),
            diag_act_abs=jnp.abs(sampled_action).mean(),
            info=info,
            raw_episode_return=raw_episode_return,
            normalization=normalization_metrics(
                normalizer_state, normalizer.config.eps
            ),
            state_after=next_state,
        )
        return next_state, metrics

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
            carry=torso.initialize_carry(jax.random.key(0), carry_shape),
            sensitivity=torso.initialize_sensitivity(jax.random.key(0), carry_shape),
            normalizer_state=normalizer_state,
        )

        def eval_step(current, step_key):
            action_key, env_key = jax.random.split(step_key)
            del action_key
            obs_s, done_s, action_s, reward_s = current.timestep.to_sequence()
            views = target.views(
                fast_params=current.params, slow_subtree=current.slow_torso
            )
            (carry, sensitivity), (dist, _, _) = forward(
                views.acting,
                obs_s,
                action_s,
                reward_s,
                done_s,
                current.carry,
                current.sensitivity,
            )
            chosen = (
                jnp.argmax(dist.logits, axis=-1)
                if hasattr(dist, "logits")
                else dist.mode()
            )
            chosen = remove_time_axis(chosen)
            if config.act_clip:
                chosen = jnp.clip(chosen, -config.act_clip, config.act_clip)
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
                carry=carry,
                sensitivity=sensitivity,
                normalizer_state=next_normalizer_state,
            ), EvalSummary(
                info={
                    **info,
                    "environment_info": info,
                    "reward": next_reward,
                },
                normalization=normalization_metrics(
                    next_normalizer_state, normalizer.config.eps
                ),
            )

        step_keys = jax.random.split(eval_key, num_steps)
        eval_state, summary = jax.lax.scan(eval_step, eval_state, step_keys)
        return eval_state, summary

    if _debug_sink is not None:
        _debug_sink.append(
            MetaDebugInterface(
                forward=forward,
                optimizer=optimizer,
                step=step_fn,
            )
        )
    return AgentProgram(
        init_fn=init_fn,
        train_epoch_fn=train_epoch_fn,
        evaluate_fn=evaluate_fn,
        state_schema=MetaState,
        metric_schema=MetaStepMetrics,
    )
