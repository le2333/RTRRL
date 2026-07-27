"""RTRRL: one recurrent torso shared by an actor and a critic.

The torso is credited by exact RTRL, so every parameter carries an eligibility
trace that the TD error weights on arrival. Forward passes read a slowly
following copy of the torso while updates land on the fast parameters, which
keeps the bootstrap target from moving with the parameters it evaluates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from flax import core, struct

from memorax.rl import (
    NormalizationConfig,
    ObjectiveDirections,
    environment_owns_normalization,
    make_exact_rtrl_credit,
    make_normalizer,
    make_obgd_rule,
    make_optax_rule,
    make_td0,
    normalization_metrics,
)
from memorax.utils import Timestep, find_leaf, tree_norm
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)

from .contract import ActionDecision, AgentProgram, EvalSummary, EvaluationConfig

RECURRENT_DOMAINS = ("feature_extractor", "torso")


@dataclass(frozen=True)
class RTRRLConfig:
    """Everything the kernel reads that does not change during a run."""

    num_envs: int
    gamma: float = 0.95
    lambda_pi: float = 0.97
    lambda_v: float = 0.9
    lambda_rnn: float = 0.945
    td_lr: float = 3e-5
    rnn_lr: float = 2e-6
    eta_pi: float = 0.38
    eta_f: float = 0.5
    entropy_rate: float = 3e-5
    update_period: float = 0.1
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    rnn_grad_clip: float = 1.0
    act_clip: float = 0.0
    freeze_gamma: bool = False
    update_trace_before_td: bool = True
    logprob_scale: float = 1.0
    update_rule: str = "adam"
    kappa: float = 2.0
    obgd_beta2: float = 0.0
    obgd_adaptive: bool = False
    actor_to_recurrent: bool = True
    critic_to_recurrent: bool = True


@dataclass(frozen=True)
class RTRRLParts:
    """The modules and environment a program is built around."""

    env: Any = None
    env_params: Any = None
    feature_extractor: Any = None
    torso: Any = None
    actor_head: Any = None
    critic_head: Any = None
    activation: Callable[[Any], Any] = jax.nn.silu
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


def make_rtrrl_trace(config):
    """Build RTRRL's three-domain, post-transition trace recurrence."""

    decays = {
        "actor": config.gamma * config.lambda_pi,
        "critic": config.gamma * config.lambda_v,
        "recurrent": config.gamma * config.lambda_rnn,
    }
    use_fresh = config.update_trace_before_td

    def rtrrl_trace(incoming, gradient, *, terminated_after, emphasis):
        carried = {
            domain: jax.tree.map(
                lambda old, grad, decay=decay: (
                    decay * (1 - _broadcast_env(terminated_after, old)) * old
                    + _broadcast_env(emphasis, grad) * grad
                ),
                incoming[domain],
                gradient[domain],
            )
            for domain, decay in decays.items()
        }
        return TraceDirections(
            carried=carried,
            update=carried if use_fresh else incoming,
        )

    return rtrrl_trace


def make_rtrrl_objective(config):
    """Route each ascent direction to the domains that should receive it.

    The recurrent domain is the interesting one: it is the only place the
    actor and the critic ask the same parameters to move. Closing either gate
    leaves that side learning from the shared torso while no longer steering
    it, which is what separates a representation the two disagree about from
    one that is simply wrong. Entropy follows the actor gate, being an
    actor-side objective.
    """

    def rtrrl_objective(*, log_prob, value, entropy):
        actor = config.eta_pi * config.logprob_scale * log_prob
        entropy_direction = config.entropy_rate * config.logprob_scale * entropy
        zero = jnp.zeros_like(value)

        recurrent_traced = zero
        recurrent_direct = zero
        if config.actor_to_recurrent:
            recurrent_traced = recurrent_traced + actor
            recurrent_direct = recurrent_direct + entropy_direction
        if config.critic_to_recurrent:
            recurrent_traced = recurrent_traced + value

        return ObjectiveDirections(
            traced_by_domain={
                "actor": actor,
                "critic": value,
                "recurrent": recurrent_traced,
            },
            direct_by_domain={
                "actor": entropy_direction,
                "critic": zero,
                "recurrent": recurrent_direct,
            },
            metrics={"entropy": entropy},
        )

    return rtrrl_objective


def slow_view(fast_params, slow_torso):
    """Read the slow torso while every other parameter stays current."""

    if isinstance(fast_params, core.FrozenDict):
        return fast_params.copy(add_or_replace={"torso": slow_torso})
    return {**fast_params, "torso": slow_torso}


def follow_torso(fast_torso, slow_torso, update_period):
    if update_period == 1.0:
        return fast_torso
    return optax.incremental_update(fast_torso, slow_torso, update_period)


def group_parameters(params):
    """Say which rule group each top-level parameter tree belongs to.

    The recurrent parameters are clipped as one and may hold a frozen decay;
    everything else steps plainly. Every RTRRL topology splits the same way,
    whichever names it gives its heads.
    """

    return {name: ("rnn" if name in RECURRENT_DOMAINS else "td") for name in params}


def group_trees(by_name, group_of):
    """Gather per-parameter trees into the groups the rules step over."""

    groups = {}
    for name, tree in by_name.items():
        groups.setdefault(group_of[name], {})[name] = tree
    return groups


def make_rtrrl_update_rules(config, abstract_params):
    """One rule per parameter group, both answering the same contract.

    Which mechanism steps is a config choice; the groups do not move, because
    the recurrent parameters are clipped together and may hold a frozen decay
    while the actor and critic never do. OBGD ignores ``rnn_grad_clip`` by
    construction, since bounding the step is the whole of what it does.
    """

    if config.update_rule == "obgd":
        if config.freeze_gamma:
            raise ValueError(
                "freeze_gamma has no effect under OBGD: the bound scales a whole "
                "group at once and cannot hold one leaf still"
            )
        return {
            group: make_obgd_rule(
                learning_rate=rate,
                kappa=config.kappa,
                beta2=config.obgd_beta2,
                eps=config.eps,
                adaptive=config.obgd_adaptive,
            )
            for group, rate in (("rnn", config.rnn_lr), ("td", config.td_lr))
        }
    if config.update_rule != "adam":
        raise ValueError(f"unknown update rule: {config.update_rule!r}")

    recurrent = []
    if config.rnn_grad_clip:
        recurrent.append(optax.clip_by_global_norm(config.rnn_grad_clip))
    recurrent.extend(
        (
            optax.scale_by_adam(b1=config.b1, b2=config.b2, eps=config.eps),
            optax.scale(config.rnn_lr),
        )
    )
    recurrent_transform = optax.chain(*recurrent)
    if config.freeze_gamma:
        labels = jax.tree_util.tree_map_with_path(
            lambda path, _leaf: (
                "frozen"
                if any(getattr(part, "key", None) == "gamma_log" for part in path)
                else "rnn"
            ),
            {name: abstract_params[name] for name in RECURRENT_DOMAINS},
        )
        recurrent_transform = optax.multi_transform(
            {"rnn": recurrent_transform, "frozen": optax.set_to_zero()},
            labels,
        )
    return {
        "rnn": make_optax_rule(recurrent_transform),
        "td": make_optax_rule(
            optax.chain(
                optax.scale_by_adam(b1=config.b1, b2=config.b2, eps=config.eps),
                optax.scale(config.td_lr),
            )
        ),
    }


@struct.dataclass(frozen=True)
class RTRRLState:
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
    emphasis: Any
    normalizer_state: Any = None


@struct.dataclass(frozen=True)
class RTRRLStepMetrics:
    """Fixed-shape observables from one transition.

    Everything here is a scalar or one step of trajectory. The kernel runs
    under ``lax.scan``, so anything returned is stacked once per step; whole
    parameter, trace, and optimiser trees used to be returned alongside these
    and cost memory proportional to epoch length times model size.
    """

    action_decision: ActionDecision | None = None
    log_prob: Any = None
    value: Any = None
    next_value: Any = None
    td_error: Any = None
    entropy: Any = None
    emphasis: Any = None
    step_size: Any = None
    observation: Any = None
    reward: Any = None
    done: Any = None
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


def build_rtrrl(config: RTRRLConfig, parts: RTRRLParts) -> AgentProgram:
    """Close over the modules, the environment, and every static choice.

    Nothing after this point reads Python state, so the returned functions are
    safe to trace once and call repeatedly.
    """

    normalizer = make_normalizer(parts.normalization or NormalizationConfig())
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
    feature_extractor = parts.feature_extractor
    torso = parts.torso
    actor_head = parts.actor_head
    critic_head = parts.critic_head
    activation = parts.activation
    if normalization_enabled and environment_owns_normalization(env):
        raise ValueError(
            "normalization owner conflict: wrapper and program normalization are both enabled"
        )
    record_trajectory = parts.record_trajectory
    credit = make_exact_rtrl_credit(torso)
    objective = make_rtrrl_objective(config)
    trace_kernel = make_rtrrl_trace(config)
    td0 = make_td0()

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
        return (next_carry, next_sensitivity), (dist, value)

    def initialize_arrays(key):
        (
            env_key,
            feat_key,
            torso_key,
            actor_key,
            critic_key,
            sens_key,
        ) = jax.random.split(key, 6)
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
    rules = make_rtrrl_update_rules(config, abstract["params"])
    group_of = group_parameters(abstract["params"])

    def regroup(by_name):
        return group_trees(by_name, group_of)

    def init_fn(key):
        arrays = initialize_arrays(key)
        params = arrays["params"]
        grouped_params = regroup(params)
        grouped_traces = regroup(arrays["traces"])
        return RTRRLState(
            step=jnp.asarray(0, dtype=jnp.int32),
            update_step=jnp.asarray(0, dtype=jnp.int32),
            timestep=arrays["timestep"],
            env_state=arrays["env_state"],
            params=params,
            slow_torso=params["torso"],
            traces=arrays["traces"],
            opt_state={
                group: rule.init(
                    params=grouped_params[group],
                    traces=grouped_traces[group],
                )
                for group, rule in rules.items()
            },
            carry=arrays["carry"],
            sensitivity=arrays["sensitivity"],
            emphasis=jnp.ones((config.num_envs,), dtype=jnp.float32),
            normalizer_state=arrays["normalizer_state"],
        )

    def step_fn(state, key):
        action_key, env_key = jax.random.split(key)
        obs, done, previous_action, reward = state.timestep.to_sequence()
        forward_params = slow_view(state.params, state.slow_torso)
        pre_carry = jax.lax.stop_gradient(state.carry)
        pre_sensitivity = jax.lax.stop_gradient(state.sensitivity)

        (carry, sensitivity), (dist, value_raw, _) = forward(
            forward_params,
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
            _bootstrap_carry,
            _bootstrap_sensitivity,
        ), (_, next_value_raw, _) = forward(
            jax.lax.stop_gradient(forward_params),
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

        def differentiation(params):
            _, (diff_dist, diff_value_raw) = forward(
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
            directions = objective(
                log_prob=diff_log_prob,
                value=diff_value,
                entropy=diff_entropy,
            )
            return (
                directions.traced_by_domain,
                directions.direct_by_domain,
            )

        traced_by_domain, direct_by_domain = jax.jacobian(differentiation)(
            forward_params
        )
        recurrent_keys = RECURRENT_DOMAINS
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
            emphasis=state.emphasis,
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
        direct_grads = {
            "actor": direct_by_domain["actor"]["actor"],
            "critic": direct_by_domain["critic"]["critic"],
            "feature_extractor": direct_by_domain["recurrent"]["feature_extractor"],
            "torso": direct_by_domain["recurrent"]["torso"],
        }
        scaled_traces = {
            name: (
                jax.tree.map(lambda trace: config.eta_f * trace, update_traces[name])
                if name in recurrent_keys
                else update_traces[name]
            )
            for name in state.params
        }
        grouped_traces = regroup(scaled_traces)
        grouped_direct = regroup(direct_grads)
        grouped_params = regroup(state.params)
        outputs = {
            group: rule.apply(
                grouped_traces[group],
                grouped_direct[group],
                state.opt_state[group],
                delta=td_error,
                step=state.update_step + 1,
                params=grouped_params[group],
            )
            for group, rule in rules.items()
        }
        updates = {
            name: outputs[group].updates[name] for name, group in group_of.items()
        }
        opt_state = {group: output.state for group, output in outputs.items()}
        fast_params = cast(dict[str, Any], optax.apply_updates(state.params, updates))
        slow_torso = follow_torso(
            fast_params["torso"], state.slow_torso, config.update_period
        )
        not_done = 1 - next_done
        next_emphasis = config.gamma * state.emphasis * not_done + next_done
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
            params=fast_params,
            slow_torso=slow_torso,
            traces=carried_traces,
            opt_state=opt_state,
            carry=carry,
            sensitivity=sensitivity,
            emphasis=next_emphasis,
            normalizer_state=normalizer_state,
        )
        nu_log = find_leaf(fast_params["torso"], "nu_log")
        gamma_log = find_leaf(fast_params["torso"], "gamma_log")
        metrics = RTRRLStepMetrics(
            action_decision=action_decision,
            log_prob=log_prob,
            value=value,
            next_value=next_value,
            td_error=td_error,
            entropy=entropy,
            emphasis=state.emphasis.mean(),
            step_size=outputs["rnn"].metrics.get("step_size"),
            observation=next_obs if record_trajectory else None,
            reward=next_reward_f if record_trajectory else None,
            done=next_done if record_trajectory else None,
            diag_lambda_max=(
                jnp.max(jnp.exp(-jnp.exp(nu_log))) if nu_log is not None else jnp.nan
            ),
            diag_gamma_max=(
                jnp.max(jnp.exp(gamma_log)) if gamma_log is not None else jnp.nan
            ),
            diag_sens_norm=tree_norm(sensitivity),
            diag_carry_norm=tree_norm(carry),
            diag_z_rnn=tree_norm(
                {name: carried_traces[name] for name in recurrent_keys}
            ),
            diag_z_actor=tree_norm(carried_traces["actor"]),
            diag_z_critic=tree_norm(carried_traces["critic"]),
            diag_grad_rnn=tree_norm(trace_gradients["recurrent"]),
            diag_grad_actor=tree_norm(trace_gradients["actor"]),
            diag_grad_critic=tree_norm(trace_gradients["critic"]),
            diag_upd_rnn=tree_norm({name: updates[name] for name in recurrent_keys}),
            diag_p_torso=tree_norm(fast_params["torso"]),
            diag_p_actor=tree_norm(fast_params["actor"]),
            diag_p_critic=tree_norm(fast_params["critic"]),
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
            (carry, sensitivity), (dist, _, _) = forward(
                slow_view(current.params, current.slow_torso),
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
            # What the environment paid, kept before normalisation overwrites
            # it. Episode returns and the score are read off the summary below,
            # and those are statements about the task, not about the scale the
            # agent happens to be learning on.
            environment_reward = jnp.asarray(next_reward, dtype=jnp.float32)
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
                    "reward": environment_reward,
                },
                normalization=normalization_metrics(
                    next_normalizer_state, normalizer.config.eps
                ),
                observation=current.timestep.obs,
                next_observation=next_obs,
                action=chosen,
                reward=environment_reward,
                done=next_done,
            )

        step_keys = jax.random.split(eval_key, num_steps)
        eval_state, summary = jax.lax.scan(eval_step, eval_state, step_keys)
        return eval_state, summary

    return AgentProgram(
        init_fn=init_fn,
        train_epoch_fn=train_epoch_fn,
        evaluate_fn=evaluate_fn,
        state_schema=RTRRLState,
        metric_schema=RTRRLStepMetrics,
    )
