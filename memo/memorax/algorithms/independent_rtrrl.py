"""Strictly independent two-path legacy RTRRL.

This preserves the update semantics of :mod:`memorax.algorithms.rtrrl`, while
giving actor and critic separate feature extractors, recurrent torsos, RTRL
state, eligibility traces, slow targets, and Adam moments.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import core, struct

from memorax.rl import (
    NormalizationConfig,
    make_normalizer,
    normalization_metrics,
)
from memorax.utils import Timestep, Transition, find_leaf, tree_norm
from memorax.utils.axes import add_time_axis, remove_feature_axis, remove_time_axis
from memorax.utils.typing import (
    Array,
    Carry,
    Discrete,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)

from .contract import AgentProgram, EvalSummary
from .rtrrl import (
    RTRRLConfig,
    follow_torso,
    group_parameters,
    group_trees,
    make_rtrrl_update_rules,
    slow_view,
)


class IndependentRecurrentKernel(Protocol):
    """Selected recurrent module surface consumed by the independent kernel."""

    def init(self, *args: Any, **kwargs: Any) -> Any: ...

    def apply(self, *args: Any, **kwargs: Any) -> Any: ...

    def initialize_carry(self, key: Any, input_shape: Any) -> Any: ...

    def initialize_sensitivity(self, key: Any, input_shape: Any) -> Any: ...


@struct.dataclass(frozen=True)
class IndependentRTRRLConfig(RTRRLConfig):
    """Legacy RTRRL hyperparameters for the independent two-path topology."""


@struct.dataclass(frozen=True)
class IndependentRTRRLState:
    """Training state with no actor/critic parameter or recurrent-state sharing."""

    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: PyTree
    critic_params: PyTree
    actor_slow_torso: PyTree
    critic_slow_torso: PyTree
    actor_traces: PyTree
    critic_traces: PyTree
    actor_opt_state: Any
    critic_opt_state: Any
    actor_carry: Carry
    critic_carry: Carry
    actor_sensitivity: Any
    critic_sensitivity: Any
    emphasis: Array
    normalizer_state: Any


@struct.dataclass
class IndependentStepMetrics:
    """Fixed metrics emitted by the independent selected-component kernel."""

    info: Any = None
    td_error: Any = None
    entropy: Any = None
    value: Any = None
    emphasis: Any = None
    diag_actor_lambda_max: Any = None
    diag_critic_lambda_max: Any = None
    diag_actor_sens_norm: Any = None
    diag_critic_sens_norm: Any = None
    diag_actor_carry_norm: Any = None
    diag_critic_carry_norm: Any = None
    diag_z_actor: Any = None
    diag_z_critic: Any = None
    diag_act_abs: Any = None
    normalization: Any = None


@dataclass
class IndependentRTRRL:
    """RTRRL with strictly independent actor and critic recurrent pathways."""

    cfg: IndependentRTRRLConfig
    env: Environment
    env_params: EnvParams
    actor_feature_extractor: nn.Module
    actor_torso: IndependentRecurrentKernel
    actor_head: nn.Module
    critic_feature_extractor: nn.Module
    critic_torso: IndependentRecurrentKernel
    critic_head: nn.Module
    activation: Callable = jax.nn.silu
    program_normalization: NormalizationConfig | None = None
    actor_rules: Any = field(default=None, init=False)
    critic_rules: Any = field(default=None, init=False)
    normalizer: Any = field(default=None, init=False)

    def __post_init__(self):
        config = self.program_normalization or NormalizationConfig()
        self.normalizer = make_normalizer(config)

    def _forward(
        self,
        feature_extractor: nn.Module,
        torso: IndependentRecurrentKernel,
        head: nn.Module,
        params: PyTree,
        obs: Array,
        action: Array,
        reward: Array,
        done: Array,
        carry: Carry,
        sensitivity: Any,
    ):
        x, _ = feature_extractor.apply(
            {"params": params["feature_extractor"]},
            observation=obs,
            action=action,
            reward=reward,
            done=done,
        )
        next_carry, h, next_sensitivity = torso.apply(
            {"params": params["torso"]},
            x,
            done,
            carry,
            sensitivity=sensitivity,
            method="local_jacobian",
        )
        output, _ = head.apply(
            {"params": params["head"]},
            self.activation(h),
            action=action,
            reward=reward,
            done=done,
        )
        return (next_carry, next_sensitivity), output

    def _actor_forward(self, params, obs, action, reward, done, carry, sensitivity):
        return self._forward(
            self.actor_feature_extractor,
            self.actor_torso,
            self.actor_head,
            params,
            obs,
            action,
            reward,
            done,
            carry,
            sensitivity,
        )

    def _critic_forward(self, params, obs, action, reward, done, carry, sensitivity):
        return self._forward(
            self.critic_feature_extractor,
            self.critic_torso,
            self.critic_head,
            params,
            obs,
            action,
            reward,
            done,
            carry,
            sensitivity,
        )

    @staticmethod
    def _init_rule_state(rules, params, traces):
        group_of = group_parameters(params)
        grouped_params = group_trees(params, group_of)
        grouped_traces = group_trees(traces, group_of)
        return {
            group: rule.init(
                params=grouped_params[group],
                traces=grouped_traces[group],
            )
            for group, rule in rules.items()
        }

    def _step_branch(self, rules, params, traces, direct, opt_state, delta, step):
        """Step one branch's parameters with the shared RTRRL update rules."""

        group_of = group_parameters(params)
        scaled = {
            name: (
                tree
                if group_of[name] == "td"
                else jax.tree.map(lambda leaf: self.cfg.eta_f * leaf, tree)
            )
            for name, tree in traces.items()
        }
        grouped_traces = group_trees(scaled, group_of)
        grouped_direct = group_trees(direct, group_of) if direct is not None else None
        grouped_params = group_trees(params, group_of)
        outputs = {
            group: rule.apply(
                grouped_traces[group],
                None if grouped_direct is None else grouped_direct[group],
                opt_state[group],
                delta=delta,
                step=step,
                params=grouped_params[group],
            )
            for group, rule in rules.items()
        }
        updates = {
            name: outputs[group].updates[name] for name, group in group_of.items()
        }
        return (
            cast(Any, optax.apply_updates(params, updates)),
            {group: output.state for group, output in outputs.items()},
        )

    def _update_traces(self, traces, grads, head_decay, done, emphasis):
        def decay_for(key):
            return head_decay if key == "head" else self.cfg.gamma * self.cfg.lambda_rnn

        def update(z, g, decay):
            trailing = z.ndim - 1
            nd = (1 - done)[(slice(None),) + (None,) * trailing]
            weight = emphasis[(slice(None),) + (None,) * trailing]
            return decay * nd * z + weight * g

        return {
            key: jax.tree.map(
                partial(lambda z, g, d: update(z, g, d), d=decay_for(key)),
                traces[key],
                grads[key],
            )
            for key in traces
        }

    def _deterministic_action(self, key: Key, state: IndependentRTRRLState):
        del key
        obs, done, action, reward = state.timestep.to_sequence()
        actor_gp = slow_view(state.actor_params, state.actor_slow_torso)
        critic_gp = slow_view(state.critic_params, state.critic_slow_torso)
        (actor_carry, actor_sensitivity), dist = self._actor_forward(
            actor_gp,
            obs,
            action,
            reward,
            done,
            state.actor_carry,
            state.actor_sensitivity,
        )
        (critic_carry, critic_sensitivity), value = self._critic_forward(
            critic_gp,
            obs,
            action,
            reward,
            done,
            state.critic_carry,
            state.critic_sensitivity,
        )
        selected = (
            jnp.argmax(dist.logits, axis=-1)
            if isinstance(self.env.action_space(self.env_params), Discrete)
            else dist.mode()
        )
        log_prob = remove_time_axis(dist.log_prob(selected))
        selected = remove_time_axis(selected)
        value = remove_feature_axis(remove_time_axis(value))
        state = cast(Any, state).replace(
            actor_carry=actor_carry,
            actor_sensitivity=actor_sensitivity,
            critic_carry=critic_carry,
            critic_sensitivity=critic_sensitivity,
        )
        return state, selected, log_prob, value

    def _step(self, state, key, *, policy):
        action_key, step_key = jax.random.split(key)
        state, action, log_prob, value = policy(action_key, state)
        clip = self.cfg.act_clip
        env_action = jnp.clip(action, -clip, clip) if clip else action
        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, env_action, self.env_params)
        normalized = self.normalizer.step(
            state.normalizer_state,
            observation=next_obs,
            reward=reward,
            done=done,
            update=self.normalizer.config.update_during_eval,
        )
        next_obs = normalized.observation
        reward = normalized.reward
        transition = Transition(
            first=state.timestep,
            second=Timestep(obs=None, action=action, reward=reward, done=done),
            aux={"log_prob": log_prob, "value": value},
        )
        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        return (
            state.replace(
                step=state.step + self.cfg.num_envs,
                timestep=Timestep(
                    obs=next_obs,
                    action=jnp.where(
                        jnp.expand_dims(done, axis=broadcast_dims),
                        jnp.zeros_like(env_action),
                        env_action,
                    ),
                    reward=jnp.where(done, jnp.zeros_like(reward), reward),
                    done=done,
                ),
                env_state=env_state,
                normalizer_state=normalized.state,
            ),
            (transition, info),
        )

    def _update_step(self, state: IndependentRTRRLState, key: Key):
        action_key, step_key = jax.random.split(key)
        obs, done, ts_action, reward = state.timestep.to_sequence()
        actor_gp = slow_view(state.actor_params, state.actor_slow_torso)
        critic_gp = slow_view(state.critic_params, state.critic_slow_torso)

        (actor_carry, actor_sensitivity), dist = self._actor_forward(
            actor_gp,
            obs,
            ts_action,
            reward,
            done,
            state.actor_carry,
            state.actor_sensitivity,
        )
        (critic_carry, critic_sensitivity), value_seq = self._critic_forward(
            critic_gp,
            obs,
            ts_action,
            reward,
            done,
            state.critic_carry,
            state.critic_sensitivity,
        )
        action, log_prob = dist.sample_and_log_prob(seed=action_key)
        entropy = (self.cfg.logprob_scale * remove_time_axis(dist.entropy())).mean()
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)
        value = remove_feature_axis(remove_time_axis(value_seq))
        clip = self.cfg.act_clip
        env_action = jnp.clip(action, -clip, clip) if clip else action

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, env_action, self.env_params)
        normalized = self.normalizer.step(
            state.normalizer_state,
            observation=next_obs,
            reward=next_reward,
            done=next_done,
            update=True,
        )
        next_obs = normalized.observation
        next_reward = normalized.reward

        next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
            obs=next_obs, action=env_action, reward=next_reward, done=next_done
        ).to_sequence()
        # Bootstrap is a probe only: its resulting critic carry/sensitivity is
        # deliberately discarded, so the committed state advances exactly once.
        _, next_value_seq = self._critic_forward(
            jax.lax.stop_gradient(critic_gp),
            next_obs_s,
            next_action_s,
            next_reward_s,
            next_done_s,
            jax.lax.stop_gradient(critic_carry),
            jax.lax.stop_gradient(critic_sensitivity),
        )
        next_value = remove_feature_axis(remove_time_axis(next_value_seq))
        td_error = next_reward + self.cfg.gamma * (1 - next_done) * next_value - value

        initial_actor_carry = jax.lax.stop_gradient(state.actor_carry)
        initial_actor_sens = jax.lax.stop_gradient(state.actor_sensitivity)
        initial_critic_carry = jax.lax.stop_gradient(state.critic_carry)
        initial_critic_sens = jax.lax.stop_gradient(state.critic_sensitivity)

        def actor_objective(params):
            _, d = self._actor_forward(
                params,
                obs,
                ts_action,
                reward,
                done,
                initial_actor_carry,
                initial_actor_sens,
            )
            return (
                self.cfg.eta_pi
                * self.cfg.logprob_scale
                * remove_time_axis(d.log_prob(add_time_axis(action)))
            )

        def critic_objective(params):
            _, v = self._critic_forward(
                params,
                obs,
                ts_action,
                reward,
                done,
                initial_critic_carry,
                initial_critic_sens,
            )
            return remove_feature_axis(remove_time_axis(v))

        def entropy_objective(params):
            _, d = self._actor_forward(
                params,
                obs,
                ts_action,
                reward,
                done,
                initial_actor_carry,
                initial_actor_sens,
            )
            return (
                self.cfg.entropy_rate
                * self.cfg.logprob_scale
                * remove_time_axis(d.entropy())
            )

        actor_grads = jax.jacobian(actor_objective)(actor_gp)
        critic_grads = jax.jacobian(critic_objective)(critic_gp)
        entropy_grads = jax.jacobian(entropy_objective)(actor_gp)
        actor_traces_new = self._update_traces(
            state.actor_traces,
            actor_grads,
            self.cfg.gamma * self.cfg.lambda_pi,
            next_done,
            state.emphasis,
        )
        critic_traces_new = self._update_traces(
            state.critic_traces,
            critic_grads,
            self.cfg.gamma * self.cfg.lambda_v,
            next_done,
            state.emphasis,
        )
        actor_traces = (
            actor_traces_new if self.cfg.update_trace_before_td else state.actor_traces
        )
        critic_traces = (
            critic_traces_new
            if self.cfg.update_trace_before_td
            else state.critic_traces
        )
        current_step = state.update_step + 1
        actor_params, actor_opt_state = self._step_branch(
            self.actor_rules,
            state.actor_params,
            actor_traces,
            entropy_grads,
            state.actor_opt_state,
            td_error,
            current_step,
        )
        critic_params, critic_opt_state = self._step_branch(
            self.critic_rules,
            state.critic_params,
            critic_traces,
            None,
            state.critic_opt_state,
            td_error,
            current_step,
        )
        actor_slow_torso = follow_torso(
            actor_params["torso"], state.actor_slow_torso, self.cfg.update_period
        )
        critic_slow_torso = follow_torso(
            critic_params["torso"], state.critic_slow_torso, self.cfg.update_period
        )

        not_done = 1 - next_done
        next_emphasis = self.cfg.gamma * state.emphasis * not_done + next_done
        actor_nu = find_leaf(actor_params["torso"], "nu_log")
        critic_nu = find_leaf(critic_params["torso"], "nu_log")

        broadcast_dims = tuple(
            range(
                cast(Any, state.timestep.done).ndim,
                cast(Any, state.timestep.action).ndim,
            )
        )
        next_reward = jnp.asarray(next_reward, dtype=jnp.float32)
        return (
            cast(Any, state).replace(
                step=state.step + self.cfg.num_envs,
                update_step=state.update_step + 1,
                timestep=Timestep(
                    obs=next_obs,
                    action=jnp.where(
                        jnp.expand_dims(next_done, axis=broadcast_dims),
                        jnp.zeros_like(env_action),
                        env_action,
                    ),
                    reward=jnp.where(
                        next_done, jnp.zeros_like(next_reward), next_reward
                    ),
                    done=next_done,
                ),
                env_state=env_state,
                actor_params=actor_params,
                critic_params=critic_params,
                actor_slow_torso=actor_slow_torso,
                critic_slow_torso=critic_slow_torso,
                actor_traces=actor_traces_new,
                critic_traces=critic_traces_new,
                actor_opt_state=actor_opt_state,
                critic_opt_state=critic_opt_state,
                actor_carry=actor_carry,
                critic_carry=critic_carry,
                actor_sensitivity=actor_sensitivity,
                critic_sensitivity=critic_sensitivity,
                emphasis=next_emphasis,
                normalizer_state=normalized.state,
            ),
            IndependentStepMetrics(
                info=info,
                td_error=td_error,
                entropy=entropy,
                value=value,
                emphasis=state.emphasis.mean(),
                diag_actor_lambda_max=(
                    jnp.max(jnp.exp(-jnp.exp(actor_nu)))
                    if actor_nu is not None
                    else jnp.nan
                ),
                diag_critic_lambda_max=(
                    jnp.max(jnp.exp(-jnp.exp(critic_nu)))
                    if critic_nu is not None
                    else jnp.nan
                ),
                diag_actor_sens_norm=tree_norm(actor_sensitivity),
                diag_critic_sens_norm=tree_norm(critic_sensitivity),
                diag_actor_carry_norm=tree_norm(actor_carry),
                diag_critic_carry_norm=tree_norm(critic_carry),
                diag_z_actor=tree_norm(actor_traces_new),
                diag_z_critic=tree_norm(critic_traces_new),
                diag_act_abs=jnp.abs(action).mean(),
                normalization=normalization_metrics(
                    normalized.state, self.normalizer.config.eps
                ),
            ),
        )

    def _init_branch(
        self,
        feature_extractor,
        torso,
        head,
        keys,
        timestep,
        carry,
    ):
        feat_key, torso_key, head_key, sens_key = keys
        obs, done, action, reward = timestep
        feat_vars = feature_extractor.init(
            {"params": feat_key},
            observation=obs,
            action=action,
            reward=reward,
            done=done,
        )
        x, _ = feature_extractor.apply(
            feat_vars,
            observation=obs,
            action=action,
            reward=reward,
            done=done,
        )
        torso_vars = torso.init({"params": torso_key}, x, done, initial_carry=carry)
        _, h = torso.apply(torso_vars, x, done, initial_carry=carry)
        head_vars = head.init(
            {"params": head_key},
            self.activation(h),
            action=action,
            reward=reward,
            done=done,
        )
        params = {
            "feature_extractor": feat_vars.get("params", core.FrozenDict()),
            "torso": torso_vars["params"],
            "head": head_vars["params"],
        }
        traces = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape)), params
        )
        sensitivity = torso.initialize_sensitivity(sens_key, (self.cfg.num_envs, None))
        return params, traces, sensitivity

    def init(self, key: Key) -> IndependentRTRRLState:
        split = jax.random.split(key, 10)
        env_key = split[0]
        actor_keys = split[1:5]
        critic_keys = split[5:9]
        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        obs, normalizer_state = self.normalizer.reset(obs, None, update=True)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()
        carry_shape = (self.cfg.num_envs, None)
        actor_carry = self.actor_torso.initialize_carry(split[9], carry_shape)
        critic_carry = self.critic_torso.initialize_carry(
            jax.random.fold_in(split[9], 1), carry_shape
        )
        actor_params, actor_traces, actor_sensitivity = self._init_branch(
            self.actor_feature_extractor,
            self.actor_torso,
            self.actor_head,
            actor_keys,
            timestep,
            actor_carry,
        )
        critic_params, critic_traces, critic_sensitivity = self._init_branch(
            self.critic_feature_extractor,
            self.critic_torso,
            self.critic_head,
            critic_keys,
            timestep,
            critic_carry,
        )
        self.actor_rules = make_rtrrl_update_rules(self.cfg, actor_params)
        self.critic_rules = make_rtrrl_update_rules(self.cfg, critic_params)
        return IndependentRTRRLState(
            step=0,
            update_step=0,
            timestep=timestep.from_sequence(),
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            actor_slow_torso=actor_params["torso"],
            critic_slow_torso=critic_params["torso"],
            actor_traces=actor_traces,
            critic_traces=critic_traces,
            actor_opt_state=self._init_rule_state(
                self.actor_rules, actor_params, actor_traces
            ),
            critic_opt_state=self._init_rule_state(
                self.critic_rules, critic_params, critic_traces
            ),
            actor_carry=actor_carry,
            critic_carry=critic_carry,
            actor_sensitivity=actor_sensitivity,
            critic_sensitivity=critic_sensitivity,
            emphasis=jnp.ones((self.cfg.num_envs,), dtype=jnp.float32),
            normalizer_state=normalizer_state,
        )

    def warmup(self, key, state, num_steps):
        del key, num_steps
        return state

    def train(self, key, state, num_steps):
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(self._update_step, state, keys)
        return state

    def _evaluate_with_summary(self, key, state, num_steps):
        reset_key, eval_key, actor_sens_key, critic_sens_key = jax.random.split(key, 4)
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
        )
        obs, normalizer_state = self.normalizer.reset(
            obs,
            None if self.normalizer.config.reset_on_start else state.normalizer_state,
            update=self.normalizer.config.update_during_eval,
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        carry_shape = (self.cfg.num_envs, None)
        state = state.replace(
            timestep=Timestep(obs=obs, action=action, reward=reward, done=done),
            env_state=env_state,
            actor_carry=self.actor_torso.initialize_carry(
                jax.random.fold_in(actor_sens_key, 0), carry_shape
            ),
            critic_carry=self.critic_torso.initialize_carry(
                jax.random.fold_in(critic_sens_key, 0), carry_shape
            ),
            actor_sensitivity=self.actor_torso.initialize_sensitivity(
                actor_sens_key, carry_shape
            ),
            critic_sensitivity=self.critic_torso.initialize_sensitivity(
                critic_sens_key, carry_shape
            ),
            normalizer_state=normalizer_state,
        )
        keys = jax.random.split(eval_key, num_steps)
        state, summaries = jax.lax.scan(
            partial(self._step, policy=self._deterministic_action), state, keys
        )
        _, info = summaries
        return state, EvalSummary(
            info=info,
            normalization=normalization_metrics(
                state.normalizer_state, self.normalizer.config.eps
            ),
        )

    def evaluate(self, key, state, num_steps):
        """Preserve the legacy state-only evaluation lifecycle."""
        return self._evaluate_with_summary(key, state, num_steps)[0]

    def as_program(self) -> AgentProgram:
        """Expose the fixed independent state schema through the common program."""

        def train_epoch_fn(key, state, num_steps):
            keys = jax.random.split(key, num_steps // self.cfg.num_envs)
            return jax.lax.scan(self._update_step, state, keys)

        def evaluate_fn(key, state, num_steps):
            return self._evaluate_with_summary(key, state, num_steps)

        return AgentProgram(
            init_fn=self.init,
            train_epoch_fn=train_epoch_fn,
            evaluate_fn=evaluate_fn,
            state_schema=IndependentRTRRLState,
            metric_schema=IndependentStepMetrics,
        )
