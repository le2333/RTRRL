"""Strictly independent two-path legacy RTRRL.

This preserves the update semantics of :mod:`memorax.algorithms.rtrrl`, while
giving actor and critic separate feature extractors, recurrent torsos, RTRL
state, eligibility traces, slow targets, and Adam moments.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Protocol, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from memorax.online_ac.types import AgentProgram, EvalSummary
from memorax.rl import (
    NormalizationConfig,
    make_normalizer,
    normalization_metrics,
)
from memorax.utils import Timestep, Transition
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

from .rtrrl import RTRRLConfig, _find_leaf, _tree_norm


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
    I: Array  # noqa: E741 - retained legacy emphasis-state name
    normalizer_state: Any


@struct.dataclass
class IndependentStepMetrics:
    """Fixed metrics emitted by the independent selected-component kernel."""

    info: Any = None
    td_error: Any = None
    entropy: Any = None
    value: Any = None
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
    actor_optimizer: Any = field(default=None, init=False)
    critic_optimizer: Any = field(default=None, init=False)
    normalizer: Any = field(default=None, init=False)

    def __post_init__(self):
        if self.cfg.pred_obs:
            raise ValueError("IndependentRTRRL does not support pred_obs.")
        config = self.program_normalization or NormalizationConfig(
            normalize_observation=self.cfg.normalize_obs,
            normalize_reward=self.cfg.normalize_reward,
            reward_gamma=self.cfg.gamma,
        )
        self.normalizer = make_normalizer(config)

    @staticmethod
    def _grad_params(params: PyTree, slow_torso: PyTree) -> PyTree:
        return {**params, "torso": slow_torso}

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

    def _make_optimizer(self, params: PyTree) -> optax.GradientTransformation:
        c = self.cfg
        td_tx = optax.chain(
            optax.scale_by_adam(b1=c.b1, b2=c.b2, eps=c.eps),
            optax.scale(c.td_lr),
        )
        rnn_chain = []
        if c.rnn_grad_clip:
            rnn_chain.append(optax.clip_by_global_norm(c.rnn_grad_clip))
        rnn_chain.extend(
            [
                optax.scale_by_adam(b1=c.b1, b2=c.b2, eps=c.eps),
                optax.scale(c.rnn_lr),
            ]
        )
        rnn_tx = optax.chain(*rnn_chain)

        def label(path, _leaf):
            if path[0].key == "head":
                return "td"
            if self.cfg.freeze_gamma and any(
                getattr(p, "key", None) == "gamma_log" for p in path
            ):
                return "frozen"
            return "rnn"

        labels = jax.tree_util.tree_map_with_path(label, params)
        return optax.multi_transform(
            {"td": td_tx, "rnn": rnn_tx, "frozen": optax.set_to_zero()},
            labels,
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

    @staticmethod
    def _delta_updates(traces, td_error, recurrent_scale):
        updates = {}
        for key, subtree in traces.items():
            scale = 1.0 if key == "head" else recurrent_scale

            def apply(z):
                trailing = z.ndim - 1
                delta = td_error[(slice(None),) + (None,) * trailing]
                return scale * delta * z

            updates[key] = jax.tree.map(apply, subtree)
        return updates

    @staticmethod
    def _mean_batch(tree):
        return jax.tree.map(lambda x: jnp.mean(x, axis=0), tree)

    def _deterministic_action(self, key: Key, state: IndependentRTRRLState):
        del key
        obs, done, action, reward = state.timestep.to_sequence()
        actor_gp = self._grad_params(state.actor_params, state.actor_slow_torso)
        critic_gp = self._grad_params(state.critic_params, state.critic_slow_torso)
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
        lox.log({"info": info})
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
        actor_gp = self._grad_params(state.actor_params, state.actor_slow_torso)
        critic_gp = self._grad_params(state.critic_params, state.critic_slow_torso)

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
        td_error = (
            next_reward
            + self.cfg.gamma * (1 - next_done) * next_value
            - value
        )

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
            state.I,
        )
        critic_traces_new = self._update_traces(
            state.critic_traces,
            critic_grads,
            self.cfg.gamma * self.cfg.lambda_v,
            next_done,
            state.I,
        )
        actor_traces = (
            actor_traces_new
            if self.cfg.update_trace_before_td
            else state.actor_traces
        )
        critic_traces = (
            critic_traces_new
            if self.cfg.update_trace_before_td
            else state.critic_traces
        )
        actor_updates = self._delta_updates(
            actor_traces, td_error, self.cfg.eta_f
        )
        actor_updates = jax.tree.map(
            lambda traced, direct: traced + direct,
            actor_updates,
            entropy_grads,
        )
        critic_updates = self._delta_updates(
            critic_traces, td_error, self.cfg.eta_f
        )
        actor_adam_updates, actor_opt_state = self.actor_optimizer.update(
            self._mean_batch(actor_updates),
            state.actor_opt_state,
            state.actor_params,
        )
        critic_adam_updates, critic_opt_state = self.critic_optimizer.update(
            self._mean_batch(critic_updates),
            state.critic_opt_state,
            state.critic_params,
        )
        actor_params = cast(
            Any, optax.apply_updates(state.actor_params, actor_adam_updates)
        )
        critic_params = cast(
            Any, optax.apply_updates(state.critic_params, critic_adam_updates)
        )

        if self.cfg.update_period == 1.0:
            actor_slow_torso = actor_params["torso"]
            critic_slow_torso = critic_params["torso"]
        else:
            actor_slow_torso = optax.incremental_update(
                actor_params["torso"],
                state.actor_slow_torso,
                self.cfg.update_period,
            )
            critic_slow_torso = optax.incremental_update(
                critic_params["torso"],
                state.critic_slow_torso,
                self.cfg.update_period,
            )

        not_done = 1 - next_done
        I_next = self.cfg.gamma * state.I * not_done + next_done
        actor_nu = _find_leaf(actor_params["torso"], "nu_log")
        critic_nu = _find_leaf(critic_params["torso"], "nu_log")
        lox.log(
            {
                "info": info,
                "critic/td_error": td_error.mean(),
                "actor/entropy": entropy,
                "critic/value": value.mean(),
                "emphasis/I": state.I.mean(),
                "diag/actor_lambda_max": (
                    jnp.max(jnp.exp(-jnp.exp(actor_nu)))
                    if actor_nu is not None
                    else jnp.nan
                ),
                "diag/critic_lambda_max": (
                    jnp.max(jnp.exp(-jnp.exp(critic_nu)))
                    if critic_nu is not None
                    else jnp.nan
                ),
                "diag/actor_sens_norm": _tree_norm(actor_sensitivity),
                "diag/critic_sens_norm": _tree_norm(critic_sensitivity),
                "diag/actor_carry_norm": _tree_norm(actor_carry),
                "diag/critic_carry_norm": _tree_norm(critic_carry),
                "diag/z_actor": _tree_norm(actor_traces_new),
                "diag/z_critic": _tree_norm(critic_traces_new),
                "diag/act_abs": jnp.abs(action).mean(),
            }
        )

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
                I=I_next,
                normalizer_state=normalized.state,
            ),
            IndependentStepMetrics(
                info=info,
                td_error=td_error,
                entropy=entropy,
                value=value,
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
        torso_vars = torso.init(
            {"params": torso_key}, x, done, initial_carry=carry
        )
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
        sensitivity = torso.initialize_sensitivity(
            sens_key, (self.cfg.num_envs, None)
        )
        return params, traces, sensitivity

    def init(self, key: Key) -> IndependentRTRRLState:
        if self.cfg.pred_obs:
            raise ValueError("IndependentRTRRL does not support pred_obs.")
        split = jax.random.split(key, 10)
        env_key = split[0]
        actor_keys = split[1:5]
        critic_keys = split[5:9]
        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        obs, normalizer_state = self.normalizer.reset(
            obs, None, update=True
        )
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
        self.actor_optimizer = self._make_optimizer(actor_params)
        self.critic_optimizer = self._make_optimizer(critic_params)
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
            actor_opt_state=self.actor_optimizer.init(actor_params),
            critic_opt_state=self.critic_optimizer.init(critic_params),
            actor_carry=actor_carry,
            critic_carry=critic_carry,
            actor_sensitivity=actor_sensitivity,
            critic_sensitivity=critic_sensitivity,
            I=jnp.ones((self.cfg.num_envs,), dtype=jnp.float32),
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
        reset_key, eval_key, actor_sens_key, critic_sens_key = jax.random.split(
            key, 4
        )
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
