"""StreamAC as upstream Memorax writes it: one-step credit, no RTRL.

This repository carried a byte-identical copy of upstream's
``memorax/algorithms/stream_ac.py`` until it was deleted along with the golden
snapshots. It is back because it is the reference: ``stream_ac_rtrl`` differs
from it in exactly one deliberate way, and without this file next to it there
is nothing to check that claim against.

The arithmetic here is upstream's, line for line -- the eligibility trace, the
overshooting bound, the objectives, the initialisation. What credits the
recurrence is the difference: here the carry is detached and one forward pass
is differentiated, so the gradient sees a single step; ``stream_ac_rtrl``
carries a sensitivity instead and sees the whole stream.

Two things about upstream's text are not reproduced, both about talking rather
than computing:

``lox.log`` calls, which logged from inside the traced region. Nothing here
logs. ``_update_step`` returns a fixed-shape record of what the transition
passed through and the evaluation step returns an ``EvalSummary``, both stacked
by the scan and reported outside JIT by whatever drives the algorithm, which is
the boundary every other algorithm in this package keeps.

``_stochastic_action`` and ``warmup``, which nothing reaches: the training step
inlines its own acting forward, evaluation acts greedily, and warmup returned
its argument.

Three pieces of the update are small methods here rather than an expression and
a closure buried in the training step: the TD error, the actor's ascent
direction, and the trace recurrence. A reference nothing can call is a reference
that has to be copied into a test to be used, and a copy drifts. The expressions
themselves are unchanged, character for character; only what they can be reached
from is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import core, struct

from memorax.utils import Timestep
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils.typing import Array, Discrete, EnvState, Key, PyTree

from .contract import ActionDecision, EvalSummary


@struct.dataclass(frozen=True)
class StreamACConfig:
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


@struct.dataclass(frozen=True)
class StreamACState:
    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_traces: core.FrozenDict[str, Any]
    actor_v: core.FrozenDict[str, Any]
    actor_carry: Array
    critic_params: core.FrozenDict[str, Any]
    critic_traces: core.FrozenDict[str, Any]
    critic_v: core.FrozenDict[str, Any]
    critic_carry: Array


@struct.dataclass(frozen=True)
class StreamACStepMetrics:
    """What one transition passed through, in the shape it is stacked in.

    The three scalars upstream handed to its logger are here, alongside the
    quantities they were computed from. The bound on the step size is not: it
    lives inside ``_obgd_update``, which is upstream's and stays that way.
    """

    action_decision: ActionDecision | None = None
    log_prob: Any = None
    entropy: Any = None
    value: Any = None
    next_value: Any = None
    td_error: Any = None
    info: Any = None


@dataclass
class StreamAC:
    cfg: StreamACConfig
    env: Any
    env_params: Any
    actor_network: Any
    critic_network: Any

    def _deterministic_action(
        self, key: Key, state: Any
    ) -> tuple[StreamACState, Array, Array, None]:
        del key
        obs, done, action, reward = state.timestep.to_sequence()
        actor_carry, (probs, _) = self.actor_network.apply(
            state.actor_params,
            observation=obs,
            action=action,
            reward=reward,
            done=done,
            initial_carry=state.actor_carry,
        )
        action = (
            jnp.argmax(probs.logits, axis=-1)
            if isinstance(self.env.action_space(self.env_params), Discrete)
            else probs.mode()
        )
        log_prob = probs.log_prob(action)
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)
        state = state.replace(actor_carry=actor_carry)
        return state, action, log_prob, None

    def _step(
        self, state: Any, key: Key, *, policy: Callable
    ) -> tuple[StreamACState, EvalSummary]:
        action_key, step_key = jax.random.split(key)
        state, action, log_prob, value = policy(action_key, state)
        del log_prob, value

        num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)

        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        next_reward = jnp.asarray(reward, dtype=jnp.float32)
        summary = EvalSummary(
            info=info,
            observation=state.timestep.obs,
            next_observation=next_obs,
            action=action,
            reward=next_reward,
            done=done,
        )
        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(
                obs=next_obs,
                action=jnp.where(
                    jnp.expand_dims(done, axis=broadcast_dims),
                    jnp.zeros_like(action),
                    action,
                ),
                reward=jnp.where(done, jnp.zeros_like(next_reward), next_reward),
                done=done,
            ),
            env_state=env_state,
        )
        return state, summary

    def _obgd_update(
        self,
        traces: PyTree,
        v: PyTree,
        td_error: Array,
        lr: float,
        kappa: float,
        step: int,
    ):
        beta2 = self.cfg.beta2
        eps = self.cfg.eps

        def _broadcast_delta(td_error, z):
            n_trailing = z.ndim - 1
            return td_error[(slice(None),) + (None,) * n_trailing]

        # Update second moment: v <- beta2*v + (1-beta2)*(delta*z)^2
        new_v = jax.tree.map(
            lambda vi, z: beta2 * vi
            + (1 - beta2) * jnp.square(_broadcast_delta(td_error, z) * z),
            v,
            traces,
        )

        if self.cfg.adaptive:
            # Bias-corrected v_hat = v / (1 - beta2^t)
            v_hat = jax.tree.map(lambda vi: vi / (1.0 - beta2**step), new_v)

            # z_sum over normalised traces: sum|z / sqrt(v_hat + eps)|
            norm_leaves = jax.tree.leaves(
                jax.tree.map(
                    lambda z, vh: jnp.abs(z) / (jnp.sqrt(vh) + eps),
                    traces,
                    v_hat,
                )
            )
            z_sum = sum(jnp.sum(z, axis=tuple(range(1, z.ndim))) for z in norm_leaves)
        else:
            v_hat = None
            z_leaves = jax.tree.leaves(traces)
            z_sum = sum(
                jnp.sum(jnp.abs(z), axis=tuple(range(1, z.ndim))) for z in z_leaves
            )

        delta_bar = jnp.maximum(jnp.abs(td_error), 1.0)
        step_size = lr / jnp.maximum(1.0, delta_bar * z_sum * lr * kappa)

        if self.cfg.adaptive:
            # Upstream names both branches' closures alike, which a type checker
            # reads as one shadowing the other. Only the name differs here.
            def compute_normalized_update(z: Array, vh: Array):
                n_trailing = z.ndim - 1
                ss = step_size[(slice(None),) + (None,) * n_trailing]
                delta = td_error[(slice(None),) + (None,) * n_trailing]
                return (ss * delta * z / (jnp.sqrt(vh) + eps)).mean(axis=0)

            updates = jax.tree.map(compute_normalized_update, traces, v_hat)
        else:

            def compute_update(z: Array):
                n_trailing = z.ndim - 1
                ss = step_size[(slice(None),) + (None,) * n_trailing]
                delta = td_error[(slice(None),) + (None,) * n_trailing]
                return (ss * delta * z).mean(axis=0)

            updates = jax.tree.map(compute_update, traces)

        return updates, new_v

    def _td_error(
        self, next_reward: Array, next_done: Array, next_value: Array, value: Array
    ) -> Array:
        gamma = self.cfg.gamma
        return next_reward + gamma * (1 - next_done) * next_value - value

    def _actor_direction(self, log_p: Array, entropy: Array, td_error: Array) -> Array:
        return log_p + self.cfg.entropy_coefficient * jnp.sign(td_error) * entropy

    def _update_trace(
        self, z: Array, g: Array, done: Array, trace_decay: Array | float
    ) -> Array:
        n_trailing = z.ndim - 1
        not_done = (1 - done)[(slice(None),) + (None,) * n_trailing]
        return trace_decay * not_done * z + g

    def _update_step(
        self, state: Any, key: Key
    ) -> tuple[StreamACState, StreamACStepMetrics]:
        action_key, step_key, actor_torso_key, critic_torso_key = jax.random.split(
            key, 4
        )

        obs, done, ts_action, reward = state.timestep.to_sequence()

        actor_carry, (probs, _) = self.actor_network.apply(
            state.actor_params,
            observation=obs,
            action=ts_action,
            reward=reward,
            done=done,
            initial_carry=state.actor_carry,
            rngs={"torso": actor_torso_key},
        )
        action, log_prob = probs.sample_and_log_prob(seed=action_key)
        entropy = remove_time_axis(probs.entropy()).mean()
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)

        critic_carry, (value, _) = self.critic_network.apply(
            state.critic_params,
            observation=obs,
            action=ts_action,
            reward=reward,
            done=done,
            initial_carry=state.critic_carry,
            rngs={"torso": critic_torso_key},
        )
        value = remove_time_axis(value)
        value = remove_feature_axis(value)

        num_envs, *_ = state.timestep.obs.shape
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
            obs=next_obs, action=action, reward=next_reward, done=next_done
        ).to_sequence()
        _, (next_value, _) = self.critic_network.apply(
            jax.lax.stop_gradient(state.critic_params),
            observation=next_obs_s,
            action=next_action_s,
            reward=next_reward_s,
            done=next_done_s,
            initial_carry=jax.lax.stop_gradient(critic_carry),
        )
        next_value = remove_time_axis(next_value)
        next_value = remove_feature_axis(next_value)

        gamma = self.cfg.gamma
        td_error = self._td_error(next_reward, next_done, next_value, value)

        initial_actor_carry = jax.lax.stop_gradient(state.actor_carry)
        initial_critic_carry = jax.lax.stop_gradient(state.critic_carry)

        def critic_loss_fn(params: PyTree):
            _, (v, _) = self.critic_network.apply(
                params,
                observation=obs,
                action=ts_action,
                reward=reward,
                done=done,
                initial_carry=initial_critic_carry,
            )
            return remove_feature_axis(remove_time_axis(v))

        def actor_loss_fn(params: PyTree):
            _, (dist, _) = self.actor_network.apply(
                params,
                observation=obs,
                action=ts_action,
                reward=reward,
                done=done,
                initial_carry=initial_actor_carry,
            )
            log_p = remove_time_axis(dist.log_prob(add_time_axis(action)))
            entropy = remove_time_axis(dist.entropy())
            return self._actor_direction(log_p, entropy, td_error)

        critic_grads = jax.jacobian(critic_loss_fn)(state.critic_params)
        actor_grads = jax.jacobian(actor_loss_fn)(state.actor_params)

        trace_decay = gamma * self.cfg.trace_lambda

        def update_trace(z: Array, g: Array):
            return self._update_trace(z, g, state.timestep.done, trace_decay)

        critic_traces = jax.tree.map(update_trace, state.critic_traces, critic_grads)
        actor_traces = jax.tree.map(update_trace, state.actor_traces, actor_grads)

        current_step = state.update_step + 1

        critic_updates, critic_v = self._obgd_update(
            critic_traces,
            state.critic_v,
            td_error,
            self.cfg.critic_lr,
            self.cfg.critic_kappa,
            current_step,
        )
        actor_updates, actor_v = self._obgd_update(
            actor_traces,
            state.actor_v,
            td_error,
            self.cfg.actor_lr,
            self.cfg.actor_kappa,
            current_step,
        )

        critic_params = jax.tree.map(
            lambda p, u: p + u, state.critic_params, critic_updates
        )
        actor_params = jax.tree.map(
            lambda p, u: p + u, state.actor_params, actor_updates
        )

        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        next_reward_f = jnp.asarray(next_reward, dtype=jnp.float32)
        persisted_action = jnp.where(
            jnp.expand_dims(next_done, axis=broadcast_dims),
            jnp.zeros_like(action),
            action,
        )
        state = state.replace(
            step=state.step + self.cfg.num_envs,
            update_step=current_step,
            timestep=Timestep(
                obs=next_obs,
                action=persisted_action,
                reward=jnp.where(
                    next_done, jnp.zeros_like(next_reward_f), next_reward_f
                ),
                done=next_done,
            ),
            env_state=env_state,
            actor_params=actor_params,
            actor_traces=actor_traces,
            actor_v=actor_v,
            actor_carry=actor_carry,
            critic_params=critic_params,
            critic_traces=critic_traces,
            critic_v=critic_v,
            critic_carry=critic_carry,
        )

        metrics = StreamACStepMetrics(
            action_decision=ActionDecision(
                sampled_action=action,
                logprob_action=action,
                env_action=action,
                bootstrap_feedback_action=action,
                persisted_feedback_action=persisted_action,
            ),
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            next_value=next_value,
            td_error=td_error,
            info=info,
        )
        return state, metrics

    def init(self, key: Key) -> StreamACState:
        (
            env_key,
            actor_key,
            actor_torso_key,
            actor_dropout_key,
            critic_key,
            critic_torso_key,
            critic_dropout_key,
        ) = jax.random.split(key, 7)

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()

        actor_carry = self.actor_network.initialize_carry((self.cfg.num_envs, None))
        critic_carry = self.critic_network.initialize_carry((self.cfg.num_envs, None))

        ts_obs, ts_done, ts_action, ts_reward = timestep
        actor_params = self.actor_network.init(
            {
                "params": actor_key,
                "torso": actor_torso_key,
                "dropout": actor_dropout_key,
            },
            observation=ts_obs,
            action=ts_action,
            reward=ts_reward,
            done=ts_done,
            initial_carry=actor_carry,
        )
        critic_params = self.critic_network.init(
            {
                "params": critic_key,
                "torso": critic_torso_key,
                "dropout": critic_dropout_key,
            },
            observation=ts_obs,
            action=ts_action,
            reward=ts_reward,
            done=ts_done,
            initial_carry=critic_carry,
        )

        actor_traces = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape)), actor_params
        )
        critic_traces = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape)), critic_params
        )
        actor_v = jax.tree.map(jnp.zeros_like, actor_traces)
        critic_v = jax.tree.map(jnp.zeros_like, critic_traces)

        return StreamACState(
            step=0,
            update_step=0,
            timestep=timestep.from_sequence(),
            env_state=env_state,
            actor_params=actor_params,
            actor_traces=actor_traces,
            actor_v=actor_v,
            actor_carry=actor_carry,
            critic_params=critic_params,
            critic_traces=critic_traces,
            critic_v=critic_v,
            critic_carry=critic_carry,
        )

    def train(self, key: Key, state: StreamACState, num_steps: int):
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        return jax.lax.scan(self._update_step, state, keys)

    def evaluate(self, key: Key, state: Any, num_steps: int):
        reset_key, eval_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        initial_actor_carry = self.actor_network.initialize_carry(
            (self.cfg.num_envs, None)
        )
        initial_critic_carry = self.critic_network.initialize_carry(
            (self.cfg.num_envs, None)
        )

        state = state.replace(
            timestep=timestep,
            actor_carry=initial_actor_carry,
            critic_carry=initial_critic_carry,
            env_state=env_state,
        )

        step_keys = jax.random.split(eval_key, num_steps)
        return jax.lax.scan(
            partial(self._step, policy=self._deterministic_action),
            state,
            step_keys,
        )
