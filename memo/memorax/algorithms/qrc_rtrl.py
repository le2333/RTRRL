"""QRC(λ) with RTU-RTRL — reproduces arXiv 2605.24709 (QRC(λ) + RTRL variant).

Forked from memorax/algorithms/qrc.py. The single change vs the parent
algorithm: both the Q-network and the H-network (correction / recurrent
state-value) forward through RNN.local_jacobian, which maintains the
forward-mode RTRL sensitivity S_t = dh_t/dpsi and injects the RFLO phantom
into the RTU carry. As a result, jax.jacobian(q_loss_fn / h_loss_fn)(params)
picks up cross-time recurrent gradients via the phantom, while encoder/head
parameters use standard 1-step autodiff.

Eligibility traces, gradient correction, optax updates and epsilon-greedy
policy are unchanged from parent QRC(λ). Each network maintains its own RTU
carry and RTRL sensitivity, reset together on episode termination (handled
inside RNN.local_jacobian); traces additionally reset on non-greedy actions.
"""
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from memorax.networks import Network, RNN
from memorax.utils import Timestep
from memorax.utils.axes import add_feature_axis, remove_feature_axis, remove_time_axis
from memorax.utils.typing import (
    Array,
    Carry,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)

from .qrc import QRCConfig


@struct.dataclass(frozen=True)
class QRCRtrlState:
    step: int
    update_step: int
    timestep: Timestep
    q_carry: Carry
    q_sensitivity: Any
    h_carry: Carry
    h_sensitivity: Any
    env_state: EnvState
    q_params: core.FrozenDict[str, Any]
    h_params: core.FrozenDict[str, Any]
    q_optimizer_state: optax.OptState
    h_optimizer_state: optax.OptState
    h_trace: Array
    q_traces: PyTree
    h_traces: PyTree


@dataclass
class QRCRtrl:
    cfg: QRCConfig
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    h_network: nn.Module
    q_optimizer: optax.GradientTransformation
    h_optimizer: optax.GradientTransformation
    epsilon_schedule: optax.Schedule

    # ------------------------------------------------------------------ forward
    def _rtrl_forward(
        self,
        network: Network,
        params: PyTree,
        obs: Array,
        done: Array,
        action: Array,
        reward: Array,
        carry: Carry,
        sensitivity: Any,
    ) -> tuple[tuple[Carry, Any], Any]:
        """encoder -> torso.local_jacobian (RTRL S_t + phantom) -> head."""
        p = params["params"] if "params" in params else params
        x, _ = network.feature_extractor.apply(
            {"params": p["feature_extractor"]},
            observation=obs,
            action=action,
            reward=reward,
            done=done,
        )
        next_carry, h, next_sensitivity = network.torso.apply(
            {"params": p["torso"]},
            x,
            done,
            carry,
            sensitivity=sensitivity,
            method=RNN.local_jacobian,
        )
        out = network.head.apply(
            {"params": p["head"]}, h, action=action, reward=reward, done=done
        )
        return (next_carry, next_sensitivity), out

    # ------------------------------------------------------------------ policies
    def _greedy_action(
        self, key: Key, state: QRCRtrlState
    ) -> tuple[QRCRtrlState, Array, Array, dict]:
        obs, done, action, reward = state.timestep.to_sequence()
        (q_carry, q_sensitivity), (q_values, _) = self._rtrl_forward(
            self.q_network, state.q_params, obs, done, action, reward,
            state.q_carry, state.q_sensitivity,
        )
        action = remove_time_axis(jnp.argmax(q_values, axis=-1))
        return (
            state.replace(q_carry=q_carry, q_sensitivity=q_sensitivity),
            action,
            jnp.zeros(self.cfg.num_envs, dtype=jnp.bool_),
            {},
        )

    def _random_action(
        self, key: Key, state: QRCRtrlState
    ) -> tuple[QRCRtrlState, Array, Array, dict]:
        keys = jax.random.split(key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(keys)
        return state, action, jnp.ones(self.cfg.num_envs, dtype=jnp.bool_), {}

    def _epsilon_greedy_action(
        self, key: Key, state: QRCRtrlState
    ) -> tuple[QRCRtrlState, Array, Array, dict]:
        random_key, greedy_key, sample_key = jax.random.split(key, 3)
        _, sampled_action, _, _ = self._random_action(random_key, state)
        state, greedy_action, _, _ = self._greedy_action(greedy_key, state)
        epsilon = self.epsilon_schedule(state.step)
        random_action = jax.random.uniform(sample_key, greedy_action.shape) < epsilon
        action = jnp.where(random_action, sampled_action, greedy_action)
        non_greedy = action != greedy_action
        return state, action, non_greedy, {}

    # ------------------------------------------------------------------ update
    def _update(
        self,
        key: Key,
        state: QRCRtrlState,
        action: Array,
        next_obs: Array,
        reward: Array,
        done: Array,
        non_greedy: Array,
        q_carry: Carry,
        h_carry: Carry,
        q_sensitivity: Any,
        h_sensitivity: Any,
    ) -> QRCRtrlState:
        second = Timestep(obs=next_obs, done=done, action=action, reward=reward)
        obs, s_done, s_action, s_reward = state.timestep.to_sequence()
        n_obs, n_done, n_action, n_reward = second.to_sequence()

        initial_q_carry = jax.lax.stop_gradient(q_carry)
        initial_q_sens = jax.lax.stop_gradient(q_sensitivity)
        initial_h_carry = jax.lax.stop_gradient(h_carry)
        initial_h_sens = jax.lax.stop_gradient(h_sensitivity)

        def q_loss_fn(params):
            _, (q_values, _) = self._rtrl_forward(
                self.q_network, params, obs, s_done, s_action, s_reward,
                initial_q_carry, initial_q_sens,
            )
            return remove_feature_axis(
                jnp.take_along_axis(q_values[:, 0], add_feature_axis(action), axis=-1)
            )

        def td_loss_fn(params):
            _, (q_values, _) = self._rtrl_forward(
                self.q_network, params, obs, s_done, s_action, s_reward,
                initial_q_carry, initial_q_sens,
            )
            q_value = remove_feature_axis(
                jnp.take_along_axis(q_values[:, 0], add_feature_axis(action), axis=-1)
            )
            _, (next_q_values, _) = self._rtrl_forward(
                self.q_network, params, n_obs, n_done, n_action, n_reward,
                initial_q_carry, initial_q_sens,
            )
            next_q_value = next_q_values[:, 0].max(axis=-1)
            return (
                second.reward
                + self.cfg.gamma * next_q_value * (1.0 - second.done)
                - q_value
            )

        def h_loss_fn(params):
            _, (h_values, _) = self._rtrl_forward(
                self.h_network, params, obs, s_done, s_action, s_reward,
                initial_h_carry, initial_h_sens,
            )
            return remove_feature_axis(
                jnp.take_along_axis(h_values[:, 0], add_feature_axis(action), axis=-1)
            )

        q_grads = jax.jacobian(q_loss_fn)(state.q_params)
        td_errors = td_loss_fn(state.q_params)
        td_grads = jax.jacobian(td_loss_fn)(state.q_params)
        correction = h_loss_fn(state.h_params)
        h_grads = jax.jacobian(h_loss_fn)(state.h_params)

        h_trace = self.cfg.gamma * self.cfg.lamda * state.h_trace + correction
        q_traces = jax.tree.map(
            lambda e, g: self.cfg.gamma * self.cfg.lamda * e + g,
            state.q_traces, q_grads,
        )
        h_traces = jax.tree.map(
            lambda e, g: self.cfg.gamma * self.cfg.lamda * e + g,
            state.h_traces, h_grads,
        )

        def broadcast(v, x):
            return v[(slice(None),) + (None,) * (x.ndim - 1)]

        q_updates = jax.tree.map(
            lambda td_g: -broadcast(h_trace, td_g) * td_g, td_grads
        )
        if self.cfg.gradient_correction:
            q_updates = jax.tree.map(
                lambda upd, eq, qg: upd
                + broadcast(td_errors, eq) * eq
                - broadcast(correction, qg) * qg,
                q_updates, q_traces, q_grads,
            )

        h_updates = jax.tree.map(
            lambda eh, hg, p: broadcast(td_errors, eh) * eh
            - broadcast(correction, hg) * hg
            - self.cfg.reg_coeff * p[None],
            h_traces, h_grads, state.h_params,
        )

        q_grads = jax.tree.map(lambda x: -x.mean(axis=0), q_updates)
        h_grads = jax.tree.map(lambda x: -x.mean(axis=0), h_updates)

        q_param_updates, q_optimizer_state = self.q_optimizer.update(
            q_grads, state.q_optimizer_state, state.q_params,
        )
        h_param_updates, h_optimizer_state = self.h_optimizer.update(
            h_grads, state.h_optimizer_state, state.h_params,
        )
        q_params = optax.apply_updates(state.q_params, q_param_updates)
        h_params = optax.apply_updates(state.h_params, h_param_updates)

        reset = done | non_greedy

        def reset_trace(trace):
            return jnp.where(
                reset[(slice(None),) + (None,) * (trace.ndim - 1)],
                jnp.zeros_like(trace), trace,
            )

        h_trace = jnp.where(reset, jnp.zeros_like(h_trace), h_trace)
        q_traces = jax.tree.map(reset_trace, q_traces)
        h_traces = jax.tree.map(reset_trace, h_traces)

        q_value = q_loss_fn(state.q_params)
        lox.log({
            "q_network/q_value": q_value.mean(),
            "q_network/td_error": td_errors.mean(),
            "h_network/h_trace": h_trace.mean(),
            "q_network/gradient_norm": optax.global_norm(q_grads),
            "h_network/correction": correction.mean(),
            "h_network/gradient_norm": optax.global_norm(h_grads),
            "training/epsilon": self.epsilon_schedule(state.step),
        })

        return state.replace(
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            h_trace=h_trace,
            q_traces=q_traces,
            h_traces=h_traces,
        )

    def _step(
        self, state: QRCRtrlState, key: Key, *, policy: Callable
    ) -> tuple[QRCRtrlState, None]:
        action_key, step_key, update_key = jax.random.split(key, 3)

        q_carry = state.q_carry
        h_carry = state.h_carry
        q_sensitivity = state.q_sensitivity
        h_sensitivity = state.h_sensitivity

        state, action, non_greedy, _ = policy(action_key, state)
        # policy (greedy / epsilon-greedy) advanced q_carry + q_sensitivity.
        q_carry = state.q_carry
        q_sensitivity = state.q_sensitivity

        step_keys = jax.random.split(step_key, self.cfg.num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)

        lox.log({"info": info})

        reward = jnp.asarray(reward, dtype=jnp.float32)
        state = self._update(
            update_key, state, action, next_obs, reward, done, non_greedy,
            q_carry, h_carry, q_sensitivity, h_sensitivity,
        )

        # Advance h-network carry + sensitivity one RTRL step.
        obs, s_done, s_action, s_reward = state.timestep.to_sequence()
        (h_carry, h_sensitivity), _ = self._rtrl_forward(
            self.h_network, state.h_params, obs, s_done, s_action, s_reward,
            h_carry, h_sensitivity,
        )

        return (
            state.replace(
                step=state.step + self.cfg.num_envs,
                update_step=state.update_step + 1,
                timestep=Timestep(obs=next_obs, action=action, reward=reward, done=done),
                env_state=env_state,
                h_carry=h_carry,
                h_sensitivity=h_sensitivity,
            ),
            None,
        )

    # ------------------------------------------------------------------ lifecycle
    def init(self, key: Key) -> QRCRtrlState:
        env_key, q_key, h_key, torso_key = jax.random.split(key, 4)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)

        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        carry_shape = (self.cfg.num_envs, None)
        q_carry = self.q_network.initialize_carry(carry_shape)
        h_carry = self.h_network.initialize_carry(carry_shape)
        q_sensitivity = self.q_network.torso.initialize_sensitivity(torso_key, carry_shape)
        h_sensitivity = self.h_network.torso.initialize_sensitivity(torso_key, carry_shape)

        q_params = self.q_network.init(
            {"params": q_key},
            *timestep.to_sequence(),
            initial_carry=q_carry,
        )
        h_params = self.h_network.init(
            {"params": h_key},
            *timestep.to_sequence(),
            initial_carry=h_carry,
        )

        q_optimizer_state = self.q_optimizer.init(q_params)
        h_optimizer_state = self.h_optimizer.init(h_params)

        return QRCRtrlState(
            step=0,
            update_step=0,
            timestep=timestep,
            q_carry=q_carry,
            q_sensitivity=q_sensitivity,
            h_carry=h_carry,
            h_sensitivity=h_sensitivity,
            env_state=env_state,
            q_params=q_params,
            h_params=h_params,
            q_optimizer_state=q_optimizer_state,
            h_optimizer_state=h_optimizer_state,
            h_trace=jnp.zeros((self.cfg.num_envs,)),
            q_traces=jax.tree.map(
                lambda p: jnp.zeros((self.cfg.num_envs, *p.shape)), q_params
            ),
            h_traces=jax.tree.map(
                lambda p: jnp.zeros((self.cfg.num_envs, *p.shape)), h_params
            ),
        )

    def warmup(self, key: Key, state: QRCRtrlState, num_steps: int) -> QRCRtrlState:
        step_keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._random_action), state, step_keys
        )
        return state

    def train(self, key: Key, state: QRCRtrlState, num_steps: int) -> QRCRtrlState:
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action), state, keys
        )
        return state

    def evaluate(self, key: Key, state: QRCRtrlState, num_steps: int) -> QRCRtrlState:
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
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
            q_carry=self.q_network.initialize_carry(carry_shape),
            h_carry=self.h_network.initialize_carry(carry_shape),
            q_sensitivity=self.q_network.torso.initialize_sensitivity(
                jax.random.key(0), carry_shape
            ),
            h_sensitivity=self.h_network.torso.initialize_sensitivity(
                jax.random.key(0), carry_shape
            ),
            env_state=env_state,
        )
        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._greedy_action), state, step_keys
        )
        return state
