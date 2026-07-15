"""Stream AC(lambda) with RTU-RTRL — reproduces arXiv 2605.24709 Algorithm 2.

Forked from memorax/algorithms/stream_ac.py. The single change vs the parent
algorithm: the network forward pass goes through RNN.local_jacobian, which
maintains the forward-mode RTRL sensitivity S_t = dh_t/dpsi and injects the
RFLO phantom into the RTU carry. As a result, jax.jacobian(loss)(params) picks
up cross-time recurrent gradients through phantom = sum(S * (psi - sg(psi))),
whose d/dpsi = S — so the recurrent parameter gradients are computed from S_t
exactly as the paper specifies ("gradients of any scalar depending on h are
computed from S"), while encoder/head parameters use standard 1-step autodiff.

Trace recursions and ObGD update rules are unchanged from parent stream AC(lambda).
Each network (actor / critic) maintains its own RTU carry and RTRL sensitivity,
reset together on episode termination (handled inside RNN.local_jacobian).

中文说明:
    StreamACRtrl 复现 arXiv 2605.24709 的算法 2,是 stream_ac.py 的 RTRL 版本,
    也是本项目的重点实时循环学习算法之一。与父类唯一区别在于网络前向走
    ``RNN.local_jacobian``:前向时同步维护前向模式 RTRL 敏感度 S_t = dh_t/dψ,
    并向 RTU carry 注入 RFLO 幻影 phantom = Σ S·(ψ - sg(ψ))。由于 d(phantom)/dψ = S,
    ``jax.jacobian(loss)(params)`` 无需时间反向传播即可从 S_t 精确得到循环参数梯度
    (正如论文所述"任何依赖 h 的标量,其梯度都可由 S 算出"),而编码器/输出头参数
    仍走普通 1 步自动微分。

    资格迹递推与 ObGD 更新规则与父类完全一致。actor/critic 各自维护自己的 RTU carry
    与 RTRL 敏感度,二者在 episode 终止时一并重置(由 ``RNN.local_jacobian`` 内部处理)。
"""
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
from flax import core, struct

from memorax.networks import Network
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils import Timestep, Transition
from memorax.utils.typing import (
    Array,
    Discrete,
    Environment,
    EnvParams,
    EnvState,
    Key,
    Carry,
    PyTree,
)

from .stream_ac import StreamACConfig


@struct.dataclass(frozen=True)
class StreamACRtrlState:
    """StreamACRtrl 状态:在 StreamACState 基础上多出 actor/critic 的 RTRL 敏感度。

    actor_sensitivity / critic_sensitivity 为各网络 torso 的前向敏感度 S_t = dh_t/dψ,
    随每步前向递推;其余字段含义同 stream_ac.py 的 StreamACState。
    """

    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    actor_params: core.FrozenDict[str, Any]
    actor_traces: core.FrozenDict[str, Any]
    actor_v: core.FrozenDict[str, Any]
    actor_carry: Carry
    actor_sensitivity: Any
    critic_params: core.FrozenDict[str, Any]
    critic_traces: core.FrozenDict[str, Any]
    critic_v: core.FrozenDict[str, Any]
    critic_carry: Carry
    critic_sensitivity: Any


@dataclass
class StreamACRtrl:
    """StreamAC(λ) 的 RTRL 版本:网络前向走 local_jacobian 维护 RTRL 敏感度。"""

    cfg: StreamACConfig
    env: Environment
    env_params: EnvParams
    actor_network: nn.Module
    critic_network: nn.Module

    # ------------------------------------------------------------------ forward
    def _rtrl_forward(
        self,
        network: Network,
        params: PyTree,
        obs: Array,
        action: Array,
        reward: Array,
        done: Array,
        carry: Carry,
        sensitivity: Any,
    ) -> tuple[tuple[Carry, Any], Any]:
        """编码器 -> torso.local_jacobian(维护 RTRL S_t 并注入 phantom)-> 输出头。

        循环 torso 在前向递推隐状态 h 的同时更新敏感度 S_t 并注入 RFLO 幻影,
        使后续 jax.jacobian 能取到跨时间的循环梯度。

        Returns:
            ((下一步 carry, 下一步 sensitivity), 输出头结果 (dist, aux))。
        """
        p = params["params"] if "params" in params else params
        # 编码器:融合 obs/action/reward/done 得到特征 x。
        x, _ = network.feature_extractor.apply(
            {"params": p["feature_extractor"]},
            observation=obs,
            action=action,
            reward=reward,
            done=done,
        )
        # method by NAME so the torso may be an RNN(RTU) or a Memoroid(LRU); both
        # expose local_jacobian with the same (inputs, done, carry, sensitivity=...)
        # signature. This makes StreamACRtrl backbone-agnostic (RTU vs LRU).
        next_carry, h, next_sensitivity = network.torso.apply(
            {"params": p["torso"]},
            x,
            done,
            carry,
            sensitivity=sensitivity,
            method="local_jacobian",
        )
        out = network.head.apply(
            {"params": p["head"]}, h, action=action, reward=reward, done=done
        )
        return (next_carry, next_sensitivity), out

    # ------------------------------------------------------------------ policies
    def _deterministic_action(
        self, key: Key, state: StreamACRtrlState
    ) -> tuple[StreamACRtrlState, Array, Array, None]:
        """确定性动作(评估):RTRL 前向 actor,离散取 argmax,连续取众数。"""
        obs, done, ts_action, reward = state.timestep.to_sequence()
        (actor_carry, actor_sensitivity), (probs, _) = self._rtrl_forward(
            self.actor_network,
            state.actor_params,
            obs,
            ts_action,
            reward,
            done,
            state.actor_carry,
            state.actor_sensitivity,
        )
        action = (
            jnp.argmax(probs.logits, axis=-1)
            if isinstance(self.env.action_space(self.env_params), Discrete)
            else probs.mode()
        )
        log_prob = probs.log_prob(action)
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)
        state = state.replace(
            actor_carry=actor_carry, actor_sensitivity=actor_sensitivity
        )
        return state, action, log_prob, None

    def _stochastic_action(
        self, key: Key, state: StreamACRtrlState
    ) -> tuple[StreamACRtrlState, Array, Array, Array]:
        """随机动作(采集):RTRL 前向 actor/critic,采样动作并算 log_prob 与价值。"""
        action_key = jax.random.split(key, 1)[0]
        obs, done, ts_action, reward = state.timestep.to_sequence()

        (actor_carry, actor_sensitivity), (probs, _) = self._rtrl_forward(
            self.actor_network,
            state.actor_params,
            obs,
            ts_action,
            reward,
            done,
            state.actor_carry,
            state.actor_sensitivity,
        )
        action, log_prob = probs.sample_and_log_prob(seed=action_key)

        (critic_carry, critic_sensitivity), (value, _) = self._rtrl_forward(
            self.critic_network,
            state.critic_params,
            obs,
            ts_action,
            reward,
            done,
            state.critic_carry,
            state.critic_sensitivity,
        )
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)
        value = remove_time_axis(value)
        value = remove_feature_axis(value)

        state = state.replace(
            actor_carry=actor_carry,
            actor_sensitivity=actor_sensitivity,
            critic_carry=critic_carry,
            critic_sensitivity=critic_sensitivity,
        )
        return state, action, log_prob, value

    # ------------------------------------------------------------------ env step
    def _step(
        self, state: StreamACRtrlState, key: Key, *, policy: Callable
    ) -> tuple[StreamACRtrlState, Transition]:
        """采集一步(仅用于评估):选动作、与环境交互,不做参数更新。"""
        action_key, step_key = jax.random.split(key)
        state, action, log_prob, value = policy(action_key, state)

        num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)

        broadcast_dims = tuple(range(state.timestep.done.ndim, state.timestep.action.ndim))
        first = Timestep(
            obs=state.timestep.obs,
            action=state.timestep.action,
            reward=state.timestep.reward,
            done=state.timestep.done,
        )
        second = Timestep(obs=None, action=action, reward=reward, done=done)
        lox.log({"info": info})

        transition = Transition(
            first=first,
            second=second,
            aux={"log_prob": log_prob, "value": value},
        )
        next_reward = jnp.asarray(reward, dtype=jnp.float32)
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
        return state, transition

    # ------------------------------------------------------------------ ObGD
    def _obgd_update(
        self, traces: PyTree, v: PyTree, td_error: Array, lr: float, kappa: float, step: int
    ):
        """ObGD 更新(与 stream_ac.py 一致):由 (δ × 迹) 计算自适应步长的增量。

        步长 = lr / max(1, |δ|·Σ|z|·lr·κ) 约束单步过冲;adaptive 时用二阶矩 v 归一化。
        """
        beta2 = self.cfg.beta2
        eps = self.cfg.eps

        def _broadcast_delta(td_error, z):
            n_trailing = z.ndim - 1
            return td_error[(slice(None),) + (None,) * n_trailing]

        # 二阶矩滑动平均 v <- beta2*v + (1-beta2)*(delta*z)^2。
        new_v = jax.tree.map(
            lambda vi, z: beta2 * vi + (1 - beta2) * jnp.square(_broadcast_delta(td_error, z) * z),
            v, traces,
        )

        if self.cfg.adaptive:
            v_hat = jax.tree.map(lambda vi: vi / (1.0 - beta2 ** step), new_v)
            norm_leaves = jax.tree.leaves(jax.tree.map(
                lambda z, vh: jnp.abs(z) / (jnp.sqrt(vh) + eps), traces, v_hat,
            ))
            z_sum = sum(jnp.sum(z, axis=tuple(range(1, z.ndim))) for z in norm_leaves)
        else:
            v_hat = None
            z_leaves = jax.tree.leaves(traces)
            z_sum = sum(jnp.sum(jnp.abs(z), axis=tuple(range(1, z.ndim))) for z in z_leaves)

        # ObGD 有效步长:分母限制过冲。
        delta_bar = jnp.maximum(jnp.abs(td_error), 1.0)
        step_size = lr / jnp.maximum(1.0, delta_bar * z_sum * lr * kappa)

        if self.cfg.adaptive:
            def compute_update(z: Array, vh: Array):
                n_trailing = z.ndim - 1
                ss = step_size[(slice(None),) + (None,) * n_trailing]
                delta = td_error[(slice(None),) + (None,) * n_trailing]
                return (ss * delta * z / (jnp.sqrt(vh) + eps)).mean(axis=0)
            updates = jax.tree.map(compute_update, traces, v_hat)
        else:
            def compute_update(z: Array):
                n_trailing = z.ndim - 1
                ss = step_size[(slice(None),) + (None,) * n_trailing]
                delta = td_error[(slice(None),) + (None,) * n_trailing]
                return (ss * delta * z).mean(axis=0)
            updates = jax.tree.map(compute_update, traces)

        return updates, new_v

    # ------------------------------------------------------------------ update
    def _update_step(
        self, state: StreamACRtrlState, key: Key
    ) -> tuple[StreamACRtrlState, None]:
        """在线单步(RTRL):前向 actor/critic -> env.step -> 计算 TD 误差 -> 更新迹与参数。"""
        action_key, step_key = jax.random.split(key)

        obs, done, ts_action, reward = state.timestep.to_sequence()

        # 各网络前向一步 RTRL(推进 carry 与敏感度 S_t)。
        (actor_carry, actor_sensitivity), (probs, _) = self._rtrl_forward(
            self.actor_network,
            state.actor_params,
            obs,
            ts_action,
            reward,
            done,
            state.actor_carry,
            state.actor_sensitivity,
        )
        action, log_prob = probs.sample_and_log_prob(seed=action_key)
        entropy = remove_time_axis(probs.entropy()).mean()
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)

        (critic_carry, critic_sensitivity), (value, _) = self._rtrl_forward(
            self.critic_network,
            state.critic_params,
            obs,
            ts_action,
            reward,
            done,
            state.critic_carry,
            state.critic_sensitivity,
        )
        value = remove_time_axis(value)
        value = remove_feature_axis(value)

        # Env step
        num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)

        # Next value (stop-grad carry + sensitivity) for TD target.
        next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
            obs=next_obs, action=action, reward=next_reward, done=next_done
        ).to_sequence()
        _, (next_value, _) = self._rtrl_forward(
            self.critic_network,
            jax.lax.stop_gradient(state.critic_params),
            next_obs_s,
            next_action_s,
            next_reward_s,
            next_done_s,
            jax.lax.stop_gradient(critic_carry),
            jax.lax.stop_gradient(critic_sensitivity),
        )
        next_value = remove_time_axis(next_value)
        next_value = remove_feature_axis(next_value)

        # 单步 TD 误差 δ = r + γ(1-done)V(s') - V(s)。
        gamma = self.cfg.gamma
        td_error = next_reward + gamma * (1 - next_done) * next_value - value

        # 损失函数走 local_jacobian:jax.jacobian 借 RFLO 幻影获得循环梯度(d phantom/dψ = S_t)。
        # 前向起点 carry/敏感度做 stop_gradient(跨时间信息已由 S_t 承载)。
        initial_actor_carry = jax.lax.stop_gradient(state.actor_carry)
        initial_actor_sens = jax.lax.stop_gradient(state.actor_sensitivity)
        initial_critic_carry = jax.lax.stop_gradient(state.critic_carry)
        initial_critic_sens = jax.lax.stop_gradient(state.critic_sensitivity)

        def critic_loss_fn(params: PyTree):
            """critic RTRL 前向,返回逐环境标量价值 V(s)。"""
            _, (v, _) = self._rtrl_forward(
                self.critic_network,
                params,
                obs,
                ts_action,
                reward,
                done,
                initial_critic_carry,
                initial_critic_sens,
            )
            return remove_feature_axis(remove_time_axis(v))

        def actor_loss_fn(params: PyTree):
            """actor RTRL 前向,返回 logπ(a|s) 加上按 δ 符号加权的熵项。"""
            _, (dist, _) = self._rtrl_forward(
                self.actor_network,
                params,
                obs,
                ts_action,
                reward,
                done,
                initial_actor_carry,
                initial_actor_sens,
            )
            log_p = remove_time_axis(dist.log_prob(add_time_axis(action)))
            ent = remove_time_axis(dist.entropy())
            return log_p + self.cfg.entropy_coefficient * jnp.sign(td_error) * ent

        # jacobian 通过 RFLO 幻影自动带入循环梯度。
        critic_grads = jax.jacobian(critic_loss_fn)(state.critic_params)
        actor_grads = jax.jacobian(actor_loss_fn)(state.actor_params)

        trace_decay = gamma * self.cfg.trace_lambda

        def update_trace(z: Array, g: Array):
            # 资格迹递推 z <- γλ·(1-done)·z + g。
            n_trailing = z.ndim - 1
            not_done = (1 - state.timestep.done)[(slice(None),) + (None,) * n_trailing]
            return trace_decay * not_done * z + g

        critic_traces = jax.tree.map(update_trace, state.critic_traces, critic_grads)
        actor_traces = jax.tree.map(update_trace, state.actor_traces, actor_grads)

        current_step = state.update_step + 1

        critic_updates, critic_v = self._obgd_update(
            critic_traces, state.critic_v, td_error,
            self.cfg.critic_lr, self.cfg.critic_kappa, current_step,
        )
        actor_updates, actor_v = self._obgd_update(
            actor_traces, state.actor_v, td_error,
            self.cfg.actor_lr, self.cfg.actor_kappa, current_step,
        )

        critic_params = jax.tree.map(lambda p, u: p + u, state.critic_params, critic_updates)
        actor_params = jax.tree.map(lambda p, u: p + u, state.actor_params, actor_updates)

        lox.log({
            "info": info,
            "critic/td_error": td_error.mean(),
            "actor/entropy": entropy,
            "critic/value": value.mean(),
        })

        broadcast_dims = tuple(range(state.timestep.done.ndim, state.timestep.action.ndim))
        next_reward_f = jnp.asarray(next_reward, dtype=jnp.float32)
        state = state.replace(
            step=state.step + self.cfg.num_envs,
            update_step=current_step,
            timestep=Timestep(
                obs=next_obs,
                action=jnp.where(
                    jnp.expand_dims(next_done, axis=broadcast_dims),
                    jnp.zeros_like(action),
                    action,
                ),
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
            actor_sensitivity=actor_sensitivity,
            critic_params=critic_params,
            critic_traces=critic_traces,
            critic_v=critic_v,
            critic_carry=critic_carry,
            critic_sensitivity=critic_sensitivity,
        )

        return state, None

    # ------------------------------------------------------------------ lifecycle
    def init(self, key: Key) -> StreamACRtrlState:
        """初始化环境、actor/critic 参数、资格迹、二阶矩以及 RTRL 敏感度。"""
        env_key, actor_key, critic_key = jax.random.split(key, 3)

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
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done).to_sequence()

        carry_shape = (self.cfg.num_envs, None)
        actor_carry = self.actor_network.initialize_carry(carry_shape)
        critic_carry = self.critic_network.initialize_carry(carry_shape)
        actor_sensitivity = self.actor_network.torso.initialize_sensitivity(
            actor_key, carry_shape
        )
        critic_sensitivity = self.critic_network.torso.initialize_sensitivity(
            critic_key, carry_shape
        )

        ts_obs, ts_done, ts_action, ts_reward = timestep
        actor_params = self.actor_network.init(
            {"params": actor_key},
            observation=ts_obs,
            action=ts_action,
            reward=ts_reward,
            done=ts_done,
            initial_carry=actor_carry,
        )
        critic_params = self.critic_network.init(
            {"params": critic_key},
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

        return StreamACRtrlState(
            step=0,
            update_step=0,
            timestep=timestep.from_sequence(),
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
        )

    def warmup(self, key: Key, state: StreamACRtrlState, num_steps: int) -> StreamACRtrlState:
        """在线算法,无需预热,直接返回原状态。"""
        return state

    def train(self, key: Key, state: StreamACRtrlState, num_steps: int) -> StreamACRtrlState:
        """训练:逐步在线 RTRL 更新,共 num_steps 环境步。"""
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(self._update_step, state, keys)
        return state

    def evaluate(self, key: Key, state: StreamACRtrlState, num_steps: int) -> StreamACRtrlState:
        """评估:重置环境、carry 与敏感度后以确定性策略运行 num_steps 步。"""
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
        )
        action = jnp.zeros(
            (self.cfg.num_envs, *self.env.action_space(self.env_params).shape),
            dtype=self.env.action_space(self.env_params).dtype,
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)

        carry_shape = (self.cfg.num_envs, None)
        state = state.replace(
            timestep=timestep,
            actor_carry=self.actor_network.initialize_carry(carry_shape),
            critic_carry=self.critic_network.initialize_carry(carry_shape),
            actor_sensitivity=self.actor_network.torso.initialize_sensitivity(
                jax.random.key(0), carry_shape
            ),
            critic_sensitivity=self.critic_network.torso.initialize_sensitivity(
                jax.random.key(0), carry_shape
            ),
            env_state=env_state,
        )

        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._deterministic_action),
            state,
            step_keys,
        )
        return state
