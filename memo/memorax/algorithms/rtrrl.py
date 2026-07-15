"""RTRRL — Real-Time Recurrent Reinforcement Learning (AAAI'25) on memorax.

Faithful re-host of streaming-rtrrl/rtrrl.py's algorithm inside memorax, using
memorax's LRU as the RTRL backbone. Distinguishing features vs the existing
StreamACRtrl (arXiv 2605.24709):

* SHARED recurrent torso.  One LRU feeds BOTH a linear actor head and a linear
  critic head (streaming-rtrrl's RNNActorCritic), instead of two independent
  networks. The RTRL sensitivity S_t = dh_t/dpsi is maintained once and shared.
* AC(lambda) with THREE eligibility traces.  actor / critic / recurrent params
  each get their own trace with its own decay (lambda_pi / lambda_v / lambda_rnn)
  and an episodic emphasis factor I (I <- gamma*I within an episode, reset at
  boundaries), matching Sutton & Barto online AC(lambda).
* adam optimizers (optax.multi_transform), NOT ObGD.  Heads share `td_lr`, the
  recurrent params use `rnn_lr` with global-norm clipping (streaming-rtrrl uses
  gradient_clip on the RNN group only). Updates are gradient *ascent*.
* Polyak-averaged target for the recurrent params (`update_period`).  The
  forward pass that computes gradients uses the slow LRU params; adam updates the
  fast LRU params; slow <- incremental_update(fast, slow, update_period).
* eta_pi scales the actor objective (folded into the loss), eta_f scales the
  whole recurrent update; entropy is a separate (non-trace) gradient added
  directly to the actor + recurrent updates.

The recurrent gradient is obtained exactly as in StreamACRtrl: the torso forward
goes through Memoroid.local_jacobian, which injects the RFLO phantom
(phantom = sum(S * (psi - sg(psi)))) so that jax.jacobian(loss)(params) picks up
d phantom/d psi = S_t. Head / feature-extractor params use standard 1-step
autodiff. Observation / reward normalisation is handled by env wrappers.

中文说明:
    RTRRL(实时循环强化学习,AAAI'25)是本项目最核心的算法。它把 streaming-rtrrl
    的 RNNActorCritic 移植到 memorax,用 memorax 的 LRU 作为 RTRL 骨干。相较
    StreamACRtrl 的关键区别:

    * 共享循环 torso:单个 LRU 同时喂给线性 actor 头与线性 critic 头,RTRL 敏感度
      S_t = dh_t/dψ 只维护一份并共享(而非两个独立网络)。
    * AC(λ) + 三条资格迹:actor / critic / 循环参数各有独立衰减(lambda_pi /
      lambda_v / lambda_rnn),并配一个 episodic emphasis 因子 I(episode 内 I<-γ·I,
      边界重置),对应 Sutton & Barto 的在线 AC(λ)。
    * 优化器用 adam(optax.multi_transform)而非 ObGD:两个头共享 td_lr,循环参数用
      rnn_lr 且带全局范数裁剪;更新为梯度"上升"。
    * 循环参数用 Polyak 平均的目标(update_period):求梯度的前向用"慢"LRU 参数,
      adam 更新"快"LRU 参数,再 slow <- incremental_update(fast, slow, update_period)。
    * eta_pi 缩放 actor 目标(并入损失),eta_f 缩放整个循环更新;熵项作为独立的
      (非资格迹)梯度直接加到 actor 与循环更新上。

    循环梯度的获取方式与 StreamACRtrl 完全相同:torso 前向走 Memoroid.local_jacobian,
    注入 RFLO 幻影使 jax.jacobian(loss)(params) 取到 d phantom/dψ = S_t;头与
    特征提取器参数走普通 1 步自动微分。观测/奖励归一化由环境 wrapper 处理。
"""
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import core, struct

from memorax.networks.sequence_models.memoroid import Memoroid
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

# Top-level parameter groups. feature_extractor + torso form the "recurrent"
# pathway (lambda_rnn / eta_f / rnn_lr); the heads are the "td" pathway.
_RNN_KEYS = ("feature_extractor", "torso")
_ACTOR_KEY = "actor"
_CRITIC_KEY = "critic"


def _tree_norm(tree) -> Array:
    """L2 norm over all leaves of a (possibly complex) pytree; 0 if empty."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.abs(l) ** 2) for l in leaves))


def _find_leaf(tree, name):
    """Return the first leaf whose flatten-path contains a dict key == name."""
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if any(getattr(k, "key", None) == name for k in path):
            return leaf
    return None


@struct.dataclass(frozen=True)
class RTRRLConfig:
    """RTRRL 超参数配置。

    关键字段:num_envs 并行环境数;gamma 折扣;lambda_pi/lambda_v/lambda_rnn 为
    actor/critic/循环三条资格迹各自的 λ;td_lr/rnn_lr 分别为头与循环参数的 adam 学习率;
    eta_pi 缩放 actor 目标、eta_f 缩放循环更新;entropy_rate 熵系数;update_period 为
    循环参数 Polyak 平均系数(1.0 表示无滞后);b1/b2/eps 为 adam 超参;rnn_grad_clip 为
    循环梯度的全局范数裁剪阈值。
    """

    num_envs: int
    gamma: float = 0.95
    # Per-component eligibility-trace decay.
    # 每个组件独立的资格迹衰减:actor / critic / 循环参数。
    lambda_pi: float = 0.97
    lambda_v: float = 0.9
    lambda_rnn: float = 0.945
    # adam learning rates: heads share td_lr, recurrent params use rnn_lr.
    td_lr: float = 3e-5
    rnn_lr: float = 2e-6
    # Objective scaling.
    eta_pi: float = 0.38
    eta_f: float = 0.5
    entropy_rate: float = 3e-5
    # Polyak averaging of the recurrent (LRU) params for the gradient forward.
    # 1.0 => slow params track fast exactly (no target lag).
    update_period: float = 0.1
    # adam hyperparameters + recurrent gradient clipping.
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    rnn_grad_clip: float = 1.0
    # Diagnostic / faithfulness switches (default off => identical to the
    # reproduction baseline). act_clip>0 clips the environment-facing action to
    # [-act_clip, act_clip] (streaming-rtrrl clips brax actions to [-1,1]); the
    # eligibility trace's log_prob still uses the UNCLIPPED sample
    # (align_action_logprob=False). freeze_gamma stops gradient on the LRU input
    # gain gamma_log (routes it to optax.set_to_zero), pinning it at init.
    act_clip: float = 0.0
    freeze_gamma: bool = False


@struct.dataclass(frozen=True)
class RTRRLState:
    """RTRRL 训练状态。

    关键字段:params 为"快"参数(feature_extractor/torso/actor/critic);slow_torso 为
    Polyak 平均后的"慢"LRU 参数(供求梯度的前向使用);traces 为与 params 同构、逐环境的
    资格迹;carry 为共享 LRU 隐状态;sensitivity 为共享 RTRL 敏感度 S_t;I 为逐环境的
    episodic emphasis 因子 [num_envs]。
    """

    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    params: core.FrozenDict[str, Any]  # fast params: feature_extractor/torso/actor/critic
    slow_torso: core.FrozenDict[str, Any]  # Polyak-averaged LRU params
    traces: core.FrozenDict[str, Any]  # per-env eligibility trace, same tree as params
    opt_state: Any
    carry: Carry
    sensitivity: Any
    I: Array  # episodic emphasis factor, [num_envs]


@dataclass
class RTRRL:
    """RTRRL 算法主体:共享 LRU torso + 线性 actor/critic 头 + 三条资格迹。

    optimizer 在 init() 中根据参数结构构建(adam multi_transform)。
    """

    cfg: RTRRLConfig
    env: Environment
    env_params: EnvParams
    feature_extractor: nn.Module
    torso: Memoroid
    actor_head: nn.Module
    critic_head: nn.Module
    activation: Callable = jax.nn.silu
    # Built in init() once the param structure is known.
    optimizer: optax.GradientTransformation = field(default=None, init=False)

    # ------------------------------------------------------------------ helpers
    def _grad_params(self, params: PyTree, slow_torso: PyTree) -> PyTree:
        """Params tree used for the forward/gradient: fast heads + slow LRU torso."""
        return {**params, "torso": slow_torso}

    def _forward(
        self,
        params: PyTree,
        obs: Array,
        action: Array,
        reward: Array,
        done: Array,
        carry: Carry,
        sensitivity: Any,
    ) -> tuple[tuple[Carry, Any], tuple[Any, Array]]:
        """feature_extractor -> LRU (Memoroid.local_jacobian, RTRL) -> silu -> heads.

        Returns ((next_carry, next_sensitivity), (dist, value)).
        """
        x, _ = self.feature_extractor.apply(
            {"params": params["feature_extractor"]},
            observation=obs,
            action=action,
            reward=reward,
            done=done,
        )
        next_carry, h, next_sensitivity = self.torso.apply(
            {"params": params["torso"]},
            x,
            done,
            carry,
            sensitivity=sensitivity,
            method=Memoroid.local_jacobian,
        )
        h = self.activation(h)
        dist, _ = self.actor_head.apply(
            {"params": params["actor"]}, h, action=action, reward=reward, done=done
        )
        value, _ = self.critic_head.apply(
            {"params": params["critic"]}, h, action=action, reward=reward, done=done
        )
        return (next_carry, next_sensitivity), (dist, value)

    def _trace_decay(self, key: str) -> float:
        """按参数组返回对应资格迹的衰减系数 γλ(actor/critic/循环各不同)。"""
        if key == _ACTOR_KEY:
            return self.cfg.gamma * self.cfg.lambda_pi
        if key == _CRITIC_KEY:
            return self.cfg.gamma * self.cfg.lambda_v
        return self.cfg.gamma * self.cfg.lambda_rnn

    def _make_optimizer(self, params: PyTree) -> optax.GradientTransformation:
        """adam multi_transform: heads='td' (td_lr), recurrent='rnn' (rnn_lr + clip).

        Uses +lr so optax.apply_updates performs gradient *ascent* on the
        trace/entropy update directions.
        """
        c = self.cfg
        td_tx = optax.chain(
            optax.scale_by_adam(b1=c.b1, b2=c.b2, eps=c.eps),
            optax.scale(c.td_lr),
        )
        rnn_chain = []
        if c.rnn_grad_clip:
            rnn_chain.append(optax.clip_by_global_norm(c.rnn_grad_clip))
        rnn_chain += [
            optax.scale_by_adam(b1=c.b1, b2=c.b2, eps=c.eps),
            optax.scale(c.rnn_lr),
        ]
        rnn_tx = optax.chain(*rnn_chain)

        def leaf_label(path, _leaf):
            # feature_extractor/torso 归为 'rnn' 组,actor/critic 头归为 'td' 组;
            # freeze_gamma 时把 torso 内的 gamma_log 叶子单独路由到 'frozen'(置零更新)。
            top = path[0].key
            if top not in _RNN_KEYS:
                return "td"
            if self.cfg.freeze_gamma and any(
                getattr(p, "key", None) == "gamma_log" for p in path
            ):
                return "frozen"
            return "rnn"

        param_labels = jax.tree_util.tree_map_with_path(leaf_label, params)
        transforms = {"td": td_tx, "rnn": rnn_tx, "frozen": optax.set_to_zero()}
        return optax.multi_transform(transforms, param_labels)

    # ------------------------------------------------------------------ policies
    def _deterministic_action(
        self, key: Key, state: RTRRLState
    ) -> tuple[RTRRLState, Array, Array, Array]:
        """确定性动作(评估):离散取 argmax,连续取众数;推进共享 carry 与敏感度。"""
        obs, done, ts_action, reward = state.timestep.to_sequence()
        gp = self._grad_params(state.params, state.slow_torso)
        (carry, sensitivity), (dist, value) = self._forward(
            gp, obs, ts_action, reward, done, state.carry, state.sensitivity
        )
        action = (
            jnp.argmax(dist.logits, axis=-1)
            if isinstance(self.env.action_space(self.env_params), Discrete)
            else dist.mode()
        )
        log_prob = dist.log_prob(action)
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)
        value = remove_feature_axis(remove_time_axis(value))
        state = state.replace(carry=carry, sensitivity=sensitivity)
        return state, action, log_prob, value

    # ------------------------------------------------------------------ env step
    def _step(
        self, state: RTRRLState, key: Key, *, policy: Callable
    ) -> tuple[RTRRLState, Transition]:
        """采集一步(仅用于评估):选动作、与环境交互,不做参数更新。"""
        action_key, step_key = jax.random.split(key)
        state, action, log_prob, value = policy(action_key, state)

        # Match training: clip the env-facing action when act_clip is enabled.
        c = self.cfg.act_clip
        action = jnp.clip(action, -c, c) if c else action

        num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)

        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        first = Timestep(
            obs=state.timestep.obs,
            action=state.timestep.action,
            reward=state.timestep.reward,
            done=state.timestep.done,
        )
        second = Timestep(obs=None, action=action, reward=reward, done=done)
        lox.log({"info": info})

        transition = Transition(
            first=first, second=second, aux={"log_prob": log_prob, "value": value}
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

    # ------------------------------------------------------------------ update
    def _update_step(self, state: RTRRLState, key: Key) -> tuple[RTRRLState, None]:
        """在线单步(RTRRL 核心):前向取动作/价值 -> env.step -> 自举 TD -> RTRL 梯度
        -> 更新三条资格迹 -> 组装上升更新 -> adam -> Polyak 平均 torso -> 更新 I。"""
        action_key, step_key = jax.random.split(key)
        obs, done, ts_action, reward = state.timestep.to_sequence()

        # 用"快"头 + "慢"LRU torso 组成求梯度的前向参数。
        gp = self._grad_params(state.params, state.slow_torso)

        # Acting + value forward (advances the shared carry + RTRL sensitivity).
        # 采样动作与价值的前向,同时推进共享 carry 与 RTRL 敏感度 S_t。
        (carry, sensitivity), (dist, value) = self._forward(
            gp, obs, ts_action, reward, done, state.carry, state.sensitivity
        )
        action, log_prob = dist.sample_and_log_prob(seed=action_key)
        entropy = remove_time_axis(dist.entropy()).mean()
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)
        value = remove_feature_axis(remove_time_axis(value))

        # Environment-facing action: optionally clipped to [-act_clip, act_clip]
        # (streaming-rtrrl clips brax actions to [-1,1] before env.step and the
        # meta-RL feedback input). The eligibility trace / log_prob below keep the
        # UNCLIPPED `action` (align_action_logprob=False in RTRRL-HOP-533).
        c = self.cfg.act_clip
        env_action = jnp.clip(action, -c, c) if c else action

        # Env step.
        num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, env_action, self.env_params)

        # Bootstrap value at s' (stop-grad carry + sensitivity + params). The
        # meta-RL input for s' uses the clipped env_action (matches what the env saw).
        next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
            obs=next_obs, action=env_action, reward=next_reward, done=next_done
        ).to_sequence()
        _, (_, next_value) = self._forward(
            jax.lax.stop_gradient(gp),
            next_obs_s,
            next_action_s,
            next_reward_s,
            next_done_s,
            jax.lax.stop_gradient(carry),
            jax.lax.stop_gradient(sensitivity),
        )
        next_value = remove_feature_axis(remove_time_axis(next_value))

        # 单步 TD 误差 δ = r + γ(1-done)V(s') - V(s)。
        gamma = self.cfg.gamma
        td_error = next_reward + gamma * (1 - next_done) * next_value - value

        # ---- gradients (RTRL through the shared torso via the RFLO phantom) ----
        # 求梯度的前向从动作前的 carry/敏感度起步,并做 stop_gradient。
        initial_carry = jax.lax.stop_gradient(state.carry)
        initial_sens = jax.lax.stop_gradient(state.sensitivity)

        def td_loss_fn(params: PyTree):
            """AC 目标:eta_pi·logπ(a|s) + V(s);其雅可比给出 actor+critic+循环梯度。"""
            _, (d, v) = self._forward(
                params, obs, ts_action, reward, done, initial_carry, initial_sens
            )
            log_p = remove_time_axis(d.log_prob(add_time_axis(action)))
            v = remove_feature_axis(remove_time_axis(v))
            # actor objective scaled by eta_pi, plus the value objective.
            return self.cfg.eta_pi * log_p + v

        def entropy_loss_fn(params: PyTree):
            """熵目标:entropy_rate·H(π);作为独立梯度直接加入更新(不进资格迹)。"""
            _, (d, _) = self._forward(
                params, obs, ts_action, reward, done, initial_carry, initial_sens
            )
            return self.cfg.entropy_rate * remove_time_axis(d.entropy())

        # jacobian 借 RFLO 幻影带入共享 torso 的循环梯度(d phantom/dψ = S_t)。
        td_grads = jax.jacobian(td_loss_fn)(gp)
        entropy_grads = jax.jacobian(entropy_loss_fn)(gp)

        # ---- eligibility traces (per component decay, I-weighted increment) ----
        # 资格迹递推:z <- decay·(1-done)·z + I·g;增量由 emphasis 因子 I 加权。
        not_done = 1 - next_done  # episode continues past this transition
        I = state.I

        def update_trace(z, g, decay):
            n_trailing = z.ndim - 1
            nd = not_done[(slice(None),) + (None,) * n_trailing]
            I_b = I[(slice(None),) + (None,) * n_trailing]
            return decay * nd * z + I_b * g

        # 三条迹用各自的衰减系数(actor/critic/循环)独立更新。
        traces = {
            k: jax.tree.map(
                partial(lambda zz, gg, d: update_trace(zz, gg, d), d=self._trace_decay(k)),
                state.traces[k],
                td_grads[k],
            )
            for k in state.traces
        }

        # ---- assemble ascent updates: delta*z (+eta_f for recurrent) + entropy ----
        def apply_delta(z, extra_scale):
            # 上升方向 = extra_scale·δ·z(循环组额外乘 eta_f)。
            n_trailing = z.ndim - 1
            delta = td_error[(slice(None),) + (None,) * n_trailing]
            return extra_scale * delta * z

        updates = {}
        for k in state.params:
            scale = self.cfg.eta_f if k in _RNN_KEYS else 1.0
            traced = jax.tree.map(lambda z: apply_delta(z, scale), traces[k])
            # entropy gradient added directly (not through the trace, not delta-scaled).
            # 熵梯度直接叠加(不经资格迹、也不乘 δ)。
            combined = jax.tree.map(lambda t, e: t + e, traced, entropy_grads[k])
            # mean over the env (batch) axis -> parameter-shaped update.
            # 对 env(batch)维取均值,得到与参数同形的更新。
            updates[k] = jax.tree.map(lambda u: jnp.mean(u, axis=0), combined)

        # adam 处理这些"上升"更新;注意 _make_optimizer 用 +lr,故 apply_updates 即上升。
        adam_updates, opt_state = self.optimizer.update(
            updates, state.opt_state, state.params
        )
        params = optax.apply_updates(state.params, adam_updates)

        # ---- Polyak-average the recurrent (LRU) params for the next grad forward.
        # 对循环参数做 Polyak 平均得到"慢"torso,供下一步求梯度的前向使用。
        if self.cfg.update_period != 1.0:
            slow_torso = optax.incremental_update(
                params["torso"], state.slow_torso, self.cfg.update_period
            )
        else:
            slow_torso = params["torso"]

        # ---- episodic emphasis factor I <- gamma*I (reset at episode boundary) ----
        # episode 内 I<-γI;终止步用 next_done 将 I 重置为 1。
        I_next = gamma * I * not_done + next_done

        # ---- per-step numerical-health probes (which component blows up first) --
        # 逐组件数值健康探针:定位发散时"谁先不正常"。train_loop 把每 epoch 的
        # 逐步值归约为 mean 与 max。|λ|=exp(-exp(nu_log))∈(0,1) 恒稳,但 nu_log→-∞
        # 时 |λ|→1、敏感度 S 无界累积;gamma 为输入增益。
        nu_log = _find_leaf(params["torso"], "nu_log")
        gamma_log = _find_leaf(params["torso"], "gamma_log")
        diag = {
            "diag/lambda_max": jnp.max(jnp.exp(-jnp.exp(nu_log))),
            "diag/gamma_max": jnp.max(jnp.exp(gamma_log)),
            "diag/sens_norm": _tree_norm(sensitivity),
            "diag/carry_norm": _tree_norm(carry),
            "diag/z_rnn": _tree_norm({k: traces[k] for k in _RNN_KEYS}),
            "diag/z_actor": _tree_norm(traces[_ACTOR_KEY]),
            "diag/z_critic": _tree_norm(traces[_CRITIC_KEY]),
            "diag/grad_rnn": _tree_norm({k: td_grads[k] for k in _RNN_KEYS}),
            "diag/grad_actor": _tree_norm(td_grads[_ACTOR_KEY]),
            "diag/grad_critic": _tree_norm(td_grads[_CRITIC_KEY]),
            "diag/upd_rnn": _tree_norm({k: adam_updates[k] for k in _RNN_KEYS}),
            "diag/p_torso": _tree_norm(params["torso"]),
            "diag/p_actor": _tree_norm(params[_ACTOR_KEY]),
            "diag/p_critic": _tree_norm(params[_CRITIC_KEY]),
            "diag/value_abs": jnp.abs(value).mean(),
            "diag/td_abs": jnp.abs(td_error).mean(),
        }

        lox.log(
            {
                "info": info,
                "critic/td_error": td_error.mean(),
                "actor/entropy": entropy,
                "critic/value": value.mean(),
                "emphasis/I": I.mean(),
                **diag,
            }
        )

        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        next_reward_f = jnp.asarray(next_reward, dtype=jnp.float32)
        state = state.replace(
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
                    next_done, jnp.zeros_like(next_reward_f), next_reward_f
                ),
                done=next_done,
            ),
            env_state=env_state,
            params=params,
            slow_torso=slow_torso,
            traces=traces,
            opt_state=opt_state,
            carry=carry,
            sensitivity=sensitivity,
            I=I_next,
        )
        return state, None

    # ------------------------------------------------------------------ lifecycle
    def init(self, key: Key) -> RTRRLState:
        """初始化环境、共享 torso 与 actor/critic 头参数、优化器、资格迹与敏感度。"""
        env_key, feat_key, torso_key, actor_key, critic_key, sens_key = jax.random.split(
            key, 6
        )

        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros((self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype)
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done).to_sequence()
        ts_obs, ts_done, ts_action, ts_reward = timestep

        carry_shape = (self.cfg.num_envs, None)
        carry = self.torso.initialize_carry(jax.random.key(0), carry_shape)
        # 初始化 RTRL 敏感度 S_0。
        sensitivity = self.torso.initialize_sensitivity(sens_key, carry_shape)

        # Sequential init: feature_extractor -> torso -> heads.
        # 依次初始化:特征提取器 -> 循环 torso -> 两个头(需前一级输出的形状)。
        feat_vars = self.feature_extractor.init(
            {"params": feat_key},
            observation=ts_obs,
            action=ts_action,
            reward=ts_reward,
            done=ts_done,
        )
        x, _ = self.feature_extractor.apply(
            feat_vars,
            observation=ts_obs,
            action=ts_action,
            reward=ts_reward,
            done=ts_done,
        )
        torso_vars = self.torso.init({"params": torso_key}, x, ts_done, initial_carry=carry)
        _, h = self.torso.apply(torso_vars, x, ts_done, initial_carry=carry)
        h = self.activation(h)
        actor_vars = self.actor_head.init(
            {"params": actor_key}, h, action=ts_action, reward=ts_reward, done=ts_done
        )
        critic_vars = self.critic_head.init(
            {"params": critic_key}, h, action=ts_action, reward=ts_reward, done=ts_done
        )

        params = {
            "feature_extractor": feat_vars["params"],
            "torso": torso_vars["params"],
            "actor": actor_vars["params"],
            "critic": critic_vars["params"],
        }

        # 资格迹与参数同构、前面多一个 env 维,初始全零。
        traces = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape)), params
        )

        # 参数结构确定后再构建 adam multi_transform 优化器。
        self.optimizer = self._make_optimizer(params)
        opt_state = self.optimizer.init(params)

        return RTRRLState(
            step=0,
            update_step=0,
            timestep=timestep.from_sequence(),
            env_state=env_state,
            params=params,
            slow_torso=params["torso"],
            traces=traces,
            opt_state=opt_state,
            carry=carry,
            sensitivity=sensitivity,
            I=jnp.ones((self.cfg.num_envs,), dtype=jnp.float32),
        )

    def warmup(self, key: Key, state: RTRRLState, num_steps: int) -> RTRRLState:
        """在线算法,无需预热,直接返回原状态。"""
        return state

    def train(self, key: Key, state: RTRRLState, num_steps: int) -> RTRRLState:
        """训练:逐步在线 RTRL 更新,共 num_steps 环境步。"""
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(self._update_step, state, keys)
        return state

    def evaluate(self, key: Key, state: RTRRLState, num_steps: int) -> RTRRLState:
        """评估:重置环境、carry 与敏感度后以确定性策略运行 num_steps 步。"""
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros((self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype)
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)

        carry_shape = (self.cfg.num_envs, None)
        state = state.replace(
            timestep=timestep,
            carry=self.torso.initialize_carry(jax.random.key(0), carry_shape),
            sensitivity=self.torso.initialize_sensitivity(jax.random.key(0), carry_shape),
            env_state=env_state,
        )

        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._deterministic_action),
            state,
            step_keys,
        )
        return state
