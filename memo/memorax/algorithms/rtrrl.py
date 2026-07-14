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


@struct.dataclass(frozen=True)
class RTRRLConfig:
    num_envs: int
    gamma: float = 0.95
    # Per-component eligibility-trace decay.
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


@struct.dataclass(frozen=True)
class RTRRLState:
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

        def label(top_key):
            return "rnn" if top_key in _RNN_KEYS else "td"

        param_labels = {
            k: jax.tree.map(lambda _: label(k), v) for k, v in params.items()
        }
        return optax.multi_transform({"td": td_tx, "rnn": rnn_tx}, param_labels)

    # ------------------------------------------------------------------ policies
    def _deterministic_action(
        self, key: Key, state: RTRRLState
    ) -> tuple[RTRRLState, Array, Array, Array]:
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
        action_key, step_key = jax.random.split(key)
        state, action, log_prob, value = policy(action_key, state)

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
        action_key, step_key = jax.random.split(key)
        obs, done, ts_action, reward = state.timestep.to_sequence()

        gp = self._grad_params(state.params, state.slow_torso)

        # Acting + value forward (advances the shared carry + RTRL sensitivity).
        (carry, sensitivity), (dist, value) = self._forward(
            gp, obs, ts_action, reward, done, state.carry, state.sensitivity
        )
        action, log_prob = dist.sample_and_log_prob(seed=action_key)
        entropy = remove_time_axis(dist.entropy()).mean()
        action = remove_time_axis(action)
        log_prob = remove_time_axis(log_prob)
        value = remove_feature_axis(remove_time_axis(value))

        # Env step.
        num_envs, *_ = state.timestep.obs.shape
        step_keys = jax.random.split(step_key, num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)

        # Bootstrap value at s' (stop-grad carry + sensitivity + params).
        next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
            obs=next_obs, action=action, reward=next_reward, done=next_done
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

        gamma = self.cfg.gamma
        td_error = next_reward + gamma * (1 - next_done) * next_value - value

        # ---- gradients (RTRL through the shared torso via the RFLO phantom) ----
        initial_carry = jax.lax.stop_gradient(state.carry)
        initial_sens = jax.lax.stop_gradient(state.sensitivity)

        def td_loss_fn(params: PyTree):
            _, (d, v) = self._forward(
                params, obs, ts_action, reward, done, initial_carry, initial_sens
            )
            log_p = remove_time_axis(d.log_prob(add_time_axis(action)))
            v = remove_feature_axis(remove_time_axis(v))
            # actor objective scaled by eta_pi, plus the value objective.
            return self.cfg.eta_pi * log_p + v

        def entropy_loss_fn(params: PyTree):
            _, (d, _) = self._forward(
                params, obs, ts_action, reward, done, initial_carry, initial_sens
            )
            return self.cfg.entropy_rate * remove_time_axis(d.entropy())

        td_grads = jax.jacobian(td_loss_fn)(gp)
        entropy_grads = jax.jacobian(entropy_loss_fn)(gp)

        # ---- eligibility traces (per component decay, I-weighted increment) ----
        not_done = 1 - next_done  # episode continues past this transition
        I = state.I

        def update_trace(z, g, decay):
            n_trailing = z.ndim - 1
            nd = not_done[(slice(None),) + (None,) * n_trailing]
            I_b = I[(slice(None),) + (None,) * n_trailing]
            return decay * nd * z + I_b * g

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
            n_trailing = z.ndim - 1
            delta = td_error[(slice(None),) + (None,) * n_trailing]
            return extra_scale * delta * z

        updates = {}
        for k in state.params:
            scale = self.cfg.eta_f if k in _RNN_KEYS else 1.0
            traced = jax.tree.map(lambda z: apply_delta(z, scale), traces[k])
            # entropy gradient added directly (not through the trace, not delta-scaled).
            combined = jax.tree.map(lambda t, e: t + e, traced, entropy_grads[k])
            # mean over the env (batch) axis -> parameter-shaped update.
            updates[k] = jax.tree.map(lambda u: jnp.mean(u, axis=0), combined)

        adam_updates, opt_state = self.optimizer.update(
            updates, state.opt_state, state.params
        )
        params = optax.apply_updates(state.params, adam_updates)

        # ---- Polyak-average the recurrent (LRU) params for the next grad forward.
        if self.cfg.update_period != 1.0:
            slow_torso = optax.incremental_update(
                params["torso"], state.slow_torso, self.cfg.update_period
            )
        else:
            slow_torso = params["torso"]

        # ---- episodic emphasis factor I <- gamma*I (reset at episode boundary) ----
        I_next = gamma * I * not_done + next_done

        lox.log(
            {
                "info": info,
                "critic/td_error": td_error.mean(),
                "actor/entropy": entropy,
                "critic/value": value.mean(),
                "emphasis/I": I.mean(),
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
                    jnp.zeros_like(action),
                    action,
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
        sensitivity = self.torso.initialize_sensitivity(sens_key, carry_shape)

        # Sequential init: feature_extractor -> torso -> heads.
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

        traces = jax.tree.map(
            lambda p: jnp.zeros((self.cfg.num_envs, *p.shape)), params
        )

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
        return state

    def train(self, key: Key, state: RTRRLState, num_steps: int) -> RTRRLState:
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(self._update_step, state, keys)
        return state

    def evaluate(self, key: Key, state: RTRRLState, num_steps: int) -> RTRRLState:
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
