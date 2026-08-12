"""S1 minimal DiagSSM qualification runner.

This is deliberately a self-contained experiment executable.  Both learners use
the same 16-complex-mode DiagSSM, heads, input stream, resets, and evaluation.
The RTRL learner carries exact state sensitivities and eligibility traces; the
BPTT learner carries neither and detaches its recurrent state every chunk.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs

jax.config.update("jax_enable_x64", False)


@dataclass(frozen=True)
class Config:
    learner: str
    seed: int
    steps: int = 300_000
    modes: int = 16
    actor_lr: float = 1e-2
    critic_lr: float = 1.0
    recurrent_lr: float = 1e-3
    clip_norm: float = 1.0
    gamma: float = 0.99
    eligibility_lambda: float = 0.9
    horizon: int = 1000
    bptt_length: int = 128
    log_every: int = 10_000
    eval_every: int = 10_000
    checkpoint_every: int = 50_000
    eval_episodes: int = 10


class Params(NamedTuple):
    recurrent: jax.Array
    actor_w: jax.Array
    actor_b: jax.Array
    critic_w: jax.Array
    critic_b: jax.Array


class TrainState(NamedTuple):
    env: Any
    params: Params
    hidden: jax.Array
    sensitivity: jax.Array
    actor_elig: tuple[jax.Array, jax.Array]
    critic_elig: tuple[jax.Array, jax.Array]
    recurrent_elig: jax.Array
    action: jax.Array
    value: jax.Array
    key: jax.Array
    previous_reward: jax.Array
    episode_age: jax.Array
    episode_return: jax.Array
    step: jax.Array


def _tree_to_host(tree: Any) -> Any:
    return jax.tree.map(lambda x: np.asarray(jax.device_get(x)), tree)


def _global_clip(tree: Any, limit: float) -> tuple[Any, jax.Array, jax.Array]:
    leaves = jax.tree.leaves(tree)
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))
    scale = jnp.minimum(1.0, limit / (norm + 1e-12))
    return jax.tree.map(lambda x: x * scale, tree), norm, scale


class S1Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.env = envs.get_environment(env_name="halfcheetah", backend="spring")
        self.mask = jnp.arange(0, self.env.observation_size, 2)
        self.action_size = self.env.action_size
        self.input_size = int(self.mask.size) + self.action_size + 1
        self.hidden_size = 2 * cfg.modes
        self.local_parameter_size = 2 + 2 * self.input_size

    def unpack_recurrent(self, p: jax.Array):
        m, u = self.cfg.modes, self.input_size
        nu = p[:m]
        theta = p[m : 2 * m]
        bre = p[2 * m : 2 * m + m * u].reshape(m, u)
        bim = p[2 * m + m * u :].reshape(m, u)
        return nu, theta, bre, bim

    def coefficients(self, p: jax.Array):
        nu, theta, bre, bim = self.unpack_recurrent(p)
        decay = jnp.exp(nu)
        radius = jnp.exp(-decay)
        omega = jnp.exp(theta)
        cosine, sine = jnp.cos(omega), jnp.sin(omega)
        gain = jnp.sqrt(jnp.maximum(1.0 - radius**2, 1e-12))
        dr_dnu = -decay * radius
        dg_dnu = -radius * dr_dnu / gain
        return bre, bim, radius, omega, cosine, sine, gain, dr_dnu, dg_dnu

    def recurrent_step(self, hidden, p, inputs, reset):
        bre, bim, r, _omega, c, s, gain, _dr, _dg = self.coefficients(p)
        hidden = jnp.where(reset, jnp.zeros_like(hidden), hidden)
        xr, xi = hidden[:, 0], hidden[:, 1]
        yr, yi = c * xr - s * xi, s * xr + c * xi
        return jnp.stack((r * yr + gain * (bre @ inputs), r * yi + gain * (bim @ inputs)), 1)

    def rtrl_step(self, hidden, sensitivity, p, inputs, reset):
        bre, bim, r, omega, c, s, gain, dr, dg = self.coefficients(p)
        hidden = jnp.where(reset, jnp.zeros_like(hidden), hidden)
        sensitivity = jnp.where(reset, jnp.zeros_like(sensitivity), sensitivity)
        xr, xi = hidden[:, 0], hidden[:, 1]
        ur, ui = bre @ inputs, bim @ inputs
        yr, yi = c * xr - s * xi, s * xr + c * xi
        new_hidden = jnp.stack((r * yr + gain * ur, r * yi + gain * ui), 1)
        ap0 = r[:, None] * (c[:, None] * sensitivity[:, 0] - s[:, None] * sensitivity[:, 1])
        ap1 = r[:, None] * (s[:, None] * sensitivity[:, 0] + c[:, None] * sensitivity[:, 1])
        local = jnp.zeros_like(sensitivity)
        local = local.at[:, 0, 0].set(dr * yr + dg * ur)
        local = local.at[:, 1, 0].set(dr * yi + dg * ui)
        local = local.at[:, 0, 1].set(r * (-s * xr - c * xi) * omega)
        local = local.at[:, 1, 1].set(r * (c * xr - s * xi) * omega)
        u = self.input_size
        local = local.at[:, 0, 2 : 2 + u].set(gain[:, None] * inputs)
        local = local.at[:, 1, 2 + u :].set(gain[:, None] * inputs)
        return new_hidden, jnp.stack((ap0, ap1), 1) + local

    def local_to_global(self, gradient):
        u = self.input_size
        return jnp.concatenate(
            (gradient[:, 0], gradient[:, 1], gradient[:, 2 : 2 + u].ravel(), gradient[:, 2 + u :].ravel())
        )

    def policy(self, params, hidden, key):
        output = hidden.reshape(-1) @ params.actor_w + params.actor_b
        mean, raw_scale = jnp.split(output, 2)
        bounded = -2.0 + 4.0 * jax.nn.sigmoid(raw_scale)
        scale = jax.nn.softplus(bounded)
        action = mean + scale * jax.random.normal(key, mean.shape)
        value = (hidden.reshape(-1) @ params.critic_w + params.critic_b)[0]
        return action, value, mean, scale

    def rtrl_sources(self, params, hidden, sensitivity, action):
        """Score/value sources for the action sampled at this recurrent state."""
        output = hidden.reshape(-1) @ params.actor_w + params.actor_b
        mean, raw_scale = jnp.split(output, 2)
        bounded = -2.0 + 4.0 * jax.nn.sigmoid(raw_scale)
        scale = jax.nn.softplus(bounded)
        executed = jax.lax.stop_gradient(jnp.clip(action, -1.0, 1.0))
        dmean = (executed - mean) / scale**2
        dscale = (executed - mean) ** 2 / scale**3 - 1.0 / scale
        sigmoid_raw = jax.nn.sigmoid(raw_scale)
        draw = (
            dscale
            * jax.nn.sigmoid(bounded)
            * 4.0
            * sigmoid_raw
            * (1.0 - sigmoid_raw)
        )
        score = jnp.concatenate((dmean, draw))
        actor_source = (jnp.outer(hidden.reshape(-1), score), score)
        critic_source = (
            hidden.reshape(-1)[:, None],
            jnp.ones((1,), jnp.float32),
        )
        hidden_source = params.actor_w @ score + params.critic_w[:, 0]
        local_gradient = jnp.einsum(
            "mi,mij->mj", hidden_source.reshape(self.cfg.modes, 2), sensitivity
        )
        return actor_source, critic_source, self.local_to_global(local_gradient)

    def init(self):
        cfg, m, u, a = self.cfg, self.cfg.modes, self.input_size, self.action_size
        keys = jax.random.split(jax.random.PRNGKey(cfg.seed), 9)
        env_state = self.env.reset(keys[0])
        uniform = jax.random.uniform(keys[1], (m,), minval=1e-6, maxval=1 - 1e-6)
        nu = jnp.log(-0.5 * jnp.log(uniform))
        theta = jnp.log(jax.random.uniform(keys[2], (m,), minval=1e-4, maxval=2 * jnp.pi))
        bre = jax.random.normal(keys[3], (m, u)) / jnp.sqrt(2 * u)
        bim = jax.random.normal(keys[4], (m, u)) / jnp.sqrt(2 * u)
        recurrent = jnp.concatenate((nu, theta, bre.ravel(), bim.ravel())).astype(jnp.float32)
        glorot = jax.nn.initializers.glorot_normal(in_axis=-1, out_axis=-2)
        params = Params(
            recurrent,
            glorot(keys[5], (2 * m, 2 * a), jnp.float32),
            jnp.zeros((2 * a,), jnp.float32),
            glorot(keys[6], (2 * m, 1), jnp.float32),
            jnp.zeros((1,), jnp.float32),
        )
        hidden = jnp.zeros((m, 2), jnp.float32)
        inputs = jnp.concatenate((env_state.obs[self.mask], jnp.zeros((a + 1,), jnp.float32)))
        sensitivity = jnp.zeros((m, 2, self.local_parameter_size), jnp.float32)
        hidden, sensitivity = self.rtrl_step(
            hidden, sensitivity, recurrent, inputs, jnp.array(True)
        )
        action, value, *_ = self.policy(params, hidden, keys[7])
        actor_elig, critic_elig, recurrent_elig = self.rtrl_sources(
            params, hidden, sensitivity, action
        )
        return TrainState(
            env_state,
            params,
            hidden,
            sensitivity,
            actor_elig,
            critic_elig,
            recurrent_elig,
            action,
            value,
            keys[8],
            jnp.array(0.0, jnp.float32),
            jnp.array(0, jnp.int32),
            jnp.array(0.0, jnp.float32),
            jnp.array(0, jnp.int32),
        )

    def transition(self, state, action):
        executed = jax.lax.stop_gradient(jnp.clip(action, -1.0, 1.0))
        next_env = self.env.step(state.env, executed)
        done = next_env.done.astype(bool) | (state.episode_age + 1 >= self.cfg.horizon)
        key, reset_key, action_key = jax.random.split(state.key, 3)
        reset_env = self.env.reset(reset_key)
        pipeline = jax.tree.map(
            lambda new, old: jnp.where(done, new, old), reset_env.pipeline_state, next_env.pipeline_state
        )
        obs = jnp.where(done, reset_env.obs, next_env.obs)
        next_env = next_env.replace(pipeline_state=pipeline, obs=obs)
        previous_action = jnp.where(done, jnp.zeros_like(executed), executed)
        previous_reward = jnp.where(done, 0.0, next_env.reward)
        inputs = jnp.concatenate((obs[self.mask], previous_action, previous_reward[None]))
        return next_env, next_env.reward, done, inputs, key, action_key

    def rtrl_train_step(self, state, _):
        cfg = self.cfg
        env_state, reward, done, inputs, key, action_key = self.transition(state, state.action)
        hidden, sensitivity = self.rtrl_step(
            state.hidden, state.sensitivity, state.params.recurrent, inputs, done
        )
        action, value, mean, scale = self.policy(state.params, hidden, action_key)
        alive = 1.0 - done.astype(jnp.float32)
        delta = reward + cfg.gamma * value * alive - state.value
        # The carried traces already contain the source for state.action at h_t.
        raw_a = jax.tree.map(lambda z: delta * z, state.actor_elig)
        raw_c = jax.tree.map(lambda z: delta * z, state.critic_elig)
        raw_r = delta * state.recurrent_elig
        upd_a, norm_a, scale_a = _global_clip(raw_a, cfg.clip_norm)
        upd_c, norm_c, scale_c = _global_clip(raw_c, cfg.clip_norm)
        upd_r, norm_r, scale_r = _global_clip(raw_r, cfg.clip_norm)
        p = state.params
        params = Params(
            p.recurrent + cfg.recurrent_lr * upd_r,
            p.actor_w + cfg.actor_lr * upd_a[0],
            p.actor_b + cfg.actor_lr * upd_a[1],
            p.critic_w + cfg.critic_lr * upd_c[0],
            p.critic_b + cfg.critic_lr * upd_c[1],
        )
        actor_source, critic_source, recurrent_source = self.rtrl_sources(
            state.params, hidden, sensitivity, action
        )
        actor_elig = jax.tree.map(
            lambda z, g: cfg.gamma * cfg.eligibility_lambda * z * alive + g,
            state.actor_elig,
            actor_source,
        )
        critic_elig = jax.tree.map(
            lambda z, g: cfg.gamma * cfg.eligibility_lambda * z * alive + g,
            state.critic_elig,
            critic_source,
        )
        recurrent_elig = (
            cfg.gamma * cfg.eligibility_lambda * state.recurrent_elig * alive
            + recurrent_source
        )
        episode_return = state.episode_return + reward
        completed_return = jnp.where(done, episode_return, jnp.nan)
        new_state = TrainState(
            env_state, params, hidden, sensitivity, actor_elig, critic_elig, recurrent_elig,
            action, value, key, jnp.where(done, 0.0, reward),
            jnp.where(done, 0, state.episode_age + 1), jnp.where(done, 0.0, episode_return),
            state.step + 1,
        )
        completed_length = jnp.where(done, state.episode_age + 1, jnp.nan)
        finite = jnp.stack(
            [jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(new_state)]
        ).all()
        metrics = self.metrics(params, hidden, sensitivity, recurrent_elig, reward, delta,
                               jnp.abs(delta), delta**2, value, value**2, mean, scale,
                               norm_a, norm_c, norm_r, scale_a, scale_c, scale_r,
                               completed_return, completed_length, finite)
        return new_state, metrics

    def metrics(self, params, hidden, sensitivity, recurrent_elig, reward, delta, td_abs,
                td_square, value, value_square, mean, scale,
                actor_norm, critic_norm, recurrent_norm, actor_scale, critic_scale, recurrent_scale,
                completed_return, completed_length, finite):
        _bre, _bim, radius, omega, *_ = self.coefficients(params.recurrent)
        timescale = -1.0 / jnp.log(radius)
        return jnp.asarray((
            reward, delta, td_abs, td_square, value, value_square,
            actor_norm, critic_norm, recurrent_norm, actor_scale < 1, critic_scale < 1,
            recurrent_scale < 1, jnp.mean(mean), jnp.linalg.norm(mean), jnp.mean(scale),
            jnp.linalg.norm(sensitivity), jnp.linalg.norm(recurrent_elig), jnp.mean(radius),
            jnp.min(radius), jnp.max(radius), jnp.mean(omega), jnp.mean(timescale),
            jnp.linalg.norm(params.recurrent), finite, completed_return, completed_length,
        ), jnp.float32)

    def bptt_chunk(self, state):
        cfg = self.cfg
        # The chunk initial state is numerically retained but severs the previous graph.
        state = state._replace(hidden=jax.lax.stop_gradient(state.hidden))
        keys = jax.random.split(state.key, cfg.bptt_length + 1)

        def loss(params):
            def one(carry, action_key):
                env_state, hidden, action, value, age, ep_return = carry
                temp = state._replace(env=env_state, hidden=hidden, action=action, value=value,
                                      episode_age=age, episode_return=ep_return, key=action_key)
                next_env, reward, done, inputs, _, policy_key = self.transition(temp, action)
                next_hidden = self.recurrent_step(hidden, params.recurrent, inputs, done)
                next_action, next_value, mean, scale = self.policy(params, next_hidden, policy_key)
                executed = jax.lax.stop_gradient(jnp.clip(action, -1, 1))
                old_mean, old_scale = self.policy(params, hidden, policy_key)[2:]
                log_prob = -0.5 * jnp.sum(((executed - old_mean) / old_scale) ** 2 + 2*jnp.log(old_scale) + jnp.log(2*jnp.pi))
                alive = 1.0 - done.astype(jnp.float32)
                target = jax.lax.stop_gradient(reward + cfg.gamma * next_value * alive)
                advantage = jax.lax.stop_gradient(target - value)
                actor_loss = -advantage * log_prob
                critic_loss = 0.5 * (value - target) ** 2
                completed = jnp.where(done, ep_return + reward, jnp.nan)
                completed_length = jnp.where(done, age + 1, jnp.nan)
                carry = (next_env, next_hidden, next_action, next_value,
                         jnp.where(done, 0, age + 1), jnp.where(done, 0.0, ep_return + reward))
                aux = (reward, target-value, value, mean, scale, completed, completed_length)
                return carry, (actor_loss + critic_loss, aux)

            initial = (state.env, state.hidden, state.action, state.value,
                       state.episode_age, state.episode_return)
            final, (losses, aux) = jax.lax.scan(one, initial, keys[1:])
            return jnp.sum(losses), (final, aux)

        (_loss, (final, aux)), grads = jax.value_and_grad(loss, has_aux=True)(state.params)
        grad_r, norm_r, scale_r = _global_clip(grads.recurrent, cfg.clip_norm)
        grad_a, norm_a, scale_a = _global_clip((grads.actor_w, grads.actor_b), cfg.clip_norm)
        grad_c, norm_c, scale_c = _global_clip((grads.critic_w, grads.critic_b), cfg.clip_norm)
        p = state.params
        params = Params(p.recurrent - cfg.recurrent_lr * grad_r,
                        p.actor_w - cfg.actor_lr * grad_a[0], p.actor_b - cfg.actor_lr * grad_a[1],
                        p.critic_w - cfg.critic_lr * grad_c[0], p.critic_b - cfg.critic_lr * grad_c[1])
        env_state, hidden, action, value, age, ep_return = final
        reward, delta, value_series, means, scales, completed, completed_length = aux
        completion_indices = jnp.where(
            jnp.isfinite(completed), jnp.arange(completed.size), -1
        )
        last_completion = jnp.max(completion_indices)
        completed_return = jnp.where(
            last_completion >= 0, completed[jnp.maximum(last_completion, 0)], jnp.nan
        )
        completed_episode_length = jnp.where(
            last_completion >= 0,
            completed_length[jnp.maximum(last_completion, 0)],
            jnp.nan,
        )
        empty_sensitivity = jnp.zeros_like(state.sensitivity)
        new_state = state._replace(env=env_state, params=params, hidden=jax.lax.stop_gradient(hidden),
                                   sensitivity=empty_sensitivity, action=action, value=value,
                                   key=keys[0], previous_reward=reward[-1], episode_age=age,
                                   episode_return=ep_return, step=state.step + cfg.bptt_length)
        finite = jnp.stack(
            [jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(new_state)]
        ).all()
        metric = self.metrics(
            params, hidden, empty_sensitivity, jnp.zeros_like(params.recurrent),
            jnp.mean(reward), jnp.mean(delta), jnp.mean(jnp.abs(delta)),
            jnp.mean(delta**2), jnp.mean(value_series), jnp.mean(value_series**2),
            jnp.mean(means, 0), jnp.mean(scales, 0), norm_a, norm_c, norm_r,
            scale_a, scale_c, scale_r, completed_return, completed_episode_length, finite,
        )
        return new_state, metric

    def evaluate(self, params):
        count = self.cfg.eval_episodes
        env_keys = jax.random.split(jax.random.PRNGKey(123456), count)
        policy_keys = jax.random.split(jax.random.PRNGKey(123457), count)
        env_state = jax.vmap(self.env.reset)(env_keys)
        hidden = jnp.zeros((count, self.cfg.modes, 2), jnp.float32)
        previous_action = jnp.zeros((count, self.action_size), jnp.float32)
        previous_reward = jnp.zeros((count,), jnp.float32)
        active = jnp.ones((count,), bool)
        total = jnp.zeros((count,), jnp.float32)

        def step(carry, age):
            env_state, hidden, previous_action, previous_reward, policy_keys, active, total = carry
            inputs = jnp.concatenate(
                (env_state.obs[:, self.mask], previous_action, previous_reward[:, None]), axis=1
            )
            hidden = jax.vmap(
                lambda h, u: self.recurrent_step(h, params.recurrent, u, age == 0)
            )(hidden, inputs)
            split_keys = jax.vmap(jax.random.split)(policy_keys)
            policy_keys, action_keys = split_keys[:, 0], split_keys[:, 1]
            action = jax.vmap(lambda h, k: self.policy(params, h, k)[0])(hidden, action_keys)
            action = jax.lax.stop_gradient(jnp.clip(action, -1, 1))
            next_env = jax.vmap(self.env.step)(env_state, action)
            total = total + jnp.where(active, next_env.reward, 0.0)
            active = active & ~next_env.done.astype(bool)
            previous_action = jnp.where(active[:, None], action, 0.0)
            previous_reward = jnp.where(active, next_env.reward, 0.0)
            return (
                next_env, hidden, previous_action, previous_reward,
                policy_keys, active, total,
            ), None

        final, _ = jax.lax.scan(
            step,
            (env_state, hidden, previous_action, previous_reward, policy_keys, active, total),
            jnp.arange(self.cfg.horizon),
        )
        return np.asarray(jax.device_get(final[-1]))


METRIC_NAMES = (
    "reward", "td_mean", "td_abs", "td_square", "value", "value_square", "actor_raw_norm",
    "critic_raw_norm", "recurrent_raw_norm", "actor_clipped", "critic_clipped", "recurrent_clipped",
    "action_mean", "policy_mean_norm", "action_std", "sensitivity_norm", "recurrent_eligibility_norm",
    "pole_radius_mean", "pole_radius_min", "pole_radius_max", "pole_frequency_mean", "timescale_mean",
    "recurrent_parameter_norm", "finite", "completed_episode_return", "completed_episode_length",
)


def save_checkpoint(path: Path, cfg: Config, state: TrainState):
    if cfg.learner == "rtrl":
        stored_state: Any = _tree_to_host(state)
    else:
        # BPTT checkpoints intentionally contain no sensitivity or eligibility state.
        omitted = {"sensitivity", "actor_elig", "critic_elig", "recurrent_elig"}
        stored_state = {
            name: _tree_to_host(value)
            for name, value in state._asdict().items()
            if name not in omitted
        }
    payload = {"schema": 1, "config": asdict(cfg), "state": stored_state}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_checkpoint(path: Path, cfg: Config, runner: S1Runner):
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted experiment artifact
    if payload["config"] != asdict(cfg):
        raise ValueError("checkpoint config does not exactly match requested config")
    stored = payload["state"]
    if cfg.learner == "rtrl":
        return jax.tree.map(jnp.asarray, stored)
    blank = runner.init()
    restored = {
        name: jax.tree.map(jnp.asarray, stored[name]) if name in stored else value
        for name, value in blank._asdict().items()
    }
    return TrainState(**restored)


def aggregate(metrics):
    values = np.asarray(jax.device_get(metrics))
    values = values.reshape(-1, values.shape[-1])
    out = {name: float(np.nanmean(values[:, i])) for i, name in enumerate(METRIC_NAMES[:-2])}
    episode_returns = values[:, -2]
    episode_lengths = values[:, -1]
    out["training_episode_returns"] = episode_returns[np.isfinite(episode_returns)].astype(float).tolist()
    out["training_episode_lengths"] = episode_lengths[np.isfinite(episode_lengths)].astype(int).tolist()
    out["td_rms"] = float(np.sqrt(out.pop("td_square")))
    out["value_rms"] = float(np.sqrt(out.pop("value_square")))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner", choices=("rtrl", "bptt128"), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = Config(learner=args.learner, seed=args.seed, steps=args.steps)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    runner = S1Runner(cfg)
    state = load_checkpoint(args.resume, cfg, runner) if args.resume else runner.init()
    step_fn = jax.jit(runner.bptt_chunk) if cfg.learner == "bptt128" else None
    rtrl_scans = {}
    final_bptt_fn = None
    started = time.perf_counter()
    window_started = started
    window_start_step = int(state.step)
    buffered = []
    metrics_file = (out / "metrics.jsonl").open("a", buffering=1)
    while int(state.step) < cfg.steps:
        previous_step = int(state.step)
        if cfg.learner == "rtrl":
            chunk = min(cfg.log_every, cfg.steps - previous_step)
            if chunk not in rtrl_scans:
                rtrl_scans[chunk] = jax.jit(
                    lambda carry, length=chunk: jax.lax.scan(
                        runner.rtrl_train_step, carry, None, length=length
                    )
                )
            state, metric = rtrl_scans[chunk](state)
        else:
            remaining = cfg.steps - previous_step
            if remaining >= cfg.bptt_length:
                state, metric = step_fn(state)
            else:
                # Only the terminal chunk may be shorter than 128.
                if final_bptt_fn is None:
                    final_runner = S1Runner(
                        Config(**{**asdict(cfg), "bptt_length": remaining})
                    )
                    final_bptt_fn = jax.jit(final_runner.bptt_chunk)
                state, metric = final_bptt_fn(state)
        jax.block_until_ready(metric)
        buffered.append(metric)
        step = int(state.step)
        crossed_log = step // cfg.log_every > previous_step // cfg.log_every
        crossed_eval = step // cfg.eval_every > previous_step // cfg.eval_every
        crossed_checkpoint = step // cfg.checkpoint_every > previous_step // cfg.checkpoint_every
        if crossed_log or step == cfg.steps:
            elapsed = time.perf_counter() - window_started
            record = {"step": step, "learner": cfg.learner, "seed": cfg.seed,
                      "nominal_log_boundary": (step // cfg.log_every) * cfg.log_every,
                      "wall_seconds": time.perf_counter() - started,
                      "environment_steps_per_second": (step - window_start_step) / elapsed,
                      "learner_update_seconds": elapsed,
                      "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      **aggregate(jnp.stack(buffered))}
            if crossed_eval or step == cfg.steps:
                eval_returns = runner.evaluate(state.params)
                record["nominal_eval_boundary"] = (step // cfg.eval_every) * cfg.eval_every
                record["fixed_eval_returns"] = eval_returns.astype(float).tolist()
                record["fixed_eval_mean"] = float(eval_returns.mean())
                record["fixed_eval_median"] = float(np.median(eval_returns))
                record["fixed_eval_std"] = float(eval_returns.std())
            metrics_file.write(json.dumps(record) + "\n")
            buffered.clear()
            window_started = time.perf_counter()
            window_start_step = step
        if crossed_checkpoint or step == cfg.steps:
            save_checkpoint(out / f"checkpoint_{step:07d}.pkl", cfg, state)
    parameter_count = sum(int(x.size) for x in jax.tree.leaves(state.params))
    trace_bytes = int(state.sensitivity.size * state.sensitivity.dtype.itemsize) if cfg.learner == "rtrl" else 0
    summary = {"learner": cfg.learner, "seed": cfg.seed, "steps": int(state.step),
               "wall_seconds": time.perf_counter() - started, "parameter_count": parameter_count,
               "rtrl_trace_elements": int(state.sensitivity.size) if cfg.learner == "rtrl" else 0,
               "rtrl_trace_bytes": trace_bytes,
               "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               "device_memory_note": "peak device allocation must be collected by the batch runtime/profiler",
               "finite": bool(all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(_tree_to_host(state))))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metrics_file.close()


if __name__ == "__main__":
    main()
