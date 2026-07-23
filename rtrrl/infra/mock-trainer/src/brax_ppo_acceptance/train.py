from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import training_sdk
from brax import envs
from brax.training.agents.ppo import train as ppo_train

from brax_ppo_acceptance.config import AcceptanceConfig

Policy = Callable[[jax.Array, jax.Array], tuple[jax.Array, Any]]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    objective: float
    checkpoint: Path
    platform: str
    device_kind: str


def _host_array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def rollout_episode(
    environment: envs.Env,
    policy: Policy,
    seed: int,
    episode_length: int,
    phase: Literal["train", "eval"],
) -> tuple[training_sdk.Episode, float]:
    """Roll out one complete episode, crossing to NumPy only at the host boundary."""
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)
    infer = jax.jit(policy)

    key = jax.random.PRNGKey(seed)
    state = reset(key)
    jax.block_until_ready(state)

    observations = [_host_array(state.obs)]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminals: list[bool] = []
    truncations: list[bool] = []

    for transition_index in range(episode_length):
        key, action_key = jax.random.split(key)
        action, _ = infer(state.obs, action_key)
        next_state = step(state, action)
        jax.block_until_ready(next_state)

        terminal = bool(_host_array(next_state.done))
        truncated = transition_index + 1 == episode_length and not terminal
        actions.append(_host_array(action))
        rewards.append(float(_host_array(next_state.reward)))
        terminals.append(terminal)
        truncations.append(truncated)
        observations.append(_host_array(next_state.obs))
        state = next_state
        if terminal:
            break

    episode = training_sdk.Episode(
        number=1 if phase == "train" else 2,
        phase=phase,
        start_env_steps=0,
        end_env_steps=len(actions),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        truncations=truncations,
    )
    episode_return = float(sum(rewards))
    if not math.isfinite(episode_return):
        raise ValueError(f"{phase} episode return must be finite")
    return episode, episode_return


def _zero_policy(environment: envs.Env) -> Policy:
    action_size = environment.action_size

    def policy(observation: jax.Array, key: jax.Array) -> tuple[jax.Array, dict[str, Any]]:
        del key
        return jnp.zeros(observation.shape[:-1] + (action_size,)), {}

    return policy


def _exercise_selected_device(environment: envs.Env, seed: int) -> None:
    warmup = jax.jit(lambda value: jnp.sin(value) + jnp.cos(value))(
        jnp.arange(1024.0)
    )
    warmup.block_until_ready()

    key = jax.random.PRNGKey(seed)
    state = jax.jit(environment.reset)(key)
    action = jnp.zeros((environment.action_size,))
    next_state = jax.jit(environment.step)(state, action)
    jax.block_until_ready(next_state)


def _write_checkpoint(path: Path, params: Any) -> None:
    leaves, _ = jax.tree_util.tree_flatten(params)
    arrays = {f"leaf_{index:04d}": _host_array(leaf) for index, leaf in enumerate(leaves)}
    if not arrays:
        raise ValueError("PPO parameters produced an empty checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _inject_failure(config: AcceptanceConfig, point: str) -> None:
    if config.failure_mode == point:
        raise RuntimeError(f"injected failure: {point}")


@contextmanager
def _brax_jax_compatibility() -> Iterator[None]:
    """Temporarily restore the JAX API still used by Brax 0.14.2."""
    if "device_put_replicated" in jax.__dict__:
        yield
        return

    from jax._src import api as jax_api

    jax.__dict__["device_put_replicated"] = jax_api.device_put_replicated
    try:
        yield
    finally:
        jax.__dict__.pop("device_put_replicated")


def train(config: AcceptanceConfig, run: training_sdk.TrainingRun) -> TrainingResult:
    """Run PPO acceptance and publish non-terminal SDK observability artifacts."""
    _inject_failure(config, "before_training")
    environment = envs.get_environment(
        env_name=config.environment_name,
        backend=config.backend,
    )

    if config.fast_mode:
        _exercise_selected_device(environment, config.seed)
        policy = _zero_policy(environment)
        params: Any = (jnp.zeros((1,)),)
    else:
        warmup = jax.jit(lambda value: jnp.sin(value) + jnp.cos(value))(
            jnp.arange(1024.0)
        )
        warmup.block_until_ready()
        with _brax_jax_compatibility():
            make_inference_fn, params, _metrics = ppo_train.train(
                environment=environment,
                num_timesteps=config.num_timesteps,
                episode_length=config.episode_length,
                num_envs=config.num_envs,
                learning_rate=config.learning_rate,
                unroll_length=4,
                batch_size=4,
                num_minibatches=1,
                num_updates_per_batch=1,
                seed=config.seed,
                num_evals=1,
                normalize_observations=True,
                reward_scaling=1.0,
            )
        policy = make_inference_fn(params, deterministic=True)

    _inject_failure(config, "after_training")

    train_episode, train_return = rollout_episode(
        environment,
        policy,
        seed=config.seed,
        episode_length=config.episode_length,
        phase="train",
    )
    train_episode = dataclasses.replace(
        train_episode,
        start_env_steps=0,
        end_env_steps=config.num_timesteps,
    )
    run.log_episode_summary(
        env_steps=config.num_timesteps,
        episode_return=train_return,
        episode_length=len(train_episode.actions),
    )

    eval_episode, objective = rollout_episode(
        environment,
        policy,
        seed=config.seed + 1,
        episode_length=config.episode_length,
        phase="eval",
    )
    eval_episode = dataclasses.replace(
        eval_episode,
        start_env_steps=config.num_timesteps,
        end_env_steps=config.num_timesteps,
    )
    run.log_episode(eval_episode)

    checkpoint = run.context.artifact_directory / "ppo-params.npz"
    _write_checkpoint(checkpoint, params)
    run.register_checkpoint(checkpoint)
    _inject_failure(config, "after_checkpoint")

    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX selected no devices")
    return TrainingResult(
        objective=objective,
        checkpoint=checkpoint,
        platform=jax.default_backend(),
        device_kind=devices[0].device_kind,
    )
