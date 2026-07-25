from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from brax import envs
from brax.training import types as brax_types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo_train
from training_sdk import Episode
from training_sdk.reporter import Reporter

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
) -> tuple[Episode, float]:
    """Roll out one complete episode, crossing to NumPy only at the host boundary."""
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)
    infer = jax.jit(policy)

    reset_key, rollout_key = jax.random.split(jax.random.PRNGKey(seed))
    state = reset(reset_key)
    jax.block_until_ready(state)

    observations = [_host_array(state.obs)]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminals: list[bool] = []
    truncations: list[bool] = []

    for transition_index in range(episode_length):
        rollout_key, action_key = jax.random.split(rollout_key)
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

    episode = Episode(
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


def _tree_descriptor(value: Any) -> dict[str, Any]:
    if isinstance(value, running_statistics.RunningStatisticsState):
        return {
            "kind": "running_statistics",
            "mode": int(value.mode),
            "fields": {
                "mean": _tree_descriptor(value.mean),
                "std": _tree_descriptor(value.std),
                "count": _tree_descriptor(value.count),
                "summed_variance": _tree_descriptor(value.summed_variance),
                "std_eps": _tree_descriptor(value.std_eps),
            },
        }
    if isinstance(value, brax_types.UInt64):
        return {
            "kind": "uint64",
            "fields": {
                "hi": _tree_descriptor(value.hi),
                "lo": _tree_descriptor(value.lo),
            },
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "children": [_tree_descriptor(child) for child in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "children": [_tree_descriptor(child) for child in value],
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("checkpoint mappings must use string keys")
        return {
            "kind": "dict",
            "keys": list(value),
            "children": [_tree_descriptor(value[key]) for key in value],
        }
    return {"kind": "leaf"}


def _restore_tree(descriptor: Mapping[str, Any], value: Any) -> Any:
    kind = descriptor.get("kind")
    if kind == "leaf":
        return value
    if kind in {"tuple", "list"}:
        children = descriptor.get("children")
        if not isinstance(children, list) or not isinstance(value, (list, tuple)):
            raise ValueError("invalid checkpoint tree sequence")
        if len(children) != len(value):
            raise ValueError("checkpoint tree sequence length mismatch")
        restored = [
            _restore_tree(child, item)
            for child, item in zip(children, value, strict=True)
        ]
        return tuple(restored) if kind == "tuple" else restored
    if kind == "dict":
        keys = descriptor.get("keys")
        children = descriptor.get("children")
        if (
            not isinstance(keys, list)
            or not all(isinstance(key, str) for key in keys)
            or len(set(keys)) != len(keys)
            or not isinstance(children, list)
            or len(keys) != len(children)
            or not isinstance(value, Mapping)
            or set(value) != set(keys)
        ):
            raise ValueError("invalid checkpoint tree mapping")
        return {
            key: _restore_tree(child, value[key])
            for key, child in zip(keys, children, strict=True)
        }
    if kind in {"running_statistics", "uint64"}:
        fields = descriptor.get("fields")
        if not isinstance(fields, Mapping) or not isinstance(value, Mapping):
            raise ValueError("invalid checkpoint custom node")
        expected = (
            {"mean", "std", "count", "summed_variance", "std_eps"}
            if kind == "running_statistics"
            else {"hi", "lo"}
        )
        if set(fields) != expected or set(value) != expected:
            raise ValueError("invalid checkpoint custom node fields")
        restored_fields = {
            name: _restore_tree(fields[name], value[name]) for name in expected
        }
        if kind == "uint64":
            return brax_types.UInt64(**restored_fields)
        mode = descriptor.get("mode")
        if type(mode) is not int:
            raise ValueError("invalid checkpoint normalization mode")
        return running_statistics.RunningStatisticsState(
            **restored_fields,
            mode=running_statistics.NormalizationMode(mode),
        )
    raise ValueError(f"unsupported checkpoint tree node: {kind!r}")


def _validated_archive_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe checkpoint archive path: {value!r}")
    return relative


def _write_checkpoint(path: Path, params: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = json.dumps(
        _tree_descriptor(params),
        allow_nan=False,
        separators=(",", ":"),
    )
    with tempfile.TemporaryDirectory(prefix=".orbax-", dir=path.parent) as temporary:
        checkpoint_directory = Path(temporary) / "checkpoint"
        checkpointer = ocp.StandardCheckpointer()
        try:
            checkpointer.save(checkpoint_directory, params)
            checkpointer.wait_until_finished()
        finally:
            checkpointer.close()

        files = sorted(item for item in checkpoint_directory.rglob("*") if item.is_file())
        if not files or any(item.is_symlink() for item in files):
            raise ValueError("Orbax checkpoint must contain regular files")
        relative_paths = [
            item.relative_to(checkpoint_directory).as_posix() for item in files
        ]
        arrays: dict[str, np.ndarray] = {
            "format_version": np.asarray(1, dtype=np.int64),
            "relative_paths": np.asarray(relative_paths, dtype=np.str_),
            "tree": np.asarray(descriptor, dtype=np.str_),
        }
        arrays.update(
            {
                f"payload_{index:06d}": np.frombuffer(item.read_bytes(), dtype=np.uint8)
                for index, item in enumerate(files)
            }
        )
        temporary_archive = Path(temporary) / path.name
        with temporary_archive.open("wb") as output:
            np.savez(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_archive, path)


def restore_checkpoint(path: str | Path) -> Any:
    """Restore a Brax PPO PyTree from a controlled single-file Orbax archive."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {"format_version", "relative_paths", "tree"}
        if not required.issubset(archive.files):
            raise ValueError("checkpoint archive is missing required metadata")
        version = archive["format_version"]
        if version.shape != () or version.dtype.kind not in "iu" or int(version) != 1:
            raise ValueError("unsupported checkpoint archive version")
        paths_array = archive["relative_paths"]
        if paths_array.ndim != 1 or paths_array.dtype.kind != "U":
            raise ValueError("invalid checkpoint archive paths")
        relative_paths = [
            _validated_archive_path(str(value)) for value in paths_array.tolist()
        ]
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("duplicate checkpoint archive path")
        expected_keys = {
            "format_version",
            "relative_paths",
            "tree",
            *(f"payload_{index:06d}" for index in range(len(relative_paths))),
        }
        if set(archive.files) != expected_keys:
            raise ValueError("checkpoint archive payload set mismatch")
        tree_array = archive["tree"]
        if tree_array.shape != () or tree_array.dtype.kind != "U":
            raise ValueError("invalid checkpoint tree metadata")
        try:
            descriptor = json.loads(str(tree_array))
        except json.JSONDecodeError as error:
            raise ValueError("invalid checkpoint tree metadata") from error
        if not isinstance(descriptor, Mapping):
            raise ValueError("invalid checkpoint tree metadata")  # noqa: TRY004
        payloads = []
        for index in range(len(relative_paths)):
            payload = archive[f"payload_{index:06d}"]
            if payload.ndim != 1 or payload.dtype != np.dtype(np.uint8):
                raise ValueError("invalid checkpoint archive payload")
            payloads.append(payload.tobytes())

    with tempfile.TemporaryDirectory(prefix=".orbax-restore-") as temporary:
        checkpoint_directory = Path(temporary) / "checkpoint"
        checkpoint_directory.mkdir()
        for relative, payload in zip(relative_paths, payloads, strict=True):
            destination = checkpoint_directory.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        checkpointer = ocp.StandardCheckpointer()
        try:
            generic = checkpointer.restore(checkpoint_directory)
        finally:
            checkpointer.close()
        restored = _restore_tree(descriptor, generic)
        jax.block_until_ready(restored)
        return restored


def _verify_checkpoint_round_trip(expected: Any, checkpoint: Path) -> None:
    restored = restore_checkpoint(checkpoint)
    if jax.tree_util.tree_structure(restored) != jax.tree_util.tree_structure(expected):
        raise ValueError("checkpoint round-trip tree structure mismatch")
    expected_leaves = jax.tree_util.tree_leaves(expected)
    restored_leaves = jax.tree_util.tree_leaves(restored)
    if len(expected_leaves) != len(restored_leaves):
        raise ValueError("checkpoint round-trip leaf count mismatch")
    for index, (expected_leaf, restored_leaf) in enumerate(
        zip(expected_leaves, restored_leaves, strict=True)
    ):
        expected_array = _host_array(expected_leaf)
        restored_array = _host_array(restored_leaf)
        if expected_array.dtype != restored_array.dtype:
            raise ValueError(f"checkpoint round-trip leaf {index} dtype mismatch")
        if expected_array.shape != restored_array.shape:
            raise ValueError(f"checkpoint round-trip leaf {index} shape mismatch")
        if not np.array_equal(expected_array, restored_array):
            raise ValueError(f"checkpoint round-trip leaf {index} value mismatch")


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


def train(config: AcceptanceConfig, reporter: Reporter) -> TrainingResult:
    """Run PPO acceptance and report contract v2 metrics."""
    _inject_failure(config, "before_training")
    environment = envs.get_environment(
        env_name=config.environment_name,
        backend=config.backend,
    )

    if config.fast_mode:
        batch = 4
        env_steps = 0
        while env_steps < config.num_timesteps:
            step_batch = min(batch, config.num_timesteps - env_steps)
            env_steps += step_batch
            _exercise_selected_device(environment, config.seed + env_steps)
            reporter.report(
                env_steps,
                {
                    "episode_return": 0.0,
                    "episode_length": float(step_batch),
                },
            )
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
    reporter.report(
        config.num_timesteps,
        {
            "episode_return": train_return,
            "episode_length": float(len(train_episode.actions)),
        },
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
    reporter.log_episode(eval_episode)
    reporter.report(
        config.num_timesteps,
        {
            "episode_return": objective,
            "episode_length": float(len(eval_episode.actions)),
        },
    )

    checkpoint = reporter.scratch / "ppo-params.npz"
    _write_checkpoint(checkpoint, params)
    try:
        _verify_checkpoint_round_trip(params, checkpoint)
    except BaseException:
        # An archive that fails its own round-trip must not outlive the check.
        # Left on disk it is indistinguishable from a good one, and whatever
        # collects artifacts would publish it as a restorable checkpoint.
        checkpoint.unlink(missing_ok=True)
        raise
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
