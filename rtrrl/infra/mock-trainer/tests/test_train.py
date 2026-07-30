from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from brax import envs
from brax.envs.base import State
from training_sdk.contract import RunConfig
from training_sdk.episode import Episode
from training_sdk.reporter import METRICS_FILENAME, Reporter

import brax_ppo_acceptance.__main__ as launcher
import brax_ppo_acceptance.train as train_module
from brax_ppo_acceptance.config import AcceptanceConfig
from brax_ppo_acceptance.train import restore_checkpoint, rollout_episode, train

VALID: dict[str, Any] = {
    "protocol_version": "1",
    "environment": {
        "name": "inverted_pendulum",
        "options": {"backend": "generalized"},
    },
    "logging": {"aim_every_env_steps": 1, "rerun_every_episodes": 1},
    "parameters": {
        "runtime": {"seed": 7},
        "algorithm": {
            "learning_rate": 0.0003,
            "num_envs": 4,
            "episode_length": 32,
            "failure_mode": "none",
        },
    },
    "training_budget": {"env_steps": 128},
}


def default_params(**overrides: Any) -> dict[str, Any]:
    params = {
        "env": "inverted_pendulum",
        "backend": "generalized",
        "total_steps": 128,
        "seed": 7,
        "learning_rate": 0.0003,
        "num_envs": 4,
        "episode_length": 32,
        "failure_mode": "none",
    }
    params.update(overrides)
    return params


def test_environ(*, fast: bool = True) -> dict[str, str]:
    environ = {"BRAX_ACCEPTANCE_TEST_MODE": "1"}
    if fast:
        environ["BRAX_ACCEPTANCE_E2E_FAST"] = "1"
    return environ


class RecordingRerun:
    def __init__(self) -> None:
        self.episodes: list[Episode] = []
        self.closed = 0

    def report(self, step: int, metrics: dict[str, float]) -> None:
        del step, metrics

    def log_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)

    def close(self) -> None:
        self.closed += 1


class FailingCloseSink(RecordingRerun):
    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def close(self) -> None:
        raise RuntimeError(self._message)


def make_run_config(
    tmp_path: Path,
    params: dict[str, Any],
    *,
    include_rerun: bool = True,
) -> RunConfig:
    trial_prefix = f"s3://bucket/trials/t{params.get('trial', 0)}"
    return RunConfig.model_validate(
        {
            "contract": 3,
            "run_id": "run-1",
            "experiment": "brax-acceptance",
            "name": "acceptance",
            "launch_id": "20260725-000000",
            "trial": 0,
            "entry": "brax_ppo_acceptance",
            "digest": "registry.example/trainer@sha256:" + "a" * 64,
            "environment": {
                "id": "brax::inverted_pendulum",
                "backend": "generalized",
                "num_envs": 4,
            },
            "budget": {
                "total_steps": int(params["total_steps"]),
                "epoch_steps": int(params["total_steps"]),
                "eval_steps": 0,
            },
            "source_hash": "sha256:test",
            "params": params,
            "logging": {
                "aim": str(tmp_path / "aim"),
                "every_steps": 1,
                "rerun_s3": f"{trial_prefix}/episodes/" if include_rerun else None,
                "rerun_every_episodes": 1 if include_rerun else None,
            },
            "score": {
                "metric": "episode_return",
                "window_steps": [0, int(params["total_steps"])],
                "reduce": "mean",
                "direction": "maximize",
                "non_finite": "worst",
                "s3": f"{trial_prefix}/score.json",
            },
        }
    )


def make_reporter(
    tmp_path: Path,
    *,
    failure_mode: str = "none",
    fast: bool = True,
    rerun: RecordingRerun | None = None,
    extra_sinks: list[Any] | None = None,
) -> tuple[Reporter, AcceptanceConfig, RecordingRerun, Path]:
    params = default_params(failure_mode=failure_mode)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    rerun_sink = rerun or RecordingRerun()
    config = make_run_config(tmp_path, params)
    reporter = Reporter(config, scratch, sinks=[rerun_sink, *(extra_sinks or [])])
    acceptance = AcceptanceConfig.from_params(params, environ=test_environ(fast=fast))
    return reporter, acceptance, rerun_sink, scratch


def write_run_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure_mode: str = "none",
    fast: bool = True,
    params_overrides: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    params = default_params(failure_mode=failure_mode, **(params_overrides or {}))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    config_path = tmp_path / "run-config.json"
    config_path.write_text(
        json.dumps(make_run_config(tmp_path, params).model_dump(mode="json")),
        encoding="utf-8",
    )
    environ = test_environ(fast=fast)
    monkeypatch.setenv("TRAINER_RUN_CONFIG", str(config_path))
    monkeypatch.setenv("TRAINER_SCRATCH", str(scratch))
    for key, value in environ.items():
        monkeypatch.setenv(key, value)
    return config_path, scratch


def write_config(
    tmp_path: Path,
    *,
    failure_mode: str = "none",
    fast: bool = True,
) -> tuple[Path, dict[str, str]]:
    payload = copy.deepcopy(VALID)
    payload["parameters"]["algorithm"]["failure_mode"] = failure_mode
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path, test_environ(fast=fast)


def read_metric_steps(scratch: Path) -> list[int]:
    metrics_path = scratch / METRICS_FILENAME
    return [
        json.loads(line)["step"]
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def zero_policy(observation: jax.Array, key: jax.Array) -> tuple[jax.Array, dict[str, Any]]:
    del key
    return jnp.zeros(observation.shape[:-1] + (1,)), {}


def test_rollout_uses_real_jitted_environment_and_builds_complete_episode() -> None:
    environment = envs.get_environment(
        env_name="inverted_pendulum",
        backend="generalized",
    )

    episode, episode_return = rollout_episode(
        environment,
        zero_policy,
        seed=11,
        episode_length=4,
        phase="eval",
    )

    assert episode.number == 2
    assert episode.phase == "eval"
    assert len(episode.observations) == len(episode.actions) + 1
    assert len(episode.actions) == 4
    assert episode.terminals[-1] or episode.truncations[-1]
    assert episode.truncations[-1]
    assert math.isfinite(episode_return)


def test_rollout_splits_reset_and_action_random_streams() -> None:
    class KeyEnvironment:
        action_size = 2

        def reset(self, key: jax.Array) -> State:
            return State(
                pipeline_state=None,
                obs=key,
                reward=jnp.array(0.0),
                done=jnp.array(0.0),
                metrics={},
                info={},
            )

        def step(self, state: State, action: jax.Array) -> State:
            return state.replace(
                obs=state.obs + 1,
                reward=jnp.sum(action.astype(jnp.float32)),
            )

    def key_policy(
        observation: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, dict[str, Any]]:
        del observation
        return key, {}

    seed = 23
    root_key = jax.random.PRNGKey(seed)
    expected_reset_key, rollout_key = jax.random.split(root_key)
    _, expected_action_key = jax.random.split(rollout_key)

    first, _ = rollout_episode(KeyEnvironment(), key_policy, seed, 1, "eval")
    repeated, _ = rollout_episode(KeyEnvironment(), key_policy, seed, 1, "eval")
    different, _ = rollout_episode(KeyEnvironment(), key_policy, seed + 1, 1, "eval")

    np.testing.assert_array_equal(first.observations[0], expected_reset_key)
    np.testing.assert_array_equal(first.actions[0], expected_action_key)
    assert not np.array_equal(first.observations[0], first.actions[0])
    np.testing.assert_array_equal(first.observations[0], repeated.observations[0])
    np.testing.assert_array_equal(first.actions[0], repeated.actions[0])
    assert not np.array_equal(first.actions[0], different.actions[0])


def test_fast_train_retains_rollout_sdk_and_checkpoint_lifecycle(tmp_path: Path) -> None:
    reporter, config, rerun, scratch = make_reporter(tmp_path)

    with reporter:
        result = train(config, reporter)

    assert math.isfinite(result.objective)
    assert result.platform == "cpu"
    assert result.checkpoint.name == "ppo-params.npz"
    assert result.checkpoint.exists()
    assert result.checkpoint == scratch / "ppo-params.npz"
    assert read_metric_steps(scratch), "trainer must write metric steps"
    assert max(read_metric_steps(scratch)) == config.num_timesteps
    assert len(rerun.episodes) == 1
    episode = rerun.episodes[0]
    assert episode.number == 2
    assert episode.phase == "eval"
    assert episode.start_env_steps == config.num_timesteps
    assert episode.end_env_steps == config.num_timesteps
    assert len(episode.observations) == len(episode.actions) + 1
    assert episode.terminals[-1] or episode.truncations[-1]
    restored = restore_checkpoint(result.checkpoint)
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(
        (jnp.zeros((1,)),)
    )
    np.testing.assert_array_equal(restored[0], jnp.zeros((1,)))
    with np.load(result.checkpoint, allow_pickle=False) as archive:
        assert archive.files
        assert all(archive[name].dtype != np.dtype("O") for name in archive.files)


def test_restore_checkpoint_rejects_archive_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.npz"
    np.savez(
        archive,
        format_version=np.asarray(1, dtype=np.int64),
        relative_paths=np.asarray(["../escape"], dtype=np.str_),
        tree=np.asarray('{"kind":"tuple","children":[]}', dtype=np.str_),
        payload_000000=np.frombuffer(b"malicious", dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="unsafe checkpoint archive path"):
        restore_checkpoint(archive)
    assert not (tmp_path / "escape").exists()


def test_restore_checkpoint_normalizes_missing_metadata_error(tmp_path: Path) -> None:
    archive = tmp_path / "missing-paths.npz"
    np.savez(
        archive,
        format_version=np.asarray(1, dtype=np.int64),
        tree=np.asarray('{"kind":"tuple","children":[]}', dtype=np.str_),
        unexpected=np.asarray(1, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="missing required metadata"):
        restore_checkpoint(archive)


def test_train_rejects_checkpoint_mismatch_before_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reporter, config, _, scratch = make_reporter(tmp_path)
    monkeypatch.setattr(
        train_module,
        "restore_checkpoint",
        lambda path: (jnp.ones((2,), dtype=jnp.float32),),
    )

    with reporter, pytest.raises(ValueError, match="checkpoint round-trip"):
        train(config, reporter)

    assert not (scratch / "ppo-params.npz").exists()
    # Training reported progress long before the checkpoint was written, so the
    # metrics stay as the record of how far the run got. Only the archive that
    # failed verification is withdrawn.
    assert read_metric_steps(scratch)


def test_normal_path_calls_real_brax_entry_point_with_fixed_micro_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, environ = write_config(tmp_path, fast=False)
    config = AcceptanceConfig.load(config_path, environ=environ)
    reporter, _, rerun, _ = make_reporter(tmp_path, fast=False)
    observed: dict[str, Any] = {}

    def recording_ppo_train(**kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
        observed.update(kwargs)

        def make_policy(params: Any, *, deterministic: bool = False) -> Any:
            del params
            assert deterministic
            return zero_policy

        return make_policy, (jnp.array([1.0]),), {"eval/episode_reward": jnp.array(0.0)}

    monkeypatch.setattr(train_module.ppo_train, "train", recording_ppo_train)

    with reporter:
        result = train(config, reporter)

    assert math.isfinite(result.objective)
    assert len(rerun.episodes) == 1
    assert observed["environment"].__class__.__module__.startswith("brax.envs")
    assert observed["num_timesteps"] == 128
    assert observed["episode_length"] == 32
    assert observed["num_envs"] == 4
    assert observed["learning_rate"] == pytest.approx(0.0003)
    assert observed["unroll_length"] == 4
    assert observed["batch_size"] == 4
    assert observed["num_minibatches"] == 1
    assert observed["num_updates_per_batch"] == 1
    assert observed["seed"] == 7
    assert observed["num_evals"] == 1
    assert observed["normalize_observations"] is True
    assert observed["reward_scaling"] == 1.0
    assert "device_put_replicated" not in jax.__dict__


def test_launcher_device_contract_is_mandatory_and_profile_bound() -> None:
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    descriptor = (
        Path(launcher.__file__).parents[2] / "scripts" / "brax_ppo_acceptance.yaml"
    ).read_text(encoding="utf-8")

    assert 'jax.default_backend() == "gpu"' in source
    assert 'any("NVIDIA L4" in device.device_kind for device in devices)' in source
    assert "jax.jit(lambda x: x @ x)(jnp.eye(64)).block_until_ready()" in source
    assert 'jax.default_backend() == "cpu"' in source
    assert "device_check" not in descriptor
    assert "skip" not in descriptor


def test_logical_gpu_fast_mode_requires_both_test_gates_to_skip_only_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Result:
        def block_until_ready(self) -> None:
            calls.append("matmul")

    monkeypatch.setattr(launcher.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(
        launcher.jax,
        "jit",
        lambda function: lambda value: Result(),
    )
    monkeypatch.setattr(launcher.jnp, "eye", lambda _size: object())

    launcher._verify_device_contract(
        "g6x",
        environ={
            "BRAX_ACCEPTANCE_TEST_MODE": "1",
            "BRAX_ACCEPTANCE_E2E_FAST": "1",
        },
    )
    assert calls == ["matmul"]

    with pytest.raises(AssertionError, match="gpu"):
        launcher._verify_device_contract(
            "g6x",
            environ={"BRAX_ACCEPTANCE_TEST_MODE": "1"},
        )


@pytest.mark.parametrize(
    ("failure_mode", "checkpoint_expected", "rerun_expected"),
    [
        ("before_training", False, False),
        ("after_training", False, False),
        ("after_checkpoint", True, True),
    ],
)
def test_launcher_fails_once_and_preserves_pre_failure_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_mode: str,
    checkpoint_expected: bool,
    rerun_expected: bool,
) -> None:
    _, scratch = write_run_env(tmp_path, monkeypatch, failure_mode=failure_mode)

    with pytest.raises(RuntimeError, match=f"injected failure: {failure_mode}"):
        launcher.main([])

    checkpoint = scratch / "ppo-params.npz"
    assert checkpoint.exists() is checkpoint_expected
    metrics_path = scratch / METRICS_FILENAME
    if failure_mode == "before_training":
        assert not metrics_path.exists()
    else:
        assert metrics_path.exists()


def test_finalization_failure_is_rethrown_without_double_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_run_env(tmp_path, monkeypatch)
    scratch = tmp_path / "scratch"
    config = RunConfig.model_validate(
        json.loads((tmp_path / "run-config.json").read_text(encoding="utf-8"))
    )
    failing = FailingCloseSink("close interrupted")
    reporter = Reporter(config, scratch, sinks=[failing])

    def fake_from_env() -> Reporter:
        return reporter

    monkeypatch.setattr(launcher.Reporter, "from_env", classmethod(lambda cls: reporter))

    with pytest.raises(RuntimeError, match="close interrupted"):
        launcher.main([])

    assert read_metric_steps(scratch)


def test_successful_launcher_uses_reporter_from_env_and_prints_runtime_summary(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, scratch = write_run_env(tmp_path, monkeypatch)

    assert launcher.main([]) == 0

    output = capsys.readouterr().out
    assert '"platform": "cpu"' in output
    assert '"device_platforms": ["cpu"]' in output
    assert max(read_metric_steps(scratch)) == 128


def test_cli_requires_sdk_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TRAINER_RUN_CONFIG", raising=False)
    monkeypatch.delenv("TRAINER_SCRATCH", raising=False)

    with pytest.raises(KeyError, match="TRAINER_RUN_CONFIG"):
        launcher.main([])


def test_reported_step_matches_the_environment_step_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The score window is expressed in total_steps units, so the reported step
    must be the environment step count, not an iteration index."""
    from aim import Repo

    aim_repo = tmp_path / "aim"
    Repo.from_path(str(aim_repo), init=True)
    write_run_env(
        tmp_path,
        monkeypatch,
        params_overrides={"total_steps": 8},
    )
    monkeypatch.setenv("TRAINER_RUN_CONFIG", str(tmp_path / "run-config.json"))
    config_path = tmp_path / "run-config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config_data["logging"]["aim"] = str(aim_repo)
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    assert launcher.main([]) == 0

    scratch = tmp_path / "scratch"
    steps = read_metric_steps(scratch)
    assert steps, "trainer must report at least one metric step"
    assert max(steps) == 8
