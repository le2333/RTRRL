from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
import training_sdk
import yaml
from brax import envs

import brax_ppo_acceptance.__main__ as launcher
import brax_ppo_acceptance.train as train_module
from brax_ppo_acceptance.config import AcceptanceConfig
from brax_ppo_acceptance.train import rollout_episode, train

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


class RecordingAim:
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        self.started = 0
        self.events: list[training_sdk.MetricEvent] = []
        self.failures: list[dict[str, str]] = []
        self.closed = 0
        self.close_error = close_error

    def start(self, context: training_sdk.RunContext) -> None:
        del context
        self.started += 1

    def send(self, event: training_sdk.MetricEvent) -> None:
        self.events.append(event)

    def fail(self, metadata: dict[str, str]) -> None:
        self.failures.append(metadata)

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


class RecordingRerun:
    def __init__(self) -> None:
        self.episodes: list[training_sdk.Episode] = []
        self.closed = 0

    def log_episode(self, episode: training_sdk.Episode) -> Path | None:
        self.episodes.append(episode)
        return None

    def close(self) -> None:
        self.closed += 1


class CountingRun(training_sdk.TrainingRun):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_calls: list[BaseException] = []
        self.finish_calls = 0

    def fail(self, error: BaseException) -> None:
        self.fail_calls.append(error)
        super().fail(error)

    def finish(self, final_metrics: dict[str, int | float]) -> None:
        self.finish_calls += 1
        super().finish(final_metrics)


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
    environ = {"BRAX_ACCEPTANCE_TEST_MODE": "1"}
    if fast:
        environ["BRAX_ACCEPTANCE_E2E_FAST"] = "1"
    return path, environ


def make_run(
    tmp_path: Path,
    *,
    aim: RecordingAim | None = None,
) -> tuple[CountingRun, RecordingAim, RecordingRerun, training_sdk.MemorySpool]:
    context = training_sdk.RunContext(
        experiment_name="brax-acceptance",
        experiment_id="experiment-1",
        group="cpu",
        script="brax_ppo_acceptance",
        run_id="run-1",
        run_number=1,
        trial_number=1,
        seed=7,
        metadata={},
        environment={"name": "inverted_pendulum", "options": {"backend": "generalized"}},
        training_budget={"env_steps": 128},
        fixed_parameters={},
        sampled_parameters={},
        final_parameters={},
        image_digest="sha256:test",
        resource_profile="cpu",
        artifact_directory=tmp_path / "artifacts",
        logging={"aim_every_env_steps": 1, "rerun_every_episodes": 1},
        objective={"metric": "eval/episode_return"},
    )
    recording_aim = aim or RecordingAim()
    rerun = RecordingRerun()
    spool = training_sdk.MemorySpool()
    run = CountingRun(context, recording_aim, rerun, spool)
    return run, recording_aim, rerun, spool


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


def test_fast_train_retains_rollout_sdk_and_checkpoint_lifecycle(tmp_path: Path) -> None:
    config_path, environ = write_config(tmp_path)
    config = AcceptanceConfig.load(config_path, environ=environ)
    run, _, rerun, spool = make_run(tmp_path)

    result = train(config, run)
    run.finish(
        {
            "eval/episode_return": result.objective,
            "runtime/device_count": len(jax.devices()),
        }
    )

    assert math.isfinite(result.objective)
    assert result.platform == "cpu"
    assert result.checkpoint.name == "ppo-params.npz"
    assert result.checkpoint.exists()
    copied = run.context.artifact_directory / "checkpoints" / result.checkpoint.name
    assert copied.is_file()
    assert copied.read_bytes() == result.checkpoint.read_bytes()
    assert [event.stream for event in spool.events].count("episode_summary") >= 1
    finals = [event for event in spool.events if event.kind == "final"]
    assert finals[-1].data["objective_metric"] == "eval/episode_return"
    assert finals[-1].data["finalized"] is True
    assert len(rerun.episodes) == 1
    episode = rerun.episodes[0]
    assert episode.number == 2
    assert episode.phase == "eval"
    assert episode.start_env_steps == config.num_timesteps
    assert episode.end_env_steps == config.num_timesteps
    assert len(episode.observations) == len(episode.actions) + 1
    assert episode.terminals[-1] or episode.truncations[-1]


def test_normal_path_calls_real_brax_entry_point_with_fixed_micro_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, _ = write_config(tmp_path, fast=False)
    config = AcceptanceConfig.load(config_path, environ={})
    run, _, rerun, _ = make_run(tmp_path)
    observed: dict[str, Any] = {}

    def recording_ppo_train(**kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
        observed.update(kwargs)

        def make_policy(params: Any, *, deterministic: bool = False) -> Any:
            del params
            assert deterministic
            return zero_policy

        return make_policy, (jnp.array([1.0]),), {"eval/episode_reward": jnp.array(0.0)}

    monkeypatch.setattr(train_module.ppo_train, "train", recording_ppo_train)

    result = train(config, run)

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
    config_path, environ = write_config(tmp_path, failure_mode=failure_mode)
    monkeypatch.setenv("BRAX_ACCEPTANCE_TEST_MODE", environ["BRAX_ACCEPTANCE_TEST_MODE"])
    monkeypatch.setenv("BRAX_ACCEPTANCE_E2E_FAST", environ["BRAX_ACCEPTANCE_E2E_FAST"])
    run, aim, rerun, spool = make_run(tmp_path)
    monkeypatch.setattr(launcher, "bootstrap_from_environment", lambda: run)

    with pytest.raises(RuntimeError, match=f"injected failure: {failure_mode}") as raised:
        launcher.main(["--config", str(config_path)])

    assert run.fail_calls == [raised.value]
    assert len(aim.failures) == 1
    assert not any(
        event.kind == "final" and event.data["finalized"] for event in spool.events
    )
    checkpoint = run.context.artifact_directory / "checkpoints" / "ppo-params.npz"
    assert checkpoint.exists() is checkpoint_expected
    assert bool(rerun.episodes) is rerun_expected


def test_finalization_failure_is_rethrown_without_double_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, environ = write_config(tmp_path)
    monkeypatch.setenv("BRAX_ACCEPTANCE_TEST_MODE", environ["BRAX_ACCEPTANCE_TEST_MODE"])
    monkeypatch.setenv("BRAX_ACCEPTANCE_E2E_FAST", environ["BRAX_ACCEPTANCE_E2E_FAST"])
    close_error = KeyboardInterrupt("close interrupted")
    run, aim, _, spool = make_run(tmp_path, aim=RecordingAim(close_error=close_error))
    monkeypatch.setattr(launcher, "bootstrap_from_environment", lambda: run)

    with pytest.raises(KeyboardInterrupt, match="close interrupted") as raised:
        launcher.main(["--config", str(config_path)])

    assert run.finish_calls == 1
    assert run.fail_calls == [raised.value]
    assert aim.failures == []
    finalized = [
        event
        for event in spool.events
        if event.kind == "final" and event.data["finalized"]
    ]
    assert len(finalized) == 1


def test_successful_launcher_finishes_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, environ = write_config(tmp_path)
    monkeypatch.setenv("BRAX_ACCEPTANCE_TEST_MODE", environ["BRAX_ACCEPTANCE_TEST_MODE"])
    monkeypatch.setenv("BRAX_ACCEPTANCE_E2E_FAST", environ["BRAX_ACCEPTANCE_E2E_FAST"])
    run, aim, _, spool = make_run(tmp_path)
    checkpoint = tmp_path / "ppo-params.npz"
    checkpoint.write_bytes(b"test")
    result = train_module.TrainingResult(
        objective=1.25,
        checkpoint=checkpoint,
        platform="cpu",
        device_kind="cpu",
    )
    monkeypatch.setattr(launcher, "bootstrap_from_environment", lambda: run)
    monkeypatch.setattr(launcher, "train", lambda config, sdk_run: result)

    assert launcher.main(["--config", str(config_path)]) == 0

    assert run.finish_calls == 1
    assert run.fail_calls == []
    assert aim.failures == []
    finalized = [
        event
        for event in spool.events
        if event.kind == "final" and event.data["finalized"]
    ]
    assert len(finalized) == 1


def test_cli_requires_sdk_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path, environ = write_config(tmp_path)
    monkeypatch.setenv("BRAX_ACCEPTANCE_TEST_MODE", environ["BRAX_ACCEPTANCE_TEST_MODE"])
    monkeypatch.setenv("BRAX_ACCEPTANCE_E2E_FAST", environ["BRAX_ACCEPTANCE_E2E_FAST"])
    monkeypatch.setattr(launcher, "bootstrap_from_environment", lambda: None)

    with pytest.raises(RuntimeError, match="TRAINER_RUN_CONTEXT_PATH is required"):
        launcher.main(["--config", str(config_path)])
