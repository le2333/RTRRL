import json
import threading
import time
from pathlib import Path

import pytest

import training_sdk.bootstrap as bootstrap_module
from training_sdk import (
    MemorySpool,
    RunContext,
    TrainingRun,
    bootstrap_from_environment,
    maybe_current_run,
    set_current_run,
)


def write_context(tmp_path: Path, name: str = "run-context.json") -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "experiment_name": "experiment",
                "experiment_id": "exp-123",
                "group": "group",
                "script": "memo_stream_ac",
                "run_id": f"run-{name}",
                "run_number": 1,
                "trial_number": 2,
                "seed": 3,
                "metadata": {},
                "environment": {"name": "memory_chain", "options": {}},
                "training_budget": {"env_steps": 100},
                "fixed_parameters": {},
                "sampled_parameters": {},
                "final_parameters": {},
                "image_digest": "sha256:abc",
                "resource_profile": "cpu-small",
                "artifact_directory": str(tmp_path / "artifacts"),
                "logging": {
                    "aim_every_env_steps": 10,
                    "rerun_every_episodes": 7,
                },
                "objective": {"metric": "eval/reward"},
            }
        )
    )
    return path


class FakeAim:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.started = []

    def start(self, context):
        self.started.append(context)
        if self.fail:
            raise RuntimeError("start failed")

    def send(self, event):
        del event


class FakeRerun:
    def log_episode(self, episode):
        del episode
        return None


@pytest.fixture(autouse=True)
def clear_current_run():
    set_current_run(None)
    yield
    set_current_run(None)


def factories(*, aim=None):
    aim = aim or FakeAim()
    rerun = FakeRerun()
    return (
        aim,
        rerun,
        lambda context, environ: aim,
        lambda context: rerun,
    )


def test_missing_context_is_local_noop():
    assert bootstrap_from_environment({}) is None
    assert maybe_current_run() is None


def test_present_invalid_context_is_a_hard_error(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{")

    with pytest.raises(json.JSONDecodeError):
        bootstrap_from_environment(
            {"TRAINER_RUN_CONTEXT_PATH": str(path)},
            aim_factory=lambda context, environ: FakeAim(),
            rerun_factory=lambda context: FakeRerun(),
        )

    assert maybe_current_run() is None


def test_bootstrap_is_idempotent_for_one_resolved_context(tmp_path):
    path = write_context(tmp_path)
    _, _, aim_factory, rerun_factory = factories()
    alias = path.parent / "." / path.name

    first = bootstrap_from_environment(
        {"TRAINER_RUN_CONTEXT_PATH": str(path)},
        aim_factory=aim_factory,
        rerun_factory=rerun_factory,
    )
    second = bootstrap_from_environment(
        {"TRAINER_RUN_CONTEXT_PATH": str(alias)},
        aim_factory=aim_factory,
        rerun_factory=rerun_factory,
    )

    assert second is first
    assert first.context_path == path.resolve()


def test_bootstrap_rejects_conflicting_contexts(tmp_path):
    first_path = write_context(tmp_path, "first.json")
    second_path = write_context(tmp_path, "second.json")
    _, _, aim_factory, rerun_factory = factories()
    bootstrap_from_environment(
        {"TRAINER_RUN_CONTEXT_PATH": str(first_path)},
        aim_factory=aim_factory,
        rerun_factory=rerun_factory,
    )

    with pytest.raises(RuntimeError, match="different run context"):
        bootstrap_from_environment(
            {"TRAINER_RUN_CONTEXT_PATH": str(second_path)},
            aim_factory=aim_factory,
            rerun_factory=rerun_factory,
        )


def test_bootstrap_starts_before_installing_current_run(tmp_path):
    path = write_context(tmp_path)
    aim = FakeAim(fail=True)

    with pytest.raises(RuntimeError, match="start failed"):
        bootstrap_from_environment(
            {"TRAINER_RUN_CONTEXT_PATH": str(path)},
            aim_factory=lambda context, environ: aim,
            rerun_factory=lambda context: FakeRerun(),
        )

    assert maybe_current_run() is None


def test_concurrent_bootstrap_creates_only_one_run(tmp_path):
    path = write_context(tmp_path)
    barrier = threading.Barrier(8)
    factory_calls = []
    results = []
    errors = []

    def aim_factory(context, environ):
        factory_calls.append((context, environ))
        time.sleep(0.05)
        return FakeAim()

    def call_bootstrap():
        barrier.wait()
        try:
            results.append(
                bootstrap_from_environment(
                    {"TRAINER_RUN_CONTEXT_PATH": str(path)},
                    aim_factory=aim_factory,
                    rerun_factory=lambda context: FakeRerun(),
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=call_bootstrap) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(factory_calls) == 1
    assert len(results) == 8
    assert all(run is results[0] for run in results)


def test_default_factories_use_context_and_facility_environment(tmp_path, monkeypatch):
    path = write_context(tmp_path)
    calls = {}
    aim = FakeAim()
    rerun = FakeRerun()

    def make_aim(**kwargs):
        calls["aim"] = kwargs
        return aim

    def make_rerun(context, **kwargs):
        calls["rerun"] = (context, kwargs)
        return rerun

    monkeypatch.setattr(bootstrap_module, "AimAdapter", make_aim)
    monkeypatch.setattr(bootstrap_module, "RerunAdapter", make_rerun)
    run = bootstrap_from_environment(
        {
            "TRAINER_RUN_CONTEXT_PATH": str(path),
            "AIM_REPO": "aim://facility",
        }
    )

    assert calls["aim"] == {"repo": "aim://facility"}
    assert calls["rerun"] == (
        run.context,
        {
            "every_episodes": 7,
            "root": run.context.artifact_directory,
        },
    )
    assert aim.started == [run.context]


def test_training_run_context_path_is_normalized_and_immutable(tmp_path):
    context = RunContext.from_path(write_context(tmp_path))
    run = TrainingRun(
        context,
        FakeAim(),
        FakeRerun(),
        MemorySpool(),
        context_path=tmp_path / "nested" / ".." / "run-context.json",
    )

    assert run.context_path == (tmp_path / "run-context.json").resolve()
    with pytest.raises(AttributeError):
        run.context_path = tmp_path / "other.json"


def test_manual_training_run_construction_remains_compatible(tmp_path):
    context = RunContext.from_path(write_context(tmp_path))

    run = TrainingRun(context, FakeAim(), FakeRerun(), MemorySpool())

    assert run.context_path is None
