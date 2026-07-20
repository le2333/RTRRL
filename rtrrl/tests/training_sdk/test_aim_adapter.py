from pathlib import Path

import pytest

from training_sdk import (
    AimAdapter,
    AimUnavailable,
    MemorySpool,
    MetricEvent,
    NullRerun,
    RunContext,
    TrainingRun,
)


def make_context(tmp_path, **overrides):
    values = {
        "experiment_name": "user-experiment",
        "experiment_id": "exp-1",
        "group": "dual",
        "script": "train.py",
        "run_id": "run-1",
        "run_number": 7,
        "trial_number": 2,
        "seed": 3,
        "metadata": {"algorithm": "rtrrl", "variant": "base"},
        "environment": {"name": "hopper"},
        "training_budget": {"env_steps": 1_000},
        "fixed_parameters": {"optimizer": {"name": "adam"}},
        "sampled_parameters": {"learning_rate": 0.001},
        "final_parameters": {"learning_rate": 0.001},
        "image_digest": "sha256:abc",
        "resource_profile": "cpu-small",
        "artifact_directory": Path(tmp_path) / "artifacts",
        "logging": {"aim_every_env_steps": 10},
        "objective": {"metric": "eval/reward"},
    }
    values.update(overrides)
    return RunContext(**values)


class FakeAim:
    def __init__(self, failures=()):
        self.failures = list(failures)
        self.experiment = None
        self.name = None
        self.hparams = None
        self.events = []
        self.event_ids = []
        self.metric_names = []

    def start(self, context):
        self.experiment = context.experiment_name
        self.name = context.run_name
        self.hparams = context.hparams

    def send(self, event):
        if self.failures:
            raise self.failures.pop(0)
        if event.event_id in self.event_ids:
            return
        self.event_ids.append(event.event_id)
        self.events.append(event)
        self.metric_names.extend(event.metrics)


def make_training_run(tmp_path, *, context=None, aim=None, spool=None):
    context = context or make_context(tmp_path)
    aim = aim or FakeAim()
    run = TrainingRun(
        context,
        aim=aim,
        rerun=NullRerun(),
        spool=spool or MemorySpool(),
    )
    run.start()
    return run


def test_start_uses_exact_experiment_run_name_and_nested_hparams(tmp_path):
    context = make_context(tmp_path)
    aim = FakeAim()
    run = make_training_run(tmp_path, context=context, aim=aim)

    assert run.aim is aim
    assert aim.experiment == "user-experiment"
    assert aim.name == "dual-0007"
    assert aim.hparams["identity"]["group"] == "dual"
    assert aim.hparams["parameters"]["fixed"]["optimizer"]["name"] == "adam"


def test_general_metrics_use_native_env_step_throttle(tmp_path):
    run = make_training_run(tmp_path)

    run.log_metrics(100, {"train/loss": 3.0})
    run.log_metrics(109, {"train/loss": 2.0})
    run.log_metrics(110, {"train/loss": 1.0})

    assert [event.env_steps for event in run.aim.events] == [100, 110]


def test_throttle_does_not_hide_invalid_metrics_or_advance_steps(tmp_path):
    run = make_training_run(tmp_path)
    run.log_metrics(100, {"train/loss": 3.0})

    with pytest.raises(ValueError, match="metric"):
        run.log_metrics(105, {"train/loss": float("nan")})

    run.log_episode_summary(
        env_steps=101, episode_return=2.5, episode_length=1
    )


def test_episode_summary_is_never_throttled_and_has_mandatory_metrics(tmp_path):
    run = make_training_run(tmp_path)

    run.log_episode_summary(
        env_steps=10, episode_return=2.5, episode_length=10
    )
    run.log_episode_summary(
        env_steps=11, episode_return=3.5, episode_length=1
    )

    assert run.aim.metric_names == [
        "train/episode_return",
        "train/episode_length",
        "train/env_steps",
    ] * 2
    assert run.aim.events[0].metrics == {
        "train/episode_return": 2.5,
        "train/episode_length": 10,
        "train/env_steps": 10,
    }


def test_env_steps_may_repeat_but_must_not_decrease(tmp_path):
    run = make_training_run(tmp_path)
    run.log_metrics(10, {"train/loss": 1.0})
    run.log_episode_summary(
        env_steps=10, episode_return=2.5, episode_length=10
    )

    with pytest.raises(ValueError, match="monotonic"):
        run.log_metrics(9, {"train/loss": 2.0})


def test_emit_appends_before_send(tmp_path):
    operations = []

    class RecordingSpool(MemorySpool):
        def append(self, event):
            operations.append(("append", event.event_id))
            super().append(event)

    class RecordingAim(FakeAim):
        def send(self, event):
            operations.append(("send", event.event_id))
            super().send(event)

    run = make_training_run(
        tmp_path, aim=RecordingAim(), spool=RecordingSpool()
    )
    run.log_metrics(10, {"train/loss": 1.0})

    assert [operation for operation, _ in operations] == ["append", "send"]


def test_only_aim_unavailable_is_suppressed(tmp_path):
    unavailable = FakeAim(failures=[AimUnavailable("down")])
    run = make_training_run(tmp_path, aim=unavailable)
    run.log_metrics(10, {"train/loss": 1.0})
    assert len(run.spool.unsent_events) == 1

    broken = FakeAim(failures=[RuntimeError("algorithm bug")])
    run = make_training_run(tmp_path, aim=broken)
    with pytest.raises(RuntimeError, match="algorithm bug"):
        run.log_metrics(10, {"train/loss": 1.0})


@pytest.mark.parametrize(
    "final_metrics",
    [
        {},
        {"other": 1.0},
        {"eval/reward": float("nan")},
        {"eval/reward": 1.0, "other": float("inf")},
        {"eval/reward": True},
    ],
)
def test_finish_rejects_missing_or_non_finite_required_values(
    tmp_path, final_metrics
):
    run = make_training_run(tmp_path)

    with pytest.raises((TypeError, ValueError), match="final|objective"):
        run.finish(final_metrics)

    assert run.aim.events == []


def test_finish_emits_descriptor_objective_and_finalized_marker(tmp_path):
    run = make_training_run(tmp_path)

    run.finish({"eval/reward": 4.0, "eval/length": 20.0})

    event = run.aim.events[-1]
    assert event.kind == "final"
    assert event.metrics == {
        "eval/reward": 4.0,
        "eval/length": 20.0,
    }
    assert event.data["objective_metric"] == "eval/reward"
    assert event.data["finalized"] is True


def test_finish_without_descriptor_objective_is_a_noop(tmp_path):
    context = make_context(tmp_path, objective={})
    run = make_training_run(tmp_path, context=context)

    run.finish({"eval/reward": 4.0})

    assert run.aim.events == []


class FakeAimRun:
    def __init__(self, *, experiment):
        self.experiment = experiment
        self.name = None
        self.values = {}
        self.tracked = []

    def __setitem__(self, key, value):
        self.values[key] = value

    def get(self, key, default=None):
        return self.values.get(key, default)

    def track(self, value, *, name, step):
        self.tracked.append((name, value, step))


def test_real_aim_adapter_maps_start_and_is_idempotent_by_event_id(tmp_path):
    created = []

    def factory(**kwargs):
        run = FakeAimRun(**kwargs)
        created.append(run)
        return run

    context = make_context(tmp_path)
    adapter = AimAdapter(run_factory=factory)
    adapter.start(context)
    event = MetricEvent.metrics_event(12, {"eval/reward": 4.0})

    adapter.send(event)
    adapter.send(event)

    backend = created[0]
    assert backend.experiment == context.experiment_name
    assert backend.name == context.run_name
    assert backend.values["hparams"] == context.hparams
    assert backend.tracked == [("eval/reward", 4.0, 12)]


def test_real_aim_adapter_marks_finalized_only_after_metrics(tmp_path):
    context = make_context(tmp_path)
    backend = FakeAimRun(experiment=context.experiment_name)
    adapter = AimAdapter(run_factory=lambda **_: backend)
    adapter.start(context)
    event = MetricEvent.final(
        env_steps=50,
        metrics={"eval/reward": 4.0},
        objective_metric="eval/reward",
    )

    adapter.send(event)

    assert backend.tracked == [("eval/reward", 4.0, 50)]
    assert backend.values["sdk/objective_metric"] == "eval/reward"
    assert backend.values["sdk/finalized"] is True
