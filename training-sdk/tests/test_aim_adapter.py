from pathlib import Path

import pytest
import training_sdk
import training_sdk.run as run_module

from training_sdk import (
    EventSpool,
    MemorySpool,
    MetricEvent,
    NullRerun,
    RunContext,
    TrainingRun,
)
from training_sdk.aim_adapter import AimAdapter
from training_sdk.spool import AimUnavailable


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
        self.metric_names.append(event.metric_name)


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


def test_checkpoint_duplicate_registration_preserves_existing_artifact(tmp_path):
    run = make_training_run(tmp_path)
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"first")
    run.register_checkpoint(source)
    target = run.context.artifact_directory / "checkpoints" / source.name
    source.write_bytes(b"second")

    with pytest.raises(FileExistsError):
        run.register_checkpoint(source)

    assert target.read_bytes() == b"first"


def test_checkpoint_source_equal_to_target_is_rejected_without_deletion(tmp_path):
    run = make_training_run(tmp_path)
    target = run.context.artifact_directory / "checkpoints" / "model.ckpt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        run.register_checkpoint(target)

    assert target.read_bytes() == b"existing"


def test_checkpoint_copy_failure_preserves_source_and_removes_only_temporary_file(
    tmp_path,
    monkeypatch,
):
    run = make_training_run(tmp_path)
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"source")
    target_directory = run.context.artifact_directory / "checkpoints"

    def fail_after_partial_copy(input_file, output_file, *, length):
        del input_file, length
        output_file.write(b"partial")
        raise OSError("copy failed")

    monkeypatch.setattr(run_module.shutil, "copyfileobj", fail_after_partial_copy)

    with pytest.raises(OSError, match="copy failed"):
        run.register_checkpoint(source)

    assert source.read_bytes() == b"source"
    assert not (target_directory / source.name).exists()
    assert list(target_directory.iterdir()) == []


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


def test_each_spooled_event_maps_to_one_aim_track(tmp_path):
    run = make_training_run(tmp_path)

    run.log_metrics(100, {"train/loss": 3.0, "train/entropy": 0.5})

    assert [(event.metric_name, event.metric_value) for event in run.aim.events] == [
        ("train/loss", 3.0),
        ("train/entropy", 0.5),
    ]
    assert all(len(event.metrics) == 1 for event in run.spool.events)


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
    assert [(event.metric_name, event.metric_value) for event in run.aim.events[:3]] == [
        ("train/episode_return", 2.5),
        ("train/episode_length", 10),
        ("train/env_steps", 10),
    ]
    assert all(len(event.metrics) == 1 for event in run.aim.events)


def test_same_env_steps_episode_summaries_use_distinct_aim_steps(tmp_path):
    context = make_context(tmp_path)
    backend = FakeAimRun(
        experiment=context.experiment_name,
        run_hash="a" * 24,
        force_resume=True,
    )
    run = make_training_run(
        tmp_path,
        context=context,
        aim=AimAdapter(run_factory=lambda **_: backend),
    )

    run.log_episode_summary(
        env_steps=10, episode_return=2.5, episode_length=10
    )
    run.log_episode_summary(
        env_steps=10, episode_return=3.5, episode_length=8
    )

    for metric_name in (
        "train/episode_return",
        "train/episode_length",
        "train/env_steps",
    ):
        points = [
            (key, value)
            for key, value in backend.points.items()
            if key[0] == metric_name
        ]
        assert len(points) == 2
        assert {key[2] for key, _ in points} == {1, 2}
    assert [
        event.metric_value
        for event in run.spool.events
        if event.metric_name == "train/env_steps"
    ] == [10, 10]


def test_summary_sequence_resumes_from_reopened_spool(tmp_path):
    path = tmp_path / "events.jsonl"
    first = make_training_run(tmp_path, spool=EventSpool(path))
    first.log_episode_summary(
        env_steps=10, episode_return=2.5, episode_length=10
    )

    second = make_training_run(tmp_path, spool=EventSpool(path))
    second.log_episode_summary(
        env_steps=10, episode_return=3.5, episode_length=8
    )

    summary_events = [
        event
        for event in EventSpool(path).events
        if event.stream == "episode_summary"
    ]
    assert [event.aim_step for event in summary_events] == [1, 1, 1, 2, 2, 2]


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
        def append_many(self, events):
            operations.append(("append_many", tuple(event.event_id for event in events)))
            super().append_many(events)

    class RecordingAim(FakeAim):
        def send(self, event):
            operations.append(("send", event.event_id))
            super().send(event)

    run = make_training_run(
        tmp_path, aim=RecordingAim(), spool=RecordingSpool()
    )
    run.log_metrics(10, {"train/loss": 1.0})

    assert [operation for operation, _ in operations] == ["append_many", "send"]


def test_failed_summary_batch_append_sends_no_aim_events(tmp_path):
    class FailingBatchSpool(MemorySpool):
        failed = False

        def append_many(self, events):
            if not self.failed:
                self.failed = True
                raise OSError("disk full before durable batch")
            super().append_many(events)

    aim = FakeAim()
    run = make_training_run(tmp_path, aim=aim, spool=FailingBatchSpool())

    with pytest.raises(OSError, match="disk full"):
        run.log_episode_summary(
            env_steps=10, episode_return=2.5, episode_length=10
        )

    assert aim.events == []
    assert run.spool.events == ()

    run.log_episode_summary(
        env_steps=10, episode_return=3.5, episode_length=8
    )
    assert {event.aim_step for event in run.spool.events} == {1}


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

    assert [(event.metric_name, event.metric_value) for event in run.aim.events] == [
        ("eval/length", 20.0),
        ("eval/reward", 4.0),
    ]
    event = run.aim.events[-1]
    assert event.kind == "final"
    assert event.data["objective_metric"] == "eval/reward"
    assert event.data["finalized"] is True
    assert all(len(item.metrics) == 1 for item in run.aim.events)


def test_fail_never_emits_objective_or_finalized_even_after_objective_metric(tmp_path):
    class LifecycleAim(FakeAim):
        def __init__(self):
            super().__init__()
            self.failure = None
            self.closed = 0

        def fail(self, metadata):
            self.failure = metadata

        def close(self):
            self.closed += 1

    class LifecycleRerun(NullRerun):
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    class LifecycleSpool(MemorySpool):
        def __init__(self):
            super().__init__()
            self.closed = 0

        def close(self):
            self.closed += 1

    aim = LifecycleAim()
    rerun = LifecycleRerun()
    spool = LifecycleSpool()
    run = TrainingRun(make_context(tmp_path), aim, rerun, spool)
    run.start()
    run.log_metrics(10, {"eval/reward": 9.0})

    run.fail(RuntimeError("secret-token=do-not-copy"))
    run.fail(RuntimeError("second"))

    assert aim.failure == {"type": "RuntimeError"}
    assert "secret-token" not in str(aim.failure)
    assert not any(event.kind == "final" for event in spool.events)
    assert aim.closed == rerun.closed == spool.closed == 1


def test_finish_validation_failure_remains_active_for_failed_terminal(tmp_path):
    class LifecycleAim(FakeAim):
        def __init__(self):
            super().__init__()
            self.failure = None

        def fail(self, metadata):
            self.failure = metadata

    aim = LifecycleAim()
    spool = MemorySpool()
    run = make_training_run(tmp_path, aim=aim, spool=spool)

    with pytest.raises(ValueError) as raised:
        run.finish({"not/the/objective": 4.0})
    run.fail(raised.value)

    assert aim.failure == {"type": "ValueError"}
    assert not any(event.kind == "final" for event in spool.events)


def test_finish_spool_append_failure_remains_active_for_failed_terminal(tmp_path):
    class AppendFailure(MemorySpool):
        def append_many(self, events):
            raise OSError("append failed")

    class LifecycleAim(FakeAim):
        def __init__(self):
            super().__init__()
            self.failure = None

        def fail(self, metadata):
            self.failure = metadata

    aim = LifecycleAim()
    spool = AppendFailure()
    run = make_training_run(tmp_path, aim=aim, spool=spool)

    with pytest.raises(OSError, match="append failed") as raised:
        run.finish({"eval/reward": 4.0})
    run.fail(raised.value)

    assert aim.failure == {"type": "OSError"}
    assert not any(event.kind == "final" for event in spool.events)


def test_finish_mark_sent_failure_cannot_transition_to_failed(tmp_path):
    class MarkSentFailure(MemorySpool):
        def mark_sent(self, event_id):
            raise OSError("mark sent failed")

    class LifecycleAim(FakeAim):
        def __init__(self):
            super().__init__()
            self.failure = None

        def fail(self, metadata):
            self.failure = metadata

    aim = LifecycleAim()
    run = TrainingRun(
        make_context(tmp_path),
        aim,
        NullRerun(),
        MarkSentFailure(),
    )
    run.start()

    with pytest.raises(OSError, match="mark sent failed") as raised:
        run.finish({"eval/reward": 4.0})
    run.fail(raised.value)

    assert aim.failure is None
    assert sum(
        event.kind == "final" and event.data["finalized"]
        for event in aim.events
    ) == 1


def test_finish_aim_send_failure_after_append_cannot_transition_to_failed(tmp_path):
    class LifecycleAim(FakeAim):
        def __init__(self):
            super().__init__([ConnectionError("aim failed")])
            self.failure = None

        def fail(self, metadata):
            self.failure = metadata

    aim = LifecycleAim()
    spool = MemorySpool()
    run = make_training_run(tmp_path, aim=aim, spool=spool)

    with pytest.raises(ConnectionError, match="aim failed") as raised:
        run.finish({"eval/reward": 4.0})
    run.fail(raised.value)

    assert aim.failure is None
    assert sum(
        event.kind == "final" and event.data["finalized"]
        for event in spool.events
    ) == 1


def test_finish_reentrant_fail_during_append_cannot_create_two_terminals(tmp_path):
    class ReentrantSpool(MemorySpool):
        run = None

        def append_many(self, events):
            assert self.run is not None
            self.run.fail(RuntimeError("reentrant"))
            super().append_many(events)

    class LifecycleAim(FakeAim):
        def __init__(self):
            super().__init__()
            self.failure = None

        def fail(self, metadata):
            self.failure = metadata

    aim = LifecycleAim()
    spool = ReentrantSpool()
    run = make_training_run(tmp_path, aim=aim, spool=spool)
    spool.run = run

    run.finish({"eval/reward": 4.0})

    assert aim.failure is None
    assert sum(
        event.kind == "final" and event.data["finalized"]
        for event in spool.events
    ) == 1


def test_finish_without_descriptor_objective_is_a_noop(tmp_path):
    context = make_context(tmp_path, objective={})
    run = make_training_run(tmp_path, context=context)

    run.finish({"eval/reward": 4.0})

    assert run.aim.events == []


class FakeAimRun:
    def __init__(self, *, experiment, run_hash, force_resume):
        self.experiment = experiment
        self.run_hash = run_hash
        self.force_resume = force_resume
        self.name = None
        self.values = {}
        self.tracked = []
        self.points = {}
        self.fail_event_marker_once = False
        self.fail_hparams_once = False
        self.closed = 0

    def __setitem__(self, key, value):
        if key == "hparams" and self.fail_hparams_once:
            self.fail_hparams_once = False
            raise RuntimeError("hparams failed")
        if key.startswith("sdk/event_ids/") and self.fail_event_marker_once:
            self.fail_event_marker_once = False
            raise ConnectionError("crash after track")
        self.values[key] = value

    def get(self, key, default=None):
        return self.values.get(key, default)

    def track(self, value, *, name, step, context, epoch):
        self.tracked.append((name, value, step, context, epoch))
        stream_identity = tuple(sorted(context.items()))
        self.points[(name, stream_identity, step)] = value

    def close(self):
        self.closed += 1


def test_aim_adapter_failure_metadata_is_nonfinal_and_closes(tmp_path):
    context = make_context(tmp_path)
    backend = FakeAimRun(
        experiment=context.experiment_name,
        run_hash="a" * 24,
        force_resume=True,
    )
    adapter = AimAdapter(run_factory=lambda **_: backend)
    adapter.start(context)

    adapter.fail({"type": "RuntimeError"})
    adapter.close()

    assert backend.values["sdk/failed"] is True
    assert backend.values["sdk/error"] == {"type": "RuntimeError"}
    assert "sdk/finalized" not in backend.values
    assert "sdk/objective_metric" not in backend.values
    assert backend.closed == 1


def test_aim_adapter_can_close_run_created_before_start_configuration_fails(
    tmp_path,
):
    context = make_context(tmp_path)
    backend = FakeAimRun(
        experiment=context.experiment_name,
        run_hash="a" * 24,
        force_resume=True,
    )
    backend.fail_hparams_once = True
    adapter = AimAdapter(run_factory=lambda **_: backend)

    with pytest.raises(RuntimeError, match="hparams failed"):
        adapter.start(context)
    adapter.close()

    assert backend.closed == 1


def test_real_aim_adapter_resumes_stable_run_identity(tmp_path):
    runs = {}
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return runs.setdefault(kwargs["run_hash"], FakeAimRun(**kwargs))

    context = make_context(tmp_path)
    first = AimAdapter(run_factory=factory)
    second = AimAdapter(run_factory=factory)
    first.start(context)
    second.start(context)
    event = MetricEvent.metrics_event(12, {"eval/reward": 4.0})

    first.send(event)
    second.send(event)

    assert calls[0]["run_hash"] == calls[1]["run_hash"]
    assert len(calls[0]["run_hash"]) == 24
    assert calls[0]["run_hash"].isalnum()
    assert calls[0]["force_resume"] is True
    backend = runs[calls[0]["run_hash"]]
    assert backend.experiment == context.experiment_name
    assert backend.name == context.run_name
    assert backend.values["hparams"] == context.hparams
    assert backend.tracked == [
        ("eval/reward", 4.0, 12, {"sdk_stream": "metrics"}, 12)
    ]


def test_cross_adapter_replay_overwrites_same_aim_sequence_point(tmp_path):
    runs = {}

    def factory(**kwargs):
        return runs.setdefault(kwargs["run_hash"], FakeAimRun(**kwargs))

    context = make_context(tmp_path)
    first = AimAdapter(run_factory=factory)
    first.start(context)
    event = MetricEvent.metrics_event(12, {"eval/reward": 4.0})
    spool_path = tmp_path / "events.jsonl"
    spool = EventSpool(spool_path)
    spool.append(event)
    backend = next(iter(runs.values()))
    backend.fail_event_marker_once = True

    spool.replay(first)
    assert EventSpool(spool_path).unsent_events == (event,)

    second = AimAdapter(run_factory=factory)
    second.start(context)
    EventSpool(spool_path).replay(second)
    EventSpool(spool_path).replay(second)

    assert backend.tracked == [
        ("eval/reward", 4.0, 12, {"sdk_stream": "metrics"}, 12),
        ("eval/reward", 4.0, 12, {"sdk_stream": "metrics"}, 12),
    ]
    assert backend.points == {
        ("eval/reward", (("sdk_stream", "metrics"),), 12): 4.0
    }
    assert backend.values[f"sdk/event_ids/{event.event_id}"] is True


def test_replaying_same_summary_event_overwrites_sequence_point(tmp_path):
    runs = {}

    def factory(**kwargs):
        return runs.setdefault(kwargs["run_hash"], FakeAimRun(**kwargs))

    context = make_context(tmp_path)
    first = AimAdapter(run_factory=factory)
    first.start(context)
    event = MetricEvent.episode_summary(
        env_steps=12,
        summary_sequence=1,
        episode_return=4.0,
        episode_length=12,
    )[0]
    path = tmp_path / "summary-events.jsonl"
    spool = EventSpool(path)
    spool.append(event)
    backend = next(iter(runs.values()))
    backend.fail_event_marker_once = True

    spool.replay(first)
    second = AimAdapter(run_factory=factory)
    second.start(context)
    EventSpool(path).replay(second)

    assert backend.points == {
        (
            "train/episode_return",
            (("sdk_stream", "episode_summary"),),
            1,
        ): 4.0
    }
    assert len(backend.tracked) == 2


@pytest.mark.filterwarnings("ignore:.*declarative_base.*:sqlalchemy.exc.MovedIn20Warning")
@pytest.mark.filterwarnings(
    "ignore:datetime.datetime.utcnow.*:DeprecationWarning"
)
def test_aim_328_overwrites_replay_and_keeps_distinct_summary_steps(tmp_path):
    import hashlib

    from aim import Run as AimRun
    from aim.storage.context import Context

    repo_path = tmp_path / "aim-repo"
    context = make_context(tmp_path)
    adapter = AimAdapter(repo=str(repo_path))
    adapter.start(context)
    first = MetricEvent.episode_summary(
        env_steps=12,
        summary_sequence=1,
        episode_return=2.5,
        episode_length=12,
    )[0]
    replay = MetricEvent.episode_summary(
        env_steps=12,
        summary_sequence=1,
        episode_return=2.5,
        episode_length=12,
    )[0]
    second = MetricEvent.episode_summary(
        env_steps=12,
        summary_sequence=2,
        episode_return=3.5,
        episode_length=8,
    )[0]

    adapter.send(first)
    adapter.send(replay)
    adapter.send(second)

    run_hash = hashlib.sha256(context.run_id.encode("utf-8")).hexdigest()[:24]
    stored_run = AimRun(run_hash=run_hash, repo=str(repo_path), read_only=True)
    metric = stored_run.get_metric(
        "train/episode_return",
        Context({"sdk_stream": "episode_summary"}),
    )
    assert metric is not None
    assert metric.data.items_list()[0] == [1, 2]
    assert metric.data.values_list()[0] == [2.5, 3.5]


def test_real_aim_adapter_marks_finalized_only_after_metrics(tmp_path):
    context = make_context(tmp_path)
    backend = FakeAimRun(
        experiment=context.experiment_name,
        run_hash="a" * 24,
        force_resume=True,
    )
    adapter = AimAdapter(run_factory=lambda **_: backend)
    adapter.start(context)
    event = MetricEvent.final(
        env_steps=50,
        metrics={"eval/reward": 4.0},
        objective_metric="eval/reward",
    )

    adapter.send(event)

    assert backend.tracked == [
        ("eval/reward", 4.0, 50, {"sdk_stream": "final"}, 50)
    ]
    assert backend.values["sdk/objective_metric"] == "eval/reward"
    assert backend.values["sdk/finalized"] is True


@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError])
def test_aim_adapter_converts_only_default_transient_errors(tmp_path, error_type):
    def factory(**_):
        raise error_type("temporary")

    adapter = AimAdapter(run_factory=factory)

    with pytest.raises(AimUnavailable):
        adapter.start(make_context(tmp_path))


@pytest.mark.parametrize("error", [PermissionError("denied"), OSError("bad data")])
def test_aim_adapter_preserves_non_transient_os_errors(tmp_path, error):
    def factory(**_):
        raise error

    adapter = AimAdapter(run_factory=factory)

    with pytest.raises(type(error), match=str(error)):
        adapter.start(make_context(tmp_path))


def test_aim_adapter_converts_injected_transient_error(tmp_path):
    class TemporaryBackendError(Exception):
        pass

    def factory(**_):
        raise TemporaryBackendError("temporary")

    adapter = AimAdapter(
        run_factory=factory,
        availability_errors=(TemporaryBackendError,),
    )

    with pytest.raises(AimUnavailable):
        adapter.start(make_context(tmp_path))


def test_public_sdk_does_not_export_aim_backend_types():
    assert not hasattr(training_sdk, "AimAdapter")
    assert not hasattr(training_sdk, "AimUnavailable")
