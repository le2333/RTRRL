import sys
from types import SimpleNamespace

import pytest

from logging_util import AimLogger, DummyLogger, MultiLogger, WandbLogger, with_logger
from training_sdk import Episode, set_current_run


def complete_episode():
    return Episode(
        number=1,
        phase="eval",
        start_env_steps=0,
        end_env_steps=1,
        observations=([0.0], [1.0]),
        actions=([0.5],),
        rewards=(1.0,),
        terminals=(True,),
        truncations=(False,),
    )


class RecordingTrainingRun:
    def __init__(self, *, objective_metric="eval/reward"):
        self.context = SimpleNamespace(objective={"metric": objective_metric})
        self.calls = []

    def log_metrics(self, env_steps, metrics):
        self.calls.append(("log_metrics", env_steps, metrics))

    def log_episode_summary(self, **summary):
        self.calls.append(("log_episode_summary", summary))

    def log_episode(self, episode):
        self.calls.append(("log_episode", episode))
        return "episode.rrd"

    def finish(self, final_metrics):
        self.calls.append(("finish", final_metrics))
        objective = self.context.objective["metric"]
        if objective not in final_metrics:
            raise ValueError(
                f"final_metrics must contain objective metric {objective!r}"
            )


def test_dummy_and_wandb_episode_methods_are_noops():
    episode = complete_episode()

    for logger in (DummyLogger(), WandbLogger()):
        assert logger.log_episode_summary(
            env_steps=10, episode_return=2.0, episode_length=10
        ) is None
        assert logger.log_episode(episode) is None


def test_existing_dummy_logger_api_is_unchanged():
    logger = DummyLogger()

    logger.log({"eval/reward": 1.0}, step=10)
    logger.log_params({"seed": 7})
    logger.finalize()
    logger.save_model(object())
    logger.log_video("eval", [])
    logger["summary"] = 3.0

    assert logger["summary"] == 3.0


def test_facility_aim_logger_does_not_create_or_reconfigure_aim_run(monkeypatch):
    def unexpected_run(**kwargs):
        raise AssertionError(f"created a second Aim run: {kwargs}")

    monkeypatch.setitem(sys.modules, "aim", SimpleNamespace(Run=unexpected_run))
    training_run = RecordingTrainingRun()

    logger = AimLogger(
        "wrong-experiment",
        repo="wrong-repo",
        hparams={"wrong": "hparams"},
        run_name="wrong-name",
        training_run=training_run,
    )
    logger.log_params({"also": "wrong"})
    logger["local-summary"] = 2.0

    assert training_run.calls == []
    assert logger["local-summary"] == 2.0


def test_facility_aim_logger_delegates_metrics_and_caches_latest_finite_scalars():
    training_run = RecordingTrainingRun()
    logger = AimLogger("ignored", training_run=training_run)
    metrics = {
        "eval/reward": 1.0,
        "eval/length": 10,
        "invalid/nan": float("nan"),
        "invalid/inf": float("inf"),
        "non_scalar": "ignored",
    }

    logger.log(metrics, step=20)
    logger.log({"eval/reward": 2.0}, step=30)
    logger.finalize()

    assert training_run.calls == [
        ("log_metrics", 20, metrics),
        ("log_metrics", 30, {"eval/reward": 2.0}),
        ("finish", {"eval/reward": 2.0, "eval/length": 10}),
    ]


def test_facility_log_uses_canonical_native_env_steps_when_step_is_none():
    training_run = RecordingTrainingRun()
    logger = AimLogger("ignored", training_run=training_run)
    metrics = {"train/env_steps": 20, "eval/reward": 1.0}

    logger.log(metrics)

    assert training_run.calls == [("log_metrics", 20, metrics)]


def test_facility_log_without_native_env_steps_fails_before_call_or_cache():
    training_run = RecordingTrainingRun()
    logger = AimLogger("ignored", training_run=training_run)

    with pytest.raises(
        ValueError,
        match="facility logging requires explicit native env_steps",
    ):
        logger.log({"eval/reward": 1.0})

    assert training_run.calls == []
    assert logger._final_metrics == {}


def test_facility_log_rejects_conflicting_native_env_steps():
    training_run = RecordingTrainingRun()
    logger = AimLogger("ignored", training_run=training_run)

    with pytest.raises(ValueError, match="conflicts"):
        logger.log({"train/env_steps": 21, "eval/reward": 1.0}, step=20)

    assert training_run.calls == []
    assert logger._final_metrics == {}


@pytest.mark.parametrize("env_steps", [True, 20.0, "20"])
def test_facility_log_rejects_non_integer_canonical_env_steps(env_steps):
    training_run = RecordingTrainingRun()
    logger = AimLogger("ignored", training_run=training_run)

    with pytest.raises(ValueError, match="train/env_steps.*integer"):
        logger.log({"train/env_steps": env_steps, "eval/reward": 1.0})

    assert training_run.calls == []
    assert logger._final_metrics == {}


def test_facility_aim_logger_propagates_missing_objective_error():
    training_run = RecordingTrainingRun()
    logger = AimLogger("ignored", training_run=training_run)
    logger.log({"train/loss": 1.0}, step=10)

    with pytest.raises(ValueError, match="objective metric 'eval/reward'"):
        logger.finalize()

    assert training_run.calls[-1] == ("finish", {"train/loss": 1.0})


def test_facility_aim_logger_delegates_episode_calls_exactly():
    training_run = RecordingTrainingRun()
    logger = AimLogger("ignored", training_run=training_run)
    episode = complete_episode()
    summary = {
        "env_steps": 10,
        "episode_return": 2.0,
        "episode_length": 10,
    }

    assert logger.log_episode_summary(**summary) is None
    assert logger.log_episode(episode) == "episode.rrd"

    assert training_run.calls == [
        ("log_episode_summary", summary),
        ("log_episode", episode),
    ]


def test_multilogger_fans_out_episode_calls_in_order():
    calls = []

    class Recorder(DummyLogger):
        def __init__(self, name):
            self.name = name

        def log_episode_summary(self, **summary):
            calls.append((self.name, "summary", summary))

        def log_episode(self, episode):
            calls.append((self.name, "episode", episode))

    episode = complete_episode()
    logger = MultiLogger([Recorder("first"), Recorder("second")])

    logger.log_episode_summary(
        env_steps=10, episode_return=2.0, episode_length=10
    )
    logger.log_episode(episode)

    assert [(name, method) for name, method, _ in calls] == [
        ("first", "summary"),
        ("second", "summary"),
        ("first", "episode"),
        ("second", "episode"),
    ]


def test_multilogger_stops_and_propagates_episode_exception():
    calls = []

    class Recorder(DummyLogger):
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail

        def log_episode(self, episode):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError("broken logger")

    logger = MultiLogger(
        [Recorder("first"), Recorder("broken", fail=True), Recorder("last")]
    )

    with pytest.raises(RuntimeError, match="broken logger"):
        logger.log_episode(complete_episode())

    assert calls == ["first", "broken"]


def test_legacy_aim_logger_preserves_run_setup_and_episode_noops(monkeypatch):
    class FakeAimRun(dict):
        hash = "abc123"

        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.name = None
            self.tracked = []
            self.finished = []

        def track(self, value, *, name, step):
            self.tracked.append((name, value, step))

        def report_successful_finish(self, *, block):
            self.finished.append(block)

    created = []

    def make_run(**kwargs):
        run = FakeAimRun(**kwargs)
        created.append(run)
        return run

    monkeypatch.setitem(sys.modules, "aim", SimpleNamespace(Run=make_run))
    logger = AimLogger(
        "legacy-experiment",
        repo="legacy-repo",
        hparams={"seed": 7},
        run_name="legacy-name",
    )

    assert len(created) == 1
    assert created[0].kwargs == {
        "experiment": "legacy-experiment",
        "repo": "legacy-repo",
        "log_system_params": True,
    }
    assert created[0]["hparams"] == {"seed": 7}
    assert created[0].name == "legacy-name abc123"
    logger.log({"eval/reward": 2.0}, step=4)
    logger.log({"eval/reward": 2.5})
    logger["summary"] = 3
    logger.finalize()
    assert created[0].tracked == [
        ("eval/reward", 2.0, 4),
        ("eval/reward", 2.5, None),
    ]
    assert logger["summary"] == 3.0
    assert created[0].finished == [False]
    assert logger.log_episode_summary(
        env_steps=1, episode_return=1.0, episode_length=1
    ) is None
    assert logger.log_episode(complete_episode()) is None
    assert len(created) == 1


def test_with_logger_passes_explicit_facility_run_to_aim():
    training_run = RecordingTrainingRun()
    received = []

    with_logger(
        lambda hparams, logger: received.append((hparams, logger)),
        {"seed": 7},
        "aim",
        "project",
        training_run=training_run,
    )

    assert received[0][0] == {"seed": 7}
    assert isinstance(received[0][1], AimLogger)
    received[0][1].log({"eval/reward": 3.0}, step=4)
    assert training_run.calls == [
        ("log_metrics", 4, {"eval/reward": 3.0})
    ]


def test_with_logger_safely_uses_current_facility_run():
    training_run = RecordingTrainingRun()
    received = []
    set_current_run(training_run)
    try:
        with_logger(
            lambda hparams, logger: received.append(logger),
            {"seed": 7},
            "aim",
            "project",
        )
    finally:
        set_current_run(None)

    received[0].log({"eval/reward": 3.0}, step=4)
    assert training_run.calls == [
        ("log_metrics", 4, {"eval/reward": 3.0})
    ]


def test_with_logger_unknown_backend_keeps_legacy_call_shape():
    calls = []

    result = with_logger(
        lambda hparams: calls.append(hparams) or "result",
        {"seed": 7},
        "unknown",
        "project",
        training_run=RecordingTrainingRun(),
    )

    assert result == "result"
    assert calls == [{"seed": 7}]
