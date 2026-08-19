import pytest
from aim import Repo

from memorax.observability import RunMetadata
from memorax.observability.sinks.aim import AimSink

pytestmark = [pytest.mark.integration, pytest.mark.service]


def test_aim_needs_only_an_endpoint_metadata_and_parameters(tmp_path):
    endpoint = str(tmp_path / "aim")
    Repo.from_path(endpoint, init=True)
    metadata = RunMetadata(
        run_id="run-t0",
        experiment="experiment",
        launch_id="launch",
        trial=0,
        seed=0,
        role="tuning",
        entry="stream_ac",
        digest="local@sha256:" + "a" * 64,
    )

    sink = AimSink(endpoint, metadata, parameters={"gamma": 0.9})
    sink.report(8, {"train/episode/return": 4.0})
    sink.close()

    run = list(Repo.from_path(endpoint).iter_runs())[0]
    assert run.name == "run-t0"
    assert run["entry"] == "stream_ac"
    assert run["params"] == {"gamma": 0.9}
    assert "train/episode/return" in {metric.name for metric in run.metrics()}


def test_aim_keeps_every_scalar_report_and_same_step_names(tmp_path):
    endpoint = str(tmp_path / "aim")
    Repo.from_path(endpoint, init=True)
    metadata = RunMetadata(
        run_id="run-t0",
        experiment="experiment",
        launch_id="launch",
        trial=0,
        seed=0,
        role="tuning",
        entry="stream_ac",
        digest="local@sha256:" + "a" * 64,
    )
    sink = AimSink(endpoint, metadata, parameters={})

    for step in (0, 300, 600, 1200):
        sink.report(step, {"train/episode/return": float(step)})
    sink.report(1200, {"eval/episode/return": 7.0})
    sink.close()

    run = list(Repo.from_path(endpoint).iter_runs())[0]
    metrics = {metric.name: metric for metric in run.metrics()}
    steps, _ = metrics["train/episode/return"].data.items_list()
    assert list(steps) == [0, 300, 600, 1200]
    assert "eval/episode/return" in metrics
