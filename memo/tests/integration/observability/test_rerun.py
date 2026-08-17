import pytest
from rerun.experimental import RrdReader

from memorax.observability import RunMetadata
from memorax.observability.sinks.rerun import RerunSink
from tests.support.observability import completed_trajectory

pytestmark = [pytest.mark.integration, pytest.mark.service]


def metadata():
    return RunMetadata(
        run_id="run-t0",
        experiment="experiment",
        launch_id="launch",
        trial=0,
        seed=0,
        role="tuning",
        entry="stream_ac",
        digest="local@sha256:" + "a" * 64,
    )


def recorded(path, entity):
    """Return one entity's logged values, one list per component."""

    columns: dict[str, list] = {}
    for chunk in RrdReader(path).store().stream().to_chunks():
        if chunk.entity_path != entity:
            continue
        batch = chunk.to_record_batch()
        for name in batch.schema.names:
            if name.startswith("rerun.controls"):
                continue
            columns.setdefault(name, []).extend(batch.column(name).to_pylist())
    assert columns, f"{entity} is not in {path.name}"
    return columns


def test_rerun_serializes_the_runtime_selected_sample_and_budget_mask(tmp_path):
    sink = RerunSink(tmp_path, metadata=metadata())

    sink.log_trajectory(
        completed_trajectory(sample_step=10_000_000, post_budget=(False, True))
    )
    sink.close()

    path = tmp_path / "train-sample-000010000000.rrd"
    summary = RrdReader(path).store().summary()
    assert "/episode/post_budget" in summary
    assert "/episode/rewards" in summary
    assert "/episode/series/td_error" in summary


def test_rerun_records_which_sample_and_which_stream_the_trajectory_answers(tmp_path):
    sink = RerunSink(tmp_path, metadata=metadata())

    sink.log_trajectory(completed_trajectory(sample_step=10_000_000, stream=3))
    sink.close()

    written = recorded(tmp_path / "train-sample-000010000000.rrd", "/episode/metadata")

    assert written["sample_step"] == [[10_000_000]]
    assert written["stream"] == [[3]]
    assert written["start_env_steps"] == [[0]]
    assert written["end_env_steps"] == [[8]]
    assert written["run_id"] == [["run-t0"]]


def test_rerun_names_one_artifact_per_requested_sample(tmp_path):
    sink = RerunSink(tmp_path, metadata=metadata())

    sink.log_trajectory(completed_trajectory(sample_step=10_000_000))
    sink.log_trajectory(completed_trajectory(sample_step=20_000_000))
    sink.close()

    assert sorted(path.name for path in tmp_path.glob("*.rrd")) == [
        "train-sample-000010000000.rrd",
        "train-sample-000020000000.rrd",
    ]


def test_rerun_writes_nothing_of_its_own_accord(tmp_path):
    """Runtime decides what is sampled, so an unasked sink produces no artifact."""

    sink = RerunSink(tmp_path, metadata=metadata())
    sink.close()

    assert not hasattr(sink, "log_episode")
    assert list(tmp_path.glob("*.rrd")) == []
