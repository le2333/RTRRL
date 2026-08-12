import pytest
from rerun.experimental import RrdReader

from memorax.observability import RunMetadata
from memorax.observability.sinks.rerun import RerunSink
from tests.support.observability import completed_episode

pytestmark = [pytest.mark.integration, pytest.mark.service]


def test_rerun_keeps_a_local_rrd_and_has_no_upload_destination(tmp_path):
    metadata = RunMetadata(
        run_id="run-t0",
        experiment="experiment",
        launch_id="launch",
        trial=0,
        entry="stream_ac",
        digest="local@sha256:" + "a" * 64,
    )
    sink = RerunSink(
        tmp_path,
        every_steps=1,
        num_envs=1,
        metadata=metadata,
    )

    sink.log_episode(completed_episode())
    sink.close()

    path = tmp_path / "train-000001.rrd"
    assert path.exists()
    summary = RrdReader(path).store().summary()
    assert "/episode/rewards" in summary
    assert "/episode/series/td_error" in summary


def recorded_numbers(sink, directory, episodes):
    for episode in episodes:
        sink.log_episode(episode)
    sink.close()
    return [
        episode.number
        for episode in episodes
        if (directory / f"{episode.phase}-{episode.number:06d}.rrd").exists()
    ]


def metadata():
    return RunMetadata(
        run_id="run-t0",
        experiment="experiment",
        launch_id="launch",
        trial=0,
        entry="stream_ac",
        digest="local@sha256:" + "a" * 64,
    )


def test_sampling_interval_is_over_environment_steps(tmp_path):
    sink = RerunSink(tmp_path, every_steps=100, num_envs=1, metadata=metadata())
    episodes = [
        completed_episode(1, span=(0, 90)),
        completed_episode(2, span=(110, 190)),
        completed_episode(3, span=(190, 260)),
    ]

    assert recorded_numbers(sink, tmp_path, episodes) == [1, 3]


def test_sample_step_selects_only_the_stream_it_belongs_to(tmp_path):
    sink = RerunSink(tmp_path, every_steps=10, num_envs=4, metadata=metadata())
    episodes = [
        completed_episode(1, span=(0, 40), stream=0),
        completed_episode(2, span=(0, 40), stream=1),
        completed_episode(3, span=(0, 40), stream=2),
    ]

    assert recorded_numbers(sink, tmp_path, episodes) == [1, 3]


def test_zero_span_evaluation_episode_contains_no_training_sample(tmp_path):
    sink = RerunSink(tmp_path, every_steps=1, num_envs=1, metadata=metadata())
    episodes = [completed_episode(1, phase="eval", span=(64, 64))]

    assert recorded_numbers(sink, tmp_path, episodes) == []


def test_phase_is_part_of_the_local_artifact_name(tmp_path):
    sink = RerunSink(tmp_path, every_steps=1, num_envs=1, metadata=metadata())
    sink.log_episode(completed_episode(1, phase="train"))
    sink.log_episode(completed_episode(1, phase="eval", span=(0, 8)))
    sink.close()

    assert (tmp_path / "train-000001.rrd").exists()
    assert (tmp_path / "eval-000001.rrd").exists()
