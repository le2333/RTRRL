"""What the run's record owes a run that stopped and started again.

The metrics artifact is the complete record: one row per episode, and every
question asked afterwards is answered from it. Two failures would make it
something else. A row written twice, because the interrupted process kept
reporting after its last snapshot and the resumed one lives through the same
episodes again -- a duplicate is indistinguishable from a second sample, and
every reduction over the file would weight that interval twice. And a window
reported as though it began at the interruption, because the accumulator it
had been filling was thrown away with the process.

Both are fixed here rather than left to the reader of the file, which could
not detect either of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memorax.observability import Reporter, WindowScope
from memorax.observability.sinks import METRICS_FILENAME, MetricsSink
from tests.support.fakes import ScalarRecorder
from tests.support.observability import completed_episode


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_record_is_cut_back_to_the_snapshot(tmp_path: Path) -> None:
    """The interval reported after the last snapshot is reported once."""

    path = tmp_path / METRICS_FILENAME
    sink = MetricsSink(path)
    sink.report(10, {"train/episode/return": 1.0})
    offset = sink.suspend()
    sink.report(20, {"train/episode/return": 2.0})
    sink.report(30, {"train/episode/return": 3.0})
    sink.close()

    resumed = MetricsSink(path)
    resumed.resume(offset)
    resumed.report(20, {"train/episode/return": 2.0})
    resumed.close()

    assert [row["step"] for row in rows(path)] == [10, 20]


def test_a_record_that_did_not_come_back_is_refused(tmp_path: Path) -> None:
    """A hole in the middle of the record, with nothing marking it.

    The snapshot says how much had been written. A shorter file means the
    archive did not travel with it, and appending would leave the run's record
    missing exactly the interval nobody would think to look for.
    """

    path = tmp_path / METRICS_FILENAME
    sink = MetricsSink(path)
    sink.report(10, {"train/episode/return": 1.0})
    offset = sink.suspend()
    sink.close()
    path.unlink()

    with pytest.raises(ValueError, match="did not come back"):
        MetricsSink(path).resume(offset)


def test_a_record_already_being_written_cannot_be_cut(tmp_path: Path) -> None:
    """Resuming happens before the first row, or it is not resuming."""

    sink = MetricsSink(tmp_path / METRICS_FILENAME)
    sink.report(10, {"train/episode/return": 1.0})

    with pytest.raises(ValueError, match="written before it was resumed"):
        sink.resume(0)


def test_a_run_with_no_record_yet_resumes_from_nothing(tmp_path: Path) -> None:
    """A snapshot taken before the first episode names no bytes."""

    path = tmp_path / METRICS_FILENAME
    assert MetricsSink(path).suspend() == 0
    MetricsSink(path).resume(0)
    assert not path.exists()


def test_an_open_window_survives_the_interruption() -> None:
    """A stretch spans an interruption exactly as it spans a chunk.

    Welford's recurrence has no closed form to recompute from and the episodes
    it pooled are gone, so a window that lost its accumulator would report the
    stretch after the interruption under the whole stretch's name.
    """

    whole = WindowScope(every_steps=100)
    for end in (20, 40, 60):
        whole.take(completed_episode(span=(end - 10, end)))
    expected = whole.close()

    stopped = WindowScope(every_steps=100)
    for end in (20, 40):
        stopped.take(completed_episode(span=(end - 10, end)))
    carried = stopped.suspend()

    resumed = WindowScope(every_steps=100)
    resumed.resume(carried)
    resumed.take(completed_episode(span=(50, 60)))

    assert resumed.close() == expected


def test_a_reporter_carries_every_destination_it_holds(tmp_path: Path) -> None:
    """Position is the identity, and it is checked rather than assumed."""

    path = tmp_path / METRICS_FILENAME
    reporter = Reporter(
        scalar_sinks=[MetricsSink(path)],
        sampled_sinks=[ScalarRecorder()],
        training_scopes=[WindowScope(every_steps=100)],
    )
    reporter.log_episode(completed_episode(span=(0, 10)))
    carried = reporter.suspend()

    assert set(carried) == {"scalar", "sampled", "episode", "trajectory", "scopes"}
    assert carried["scalar"] == (path.stat().st_size,)
    # A recorder is not resumable, so it carries nothing and is asked for
    # nothing on the way back.
    assert carried["sampled"] == (None,)

    Reporter(
        scalar_sinks=[MetricsSink(path)],
        sampled_sinks=[ScalarRecorder()],
        training_scopes=[WindowScope(every_steps=100)],
    ).resume(carried)


def test_a_reporter_of_a_different_shape_is_refused(tmp_path: Path) -> None:
    """A run document that changed between the two processes.

    Handing one sink's state to another is how a resumed run would truncate
    the wrong file, or report a window it never accumulated.
    """

    reporter = Reporter(scalar_sinks=[MetricsSink(tmp_path / METRICS_FILENAME)])
    carried = reporter.suspend()

    with pytest.raises(ValueError, match="1 scalar destinations"):
        Reporter(scalar_sinks=[]).resume(carried)


def test_a_destination_that_lost_its_state_is_refused(tmp_path: Path) -> None:
    """Something was resumable when it was written down and is not now."""

    with pytest.raises(ValueError, match="does not implement suspend/resume"):
        Reporter(sampled_sinks=[ScalarRecorder()]).resume({"sampled": (17,)})
