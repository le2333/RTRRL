import json

import pytest
import training_sdk.spool as spool_module

from training_sdk import (
    EventSpool,
    MetricEvent,
    SpoolCorruptionError,
)
from training_sdk.spool import AimUnavailable


class IdempotentSink:
    def __init__(self, *, unavailable_calls=0):
        self.unavailable_calls = unavailable_calls
        self.calls = 0
        self.event_ids = []
        self._seen = set()

    def send(self, event):
        self.calls += 1
        if self.calls <= self.unavailable_calls:
            raise AimUnavailable("temporary outage")
        if event.event_id in self._seen:
            return
        self._seen.add(event.event_id)
        self.event_ids.append(event.event_id)


def test_metric_event_has_unique_event_id_and_json_payload():
    first = MetricEvent.metrics_event(5, {"eval/reward": 1.5})
    second = MetricEvent.metrics_event(5, {"eval/reward": 1.5})

    assert first.event_id != second.event_id
    json.dumps(first.to_dict(), allow_nan=False)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -float("inf")])
def test_metric_event_rejects_bool_and_non_finite_metric_values(value):
    with pytest.raises((TypeError, ValueError), match="metric"):
        MetricEvent.metrics_event(5, {"eval/reward": value})


def test_metric_event_rejects_non_json_serializable_data():
    with pytest.raises(TypeError, match="JSON"):
        MetricEvent(
            event_id="event-1",
            kind="metrics",
            env_steps=5,
            aim_step=5,
            stream="metrics",
            data={"metrics": {"eval/reward": object()}},
        )


def test_metric_event_rejects_negative_or_bool_env_steps():
    with pytest.raises(ValueError, match="env_steps"):
        MetricEvent.metrics_event(-1, {"eval/reward": 1.0})
    with pytest.raises(TypeError, match="env_steps"):
        MetricEvent.metrics_event(True, {"eval/reward": 1.0})


def test_spool_appends_before_attempting_send(tmp_path):
    path = tmp_path / "events.jsonl"
    spool = EventSpool(path)
    event = MetricEvent.metrics_event(10, {"eval/reward": 2.0})

    class InspectingSink:
        def send(self, sent_event):
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert records[0]["event"]["event_id"] == sent_event.event_id

    spool.append(event)
    spool.replay(InspectingSink())


def test_spool_persists_unsent_events_and_sent_status(tmp_path):
    path = tmp_path / "events.jsonl"
    event = MetricEvent.metrics_event(10, {"eval/reward": 2.0})
    EventSpool(path).append(event)
    sink = IdempotentSink()

    EventSpool(path).replay(sink)
    EventSpool(path).replay(sink)

    assert sink.event_ids == [event.event_id]
    assert EventSpool(path).unsent_events == ()


def test_spool_replays_only_unsent_events_after_temporary_outage(tmp_path):
    path = tmp_path / "events.jsonl"
    event = MetricEvent.metrics_event(10, {"eval/reward": 2.0})
    spool = EventSpool(path)
    spool.append(event)
    sink = IdempotentSink(unavailable_calls=1)

    spool.replay(sink)
    assert EventSpool(path).unsent_events == (event,)

    EventSpool(path).replay(sink)
    EventSpool(path).replay(sink)
    assert sink.event_ids == [event.event_id]


def test_spool_stops_replay_at_first_unavailable_event(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        MetricEvent.metrics_event(10, {"train/loss": 2.0}),
        MetricEvent.metrics_event(20, {"eval/reward": 3.0}),
        MetricEvent.final(
            env_steps=20,
            metrics={"eval/reward": 3.0},
            objective_metric="eval/reward",
        ),
    ]
    spool = EventSpool(path)
    for event in events:
        spool.append(event)
    sink = IdempotentSink(unavailable_calls=1)

    spool.replay(sink)

    assert sink.calls == 1
    assert EventSpool(path).unsent_events == tuple(events)


def test_first_spool_creation_fsyncs_new_directory_entries(tmp_path, monkeypatch):
    fsynced_directories = []
    monkeypatch.setattr(
        spool_module,
        "_fsync_directory",
        lambda path: fsynced_directories.append(path),
        raising=False,
    )
    parent = tmp_path / "new-parent"
    spool = EventSpool(parent / "events.jsonl")

    spool.append(MetricEvent.metrics_event(1, {"train/loss": 1.0}))
    first_append_calls = list(fsynced_directories)
    spool.append(MetricEvent.metrics_event(2, {"train/loss": 0.5}))

    assert first_append_calls == [tmp_path, parent]
    assert fsynced_directories == first_append_calls


def test_spool_corruption_is_reported_explicitly(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"record":"event","event":')

    with pytest.raises(SpoolCorruptionError, match="line 1"):
        EventSpool(path)


def test_spool_invalid_utf8_is_reported_as_corruption(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"\xff")

    with pytest.raises(SpoolCorruptionError, match="corrupt spool"):
        EventSpool(path)


def test_spool_rejects_sent_marker_for_unknown_event(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"record": "sent", "event_id": "missing"}) + "\n")

    with pytest.raises(SpoolCorruptionError, match="unknown event"):
        EventSpool(path)
