from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, cast
from uuid import uuid4


class AimUnavailable(RuntimeError):
    """A transient failure communicating with Aim."""


class SpoolCorruptionError(ValueError):
    """The durable event spool cannot be decoded safely."""


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _validated_metrics(metrics: Mapping[str, int | float]) -> dict[str, int | float]:
    if len(metrics) != 1:
        raise ValueError("each MetricEvent must contain exactly one metric")
    result: dict[str, int | float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise TypeError("metric names must be non-empty strings")
        if type(value) not in (int, float):
            raise TypeError(
                f"metric {name!r} must be a JSON int or float, not bool"
            )
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
        result[name] = value
    return result


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class MetricEvent:
    event_id: str
    kind: str
    env_steps: int
    aim_step: int
    stream: str
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("event_id must be a non-empty string")
        if self.kind not in {"metrics", "episode_summary", "final"}:
            raise ValueError(f"unsupported metric event kind: {self.kind!r}")
        if type(self.env_steps) is not int:
            raise TypeError("env_steps must be an integer")
        if self.env_steps < 0:
            raise ValueError("env_steps must be non-negative")
        if type(self.aim_step) is not int:
            raise TypeError("aim_step must be an integer")
        if self.aim_step < 0:
            raise ValueError("aim_step must be non-negative")
        if self.stream != self.kind:
            raise ValueError("MetricEvent stream must match its stable event kind")
        if self.stream == "episode_summary" and self.aim_step < 1:
            raise ValueError("episode summary aim_step must be one-based")
        if self.stream != "episode_summary" and self.aim_step != self.env_steps:
            raise ValueError("non-summary aim_step must equal native env_steps")

        copied = dict(self.data)
        metrics = copied.get("metrics")
        if not isinstance(metrics, dict):
            raise TypeError("MetricEvent data must contain a metrics mapping")
        copied["metrics"] = _validated_metrics(metrics)
        try:
            encoded = json.dumps(copied, allow_nan=False)
            copied = cast(dict[str, Any], json.loads(encoded))
        except (TypeError, ValueError) as exc:
            raise TypeError("MetricEvent data must be finite JSON data") from exc
        object.__setattr__(self, "data", _freeze(copied))

    @property
    def metrics(self) -> dict[str, int | float]:
        return dict(cast(Mapping[str, int | float], self.data["metrics"]))

    @property
    def metric_name(self) -> str:
        return next(iter(self.metrics))

    @property
    def metric_value(self) -> int | float:
        return self.metrics[self.metric_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "env_steps": self.env_steps,
            "aim_step": self.aim_step,
            "stream": self.stream,
            "data": _thaw(self.data),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MetricEvent:
        try:
            return cls(
                event_id=payload["event_id"],
                kind=payload["kind"],
                env_steps=payload["env_steps"],
                aim_step=payload["aim_step"],
                stream=payload["stream"],
                data=payload["data"],
            )
        except KeyError as exc:
            raise ValueError(f"event is missing field {exc.args[0]!r}") from exc

    @classmethod
    def metrics_event(
        cls, env_steps: int, metrics: Mapping[str, int | float]
    ) -> MetricEvent:
        return cls(
            event_id=str(uuid4()),
            kind="metrics",
            env_steps=env_steps,
            aim_step=env_steps,
            stream="metrics",
            data={"metrics": dict(metrics)},
        )

    @classmethod
    def episode_summary(
        cls,
        *,
        env_steps: int,
        summary_sequence: int,
        episode_return: int | float,
        episode_length: int,
    ) -> tuple[MetricEvent, ...]:
        return tuple(
            cls(
                event_id=str(uuid4()),
                kind="episode_summary",
                env_steps=env_steps,
                aim_step=summary_sequence,
                stream="episode_summary",
                data={"metrics": {name: value}},
            )
            for name, value in (
                ("train/episode_return", episode_return),
                ("train/episode_length", episode_length),
                ("train/env_steps", env_steps),
            )
        )

    @classmethod
    def final(
        cls,
        *,
        env_steps: int,
        metrics: Mapping[str, int | float],
        objective_metric: str,
        finalized: bool = True,
    ) -> MetricEvent:
        if not isinstance(objective_metric, str) or not objective_metric:
            raise ValueError("objective_metric must be a non-empty string")
        return cls(
            event_id=str(uuid4()),
            kind="final",
            env_steps=env_steps,
            aim_step=env_steps,
            stream="final",
            data={
                "metrics": dict(metrics),
                "objective_metric": objective_metric,
                "finalized": finalized,
            },
        )


class EventSink(Protocol):
    def send(self, event: MetricEvent) -> None: ...


class EventSpool:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._events: list[MetricEvent] = []
        self._events_by_id: dict[str, MetricEvent] = {}
        self._sent_event_ids: set[str] = set()
        self._load()

    @property
    def events(self) -> tuple[MetricEvent, ...]:
        return tuple(self._events)

    @property
    def unsent_events(self) -> tuple[MetricEvent, ...]:
        return tuple(
            event
            for event in self._events
            if event.event_id not in self._sent_event_ids
        )

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw_contents = self.path.read_bytes()
            decoded_contents = raw_contents.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SpoolCorruptionError(f"corrupt spool encoding: {exc}") from exc
        if raw_contents and not raw_contents.endswith(b"\n"):
            durable_size = raw_contents.rfind(b"\n") + 1
            with self.path.open("r+b") as spool_file:
                spool_file.truncate(durable_size)
                spool_file.flush()
                os.fsync(spool_file.fileno())
            decoded_contents = raw_contents[:durable_size].decode("utf-8")
        lines = decoded_contents.splitlines(keepends=True)
        for line_number, line in enumerate(lines, start=1):
            if not line.endswith("\n"):
                if line_number == len(lines):
                    break
                raise SpoolCorruptionError(
                    f"corrupt unterminated spool record at line {line_number}"
                )
            try:
                record = json.loads(line)
                self._apply_record(record)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                raise SpoolCorruptionError(
                    f"corrupt spool record at line {line_number}: {exc}"
                ) from exc

    def _apply_record(self, record: Any) -> None:
        if not isinstance(record, dict):
            raise TypeError("record must be a JSON object")
        record_type = record["record"]
        if record_type == "event":
            self._add_loaded_events((MetricEvent.from_dict(record["event"]),))
            return
        if record_type == "batch":
            payloads = record["events"]
            if not isinstance(payloads, list) or not payloads:
                raise ValueError("batch record must contain a non-empty events list")
            self._add_loaded_events(
                tuple(MetricEvent.from_dict(payload) for payload in payloads)
            )
            return
        if record_type == "sent":
            event_id = record["event_id"]
            if event_id not in self._events_by_id:
                raise ValueError(f"sent marker references unknown event {event_id!r}")
            self._sent_event_ids.add(event_id)
            return
        raise ValueError(f"unknown spool record type {record_type!r}")

    def _validate_new_events(self, events: tuple[MetricEvent, ...]) -> None:
        event_ids = set(self._events_by_id)
        for event in events:
            if not isinstance(event, MetricEvent):
                raise TypeError("spool batches may contain only MetricEvent values")
            if event.event_id in event_ids:
                raise ValueError(f"duplicate event ID {event.event_id!r}")
            event_ids.add(event.event_id)

    def _add_loaded_events(self, events: tuple[MetricEvent, ...]) -> None:
        self._validate_new_events(events)
        self._events.extend(events)
        self._events_by_id.update((event.event_id, event) for event in events)

    def _append_record(self, record: Mapping[str, Any]) -> None:
        missing_directories: list[Path] = []
        candidate = self.path.parent
        while not candidate.exists():
            missing_directories.append(candidate)
            candidate = candidate.parent
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for directory in reversed(missing_directories):
            _fsync_directory(directory.parent)

        creating_spool = not self.path.exists()
        with self.path.open("a", encoding="utf-8") as spool_file:
            encoded_record = (
                json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
            )
            spool_file.write(encoded_record)
            spool_file.flush()
            os.fsync(spool_file.fileno())
        if creating_spool:
            _fsync_directory(self.path.parent)

    def append(self, event: MetricEvent) -> None:
        self.append_many((event,))

    def append_many(self, events: tuple[MetricEvent, ...]) -> None:
        events = tuple(events)
        if not events:
            return
        self._validate_new_events(events)
        self._append_record(
            {
                "record": "batch",
                "events": [event.to_dict() for event in events],
            }
        )
        self._events.extend(events)
        self._events_by_id.update((event.event_id, event) for event in events)

    def mark_sent(self, event_id: str) -> None:
        if event_id not in self._events_by_id:
            raise ValueError(f"cannot mark unknown event {event_id!r} sent")
        if event_id in self._sent_event_ids:
            return
        self._append_record({"record": "sent", "event_id": event_id})
        self._sent_event_ids.add(event_id)

    def replay(self, sink: EventSink) -> None:
        for event in self.unsent_events:
            try:
                sink.send(event)
            except AimUnavailable:
                break
            self.mark_sent(event.event_id)


class MemorySpool:
    def __init__(self) -> None:
        self._events: list[MetricEvent] = []
        self._sent_event_ids: set[str] = set()

    @property
    def events(self) -> tuple[MetricEvent, ...]:
        return tuple(self._events)

    @property
    def unsent_events(self) -> tuple[MetricEvent, ...]:
        return tuple(
            event
            for event in self._events
            if event.event_id not in self._sent_event_ids
        )

    def append(self, event: MetricEvent) -> None:
        self.append_many((event,))

    def append_many(self, events: tuple[MetricEvent, ...]) -> None:
        events = tuple(events)
        existing_ids = {event.event_id for event in self._events}
        batch_ids: set[str] = set()
        for event in events:
            if not isinstance(event, MetricEvent):
                raise TypeError("spool batches may contain only MetricEvent values")
            if event.event_id in existing_ids or event.event_id in batch_ids:
                raise ValueError(f"duplicate event ID {event.event_id!r}")
            batch_ids.add(event.event_id)
        self._events.extend(events)

    def mark_sent(self, event_id: str) -> None:
        if not any(event.event_id == event_id for event in self._events):
            raise ValueError(f"cannot mark unknown event {event_id!r} sent")
        self._sent_event_ids.add(event_id)

    def replay(self, sink: EventSink) -> None:
        for event in self.unsent_events:
            try:
                sink.send(event)
            except AimUnavailable:
                break
            self.mark_sent(event.event_id)
