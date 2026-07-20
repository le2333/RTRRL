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


@dataclass(frozen=True)
class MetricEvent:
    event_id: str
    kind: str
    env_steps: int
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "env_steps": self.env_steps,
            "data": _thaw(self.data),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MetricEvent:
        try:
            return cls(
                event_id=payload["event_id"],
                kind=payload["kind"],
                env_steps=payload["env_steps"],
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
            data={"metrics": dict(metrics)},
        )

    @classmethod
    def episode_summary(
        cls,
        *,
        env_steps: int,
        episode_return: int | float,
        episode_length: int,
    ) -> MetricEvent:
        return cls(
            event_id=str(uuid4()),
            kind="episode_summary",
            env_steps=env_steps,
            data={
                "metrics": {
                    "train/episode_return": episode_return,
                    "train/episode_length": episode_length,
                    "train/env_steps": env_steps,
                }
            },
        )

    @classmethod
    def final(
        cls,
        *,
        env_steps: int,
        metrics: Mapping[str, int | float],
        objective_metric: str,
    ) -> MetricEvent:
        if not isinstance(objective_metric, str) or not objective_metric:
            raise ValueError("objective_metric must be a non-empty string")
        return cls(
            event_id=str(uuid4()),
            kind="final",
            env_steps=env_steps,
            data={
                "metrics": dict(metrics),
                "objective_metric": objective_metric,
                "finalized": True,
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
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise SpoolCorruptionError(f"corrupt spool encoding: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
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
            event = MetricEvent.from_dict(record["event"])
            if event.event_id in self._events_by_id:
                raise ValueError(f"duplicate event ID {event.event_id!r}")
            self._events.append(event)
            self._events_by_id[event.event_id] = event
            return
        if record_type == "sent":
            event_id = record["event_id"]
            if event_id not in self._events_by_id:
                raise ValueError(f"sent marker references unknown event {event_id!r}")
            self._sent_event_ids.add(event_id)
            return
        raise ValueError(f"unknown spool record type {record_type!r}")

    def _append_record(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as spool_file:
            spool_file.write(json.dumps(record, allow_nan=False, separators=(",", ":")))
            spool_file.write("\n")
            spool_file.flush()
            os.fsync(spool_file.fileno())

    def append(self, event: MetricEvent) -> None:
        if event.event_id in self._events_by_id:
            raise ValueError(f"duplicate event ID {event.event_id!r}")
        self._append_record({"record": "event", "event": event.to_dict()})
        self._events.append(event)
        self._events_by_id[event.event_id] = event

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
                continue
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
        if any(existing.event_id == event.event_id for existing in self._events):
            raise ValueError(f"duplicate event ID {event.event_id!r}")
        self._events.append(event)

    def mark_sent(self, event_id: str) -> None:
        if not any(event.event_id == event_id for event in self._events):
            raise ValueError(f"cannot mark unknown event {event_id!r} sent")
        self._sent_event_ids.add(event_id)

    def replay(self, sink: EventSink) -> None:
        for event in self.unsent_events:
            try:
                sink.send(event)
            except AimUnavailable:
                continue
            self.mark_sent(event.event_id)
