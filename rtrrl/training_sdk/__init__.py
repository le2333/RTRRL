from __future__ import annotations

from .context import JsonValue, RunContext
from .run import NullRerun, TrainingRun
from .spool import (
    EventSpool,
    MemorySpool,
    MetricEvent,
    SpoolCorruptionError,
)
from .types import Episode


_current_run: TrainingRun | None = None


def set_current_run(run: TrainingRun | None) -> None:
    global _current_run
    _current_run = run


def current_run() -> TrainingRun:
    if _current_run is None:
        raise RuntimeError("no training run has been initialized")
    return _current_run


__all__ = [
    "Episode",
    "EventSpool",
    "JsonValue",
    "MemorySpool",
    "MetricEvent",
    "NullRerun",
    "RunContext",
    "SpoolCorruptionError",
    "TrainingRun",
    "current_run",
    "set_current_run",
]
