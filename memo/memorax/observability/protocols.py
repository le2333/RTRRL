"""The two observation shapes a backend can consume."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from memorax.runtime.episode import Episode


class ScalarSink(Protocol):
    def report(self, step: int, metrics: Mapping[str, float]) -> None: ...
    def close(self) -> None: ...


class EpisodeSink(Protocol):
    def log_episode(self, episode: Episode) -> None: ...
    def close(self) -> None: ...
