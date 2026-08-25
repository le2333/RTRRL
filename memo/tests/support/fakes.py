"""In-memory observers and storage used at facility boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeRecorder:
    """Both of Runtime's reporting occasions, kept apart the way they arrive."""

    episodes: list[Any] = field(default_factory=list)
    trajectories: list[Any] = field(default_factory=list)

    def log_episode(self, episode: Any) -> None:
        self.episodes.append(episode)

    def log_trajectory(self, trajectory: Any) -> None:
        self.trajectories.append(trajectory)

    def of(self, phase: str) -> list[Any]:
        return [
            episode for episode in self.episodes if getattr(episode, "phase") == phase
        ]


@dataclass
class ResumableRecorder(EpisodeRecorder):
    """A destination that can be cut back to a snapshot, as an artifact is.

    The metrics artifact is truncated to the byte offset the snapshot named,
    so that the interval the interrupted process reported after its last
    snapshot is reported once rather than twice. This is the same rule in
    memory: what a resumed run adds is compared against what one uninterrupted
    run would have written, and that only means anything if the replayed
    interval is dropped here too.
    """

    def suspend(self) -> tuple[int, int]:
        return len(self.episodes), len(self.trajectories)

    def resume(self, state: tuple[int, int]) -> None:
        episodes, trajectories = state
        del self.episodes[episodes:]
        del self.trajectories[trajectories:]


@dataclass
class TrajectoryRecorder:
    trajectories: list[Any] = field(default_factory=list)
    closed: bool = False

    def log_trajectory(self, trajectory: Any) -> None:
        self.trajectories.append(trajectory)

    def close(self) -> None:
        self.closed = True


@dataclass
class ScalarRecorder:
    reports: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    closed: bool = False

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        self.reports.append((step, dict(metrics)))

    def close(self) -> None:
        self.closed = True


@dataclass
class MemoryObjectStore:
    objects: dict[str, bytes] = field(default_factory=dict)

    def put_bytes(self, uri: str, payload: bytes) -> None:
        self.objects[uri] = payload

    def put_file(self, uri: str, path: Path) -> None:
        self.put_bytes(uri, path.read_bytes())

    def get_bytes(self, uri: str) -> bytes:
        return self.objects[uri]

    def exists(self, uri: str) -> bool:
        return uri in self.objects
