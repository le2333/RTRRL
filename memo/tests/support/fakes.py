"""In-memory observers and storage used at facility boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeRecorder:
    episodes: list[Any] = field(default_factory=list)

    def log_episode(self, episode: Any) -> None:
        self.episodes.append(episode)

    def of(self, phase: str) -> list[Any]:
        return [
            episode for episode in self.episodes if getattr(episode, "phase") == phase
        ]


@dataclass
class ScalarRecorder:
    reports: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    closed: bool = False

    def report(self, step: int, metrics: dict[str, float]) -> None:
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
