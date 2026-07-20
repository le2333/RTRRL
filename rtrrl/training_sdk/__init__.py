from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from .context import JsonValue, RunContext
from .types import Episode


class TrainingRun(Protocol):
    def log_metrics(self, env_steps: int, metrics: Mapping[str, float]) -> None: ...

    def log_episode_summary(
        self,
        *,
        env_steps: int,
        episode_return: float,
        episode_length: int,
    ) -> None: ...

    def log_episode(self, episode: Episode) -> None: ...

    def register_checkpoint(self, path: Path) -> None: ...

    def finish(self, final_metrics: Mapping[str, float]) -> None: ...


_current_run: TrainingRun | None = None


def current_run() -> TrainingRun:
    if _current_run is None:
        raise RuntimeError("no training run has been initialized")
    return _current_run


__all__ = [
    "Episode",
    "JsonValue",
    "RunContext",
    "TrainingRun",
    "current_run",
]
