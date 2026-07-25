from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import IO


class MetricsSink:
    """Append-only record of every report; also the worker's heartbeat."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] = self._path.open("a", encoding="utf-8")

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        line = json.dumps(
            {"step": int(step), "metrics": {str(k): float(v) for k, v in metrics.items()}},
            sort_keys=True,
        )
        self._handle.write(line + "\n")
        self._handle.flush()

    def log_episode(self, episode: object) -> None:
        return None

    def close(self) -> None:
        self._handle.close()
