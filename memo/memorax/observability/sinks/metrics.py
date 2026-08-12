"""Mandatory append-only scalar artifact for one run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import IO

METRICS_FILENAME = "metrics.jsonl"


class MetricsSink:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._handle: IO[str] | None = None

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        line = json.dumps(
            {
                "step": int(step),
                "metrics": {str(key): float(value) for key, value in metrics.items()},
            },
            sort_keys=True,
        )
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
