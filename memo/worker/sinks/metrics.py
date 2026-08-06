from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import IO


class MetricsSink:
    """Append-only record of every report; also the worker's heartbeat.

    The file appears with the first report rather than with the reporter, so its
    presence means the run produced at least one metric. A run that dies during
    startup leaves nothing behind to be mistaken for a run that produced no
    numbers. The worker's heartbeat already treats a missing file as silence, so
    it reads the startup grace period either way.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._handle: IO[str] | None = None

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        line = json.dumps(
            {
                "step": int(step),
                "metrics": {str(k): float(v) for k, v in metrics.items()},
            },
            sort_keys=True,
        )
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
        self._handle.write(line + "\n")
        self._handle.flush()

    def log_episode(self, episode: object) -> None:
        return None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
