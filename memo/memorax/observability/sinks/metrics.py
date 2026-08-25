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

    def suspend(self) -> int:
        """How much of the record had been written when the run was suspended.

        A byte offset and not a row count, because that is what putting the
        file back needs and it costs no walk of the file to take. Every row is
        flushed as it is written, so the size on disk is the record.
        """

        if self._handle is not None:
            self._handle.flush()
        return self._path.stat().st_size if self._path.exists() else 0

    def resume(self, state: int) -> None:
        """Cut the record back to the snapshot, so no row is written twice.

        Between the last snapshot and the interruption the previous process
        kept reporting, and those rows are about episodes the resumed run is
        going to live through again. Appending to them would put two lines in
        the run's complete record for one episode, which is the one thing this
        artifact may not contain.

        A file shorter than the snapshot says is a file that did not come back
        with the snapshot. Continuing would leave a hole in the middle of the
        record with nothing marking it, so it is refused here instead.
        """

        offset = int(state)
        if self._handle is not None:
            raise ValueError("the metrics artifact was written before it was resumed")
        size = self._path.stat().st_size if self._path.exists() else 0
        if size < offset:
            raise ValueError(
                f"{self._path} holds {size} bytes and the snapshot was taken at "
                f"{offset}; the artifact did not come back with the snapshot"
            )
        if size > offset:
            with self._path.open("r+b") as handle:
                handle.truncate(offset)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
