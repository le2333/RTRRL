"""Aim scalar storage with no deployment-contract dependency."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aim.sdk.run import Run

from ..metadata import RunMetadata


class AimSink:
    def __init__(
        self,
        endpoint: str,
        metadata: RunMetadata,
        *,
        parameters: Mapping[str, Any],
    ) -> None:
        self._run = Run(repo=endpoint, experiment=metadata.experiment)
        self._run.name = metadata.run_id
        self._run["launch_id"] = metadata.launch_id
        self._run["trial"] = metadata.trial
        self._run["entry"] = metadata.entry
        self._run["digest"] = metadata.digest
        self._run["params"] = dict(parameters)

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            self._run.track(float(value), name=str(name), step=int(step))

    def close(self) -> None:
        self._run.close()
