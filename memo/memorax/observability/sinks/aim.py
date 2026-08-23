"""Aim scalar storage with no deployment-contract dependency."""

from __future__ import annotations

from collections.abc import Mapping

from aim.sdk.run import Run

from ..metadata import RunMetadata


class AimSink:
    def __init__(self, endpoint: str, metadata: RunMetadata) -> None:
        self._run = Run(repo=endpoint, experiment=metadata.experiment)
        # The run name is the configured human-facing identity. Deployment
        # metadata and sampled parameters remain in worker artifacts, not in
        # the dashboard where they would expose implementation details.
        self._run.name = metadata.run_id

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            self._run.track(float(value), name=str(name), step=int(step))

    def close(self) -> None:
        self._run.close()
