"""Aim scalar storage with no deployment-contract dependency."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aim.sdk.run import Run

from ..metadata import RunMetadata


class AimSink:
    """One Aim run per training run, including one that was interrupted.

    The run is opened when there is something to put in it rather than when
    this object is built, which is what lets a resumed process continue the
    *same* Aim run instead of standing a second one beside it. Two runs for
    one job would divide every curve in the dashboard at the point of the
    interruption and give the launch a member that is not a member.

    What Aim cannot be asked to do is forget. Between the last snapshot and
    the interruption the previous process kept reporting, and those points
    stay where they are; the resumed process reports the same interval again
    and the dashboard holds both. The complete record is the metrics artifact,
    which is cut back to the snapshot exactly, and this is the dashboard.
    """

    def __init__(
        self,
        endpoint: str,
        metadata: RunMetadata,
        *,
        parameters: Mapping[str, Any],
    ) -> None:
        self._endpoint = endpoint
        self._metadata = metadata
        self._parameters = dict(parameters)
        self._hash: str | None = None
        self._opened: Run | None = None

    def _run(self) -> Run:
        if self._opened is not None:
            return self._opened
        metadata = self._metadata
        if self._hash is not None:
            # Reattaching by hash rather than by name: a name is what the
            # dashboard shows and nothing stops two runs from carrying one.
            run = Run(run_hash=self._hash, repo=self._endpoint)
            self._opened = run
            return run
        run = Run(repo=self._endpoint, experiment=metadata.experiment)
        run.name = metadata.run_id
        run["launch_id"] = metadata.launch_id
        run["trial"] = metadata.trial
        # Without these a formal launch is ten indistinguishable curves of one
        # trial, and nothing on the dashboard says which of them may be read
        # as a result.
        run["seed"] = metadata.seed
        run["role"] = metadata.role
        run["entry"] = metadata.entry
        run["digest"] = metadata.digest
        run["params"] = dict(self._parameters)
        self._hash = run.hash
        self._opened = run
        return run

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        run = self._run()
        for name, value in metrics.items():
            run.track(float(value), name=str(name), step=int(step))

    def suspend(self) -> str | None:
        """Which Aim run this is, so the next process writes into it."""

        return self._hash

    def resume(self, state: str | None) -> None:
        if self._opened is not None:
            raise ValueError("the Aim run was opened before it was resumed")
        self._hash = state

    def close(self) -> None:
        if self._opened is not None:
            self._opened.close()
            self._opened = None
