from __future__ import annotations

from collections.abc import Mapping

from aim import Run

from training_sdk.contract import RunConfig


def close_aim_run(run: Run) -> None:
    """Close an Aim run, patching Aim 3.28 read-only ``sequence_infos`` bug."""
    tracker = run._tracker
    if not hasattr(tracker, "sequence_infos"):
        tracker.sequence_infos = {}
    run.close()


class AimSink:
    def __init__(self, config: RunConfig, repo: str) -> None:
        self._run = Run(repo=repo, experiment=config.experiment)
        self._run.name = config.run_id
        self._run["launch_id"] = config.launch_id
        self._run["trial"] = config.trial
        self._run["entry"] = config.entry
        self._run["digest"] = config.digest
        self._run["source_hash"] = config.source_hash
        self._run["params"] = dict(config.params)
        self._every = config.logging.every_steps
        self._last: int | None = None

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        # Reports at a step already recorded always go through. A caller that
        # splits one step across several reports -- training metrics and then
        # evaluation metrics for the same epoch -- is not reporting too often,
        # and dropping the second one loses a whole family of metrics silently.
        if self._last is not None and step != self._last and step - self._last < self._every:
            return
        self._last = step
        for name, value in metrics.items():
            self._run.track(float(value), name=str(name), step=int(step))

    def log_episode(self, episode: object) -> None:
        return None

    def close(self) -> None:
        self._run.close()
