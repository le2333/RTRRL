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
        self._run["params"] = dict(config.params)

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        # Everything reported is kept. The thinning is the episode itself: two
        # streams that finished at the same step are two answers about it, and
        # a stride over an already-reduced number thinned nothing that was not
        # already thin.
        for name, value in metrics.items():
            self._run.track(float(value), name=str(name), step=int(step))

    def log_episode(self, episode: object) -> None:
        return None

    def close(self) -> None:
        self._run.close()
