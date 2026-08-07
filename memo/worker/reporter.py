from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from memorax.runtime.episode import Episode, statistics
from worker.contract import RunConfig
from worker.sinks.metrics import MetricsSink

METRICS_FILENAME = "metrics.jsonl"


class Sink(Protocol):
    def report(self, step: int, metrics: Mapping[str, float]) -> None: ...
    def log_episode(self, episode: Episode) -> None: ...
    def close(self) -> None: ...


class Reporter:
    def __init__(
        self,
        config: RunConfig,
        scratch: Path,
        sinks: Sequence[Sink] | None = None,
    ) -> None:
        self.config = config
        self.scratch = Path(scratch)
        metrics = MetricsSink(self.scratch / METRICS_FILENAME)
        self._sinks: tuple[Sink, ...] = (metrics, *(sinks or ()))

    @classmethod
    def from_env(cls) -> "Reporter":
        config_path = Path(os.environ["TRAINER_RUN_CONFIG"])
        scratch = Path(os.environ["TRAINER_SCRATCH"])
        config = RunConfig.model_validate(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        return cls(config, scratch, sinks=build_default_sinks(config, scratch))

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        for sink in self._sinks:
            sink.report(step, metrics)

    def log_episode(self, episode: Episode) -> None:
        # One occasion read two ways. The statistics travel as scalars, dated by
        # the step the episode ended on, and the episode itself travels to
        # whatever sink keeps series. Neither sink has to know about the other.
        self.report(episode.end_env_steps, statistics(episode))
        for sink in self._sinks:
            sink.log_episode(episode)

    def close(self) -> None:
        failure: BaseException | None = None
        for sink in self._sinks:
            try:
                sink.close()
            except BaseException as error:  # noqa: BLE001 - every sink must be closed
                failure = failure or error
        if failure is not None:
            raise failure

    def __enter__(self) -> "Reporter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_default_sinks(config: RunConfig, scratch: Path) -> tuple[Sink, ...]:
    from worker.sinks.aim import AimSink
    from worker.sinks.rerun import RerunSink

    sinks: list[Sink] = [AimSink(config, repo=config.logging.aim)]
    if config.logging.enable_rerun:
        sinks.append(RerunSink(config, scratch))
    return tuple(sinks)
