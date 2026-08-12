"""Project the deployment run onto Memorax observation backends."""

from __future__ import annotations

import json
import os
from pathlib import Path

from memorax.observability import Reporter, RunMetadata
from memorax.observability.sinks import (
    METRICS_FILENAME,
    AimSink,
    MetricsSink,
    RerunSink,
)
from worker.contract import RunConfig


def load_run() -> tuple[RunConfig, Path]:
    config_path = Path(os.environ["TRAINER_RUN_CONFIG"])
    scratch = Path(os.environ["TRAINER_SCRATCH"])
    config = RunConfig.model_validate(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    return config, scratch


def build_reporter(config: RunConfig, scratch: Path) -> Reporter:
    metadata = RunMetadata(
        run_id=config.run_id,
        experiment=config.experiment,
        launch_id=config.launch_id,
        trial=config.trial,
        entry=config.entry,
        digest=config.digest,
    )
    scalar_sinks = [
        MetricsSink(Path(scratch) / METRICS_FILENAME),
        AimSink(config.logging.aim, metadata, parameters=config.params),
    ]
    episode_sinks = []
    if config.logging.enable_rerun:
        every_steps = config.logging.rerun_every_steps
        if every_steps is None:
            raise ValueError("enabled Rerun logging requires rerun_every_steps")
        episode_sinks.append(
            RerunSink(
                scratch,
                every_steps=every_steps,
                num_envs=config.training.num_envs,
                metadata=metadata,
            )
        )
    return Reporter(scalar_sinks=scalar_sinks, episode_sinks=episode_sinks)
