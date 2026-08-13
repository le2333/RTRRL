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

from ._contract import RunSpec


def load_run() -> tuple[RunSpec, Path]:
    config_path = Path(os.environ["TRAINER_RUN_CONFIG"])
    scratch = Path(os.environ["TRAINER_SCRATCH"])
    config = RunSpec.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    return config, scratch


def build_reporter(config: RunSpec, scratch: Path) -> Reporter:
    identity = config.identity
    artifacts = Path(scratch) / "artifacts"
    metadata = RunMetadata(
        run_id=identity.run_id,
        experiment=identity.experiment,
        launch_id=identity.launch_id,
        trial=identity.trial,
        entry=config.entry,
        digest=identity.digest,
    )
    scalar_sinks = [
        MetricsSink(artifacts / METRICS_FILENAME),
        AimSink(
            config.logging.aim.url,
            metadata,
            parameters=config.algorithm.parameters,
        ),
    ]
    episode_sinks = []
    if config.logging.rerun is not None:
        episode_sinks.append(
            RerunSink(
                artifacts / "rerun",
                every_steps=config.logging.rerun.every_steps,
                num_envs=config.algorithm.num_envs,
                metadata=metadata,
            )
        )
    return Reporter(scalar_sinks=scalar_sinks, episode_sinks=episode_sinks)
