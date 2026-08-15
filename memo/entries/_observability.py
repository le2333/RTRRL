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
from ._schedule import training_every_steps


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
    # The metrics artifact is the run's complete record, so it is neither
    # optional nor sampled. Aim is the dashboard: every evaluation, and
    # training only as often as the document asked for.
    scalar_sinks = [MetricsSink(artifacts / METRICS_FILENAME)]
    sampled_sinks = [
        AimSink(
            config.logging.aim.url,
            metadata,
            parameters=config.algorithm.parameters,
        )
    ]
    trajectory_sinks = []
    if config.logging.rerun is not None:
        trajectory_sinks.append(RerunSink(artifacts / "rerun", metadata=metadata))
    return Reporter(
        scalar_sinks=scalar_sinks,
        sampled_sinks=sampled_sinks,
        trajectory_sinks=trajectory_sinks,
        training_every_steps=training_every_steps(config),
    )
