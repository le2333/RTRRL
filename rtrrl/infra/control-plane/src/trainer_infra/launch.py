from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from training_sdk import objects
from training_sdk.contract import (
    CONTRACT_VERSION,
    EvaluationConfig,
    EnvironmentConfig,
    LoggingConfig,
    RunConfig,
    Scalar,
    ScoreConfig,
    TrainingConfig,
)

from trainer_infra.preflight import LaunchPlan


@dataclass(frozen=True)
class Launch:
    plan: LaunchPlan
    launch_id: str
    archive: Path
    prefix: str


def create_launch(
    plan: LaunchPlan, archive_root: Path, source: Path, now: datetime
) -> Launch:
    experiment = plan.experiment
    launch_id = now.strftime("%Y%m%d-%H%M%S")
    archive = (
        Path(archive_root) / experiment.experiment / experiment.name / launch_id
    )
    archive.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"{experiment.storage.rstrip('/')}/{experiment.experiment}"
        f"/{experiment.name}/{launch_id}"
    )

    space_payload = {
        key: (list(spec.choices) if hasattr(spec, "choices") else spec.model_dump())
        for key, spec in plan.parameters.overrides.items()
    }
    launch_payload = {
        "contract": CONTRACT_VERSION,
        "experiment": experiment.experiment,
        "name": experiment.name,
        "description": experiment.description,
        "launch_id": launch_id,
        "entry": plan.entry_name,
        "environment": experiment.environment.model_dump(mode="json"),
        "training": experiment.training.model_dump(mode="json"),
        "evaluation": experiment.evaluation.model_dump(mode="json"),
        "digest": plan.digest,
        "queue": plan.queue,
        "job_definition": plan.job_definition,
        "rounds": experiment.hpo.rounds,
        "trials_per_round": experiment.hpo.trials_per_round,
        "parallel_jobs": experiment.hpo.parallel_jobs,
    }
    documents = {
        "experiment.yaml": Path(source).read_bytes(),
        "space.json": json.dumps(space_payload, sort_keys=True).encode(),
        "launch.json": json.dumps(launch_payload, sort_keys=True).encode(),
    }
    for name, payload in documents.items():
        (archive / name).write_bytes(payload)
        objects.put_bytes(f"{prefix}/{name}", payload)
    return Launch(plan=plan, launch_id=launch_id, archive=archive, prefix=prefix)


def config_uri(launch: Launch, trial: int) -> str:
    return f"{launch.prefix}/trials/t{trial}/config.json"


def build_run_config(
    launch: Launch, trial: int, params: Mapping[str, Scalar]
) -> RunConfig:
    experiment = launch.plan.experiment
    trial_prefix = f"{launch.prefix}/trials/t{trial}"
    rerun_s3 = (
        f"{trial_prefix}/episodes/" if experiment.logging.enable_rerun else None
    )
    return RunConfig(
        contract=CONTRACT_VERSION,
        run_id=f"{experiment.name}-{launch.launch_id}-t{trial}",
        experiment=experiment.experiment,
        name=experiment.name,
        launch_id=launch.launch_id,
        trial=trial,
        entry=launch.plan.entry_name,
        digest=launch.plan.digest,
        environment=EnvironmentConfig(
            id=experiment.environment.id,
            backend=experiment.environment.backend,
            seed=experiment.environment.seed,
            episode_length=experiment.environment.episode_length,
            observed=experiment.environment.observed,
        ),
        training=TrainingConfig(
            num_envs=experiment.training.num_envs,
            total_steps=experiment.training.total_steps,
            epoch_steps=experiment.training.epoch_steps,
            chunk_steps=experiment.training.chunk_steps,
            early_stop_patience=experiment.training.early_stop_patience,
        ),
        evaluation=EvaluationConfig(
            steps=experiment.evaluation.steps,
            num_envs=experiment.evaluation.num_envs,
        ),
        params=dict(params),
        logging=LoggingConfig(
            aim=experiment.logging.aim,
            enable_rerun=experiment.logging.enable_rerun,
            rerun_s3=rerun_s3,
            rerun_every_steps=experiment.logging.rerun_every_steps,
        ),
        score=ScoreConfig(
            metric=experiment.score.metric,
            window_steps=experiment.score.window_steps,
            reduce=experiment.score.reduce,
            direction=experiment.score.direction,
            non_finite=experiment.score.non_finite,
            s3=f"{trial_prefix}/score.json",
        ),
    )
