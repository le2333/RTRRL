from __future__ import annotations

from pathlib import Path

from training_sdk import RunContext
from training_sdk.execution import (
    CanonicalRecord,
    CompletionMarker,
    JobBundle,
    JobQuery,
    RunBundle,
    require_canonical_image_digest,
)

from trainer_infra.models import ConcreteRun, thaw_json

__all__ = [
    "CanonicalRecord",
    "CompletionMarker",
    "JobBundle",
    "JobQuery",
    "RunBundle",
    "build_run_context",
]


def build_run_context(
    experiment_name: str,
    experiment_id: str,
    group: str,
    concrete_run: ConcreteRun,
    artifact_prefix: str | Path,
) -> RunContext:
    if not experiment_name or not experiment_id:
        raise ValueError("experiment_name and experiment_id must be nonempty")
    if not group:
        raise ValueError("group must be nonempty")
    expected_study_key = f"{experiment_id}:{group}"
    if concrete_run.study_key != expected_study_key:
        raise ValueError(
            f"study_key {concrete_run.study_key!r} does not match experiment_id/group "
            f"identity {expected_study_key!r}"
        )
    expected_run_id = f"{concrete_run.study_key}:{concrete_run.run_number:04d}"
    if concrete_run.run_id != expected_run_id:
        raise ValueError(
            f"run_id {concrete_run.run_id!r} does not match {expected_run_id!r}"
        )
    context_group = concrete_run.context.get("group")
    context_run_id = concrete_run.context.get("run_id")
    if context_group != group or context_run_id != concrete_run.run_id:
        raise ValueError("concrete run context identity does not match study_key/run_id")
    seed = concrete_run.final_parameters.get("seed")
    if type(seed) is not int:
        raise ValueError("concrete run requires an integer seed")
    image_digest = require_canonical_image_digest(concrete_run.image)
    return RunContext(
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        group=group,
        script=concrete_run.script,
        run_id=concrete_run.run_id,
        run_number=concrete_run.run_number,
        trial_number=concrete_run.trial_number,
        seed=seed,
        metadata=thaw_json(concrete_run.metadata),
        environment=concrete_run.environment.model_dump(mode="json"),
        training_budget=concrete_run.training_budget.model_dump(mode="json"),
        fixed_parameters=thaw_json(concrete_run.fixed_parameters),
        sampled_parameters=thaw_json(concrete_run.sampled_parameters),
        final_parameters=thaw_json(concrete_run.final_parameters),
        image_digest=image_digest,
        resource_profile=concrete_run.resources.profile,
        artifact_directory=Path(artifact_prefix),
        logging=concrete_run.logging.model_dump(mode="json"),
        objective=concrete_run.objective.model_dump(mode="json"),
    )
