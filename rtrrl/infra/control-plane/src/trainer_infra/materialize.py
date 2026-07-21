from __future__ import annotations

from typing import Mapping

from optuna.trial import Trial

from trainer_infra.identities import canonical_json, canonical_yaml, sha256_text
from trainer_infra.models import (
    ConcreteRun,
    JsonScalar,
    ResolvedGroup,
    freeze_json,
    thaw_json,
)


def _set_parameter_path(target: dict, path: str, value: JsonScalar) -> None:
    segments = path.split(".")
    if any(
        not segment.isidentifier() or segment.startswith("__")
        for segment in segments
    ):
        raise ValueError(f"unsafe parameter path {path!r}")
    current = target
    for segment in segments[:-1]:
        existing = current.get(segment)
        if existing is None:
            nested: dict = {}
            current[segment] = nested
            current = nested
        elif isinstance(existing, dict):
            current = existing
        else:
            raise ValueError(f"parameter path conflict at {path!r}")
    leaf = segments[-1]
    if leaf in current:
        raise ValueError(f"parameter path conflict at {path!r}")
    current[leaf] = value


def _materialized_parameters(
    values: Mapping[str, JsonScalar],
    paths: Mapping[str, str],
) -> dict:
    result: dict = {}
    for name, value in values.items():
        _set_parameter_path(result, paths.get(name, name), value)
    return result


def materialize_run(
    group: ResolvedGroup,
    trial: Trial,
    sampled: Mapping[str, JsonScalar],
    *,
    run_number: int,
) -> ConcreteRun:
    if type(run_number) is not int:
        raise TypeError("run_number must be an integer")
    if not 1 <= run_number <= 9999:
        raise ValueError("run_number must be between 1 and 9999")

    fixed = dict(group.fixed_parameters)
    for name, value in fixed.items():
        if name in sampled and sampled[name] != value:
            raise ValueError(f"sampled values override fixed parameter '{name}'")
    sampled_values = {}
    for name in group.searchable_parameters():
        if name not in sampled:
            raise ValueError(f"missing searchable parameter '{name}'")
        sampled_values[name] = sampled[name]
    final = {**fixed, **sampled_values}
    fixed_config = _materialized_parameters(fixed, group.parameter_paths)
    sampled_config = _materialized_parameters(
        sampled_values, group.parameter_paths
    )
    final_config = _materialized_parameters(final, group.parameter_paths)
    run_name = f"{group.name}-{run_number:04d}"
    run_id = f"{group.study_key}:{run_number:04d}"

    environment = {
        "name": group.environment.name,
        "options": thaw_json(group.environment.options),
    }
    config = {
        "protocol_version": group.sdk_protocol_version,
        "environment": environment,
        "logging": group.logging.model_dump(mode="json"),
        "parameters": final_config,
        "training_budget": group.training_budget.model_dump(mode="json"),
    }
    config_yaml = canonical_yaml(config)
    config_sha256 = sha256_text(config_yaml)

    experiment_id = group.study_key.rsplit(":", 1)[0]
    context = {
        "environment": environment,
        "experiment_id": experiment_id,
        "group": group.name,
        "run_id": run_id,
        "run_name": run_name,
        "run_number": run_number,
        "script": group.script,
        "trial_number": trial.number,
    }
    manifest = {
        "argv": list(group.argv),
        "config_sha256": config_sha256,
        "context": context,
        "execution": group.execution.model_dump(mode="json"),
        "hpo": group.hpo.model_dump(mode="json"),
        "image": group.image,
        "logging": group.logging.model_dump(mode="json"),
        "metadata": group.metadata,
        "objective": group.objective.model_dump(mode="json"),
        "parameters": {
            "final": final_config,
            "fixed": fixed_config,
            "sampled": sampled_config,
        },
        "resources": group.resources.model_dump(mode="json"),
        "sdk_protocol_version": group.sdk_protocol_version,
        "training_budget": group.training_budget.model_dump(mode="json"),
    }
    run_json = canonical_json(manifest)

    return ConcreteRun(
        study_key=group.study_key,
        run_id=run_id,
        run_name=run_name,
        run_number=run_number,
        trial_number=trial.number,
        image=group.image,
        script=group.script,
        argv=group.argv,
        sdk_protocol_version=group.sdk_protocol_version,
        objective=group.objective,
        environment=group.environment,
        training_budget=group.training_budget,
        logging=group.logging,
        resources=group.resources,
        hpo=group.hpo,
        execution=group.execution,
        metadata=freeze_json(dict(group.metadata)),
        fixed_parameters=freeze_json(fixed_config),
        sampled_parameters=freeze_json(sampled_config),
        final_parameters=freeze_json(final_config),
        context=freeze_json(context),
        config_yaml=config_yaml,
        config_sha256=config_sha256,
        run_json=run_json,
        run_sha256=sha256_text(run_json),
    )
