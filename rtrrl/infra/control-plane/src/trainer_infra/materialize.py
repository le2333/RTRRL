from __future__ import annotations

from typing import Mapping

from optuna.trial import Trial

from trainer_infra.identities import canonical_json, canonical_yaml, sha256_text
from trainer_infra.models import ConcreteRun, JsonScalar, ResolvedGroup, freeze_json


def materialize_run(
    group: ResolvedGroup,
    trial: Trial,
    sampled: Mapping[str, JsonScalar],
    run_number: int | None = None,
) -> ConcreteRun:
    sequence = trial.number + 1 if run_number is None else run_number
    if type(sequence) is not int or not 1 <= sequence <= 9999:
        raise ValueError("run_number must be a one-based integer no greater than 9999")

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
    run_name = f"{group.name}-{group.script}-{sequence:04d}"
    run_id = f"{group.study_key}:{sequence:04d}"

    config = {
        "environment": group.environment.model_dump(mode="json"),
        "logging": group.logging.model_dump(mode="json"),
        "parameters": final,
        "training_budget": group.training_budget.model_dump(mode="json"),
    }
    config_yaml = canonical_yaml(config)
    config_sha256 = sha256_text(config_yaml)

    experiment_id = group.study_key.rsplit(":", 1)[0]
    context = {
        "experiment_id": experiment_id,
        "group": group.name,
        "run_id": run_id,
        "run_name": run_name,
        "run_number": sequence,
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
            "final": final,
            "fixed": fixed,
            "sampled": sampled_values,
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
        run_number=sequence,
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
        fixed_parameters=freeze_json(fixed),
        sampled_parameters=freeze_json(sampled_values),
        final_parameters=freeze_json(final),
        context=freeze_json(context),
        config_yaml=config_yaml,
        config_sha256=config_sha256,
        run_json=run_json,
        run_sha256=sha256_text(run_json),
    )
