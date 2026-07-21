from __future__ import annotations

import copy
import math
from types import MappingProxyType
from typing import Any, Mapping

from trainer_infra.models import (
    ContinuousSearch,
    DiscreteDomain,
    DiscreteSearch,
    ExperimentSpec,
    FieldConstraints,
    FieldDescriptor,
    JsonScalar,
    ParameterDomain,
    ParameterPolicy,
    ResolvedConfiguration,
    ResolvedExperiment,
    ResolvedGroup,
    ResolvedParameter,
    ScriptCatalog,
    SearchDomain,
    freeze_json,
)


def merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key == "parameters":
            params = merged.setdefault("parameters", {})
            for field_name, domain in value.items():
                params[field_name] = copy.deepcopy(domain)
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _matches_type(value: JsonScalar, declared_type: str) -> bool:
    if declared_type == "bool":
        return isinstance(value, bool)
    if declared_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared_type == "str":
        return isinstance(value, str)
    return False


def _satisfies_constraints(value: JsonScalar, constraints: FieldConstraints) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return all(
            bound is None
            for bound in (constraints.gt, constraints.ge, constraints.lt, constraints.le)
        )
    return (
        (constraints.gt is None or value > constraints.gt)
        and (constraints.ge is None or value >= constraints.ge)
        and (constraints.lt is None or value < constraints.lt)
        and (constraints.le is None or value <= constraints.le)
    )


def _validate_value(
    value: JsonScalar,
    descriptor: FieldDescriptor,
    *,
    group_name: str,
    field_name: str,
) -> None:
    if not _matches_type(value, descriptor.type):
        raise ValueError(
            f"group '{group_name}' field '{field_name}' must have type {descriptor.type}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"group '{group_name}' field '{field_name}' must be finite")
    if not _satisfies_constraints(value, descriptor.constraints):
        raise ValueError(
            f"group '{group_name}' field '{field_name}' violates declared constraints"
        )
    if descriptor.choices is not None and not any(
        value == choice for choice in descriptor.choices
    ):
        raise ValueError(
            f"group '{group_name}' field '{field_name}' must be one of "
            f"{list(descriptor.choices)!r}"
        )


def _to_search(
    domain: ParameterDomain,
    descriptor: FieldDescriptor,
    *,
    group_name: str,
    field_name: str,
) -> SearchDomain:
    if not descriptor.searchable:
        raise ValueError(f"group '{group_name}' field '{field_name}' is not searchable")
    if isinstance(domain, DiscreteDomain):
        for value in domain.values:
            _validate_value(
                value,
                descriptor,
                group_name=group_name,
                field_name=field_name,
            )
        return DiscreteSearch(tuple(domain.values))

    if descriptor.choices is not None:
        raise ValueError(
            f"group '{group_name}' field '{field_name}' has choices and only accepts "
            "a discrete domain or singleton value"
        )
    if descriptor.type not in ("int", "float"):
        raise ValueError(
            f"group '{group_name}' field '{field_name}' cannot use a continuous domain"
        )
    _validate_value(
        domain.min,
        descriptor,
        group_name=group_name,
        field_name=field_name,
    )
    _validate_value(
        domain.max,
        descriptor,
        group_name=group_name,
        field_name=field_name,
    )
    integer = descriptor.type == "int"
    if integer and (not domain.min.is_integer() or not domain.max.is_integer()):
        raise ValueError(f"group '{group_name}' field '{field_name}' requires integer bounds")
    return ContinuousSearch(
        low=domain.min,
        high=domain.max,
        log=domain.scale == "log",
        integer=integer,
        step=1 if integer else None,
    )


def _resolve_parameter(
    descriptor: FieldDescriptor,
    domain: ParameterDomain | None,
    policy: ParameterPolicy,
    *,
    group_name: str,
    field_name: str,
) -> ResolvedParameter:
    if domain is not None:
        if isinstance(domain, DiscreteDomain) and len(domain.values) == 1:
            value = domain.values[0]
            _validate_value(
                value,
                descriptor,
                group_name=group_name,
                field_name=field_name,
            )
            return ResolvedParameter(fixed_value=value, search_domain=None)
        return ResolvedParameter(
            fixed_value=None,
            search_domain=_to_search(
                domain,
                descriptor,
                group_name=group_name,
                field_name=field_name,
            ),
        )

    if descriptor.searchable and policy == ParameterPolicy.SCAN_UNFIXED:
        if descriptor.default_search is None:
            raise ValueError(
                f"group '{group_name}' field '{field_name}' has no finite default search domain"
            )
        search = _to_search(
            descriptor.default_search,
            descriptor,
            group_name=group_name,
            field_name=field_name,
        )
        if isinstance(search, DiscreteSearch) and len(search.values) == 1:
            return ResolvedParameter(fixed_value=search.values[0], search_domain=None)
        return ResolvedParameter(fixed_value=None, search_domain=search)

    _validate_value(
        descriptor.default,
        descriptor,
        group_name=group_name,
        field_name=field_name,
    )
    return ResolvedParameter(fixed_value=descriptor.default, search_domain=None)


def _resolve_parameters(
    configured: Mapping[str, ParameterDomain],
    fields: Mapping[str, FieldDescriptor],
    policy: ParameterPolicy,
    *,
    group_name: str,
) -> Mapping[str, ResolvedParameter]:
    unknown = configured.keys() - fields.keys()
    if unknown:
        field_name = sorted(unknown)[0]
        raise ValueError(f"group '{group_name}' has unknown field '{field_name}'")
    resolved = {
        field_name: _resolve_parameter(
            descriptor,
            configured.get(field_name),
            policy,
            group_name=group_name,
            field_name=field_name,
        )
        for field_name, descriptor in fields.items()
    }
    return MappingProxyType(resolved)


def resolve_experiment(
    spec: ExperimentSpec,
    catalogs: Mapping[str, ScriptCatalog],
) -> ResolvedExperiment:
    resolved_groups: list[ResolvedGroup] = []
    defaults = spec.defaults.model_dump(exclude_none=True)
    experiment_metadata = copy.deepcopy(spec.experiment.metadata)

    for group_name, group in spec.groups.items():
        image = group.image or spec.defaults.image
        if "image" in group.overrides:
            image = group.overrides["image"]
        if not isinstance(image, str):
            raise ValueError(f"group '{group_name}' has invalid image reference")
        catalog = catalogs.get(image)
        if catalog is None:
            raise ValueError(f"no catalog for exact image reference '{image}'")
        descriptor = catalog.scripts.get(group.script)
        if descriptor is None:
            raise ValueError(f"group '{group_name}' has unknown script '{group.script}'")

        merged = merge_mapping(descriptor.defaults.model_dump(), defaults)
        group_fields = group.model_dump(
            exclude={"script", "metadata", "overrides"},
            exclude_none=True,
        )
        merged = merge_mapping(merged, group_fields)
        override_values = copy.deepcopy(group.overrides)
        override_metadata = override_values.pop("metadata", {})
        merged = merge_mapping(merged, override_values)
        try:
            configuration = ResolvedConfiguration.model_validate(merged)
        except ValueError as error:
            raise ValueError(f"group '{group_name}' has invalid configuration: {error}") from error
        if configuration.environment.name not in descriptor.environments:
            raise ValueError(
                f"group '{group_name}' script '{descriptor.name}' does not support "
                f"environment '{configuration.environment.name}'"
            )

        metadata = merge_mapping(experiment_metadata, group.metadata)
        metadata = merge_mapping(metadata, override_metadata)
        parameters = _resolve_parameters(
            configuration.parameters,
            descriptor.fields,
            configuration.hpo.parameter_policy,
            group_name=group_name,
        )
        resolved_groups.append(
            ResolvedGroup(
                name=group_name,
                study_key=f"{spec.experiment.name}:{group_name}",
                image=configuration.image,
                script=group.script,
                argv=tuple(descriptor.argv),
                sdk_protocol_version=descriptor.sdk_protocol_version,
                objective=descriptor.objective,
                environment=configuration.environment,
                training_budget=configuration.training_budget,
                logging=configuration.logging,
                resources=configuration.resources,
                hpo=configuration.hpo,
                execution=configuration.execution,
                metadata=freeze_json(metadata),
                parameters=parameters,
                parameter_paths=MappingProxyType(
                    {
                        field_name: field_descriptor.path
                        for field_name, field_descriptor in descriptor.fields.items()
                    }
                ),
            )
        )

    return ResolvedExperiment(
        name=spec.experiment.name,
        description=spec.experiment.description,
        metadata=freeze_json(experiment_metadata),
        groups=tuple(resolved_groups),
    )
