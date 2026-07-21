import json
from pathlib import Path

import pytest

from trainer_infra.models import (
    ContinuousSearch,
    DiscreteSearch,
    EnvironmentSpec,
    ExperimentSpec,
    ScriptCatalog,
)
from trainer_infra.resolve import resolve_experiment

IMAGE = "repo/image@sha256:" + "a" * 64
OVERRIDE_IMAGE = "repo/image@sha256:" + "b" * 64


def test_environment_is_generic_and_immutable() -> None:
    spec = EnvironmentSpec(
        name="memory_chain",
        options={"length": 75, "nested": {"observe": ["query"]}},
    )

    with pytest.raises(TypeError):
        spec.options["nested"]["observe"][0] = "answer"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), Path("x"), {1, 2}])
def test_environment_options_reject_non_json_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="JSON|finite"):
        EnvironmentSpec(name="memory_chain", options={"bad": value})


@pytest.mark.parametrize(
    ("options", "path"),
    [
        ({"nested": {"bad": float("nan")}}, r"options\.nested\.bad"),
        ({"nested": [{"bad": Path("x")}]}, r"options\.nested\[0\]\.bad"),
    ],
)
def test_environment_option_errors_include_nested_path(
    options: dict[str, object],
    path: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=path):
        EnvironmentSpec(name="memory_chain", options=options)


def catalog_data() -> dict[str, object]:
    return {
        "protocol_version": "1",
        "scripts": {
            "rtrrl": {
                "name": "rtrrl",
                "argv": ["python", "-m", "train"],
                "sdk_protocol_version": "1",
                "defaults": {
                    "environment": {
                        "name": "brax-hopper",
                        "options": {
                            "backend": "spring",
                            "observation_mode": "P",
                            "max_episode_steps": 1000,
                        },
                    },
                    "training_budget": {"env_steps": 2_000_000},
                    "logging": {
                        "aim_every_env_steps": 10_000,
                        "rerun_every_episodes": 100,
                    },
                },
                "objective": {
                    "metric": "reward",
                    "direction": "maximize",
                    "reduction": "last",
                },
                "environments": ["brax-hopper"],
                "fields": {
                    "seed": {
                        "path": "seed",
                        "type": "int",
                        "default": 0,
                        "searchable": False,
                        "constraints": {"ge": 0},
                    },
                    "topology": {
                        "path": "topology",
                        "type": "str",
                        "default": "shared",
                        "searchable": True,
                        "default_search": {"values": ["shared", "dual"]},
                        "choices": ["shared", "dual"],
                    },
                    "learning_rate": {
                        "path": "optimizer.learning_rate",
                        "type": "float",
                        "default": 0.001,
                        "searchable": True,
                        "constraints": {"gt": 0},
                        "default_search": {
                            "min": 1e-5,
                            "max": 1e-2,
                            "scale": "log",
                        },
                    },
                },
            }
        },
    }


@pytest.fixture
def catalog() -> ScriptCatalog:
    return ScriptCatalog.model_validate(catalog_data())


@pytest.mark.parametrize("environments", [[], ["brax-hopper", "brax-hopper"]])
def test_descriptor_requires_non_empty_unique_environments(
    environments: list[str],
) -> None:
    data = catalog_data()
    data["scripts"]["rtrrl"]["environments"] = environments  # type: ignore[index]

    with pytest.raises(ValueError, match="environments.*empty|duplicate environment"):
        ScriptCatalog.model_validate(data)


@pytest.mark.parametrize(
    "choices",
    [
        (float("nan"),),
        (float("inf"),),
        (True, 1),
        (1, 1.0),
        ("shared", "shared"),
    ],
)
def test_field_choices_are_finite_and_unique(choices: tuple[object, ...]) -> None:
    data = catalog_data()
    data["scripts"]["rtrrl"]["fields"]["topology"]["choices"] = choices  # type: ignore[index]

    with pytest.raises(ValueError, match="choices.*finite|duplicate choice"):
        ScriptCatalog.model_validate(data)


@pytest.mark.parametrize(
    ("field_key", "value"),
    [
        ("default", "unsupported"),
        ("default_search", {"values": ["shared", "unsupported"]}),
    ],
)
def test_descriptor_choice_restrictions_cover_defaults_and_default_search(
    field_key: str,
    value: object,
) -> None:
    data = catalog_data()
    data["scripts"]["rtrrl"]["fields"]["topology"][field_key] = value  # type: ignore[index]

    with pytest.raises(ValueError, match="topology|choice|choices"):
        ScriptCatalog.model_validate(data)


def resolve_one_group(
    catalog: ScriptCatalog,
    *,
    policy: str,
    parameters: dict[str, object],
):
    spec = ExperimentSpec.model_validate(
        {
            "experiment": {"name": "hopper"},
            "defaults": {
                "image": IMAGE,
                "resources": {"profile": "g6x"},
                "hpo": {
                    "total_trials": 5,
                    "configs_per_batch": 2,
                    "parameter_policy": policy,
                },
                "execution": {"runs_per_job": 2},
            },
            "groups": {"shared": {"script": "rtrrl", "parameters": parameters}},
        }
    )
    return resolve_experiment(spec, {IMAGE: catalog}).groups[0]


def test_group_is_independent_and_defaults_are_resolved(catalog: ScriptCatalog) -> None:
    spec = ExperimentSpec.model_validate(
        {
            "experiment": {"name": "hopper"},
            "defaults": {
                "image": IMAGE,
                "resources": {"profile": "g6x"},
                "hpo": {"total_trials": 5, "configs_per_batch": 2},
                "execution": {"runs_per_job": 2},
                "parameters": {"seed": {"values": [7]}},
            },
            "groups": {
                "shared": {
                    "script": "rtrrl",
                    "parameters": {"topology": {"values": ["shared"]}},
                },
                "dual": {
                    "script": "rtrrl",
                    "parameters": {"topology": {"values": ["dual"]}},
                },
            },
        }
    )

    resolved = resolve_experiment(spec, {IMAGE: catalog})

    assert [group.name for group in resolved.groups] == ["shared", "dual"]
    assert resolved.groups[0].study_key != resolved.groups[1].study_key
    assert resolved.groups[0].parameters["seed"].fixed_value == 7
    assert resolved.groups[0].environment.name == "brax-hopper"
    assert resolved.groups[0].environment.options["backend"] == "spring"
    assert resolved.groups[0].image == IMAGE


def test_descriptor_rejects_unlisted_environment(catalog: ScriptCatalog) -> None:
    spec = ExperimentSpec.model_validate(
        {
            "experiment": {"name": "hopper"},
            "defaults": {
                "image": IMAGE,
                "resources": {"profile": "g6x"},
                "hpo": {"total_trials": 1, "configs_per_batch": 1},
                "execution": {"runs_per_job": 1},
            },
            "groups": {
                "shared": {
                    "script": "rtrrl",
                    "environment": {"name": "unknown", "options": {}},
                }
            },
        }
    )

    with pytest.raises(ValueError, match="rtrrl.*unknown"):
        resolve_experiment(spec, {IMAGE: catalog})


def test_scan_unfixed_uses_default_search(catalog: ScriptCatalog) -> None:
    group = resolve_one_group(catalog, policy="scan_unfixed", parameters={})

    assert group.parameters["learning_rate"].search_domain == ContinuousSearch(
        low=1e-5, high=1e-2, log=True, integer=False, step=None
    )


def test_experiment_domain_replaces_default_search(catalog: ScriptCatalog) -> None:
    group = resolve_one_group(
        catalog,
        policy="scan_unfixed",
        parameters={"learning_rate": {"min": 1e-7, "max": 0.5, "scale": "log"}},
    )

    assert group.parameters["learning_rate"].search_domain == ContinuousSearch(
        low=1e-7, high=0.5, log=True, integer=False, step=None
    )


def test_explicit_scan_fixes_omitted_fields_and_scans_multi_value_domains(
    catalog: ScriptCatalog,
) -> None:
    group = resolve_one_group(
        catalog,
        policy="explicit_scan",
        parameters={"topology": {"values": ["shared", "dual"]}},
    )

    assert group.parameters["learning_rate"].fixed_value == 0.001
    assert group.parameters["learning_rate"].search_domain is None
    assert group.parameters["topology"].search_domain == DiscreteSearch(("shared", "dual"))


def test_singleton_domain_is_fixed_and_not_searchable(catalog: ScriptCatalog) -> None:
    group = resolve_one_group(
        catalog,
        policy="scan_unfixed",
        parameters={"learning_rate": {"values": [0.25]}},
    )

    assert group.parameters["learning_rate"].fixed_value == 0.25
    assert "learning_rate" not in group.searchable_parameters()


@pytest.mark.parametrize(
    "parameters",
    [
        {"topology": {"values": ["unsupported"]}},
        {"topology": {"values": ["shared", "unsupported"]}},
    ],
)
def test_choice_restrictions_cover_fixed_and_search_domains(
    catalog: ScriptCatalog,
    parameters: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="topology.*shared.*dual"):
        resolve_one_group(catalog, policy="explicit_scan", parameters=parameters)


def test_choices_reject_continuous_experiment_domain(catalog: ScriptCatalog) -> None:
    data = catalog.model_dump()
    learning_rate = data["scripts"]["rtrrl"]["fields"]["learning_rate"]
    learning_rate["choices"] = [0.001, 0.01]
    learning_rate["default_search"] = {"values": [0.001, 0.01]}
    restricted_catalog = ScriptCatalog.model_validate(data)

    with pytest.raises(
        ValueError,
        match=r"group 'shared'.*field 'learning_rate'.*choices.*discrete.*singleton",
    ):
        resolve_one_group(
            restricted_catalog,
            policy="explicit_scan",
            parameters={"learning_rate": {"min": 0.001, "max": 0.01}},
        )


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"unknown": {"values": [1]}}, "group 'shared'.*field 'unknown'"),
        ({"seed": {"values": [-1]}}, "group 'shared'.*field 'seed'"),
        ({"seed": {"values": [1, 2]}}, "group 'shared'.*field 'seed'"),
    ],
)
def test_rejects_unknown_invalid_or_unsearchable_domains(
    catalog: ScriptCatalog,
    parameters: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_one_group(catalog, policy="scan_unfixed", parameters=parameters)


def test_catalog_lookup_requires_exact_image_reference(catalog: ScriptCatalog) -> None:
    spec = ExperimentSpec.model_validate(
        {
            "experiment": {"name": "hopper"},
            "defaults": {
                "image": IMAGE,
                "resources": {"profile": "g6x"},
                "hpo": {"total_trials": 1, "configs_per_batch": 1},
                "execution": {"runs_per_job": 1},
            },
            "groups": {"shared": {"script": "rtrrl"}},
        }
    )

    with pytest.raises(ValueError, match="catalog.*repo/image@sha256"):
        resolve_experiment(spec, {"repo/image:dev": catalog})


def test_group_override_image_selects_matching_catalog(catalog: ScriptCatalog) -> None:
    spec = ExperimentSpec.model_validate(
        {
            "experiment": {"name": "hopper"},
            "defaults": {
                "image": IMAGE,
                "resources": {"profile": "g6x"},
                "hpo": {"total_trials": 1, "configs_per_batch": 1},
                "execution": {"runs_per_job": 1},
            },
            "groups": {
                "shared": {
                    "script": "rtrrl",
                    "overrides": {"image": OVERRIDE_IMAGE},
                }
            },
        }
    )

    group = resolve_experiment(spec, {OVERRIDE_IMAGE: catalog}).groups[0]

    assert group.image == OVERRIDE_IMAGE


def test_resolved_metadata_is_recursively_immutable_and_json_serializable(
    catalog: ScriptCatalog,
) -> None:
    spec = ExperimentSpec.model_validate(
        {
            "experiment": {
                "name": "hopper",
                "metadata": {"labels": ["baseline"], "owner": {"name": "research"}},
            },
            "defaults": {
                "image": IMAGE,
                "resources": {"profile": "g6x"},
                "hpo": {"total_trials": 1, "configs_per_batch": 1},
                "execution": {"runs_per_job": 1},
            },
            "groups": {"shared": {"script": "rtrrl"}},
        }
    )

    metadata = resolve_experiment(spec, {IMAGE: catalog}).groups[0].metadata

    with pytest.raises(TypeError):
        metadata["labels"].append("changed")
    with pytest.raises(TypeError):
        metadata["owner"]["name"] = "changed"
    assert json.loads(json.dumps(metadata)) == {
        "labels": ["baseline"],
        "owner": {"name": "research"},
    }
