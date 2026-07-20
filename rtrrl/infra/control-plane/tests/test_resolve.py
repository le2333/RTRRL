import json

import pytest

from trainer_infra.models import (
    ContinuousSearch,
    DiscreteSearch,
    ExperimentSpec,
    ScriptCatalog,
)
from trainer_infra.resolve import resolve_experiment

IMAGE = "repo/image@sha256:" + "a" * 64
OVERRIDE_IMAGE = "repo/image@sha256:" + "b" * 64


@pytest.fixture
def catalog() -> ScriptCatalog:
    return ScriptCatalog.model_validate(
        {
            "protocol_version": "1",
            "scripts": {
                "rtrrl": {
                    "name": "rtrrl",
                    "argv": ["python", "-m", "train"],
                    "sdk_protocol_version": "1",
                    "defaults": {
                        "environment": {
                            "env_name": "hopper",
                            "backend": "spring",
                            "observation_mode": "P",
                            "max_episode_steps": 1000,
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
    )


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
                "resources": {"profile": "gpu"},
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
                "resources": {"profile": "gpu"},
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
    assert resolved.groups[0].environment.env_name == "hopper"
    assert resolved.groups[0].image == IMAGE


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
                "resources": {"profile": "gpu"},
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
                "resources": {"profile": "gpu"},
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
                "resources": {"profile": "gpu"},
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
