from pathlib import Path

import pytest
from pydantic import ValidationError

from trainer_infra.loaders import load_experiment, load_script_catalog
from trainer_infra.models import (
    ContinuousDomain,
    DiscreteDomain,
    ExecutionSpec,
    HpoSpec,
    ResourcesSpec,
)


@pytest.mark.parametrize("profile", ["c7am", "c7al", "c7ax", "g6x"])
def test_resources_accepts_only_declared_profiles(profile: str) -> None:
    assert ResourcesSpec(profile=profile).profile == profile


@pytest.mark.parametrize("profile", ["cpu", "gpu", "g6f", "c7a.2xlarge"])
def test_resources_rejects_legacy_or_undeclared_profiles(profile: str) -> None:
    with pytest.raises(ValidationError):
        ResourcesSpec(profile=profile)


def test_domains_reject_empty_or_invalid_bounds() -> None:
    with pytest.raises(ValidationError, match="values must not be empty"):
        DiscreteDomain(values=[])
    with pytest.raises(ValidationError, match="continuous bounds must be finite and min < max"):
        ContinuousDomain(min=1.0, max=1.0)
    with pytest.raises(ValidationError, match="log domains require min > 0"):
        ContinuousDomain(min=0.0, max=1.0, scale="log")


def test_hpo_and_execution_defaults_are_exact() -> None:
    hpo = HpoSpec(total_trials=5, configs_per_batch=2)
    execution = ExecutionSpec(runs_per_job=2)

    assert hpo.parameter_policy == "scan_unfixed"
    assert execution.max_infra_retries == 2
    assert execution.max_algorithm_retries == 0
    assert execution.retry_backoff_seconds == 30
    assert execution.aim_result_timeout_seconds == 600

    with pytest.raises(ValidationError, match="configs_per_batch must not exceed total_trials"):
        HpoSpec(total_trials=1, configs_per_batch=2)


def test_models_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExecutionSpec(runs_per_job=1, surprise=True)


def test_yaml_loaders_validate_contracts(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """
experiment:
  name: hopper
defaults:
  image: repo/image@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  resources: {profile: g6x}
  hpo: {total_trials: 5, configs_per_batch: 2}
  execution: {runs_per_job: 2}
groups:
  shared: {script: rtrrl}
""",
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        """
protocol_version: "1"
scripts:
  rtrrl:
    name: rtrrl
    argv: [python, -m, train]
    sdk_protocol_version: "1"
    defaults:
      environment:
        name: brax-hopper
        options:
          backend: spring
          observation_mode: P
          max_episode_steps: 1000
      training_budget: {env_steps: 2000000}
      logging: {aim_every_env_steps: 10000, rerun_every_episodes: 100}
    objective: {metric: reward, direction: maximize, reduction: last}
    environments: [brax-hopper]
    fields: {}
""",
        encoding="utf-8",
    )

    assert load_experiment(experiment_path).experiment.name == "hopper"
    assert load_script_catalog(catalog_path).scripts["rtrrl"].argv == (
        "python",
        "-m",
        "train",
    )
