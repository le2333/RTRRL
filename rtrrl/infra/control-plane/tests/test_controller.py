from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Any

import pytest
import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState
import yaml

from trainer_infra.adapters.aws_batch import SubmittedJob, ValidatedJobDefinition
from trainer_infra.controller import (
    ExperimentController,
    ExperimentRunError,
    SamplingExhaustedError,
)
from trainer_infra.execution import CompletionMarker, JobQuery
from trainer_infra.image_catalog import ResolvedImage
from trainer_infra.models import (
    DiscreteSearch,
    ResolvedParameter,
    ScriptCatalog,
)
from trainer_infra.materialize import materialize_run as real_materialize_run
from test_materialize import make_group
from test_resolve import catalog_data

IMAGE_TAG = "123456789012.dkr.ecr.eu-north-1.amazonaws.com/memo:latest"
IMAGE = "123456789012.dkr.ecr.eu-north-1.amazonaws.com/memo@sha256:" + "a" * 64


def write_experiment(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "automatic"},
                "defaults": {
                    "image": IMAGE_TAG,
                    "resources": {"profile": "g6x"},
                    "hpo": {
                        "total_trials": 5,
                        "configs_per_batch": 2,
                        "parameter_policy": "explicit_scan",
                    },
                    "execution": {
                        "runs_per_job": 2,
                        "aim_result_timeout_seconds": 10,
                    },
                    "parameters": {"seed": {"values": [7]}},
                },
                "groups": {
                    "first": {
                        "script": "rtrrl",
                        "parameters": {
                            "topology": {
                                "values": ["one", "two", "three", "four", "five"]
                            }
                        },
                    },
                    "second": {
                        "script": "rtrrl",
                        "parameters": {
                            "topology": {
                                "values": ["one", "two", "three", "four", "five"]
                            }
                        },
                    },
                },
            },
            sort_keys=False,
        )
    )


def make_catalog() -> ScriptCatalog:
    data = catalog_data()
    script = data["scripts"]["rtrrl"]  # type: ignore[index]
    script["argv"] = ["python", "-m", "train", "--config", "{config_path}"]
    script["fields"]["topology"]["choices"] = ["one", "two", "three", "four", "five"]
    script["fields"]["topology"]["default"] = "one"
    script["fields"]["topology"]["default_search"] = {
        "values": ["one", "two", "three", "four", "five"]
    }
    script["fields"]["width"] = {
        "path": "network.width",
        "type": "int",
        "default": 1,
        "searchable": True,
        "constraints": {"gt": 0},
        "default_search": {"values": [1, 2, 3]},
    }
    return ScriptCatalog.model_validate(data)


class CatalogReader:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def resolve_and_fetch(self, reference: str) -> ResolvedImage:
        self.calls.append(f"catalog:{reference}")
        return ResolvedImage(
            reference=IMAGE,
            repository=IMAGE.split("@")[0],
            digest="sha256:" + "a" * 64,
            catalog=make_catalog(),
        )


def definition() -> ValidatedJobDefinition:
    return ValidatedJobDefinition(
        arn=(
            "arn:aws:batch:eu-north-1:123456789012:job-definition/"
            f"trainer-g6x-{'a' * 64}:1"
        ),
        image_digest=IMAGE,
        resource_profile="g6x",
    )


class Preflight:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def validate(self, resolved: object) -> dict[str, ValidatedJobDefinition]:
        del resolved
        self.calls.append("preflight")
        return {"g6x": definition()}


class Trial:
    def __init__(
        self,
        number: int,
        *,
        forced: dict[str, object] | None = None,
        repeat: bool = False,
    ) -> None:
        self.number = number
        self.forced = forced or {}
        self.repeat = repeat

    def suggest_categorical(self, name: str, values: tuple[object, ...]) -> object:
        if name in self.forced:
            return self.forced[name]
        return values[0 if self.repeat else self.number % len(values)]

    def suggest_float(
        self, name: str, low: float, high: float, *, log: bool
    ) -> float:
        del low, high, log
        return float(self.forced.get(name, 0.001))

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int,
        log: bool,
    ) -> int:
        del low, high, step, log
        return int(self.forced.get(name, 1))


class Study:
    def __init__(self, name: str, owner: int, *, repeat: bool = False) -> None:
        self.name = name
        self.owner = owner
        self.told: list[tuple[int, object, object, int]] = []
        self._next = 0
        self._queued: list[dict[str, object]] = []
        self._repeat = repeat

    def ask(self) -> Trial:
        forced = self._queued.pop(0) if self._queued else None
        trial = Trial(self._next, forced=forced, repeat=self._repeat)
        self._next += 1
        return trial

    def enqueue_trial(self, params: dict[str, object]) -> None:
        self._queued.append(params)

    def tell(
        self,
        trial: Trial,
        values: float | None = None,
        state: TrialState | None = None,
    ) -> None:
        self.told.append((trial.number, values, state, threading.get_ident()))


class Store:
    def __init__(self, prefix: str, mode: str = "ok") -> None:
        self.prefix = prefix
        self.mode = mode
        self.values: dict[str, Any] = {}
        self.puts: list[str] = []

    def put_bytes(self, uri: str, data: bytes) -> str:
        self.puts.append(uri)
        self.values[uri] = data
        import hashlib

        return hashlib.sha256(data).hexdigest()

    def put_json(self, uri: str, value: Any) -> str:
        self.puts.append(uri)
        if (
            self.mode in {"persist-state-failed", "persist-both-failed"}
            and uri.endswith("state/final.json")
        ):
            raise OSError("state persistence failed")
        if (
            self.mode in {"persist-report-failed", "persist-both-failed"}
            and uri.endswith("report.json")
        ):
            raise PermissionError("report persistence failed")
        self.values[uri] = json.loads(json.dumps(value))
        return "digest"

    def get_json(self, uri: str, *, expected_sha256: str | None = None) -> Any:
        del expected_sha256
        if self.mode == "missing-marker" and "/status/" in uri:
            raise FileNotFoundError(uri)
        return self.values[uri]


class Batch:
    def __init__(self, store: Store, mode: str = "ok") -> None:
        self.store = store
        self.mode = mode
        self.submitted: list[object] = []
        self.round_sizes: dict[str, list[int]] = {}

    def submit(self, bundle: Any, profile: object, job_definition: object) -> SubmittedJob:
        del profile, job_definition
        self.submitted.append(bundle)
        group = bundle.runs[0].run_context["group"]
        self.round_sizes.setdefault(group, []).append(len(bundle.runs))
        for run in bundle.runs:
            marker = CompletionMarker(
                run_id="tampered:run:9999" if self.mode == "tampered-marker" else run.run_id,
                exit_code=3 if self.mode == "child-failed" else 0,
                started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                finished_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            root = run.artifact_prefix.removesuffix("input/")
            self.store.values[
                f"s3://bucket/{root}status/attempt-0.json"
            ] = marker.model_dump(mode="json")
        return SubmittedJob(job_id=f"aws-{len(self.submitted)}", bundle_id=bundle.job_id)

    def query(self, job_ids: list[str]) -> tuple[JobQuery, ...]:
        status = "FAILED" if self.mode == "batch-failed" else "SUCCEEDED"
        return tuple(JobQuery(job_id=job_id, status=status) for job_id in job_ids)


class Aim:
    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.calls: list[str] = []

    def wait_for_result(self, run_id: str, objective: str, timeout: float) -> float:
        del objective, timeout
        self.calls.append(run_id)
        if self.mode == "aim-failed":
            raise RuntimeError("Aim failed")
        return float(len(self.calls))


class RepeatingSampler(optuna.samplers.BaseSampler):
    def infer_relative_search_space(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> dict[str, optuna.distributions.BaseDistribution]:
        del study, trial
        return {}

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, object]:
        del study, trial, search_space
        return {}

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> object:
        del study, trial, param_name
        if isinstance(param_distribution, CategoricalDistribution):
            return param_distribution.choices[0]
        if isinstance(param_distribution, IntDistribution):
            return param_distribution.low
        assert isinstance(param_distribution, FloatDistribution)
        return param_distribution.low


def make_controller(
    calls: list[str],
    *,
    mode: str = "ok",
    aim_mode: str | None = None,
    ids: list[str] | None = None,
    custom_study_factory: Any | None = None,
) -> tuple[ExperimentController, Store, Batch, list[Any]]:
    owner = threading.get_ident()
    stores: list[Store] = []
    batches: list[Batch] = []
    studies: list[Any] = []
    identifiers = iter(ids or ["fresh-1"])

    def store_factory(prefix: str) -> Store:
        calls.append("store")
        store = Store(prefix, mode)
        stores.append(store)
        return store

    def batch_factory(prefix: str, store: Store) -> Batch:
        del prefix
        calls.append("batch")
        batch = Batch(store, mode)
        batches.append(batch)
        return batch

    def aim_reader_factory(store: Store) -> Aim:
        assert store in stores
        return Aim(aim_mode or mode)

    def study_factory(**kwargs: object) -> Study:
        calls.append("study")
        if custom_study_factory is not None:
            study = custom_study_factory(**kwargs)
            studies.append(study)
            return study
        study = Study(str(kwargs["study_name"]), owner)
        studies.append(study)
        return study

    controller = ExperimentController(
        catalog_reader=CatalogReader(calls),
        preflight=Preflight(calls),
        store_factory=store_factory,
        batch_factory=batch_factory,
        aim_reader_factory=aim_reader_factory,
        study_factory=study_factory,
        experiment_id_factory=lambda: next(identifiers),
        bucket="bucket",
        poll_interval=0.01,
        batch_timeout=1,
        sleep=lambda _: None,
    )
    return controller, stores, batches, studies


def test_validate_is_completely_read_only_and_returns_stable_json(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    calls: list[str] = []
    controller, stores, batches, studies = make_controller(calls)

    report = controller.validate(experiment)

    assert report.status == "valid"
    assert [group.name for group in report.groups] == ["first", "second"]
    assert report.to_json() == report.to_json()
    assert calls == [f"catalog:{IMAGE_TAG}", "preflight"]
    assert stores == batches == studies == []


def test_two_groups_run_automatic_two_two_one_with_controller_only_tell(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    calls: list[str] = []
    controller, stores, batches, studies = make_controller(calls)
    owner = threading.get_ident()

    report = controller.run(experiment)

    assert report.status == "succeeded"
    assert report.experiment_id == "fresh-1"
    assert len(batches[0].submitted) == 5
    assert all(
        {run.run_context["group"] for run in bundle.runs} == {"first", "second"}
        for bundle in batches[0].submitted
    )
    assert {
        run.run_context["experiment_name"]
        for bundle in batches[0].submitted
        for run in bundle.runs
    } == {"automatic"}
    assert {study.name for study in studies} == {"fresh-1:first", "fresh-1:second"}
    assert [len(study.told) for study in studies] == [5, 5]
    assert {thread for study in studies for _, _, _, thread in study.told} == {owner}
    assert {
        state for study in studies for _, _, state, _ in study.told
    } == {None}
    assert len(report.submitted_job_ids) == 5
    assert stores[0].puts[-2:] == [
        "s3://bucket/experiments/fresh-1/state/final.json",
        "s3://bucket/experiments/fresh-1/report.json",
    ]


def test_each_run_invocation_generates_one_fresh_experiment_id(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    calls: list[str] = []
    controller, _, _, _ = make_controller(calls, ids=["fresh-1", "fresh-2"])

    assert controller.run(experiment).experiment_id == "fresh-1"
    assert controller.run(experiment).experiment_id == "fresh-2"


@pytest.mark.parametrize(
    "mode",
    [
        "batch-failed",
        "child-failed",
        "missing-marker",
        "tampered-marker",
        "aim-failed",
    ],
)
def test_failure_boundaries_stop_future_batches_and_keep_submitted_ids(
    tmp_path: Path, mode: str
) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    calls: list[str] = []
    controller, stores, batches, studies = make_controller(calls, mode=mode)

    with pytest.raises(ExperimentRunError) as raised:
        controller.run(experiment)

    assert raised.value.report.status == "failed"
    assert raised.value.report.submitted_job_ids == ("aws-1", "aws-2")
    assert len(batches[0].submitted) == 2
    assert [
        (value, state)
        for study in studies
        for trial in study.told
        for _, value, state, _ in (trial,)
    ] == [(None, TrialState.FAIL)] * 4
    assert all(len({number for number, _, _, _ in study.told}) == 2 for study in studies)
    assert {thread for study in studies for _, _, _, thread in study.told} == {
        threading.get_ident()
    }
    assert stores[0].puts[-2:] == [
        "s3://bucket/experiments/fresh-1/state/final.json",
        "s3://bucket/experiments/fresh-1/report.json",
    ]


def test_failed_concurrent_round_retains_every_submitted_id(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    payload = yaml.safe_load(experiment.read_text())
    payload["defaults"]["execution"]["runs_per_job"] = 1
    experiment.write_text(yaml.safe_dump(payload))
    calls: list[str] = []
    controller, _, batches, _ = make_controller(calls, mode="batch-failed")

    with pytest.raises(ExperimentRunError) as raised:
        controller.run(experiment)

    assert raised.value.report.submitted_job_ids == (
        "aws-1",
        "aws-2",
        "aws-3",
        "aws-4",
    )
    assert len(batches[0].submitted) == 4


@pytest.mark.parametrize(
    ("mode", "error_types"),
    [
        ("persist-state-failed", (OSError,)),
        ("persist-report-failed", (PermissionError,)),
        ("persist-both-failed", (OSError, PermissionError)),
    ],
)
def test_final_persistence_attempts_both_writes_once_and_preserves_report(
    tmp_path: Path,
    mode: str,
    error_types: tuple[type[BaseException], ...],
) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    calls: list[str] = []
    controller, stores, _, _ = make_controller(calls, mode=mode)

    with pytest.raises(ExperimentRunError) as raised:
        controller.run(experiment)

    assert raised.value.report.status == "failed"
    assert raised.value.report.submitted_job_ids
    assert raised.value.original_cause is None
    assert tuple(type(error) for error in raised.value.persistence_errors) == error_types
    assert stores[0].puts[-2:] == [
        "s3://bucket/experiments/fresh-1/state/final.json",
        "s3://bucket/experiments/fresh-1/report.json",
    ]


def test_runtime_and_both_persistence_failures_are_all_preserved(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    calls: list[str] = []
    controller, stores, _, _ = make_controller(
        calls,
        mode="persist-both-failed",
        aim_mode="aim-failed",
    )

    with pytest.raises(ExperimentRunError) as raised:
        controller.run(experiment)

    assert isinstance(raised.value.original_cause, RuntimeError)
    assert "Aim failed" in str(raised.value.original_cause)
    assert tuple(type(error) for error in raised.value.persistence_errors) == (
        OSError,
        PermissionError,
    )
    assert raised.value.report.submitted_job_ids == ("aws-1", "aws-2")
    assert stores[0].puts[-2:] == [
        "s3://bucket/experiments/fresh-1/state/final.json",
        "s3://bucket/experiments/fresh-1/report.json",
    ]


def test_estimated_jobs_sums_each_round_partition(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    payload = yaml.safe_load(experiment.read_text())
    payload["defaults"]["hpo"] = {
        "total_trials": 6,
        "configs_per_batch": 4,
        "parameter_policy": "explicit_scan",
    }
    payload["defaults"]["execution"]["runs_per_job"] = 3
    experiment.write_text(yaml.safe_dump(payload))
    calls: list[str] = []
    controller, _, _, _ = make_controller(calls)

    report = controller.validate(experiment)

    assert [group.estimated_jobs for group in report.groups] == [3, 3]


def test_real_optuna_completes_two_two_one_without_running_trials(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    calls: list[str] = []

    def real_study(**kwargs: object) -> optuna.Study:
        return optuna.create_study(
            **kwargs,
            sampler=optuna.samplers.TPESampler(seed=4),
        )

    controller, _, batches, studies = make_controller(
        calls, custom_study_factory=real_study
    )

    report = controller.run(experiment)

    assert report.completed_runs == 10
    assert len(batches[0].submitted) == 5
    assert all(
        {run.run_context["group"] for run in bundle.runs} == {"first", "second"}
        for bundle in batches[0].submitted
    )
    for study in studies:
        assert sum(trial.state == TrialState.COMPLETE for trial in study.trials) == 5
        assert not any(trial.state == TrialState.RUNNING for trial in study.trials)


def test_real_optuna_marks_current_failed_round_once(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    calls: list[str] = []

    def real_study(**kwargs: object) -> optuna.Study:
        return optuna.create_study(
            **kwargs,
            sampler=optuna.samplers.RandomSampler(seed=2),
        )

    controller, _, _, studies = make_controller(
        calls,
        mode="batch-failed",
        custom_study_factory=real_study,
    )

    with pytest.raises(ExperimentRunError):
        controller.run(experiment)

    states = [trial.state for trial in studies[0].trials]
    assert states.count(TrialState.FAIL) == 2
    assert TrialState.RUNNING not in states


def test_real_optuna_fallback_preserves_values_through_every_execution_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    payload = yaml.safe_load(experiment.read_text())
    payload["groups"] = {"first": payload["groups"]["first"]}
    experiment.write_text(yaml.safe_dump(payload))
    calls: list[str] = []
    concrete_runs: list[Any] = []

    def recording_materialize(*args: Any, **kwargs: Any) -> Any:
        concrete = real_materialize_run(*args, **kwargs)
        concrete_runs.append(concrete)
        return concrete

    monkeypatch.setattr("trainer_infra.controller.materialize_run", recording_materialize)

    def repeating_study(**kwargs: object) -> optuna.Study:
        return optuna.create_study(**kwargs, sampler=RepeatingSampler())

    controller, _, batches, studies = make_controller(
        calls, custom_study_factory=repeating_study
    )

    report = controller.run(experiment)

    assert report.completed_runs == 5
    study = studies[0]
    complete = {
        trial.number: trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    }
    assert len(complete) == 5
    assert any(trial.state == TrialState.PRUNED for trial in study.trials)
    bundles = [bundle for bundle in batches[0].submitted]
    run_bundles = [run for bundle in bundles for run in bundle.runs]
    assert len(concrete_runs) == len(run_bundles) == 5
    for concrete, run_bundle in zip(concrete_runs, run_bundles, strict=True):
        topology = complete[concrete.trial_number].params["topology"]
        config = yaml.safe_load(run_bundle.config_yaml)
        assert concrete.sampled_parameters == {"topology": topology}
        assert concrete.final_parameters["topology"] == topology
        assert run_bundle.run_context["sampled_parameters"] == {
            "topology": topology
        }
        assert run_bundle.run_context["final_parameters"]["topology"] == topology
        assert config["parameters"]["topology"] == topology


def test_continuous_duplicate_limit_fails_without_hanging_or_running_trials(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    payload = yaml.safe_load(experiment.read_text())
    payload["groups"] = {
        "first": {
            "script": "rtrrl",
            "parameters": {
                "topology": {"values": ["one"]},
                "learning_rate": {
                    "min": 0.00001,
                    "max": 0.01,
                    "scale": "log",
                },
            },
        }
    }
    payload["defaults"]["hpo"]["total_trials"] = 2
    experiment.write_text(yaml.safe_dump(payload))
    calls: list[str] = []
    owner = threading.get_ident()

    def repeating_study(**kwargs: object) -> Study:
        return Study(str(kwargs["study_name"]), owner, repeat=True)

    controller, _, batches, studies = make_controller(
        calls, custom_study_factory=repeating_study
    )

    with pytest.raises(ExperimentRunError, match="duplicate sampling limit"):
        controller.run(experiment)

    assert batches[0].submitted == []
    told = studies[0].told
    assert len(told) <= 34
    assert sum(state == TrialState.FAIL for _, _, state, _ in told) == 1
    assert all(state in {TrialState.PRUNED, TrialState.FAIL} for _, _, state, _ in told)


def test_integer_range_is_enumerated_as_finite_fallback(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    payload = yaml.safe_load(experiment.read_text())
    payload["groups"] = {
        "first": {
            "script": "rtrrl",
            "parameters": {
                "topology": {"values": ["one"]},
                "width": {"min": 1, "max": 3, "scale": "linear"},
            },
        }
    }
    payload["defaults"]["hpo"]["total_trials"] = 3
    experiment.write_text(yaml.safe_dump(payload))
    calls: list[str] = []
    owner = threading.get_ident()

    def repeating_study(**kwargs: object) -> Study:
        return Study(str(kwargs["study_name"]), owner, repeat=True)

    controller, _, _, studies = make_controller(
        calls, custom_study_factory=repeating_study
    )

    report = controller.run(experiment)

    assert report.completed_runs == 3
    assert len([item for item in studies[0].told if item[2] is None]) == 3


def test_trillion_integer_range_is_never_iterated_or_materialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    payload = yaml.safe_load(experiment.read_text())
    payload["groups"] = {
        "first": {
            "script": "rtrrl",
            "parameters": {
                "topology": {"values": ["one"]},
                "width": {
                    "min": 1,
                    "max": 1_000_000_000_000,
                    "scale": "linear",
                },
            },
        }
    }
    payload["defaults"]["hpo"]["total_trials"] = 2
    experiment.write_text(yaml.safe_dump(payload))
    real_range = range

    class NeverIterateHugeRange:
        def __len__(self) -> int:
            return 1_000_000_000_000

        def __iter__(self) -> object:
            raise AssertionError("huge integer range was iterated")

    def guarded_range(*args: int) -> object:
        candidate = real_range(*args)
        if len(candidate) > 10_000:
            return NeverIterateHugeRange()
        return candidate

    monkeypatch.setattr("trainer_infra.controller.range", guarded_range, raising=False)
    monkeypatch.setattr(
        "trainer_infra.controller.product",
        lambda *_: (_ for _ in ()).throw(AssertionError("product was created")),
    )
    calls: list[str] = []
    owner = threading.get_ident()

    def repeating_study(**kwargs: object) -> Study:
        return Study(str(kwargs["study_name"]), owner, repeat=True)

    controller, _, batches, studies = make_controller(
        calls, custom_study_factory=repeating_study
    )

    with pytest.raises(ExperimentRunError, match="too large|32") as raised:
        controller.run(experiment)

    assert isinstance(raised.value.original_cause, SamplingExhaustedError)
    assert batches[0].submitted == []
    assert len(studies[0].told) == 33


def test_huge_multidimensional_choices_compute_cardinality_without_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = tuple(range(1_001))
    group = replace(
        make_group(),
        parameters=MappingProxyType(
            {
                "first": ResolvedParameter(
                    fixed_value=None,
                    search_domain=DiscreteSearch(choices),
                ),
                "second": ResolvedParameter(
                    fixed_value=None,
                    search_domain=DiscreteSearch(choices),
                ),
            }
        ),
    )
    monkeypatch.setattr(
        "trainer_infra.controller.product",
        lambda *_: (_ for _ in ()).throw(AssertionError("product was created")),
    )

    finite = ExperimentController._finite_space(group)

    assert finite is not None
    assert finite.too_large is True
    assert finite.cardinality > 10_000
