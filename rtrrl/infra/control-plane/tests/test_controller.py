from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

import pytest
import yaml

from trainer_infra.adapters.aws_batch import SubmittedJob, ValidatedJobDefinition
from trainer_infra.controller import ExperimentController, ExperimentRunError
from trainer_infra.execution import CompletionMarker, JobQuery
from trainer_infra.image_catalog import ResolvedImage
from trainer_infra.models import ScriptCatalog
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
    def __init__(self, number: int) -> None:
        self.number = number

    def suggest_categorical(self, name: str, values: tuple[object, ...]) -> object:
        del name
        return values[self.number % len(values)]


class Study:
    def __init__(self, name: str, owner: int) -> None:
        self.name = name
        self.owner = owner
        self.told: list[tuple[int, float, int]] = []
        self._next = 0

    def ask(self) -> Trial:
        trial = Trial(self._next)
        self._next += 1
        return trial

    def tell(self, trial: Trial, value: float) -> None:
        self.told.append((trial.number, value, threading.get_ident()))


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
        if self.mode == "persist-failed" and uri.endswith("state/final.json"):
            raise OSError("final persistence failed")
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


def make_controller(
    calls: list[str],
    *,
    mode: str = "ok",
    ids: list[str] | None = None,
) -> tuple[ExperimentController, Store, Batch, list[Study]]:
    owner = threading.get_ident()
    stores: list[Store] = []
    batches: list[Batch] = []
    studies: list[Study] = []
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

    def study_factory(**kwargs: object) -> Study:
        calls.append("study")
        study = Study(str(kwargs["study_name"]), owner)
        studies.append(study)
        return study

    controller = ExperimentController(
        catalog_reader=CatalogReader(calls),
        preflight=Preflight(calls),
        store_factory=store_factory,
        batch_factory=batch_factory,
        aim_reader=Aim(mode),
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
    assert batches[0].round_sizes == {"first": [2, 2, 1], "second": [2, 2, 1]}
    assert [len(study.told) for study in studies] == [5, 5]
    assert {thread for study in studies for _, _, thread in study.told} == {owner}
    assert len(report.submitted_job_ids) == 6
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
    controller, stores, batches, _ = make_controller(calls, mode=mode)

    with pytest.raises(ExperimentRunError) as raised:
        controller.run(experiment)

    assert raised.value.report.status == "failed"
    assert raised.value.report.submitted_job_ids == ("aws-1",)
    assert len(batches[0].submitted) == 1
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

    assert raised.value.report.submitted_job_ids == ("aws-1", "aws-2")
    assert len(batches[0].submitted) == 2


def test_final_persistence_failure_propagates_once_without_retry(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    write_experiment(experiment)
    calls: list[str] = []
    controller, stores, _, _ = make_controller(calls, mode="persist-failed")

    with pytest.raises(OSError, match="final persistence failed"):
        controller.run(experiment)

    assert (
        stores[0].puts.count(
            "s3://bucket/experiments/fresh-1/state/final.json"
        )
        == 1
    )
