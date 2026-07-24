from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import optuna
from optuna.trial import TrialState
import pytest
import yaml

from trainer_infra.adapters.aws_batch import SubmittedJob, ValidatedJobDefinition
from trainer_infra.aim_reader import AimReader
from trainer_infra.controller import ExperimentController, ExperimentRunError
from trainer_infra.execution import JobQuery
from trainer_infra.identities import canonical_json
from trainer_infra.image_catalog import (
    EcrCatalogReader,
    LABEL,
    encode_catalog,
    load_catalog_index,
)
from trainer_infra.models import ScriptCatalog
from training_sdk.spool import EventSpool


REPOSITORY_ROOT = Path(__file__).parents[4]
WORKER_PATH = REPOSITORY_ROOT / "rtrrl" / "infra" / "worker" / "worker.py"
MOCK_TRAINER_ROOT = REPOSITORY_ROOT / "rtrrl" / "infra" / "mock-trainer"
CPU_IMAGE_TAG = "123456789012.dkr.ecr.eu-north-1.amazonaws.com/acceptance:cpu"
GPU_IMAGE_TAG = "123456789012.dkr.ecr.eu-north-1.amazonaws.com/acceptance:gpu"
CPU_IMAGE_DIGEST = (
    "123456789012.dkr.ecr.eu-north-1.amazonaws.com/acceptance@sha256:" + "a" * 64
)
GPU_IMAGE_DIGEST = (
    "123456789012.dkr.ecr.eu-north-1.amazonaws.com/acceptance@sha256:" + "b" * 64
)
CPU_CONFIG_DIGEST = "sha256:" + "c" * 64
GPU_CONFIG_DIGEST = "sha256:" + "d" * 64

WORKER_SPEC = importlib.util.spec_from_file_location("task6_trainer_worker", WORKER_PATH)
assert WORKER_SPEC is not None and WORKER_SPEC.loader is not None
worker = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(worker)


class FakeEcr:
    def __init__(self, catalog: ScriptCatalog) -> None:
        self.label = encode_catalog(catalog)
        self.calls: list[tuple[Any, ...]] = []

    def resolve_tag(self, reference: str) -> str:
        self.calls.append(("resolve_tag", reference))
        return {
            CPU_IMAGE_TAG: CPU_IMAGE_DIGEST,
            GPU_IMAGE_TAG: GPU_IMAGE_DIGEST,
        }[reference]

    def get_manifest(self, reference: str) -> Mapping[str, Any]:
        self.calls.append(("get_manifest", reference))
        return {
            "config": {
                "digest": {
                    CPU_IMAGE_DIGEST: CPU_CONFIG_DIGEST,
                    GPU_IMAGE_DIGEST: GPU_CONFIG_DIGEST,
                }[reference]
            }
        }

    def get_config_blob(self, repository: str, digest: str) -> Mapping[str, Any]:
        self.calls.append(("get_config_blob", repository, digest))
        return {"config": {"Labels": {LABEL: self.label}}}


class FakeStore:
    def __init__(self, prefix: str, mode: str = "ok") -> None:
        self.prefix = prefix
        self.mode = mode
        self.objects: dict[str, bytes] = {}
        self.digests: dict[str, str] = {}
        self.put_json_calls: list[str] = []
        self.marker_writes: list[str] = []

    def _put(self, uri: str, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        self.objects[uri] = data
        self.digests[uri] = digest
        return digest

    def put_bytes(self, uri: str, data: bytes) -> str:
        if self.mode == "input-upload" and "/input/" in uri:
            raise OSError("run input upload failed")
        return self._put(uri, data)

    def get_bytes(self, uri: str, *, expected_sha256: str | None = None) -> bytes:
        data = self.objects[uri]
        actual = hashlib.sha256(data).hexdigest()
        if self.digests[uri] != actual:
            raise ValueError(f"stored digest mismatch for {uri}")
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError(f"expected digest mismatch for {uri}")
        return data

    def put_json(self, uri: str, value: Any) -> str:
        self.put_json_calls.append(uri)
        if self.mode in {"persist-state", "persist-both"} and uri.endswith(
            "state/final.json"
        ):
            raise OSError("state persistence failed")
        if self.mode in {"persist-report", "persist-both"} and uri.endswith(
            "report.json"
        ):
            raise PermissionError("report persistence failed")
        if self.mode == "job-upload" and "/jobs/" in uri:
            raise OSError("job bundle upload failed")
        if "/status/" in uri:
            self.marker_writes.append(uri)
        return self._put(uri, canonical_json(value).encode())

    def get_json(self, uri: str, *, expected_sha256: str | None = None) -> Any:
        if self.mode == "marker-missing" and "/status/" in uri:
            raise FileNotFoundError(uri)
        value = json.loads(self.get_bytes(uri, expected_sha256=expected_sha256))
        if self.mode == "marker-tamper" and "/status/" in uri:
            value["run_id"] = "tampered:run:0001"
        return value

    def put_file(self, uri: str, path: Path) -> str:
        if self.mode == "artifact-upload":
            raise OSError("artifact upload failed")
        return self._put(uri, Path(path).read_bytes())


@dataclass
class FakeMetricData:
    value: float

    def values_list(self) -> tuple[list[float]]:
        return ([self.value],)


@dataclass
class FakeMetric:
    value: float

    @property
    def data(self) -> FakeMetricData:
        return FakeMetricData(self.value)


class FakeAimRun:
    def __init__(self, record: Mapping[str, Any]) -> None:
        self.record = record

    def get(self, key: str, default: Any = None) -> Any:
        return self.record.get(key, default)

    def get_metric(self, name: str, context: object) -> FakeMetric | None:
        del context
        objective = self.record.get("objective")
        if name != self.record.get("sdk/objective_metric") or objective is None:
            return None
        return FakeMetric(float(objective))

    def close(self) -> None:
        pass


class FakeAim:
    def __init__(self, store: FakeStore, temporary: Path, mode: str = "ok") -> None:
        self.store = store
        self.temporary = temporary
        self.mode = mode
        self.records: dict[str, dict[str, Any]] = {}
        self.replayed: list[str] = []
        self.now = 0.0
        self.reader = AimReader(
            run_factory=self._open,
            replay_spool=self.replay,
            clock=lambda: self.now,
            sleep=self._sleep,
            poll_interval=0.1,
        )

    def _sleep(self, duration: float) -> None:
        self.now += duration

    def _open(self, **options: Any) -> FakeAimRun:
        run_hash = options["run_hash"]
        return FakeAimRun(self.records.get(run_hash, {}))

    @staticmethod
    def _hash(run_id: str) -> str:
        return hashlib.sha256(run_id.encode()).hexdigest()[:24]

    def replay(self, run_id: str) -> None:
        self.replayed.append(run_id)
        matching_contexts = [
            (uri, data)
            for uri, data in self.store.objects.items()
            if uri.endswith(f"/runs/{run_id}/input/run-context.json")
        ]
        assert len(matching_contexts) == 1
        context_uri, context_data = matching_contexts[0]
        context = json.loads(context_data)
        record: dict[str, Any] = {"hparams": {"identity": {"run_id": run_id}}}
        if self.mode == "aim-failed":
            record["sdk/failed"] = True
            self.records[self._hash(run_id)] = record
            return
        if self.mode == "aim-timeout":
            self.records[self._hash(run_id)] = record
            return

        spool_uri = context_uri.replace(
            "input/run-context.json", "aim-buffer/events.jsonl"
        )
        spool_path = self.temporary / f"{self._hash(run_id)}.jsonl"
        spool_path.write_bytes(self.store.objects[spool_uri])
        events = EventSpool(spool_path).events
        finals = [event for event in events if event.kind == "final"]
        assert finals
        finalized = next(event for event in finals if event.data["finalized"])
        record["sdk/objective_metric"] = finalized.data["objective_metric"]
        record["sdk/finalized"] = True
        record["objective"] = (
            float("nan") if self.mode == "aim-nonfinite" else finalized.metric_value
        )
        record["context"] = context
        self.records[self._hash(run_id)] = record

    def wait_for_result(self, run_id: str, objective: str, timeout: float) -> float:
        return self.reader.wait_for_result(run_id, objective, timeout)

    def objective_for_run(self, run_id: str, metric: str) -> float:
        record = self.records[self._hash(run_id)]
        assert record["sdk/objective_metric"] == metric
        return float(record["objective"])

    def spool_events_for_run(self, run_id: str) -> tuple[Any, ...]:
        return EventSpool(self.temporary / f"{self._hash(run_id)}.jsonl").events


class FakePreflight:
    def validate(self, resolved: Any) -> dict[str, ValidatedJobDefinition]:
        assert {group.resources.profile for group in resolved.groups} == {"c7am", "g6x"}
        return {
            "c7am": ValidatedJobDefinition(
                arn=(
                    "arn:aws:batch:eu-north-1:123456789012:job-definition/"
                    f"trainer-c7am-{'a' * 64}:1"
                ),
                image_digest=CPU_IMAGE_DIGEST,
                resource_profile="c7am",
            ),
            "g6x": ValidatedJobDefinition(
                arn=(
                    "arn:aws:batch:eu-north-1:123456789012:job-definition/"
                    f"trainer-g6x-{'b' * 64}:1"
                ),
                image_digest=GPU_IMAGE_DIGEST,
                resource_profile="g6x",
            ),
        }


class FakeBatch:
    def __init__(
        self,
        prefix: str,
        store: FakeStore,
        mode: str = "ok",
    ) -> None:
        self.prefix = prefix
        self.store = store
        self.mode = mode
        self.submitted: list[Any] = []
        self.pending: dict[str, tuple[Any, str]] = {}
        self.statuses: dict[str, JobQuery] = {}
        self.query_calls: list[tuple[str, ...]] = []
        self.events: list[tuple[str, str | tuple[str, ...]]] = []
        self.submit_attempts = 0
        self.resubmit_calls = 0
        self.cancel_calls = 0
        self.retry_calls = 0

    def submit(
        self,
        bundle: Any,
        profile: object,
        job_definition: object,
    ) -> SubmittedJob:
        del profile, job_definition
        self.submit_attempts += 1
        if self.mode == "partial-submit" and self.submit_attempts == 2:
            raise RuntimeError("second Batch submit failed")
        job_id = f"batch-{self.submit_attempts}"
        uri = f"{self.prefix}jobs/{bundle.job_id}/bundle.json"
        self.submitted.append(bundle)
        self.pending[job_id] = (bundle, uri)
        self.events.append(("submit", job_id))
        return SubmittedJob(job_id=job_id, bundle_id=bundle.job_id)

    def _execute(self, job_id: str) -> JobQuery:
        _, uri = self.pending[job_id]
        environment = {
            "PATH": os.pathsep.join(
                (
                    str(MOCK_TRAINER_ROOT / ".venv" / "bin"),
                    os.environ["PATH"],
                )
            ),
            "PYTHONPATH": os.pathsep.join(
                (
                    str(REPOSITORY_ROOT / "training-sdk" / "src"),
                    str(MOCK_TRAINER_ROOT / "src"),
                )
            ),
            "JAX_PLATFORM_NAME": "cpu",
            "BRAX_ACCEPTANCE_TEST_MODE": "1",
            "BRAX_ACCEPTANCE_E2E_FAST": "1",
        }
        previous = {name: os.environ.get(name) for name in environment}
        os.environ.update(environment)
        try:
            try:
                returncode = worker.execute_bundle(uri, self.store)
                failed_reason = (
                    f"worker exited {returncode}" if returncode else None
                )
            except BaseException as error:
                returncode = 1
                failed_reason = f"{type(error).__name__}: {error}"
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        if self.mode == "batch-failed":
            returncode = 1
            failed_reason = "injected Batch FAILED"
        status = "FAILED" if returncode else "SUCCEEDED"
        if self.mode == "child-nonzero":
            # Surface the real completion marker to the controller as an
            # algorithm failure rather than collapsing it into Batch status.
            status = "SUCCEEDED"
        if self.mode == "batch-timeout":
            status = "RUNNING"
        self.statuses[job_id] = JobQuery(
            job_id=job_id,
            status=status,
            status_reason=failed_reason,
        )
        return self.statuses[job_id]

    def query(self, job_ids: list[str]) -> tuple[JobQuery, ...]:
        requested = tuple(job_ids)
        self.query_calls.append(requested)
        self.events.append(("query", requested))
        return tuple(
            self.statuses[job_id] if job_id in self.statuses else self._execute(job_id)
            for job_id in job_ids
        )

    def resubmit(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.resubmit_calls += 1

    def cancel(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.cancel_calls += 1

    def retry(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.retry_calls += 1


def test_fake_batch_submit_defers_worker_until_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        worker,
        "execute_bundle",
        lambda uri, store: calls.append(uri) or 0,
    )
    batch = FakeBatch("s3://bucket/experiments/test/", FakeStore(str(tmp_path)))
    submitted = batch.submit(
        SimpleNamespace(job_id="bundle-1"),
        SimpleNamespace(),
        SimpleNamespace(),
    )

    assert calls == []
    assert batch.query([submitted.job_id]) == (
        JobQuery(job_id=submitted.job_id, status="SUCCEEDED"),
    )
    assert calls == ["s3://bucket/experiments/test/jobs/bundle-1/bundle.json"]


@dataclass
class Harness:
    controller: ExperimentController
    experiment: Path
    stores: list[FakeStore]
    batches: list[FakeBatch]
    aims: list[FakeAim]
    ecr: FakeEcr
    studies: list[optuna.Study]


def _catalog(*, allow_injected_failure: bool) -> ScriptCatalog:
    catalog = load_catalog_index(MOCK_TRAINER_ROOT / "scripts" / "index.yaml")
    if not allow_injected_failure:
        return catalog
    descriptor = catalog.scripts["brax_ppo_acceptance"]
    fields = dict(descriptor.fields)
    fields["failure_mode"] = fields["failure_mode"].model_copy(
        update={
            "choices": (
                "none",
                "before_training",
                "after_training",
                "after_checkpoint",
            )
        }
    )
    return catalog.model_copy(
        update={
            "scripts": {
                "brax_ppo_acceptance": descriptor.model_copy(
                    update={"fields": fields}
                )
            }
        }
    )


def _write_experiment(path: Path, *, failure_mode: str = "none") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "infra-brax-ppo-acceptance",
                    "metadata": {"purpose": "infra-acceptance"},
                },
                "defaults": {
                    "image": CPU_IMAGE_TAG,
                    "environment": {
                        "name": "inverted_pendulum",
                        "options": {"backend": "generalized"},
                    },
                    "resources": {"profile": "c7am"},
                    "training_budget": {"env_steps": 128},
                    "logging": {
                        "aim_every_env_steps": 1,
                        "rerun_every_episodes": 1,
                    },
                    "hpo": {
                        "total_trials": 5,
                        "configs_per_batch": 2,
                        "parameter_policy": "explicit_scan",
                    },
                    "execution": {
                        "runs_per_job": 2,
                        "aim_result_timeout_seconds": 1,
                    },
                    "parameters": {"failure_mode": {"values": [failure_mode]}},
                },
                "groups": {
                    "cpu": {
                        "script": "brax_ppo_acceptance",
                        "parameters": {
                            "learning_rate": {
                                "values": [0.0002, 0.0003, 0.0004, 0.0005, 0.0006]
                            }
                        },
                    },
                    "gpu": {
                        "script": "brax_ppo_acceptance",
                        "image": GPU_IMAGE_TAG,
                        "resources": {"profile": "g6x"},
                        "parameters": {
                            "learning_rate": {
                                "values": [0.0002, 0.0003, 0.0004, 0.0005, 0.0006]
                            }
                        },
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _make_harness(tmp_path: Path, mode: str = "ok") -> Harness:
    ecr = FakeEcr(_catalog(allow_injected_failure=mode == "child-nonzero"))
    catalog_reader = EcrCatalogReader(ecr)
    stores: list[FakeStore] = []
    batches: list[FakeBatch] = []
    studies: list[optuna.Study] = []

    store_mode = mode if mode in {
        "artifact-upload",
        "input-upload",
        "job-upload",
        "marker-missing",
        "marker-tamper",
        "persist-state",
        "persist-report",
        "persist-both",
    } else "ok"

    def store_factory(prefix: str) -> FakeStore:
        store = FakeStore(prefix, store_mode)
        stores.append(store)
        return store

    def batch_factory(prefix: str, store: FakeStore) -> FakeBatch:
        batch = FakeBatch(
            prefix,
            store,
            mode if mode in {
                "batch-failed",
                "batch-timeout",
                "child-nonzero",
                "partial-submit",
            } else "ok",
        )
        batches.append(batch)
        return batch

    def study_factory(**kwargs: Any) -> optuna.Study:
        study = optuna.create_study(
            **kwargs,
            sampler=optuna.samplers.RandomSampler(seed=len(studies) + 17),
        )
        studies.append(study)
        return study

    aims: list[FakeAim] = []
    batch_clock = 0.0

    def clock() -> float:
        nonlocal batch_clock
        value = batch_clock
        batch_clock += 1.0
        return value

    def aim_reader_factory(store: FakeStore) -> FakeAim:
        aim_mode = mode if mode in {
            "aim-failed",
            "aim-timeout",
            "aim-nonfinite",
        } else "ok"
        aim = FakeAim(store, tmp_path, aim_mode)
        aims.append(aim)
        return aim

    controller = ExperimentController(
        catalog_reader=catalog_reader,
        preflight=FakePreflight(),
        store_factory=store_factory,
        batch_factory=batch_factory,
        aim_reader_factory=aim_reader_factory,
        study_factory=study_factory,
        experiment_id_factory=lambda: "task6-exp-001",
        bucket="bucket",
        poll_interval=0.01,
        batch_timeout=1,
        clock=clock,
        sleep=lambda _: None,
    )

    experiment = tmp_path / "experiment.yaml"
    _write_experiment(
        experiment,
        failure_mode="before_training" if mode == "child-nonzero" else "none",
    )
    # Resolve/validate creates neither store nor mutable runtime adapter.
    validation = controller.validate(experiment)
    assert validation.experiment_name == "infra-brax-ppo-acceptance"
    assert stores == batches == studies == []
    return Harness(controller, experiment, stores, batches, aims, ecr, studies)


def _run(harness: Harness) -> Any:
    return harness.controller.run(harness.experiment)


def test_real_facility_lifecycle_mixes_groups_and_preserves_artifact_identity(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)

    report = _run(harness)

    store = harness.stores[0]
    batch = harness.batches[0]
    aim = harness.aims[0]
    assert report.status == "succeeded"
    assert report.experiment_id == "task6-exp-001"
    assert report.experiment_name == "infra-brax-ppo-acceptance"
    assert report.experiment_metadata == {"purpose": "infra-acceptance"}
    assert report.completed_runs == 10
    assert len(report.submitted_job_ids) == 6
    assert report.submitted_job_ids == (
        "batch-1",
        "batch-2",
        "batch-3",
        "batch-4",
        "batch-5",
        "batch-6",
    )
    assert [len(call) for call in batch.query_calls] == [2, 2, 2]
    assert batch.events == [
        ("submit", "batch-1"),
        ("submit", "batch-2"),
        ("query", ("batch-1", "batch-2")),
        ("submit", "batch-3"),
        ("submit", "batch-4"),
        ("query", ("batch-3", "batch-4")),
        ("submit", "batch-5"),
        ("submit", "batch-6"),
        ("query", ("batch-5", "batch-6")),
    ]

    bundles = batch.submitted
    assert [len(bundle.runs) for bundle in bundles] == [2, 2, 2, 2, 1, 1]
    assert {run.run_context["group"] for bundle in bundles for run in bundle.runs} == {
        "cpu",
        "gpu",
    }
    assert sum(bundle.resource_profile == "c7am" for bundle in bundles) == 3
    assert sum(bundle.resource_profile == "g6x" for bundle in bundles) == 3
    for group, profile, digest in (
        ("cpu", "c7am", CPU_IMAGE_DIGEST),
        ("gpu", "g6x", GPU_IMAGE_DIGEST),
    ):
        group_bundles = [
            bundle
            for bundle in bundles
            if bundle.runs[0].run_context["group"] == group
        ]
        assert [
            (bundle.resource_profile, bundle.image_digest, len(bundle.runs))
            for bundle in group_bundles
        ] == [(profile, digest, 2), (profile, digest, 2), (profile, digest, 1)]
    assert [
        sum(run.run_context["group"] == group for bundle in bundles for run in bundle.runs)
        for group in ("cpu", "gpu")
    ] == [5, 5]
    assert {
        run.run_context["run_number"]
        for bundle in bundles
        for run in bundle.runs
        if run.run_context["group"] == "cpu"
    } == {1, 2, 3, 4, 5}

    for bundle in bundles:
        expected_image, expected_profile = {
            "cpu": (CPU_IMAGE_DIGEST, "c7am"),
            "gpu": (GPU_IMAGE_DIGEST, "g6x"),
        }[bundle.runs[0].run_context["group"]]
        assert bundle.image_digest == expected_image
        assert bundle.resource_profile == expected_profile
        assert {run.run_context["group"] for run in bundle.runs} == {
            bundle.runs[0].run_context["group"]
        }
        for run in bundle.runs:
            context = run.run_context
            config = yaml.safe_load(run.config_yaml)
            assert run.attempt == 0
            assert run.image_digest == expected_image
            assert run.resource_profile == expected_profile
            assert context["experiment_name"] == "infra-brax-ppo-acceptance"
            assert context["experiment_id"] == "task6-exp-001"
            assert context["run_id"] == run.run_id
            assert context["script"] == "brax_ppo_acceptance"
            assert context["metadata"] == {"purpose": "infra-acceptance"}
            assert context["image_digest"] == expected_image
            assert context["resource_profile"] == expected_profile
            assert config["protocol_version"] == "1"
            assert config["environment"] == {
                "name": "inverted_pendulum",
                "options": {"backend": "generalized"},
            }
            assert config["parameters"]["runtime"]["seed"] == 0
            assert config["parameters"]["algorithm"]["failure_mode"] == "none"
            assert config["parameters"]["algorithm"]["learning_rate"] in {
                0.0002,
                0.0003,
                0.0004,
                0.0005,
                0.0006,
            }

    expected_marker_order = [
        "s3://bucket/"
        + run.artifact_prefix.removesuffix("input/")
        + "status/attempt-0.json"
        for bundle in bundles
        for run in bundle.runs
    ]
    assert store.marker_writes == expected_marker_order
    artifact_owners: dict[str, str] = {}
    for bundle in bundles:
        for run in bundle.runs:
            run_root = "s3://bucket/" + run.artifact_prefix.removesuffix("input/")
            marker = store.get_json(f"{run_root}status/attempt-0.json")
            assert marker["run_id"] == run.run_id
            assert marker["attempt"] == 0
            assert marker["exit_code"] == 0
            artifacts = tuple(marker["artifacts"])
            assert len(artifacts) == 3
            assert all(uri.startswith(run_root) for uri in artifacts)
            assert sum(uri == f"{run_root}aim-buffer/events.jsonl" for uri in artifacts) == 1
            assert sum(
                uri == f"{run_root}checkpoints/ppo-params.npz"
                for uri in artifacts
            ) == 1
            assert sum(
                uri.startswith(f"{run_root}rerun/")
                and uri.endswith("/episode-000002.rrd")
                for uri in artifacts
            ) == 1
            for uri in artifacts:
                assert uri not in artifact_owners
                artifact_owners[uri] = run.run_id
                assert uri in store.objects
                assert store.digests[uri] == hashlib.sha256(store.objects[uri]).hexdigest()
    assert len(artifact_owners) == 30
    assert set(aim.replayed) == {
        run.run_id for bundle in bundles for run in bundle.runs
    }
    assert {
        trial.state
        for study in harness.studies
        for trial in study.trials
        if trial.state != TrialState.PRUNED
    } == {TrialState.COMPLETE}
    studies = {
        study.study_name.rsplit(":", 1)[1]: study for study in harness.studies
    }
    for bundle in bundles:
        for run in bundle.runs:
            run_id = run.run_id
            aim_objective = aim.objective_for_run(
                run_id,
                "eval/episode_return",
            )
            spool_events = aim.spool_events_for_run(run_id)
            finalized = next(
                event
                for event in spool_events
                if event.kind == "final" and event.data["finalized"]
            )
            spool_objective = float(finalized.metric_value)
            trial = studies[run.run_context["group"]].trials[
                run.run_context["trial_number"]
            ]
            assert trial.state == TrialState.COMPLETE
            assert math.isfinite(spool_objective)
            assert spool_objective == aim_objective == trial.value
    persisted_report = store.get_json(
        "s3://bucket/experiments/task6-exp-001/report.json"
    )
    assert persisted_report["experiment_name"] == "infra-brax-ppo-acceptance"
    assert persisted_report["experiment_metadata"] == {"purpose": "infra-acceptance"}
    assert batch.resubmit_calls == batch.cancel_calls == batch.retry_calls == 0
    one_resolution = [
        ("resolve_tag", CPU_IMAGE_TAG),
        ("get_manifest", CPU_IMAGE_DIGEST),
        (
            "get_config_blob",
            CPU_IMAGE_DIGEST.split("@")[0],
            CPU_CONFIG_DIGEST,
        ),
        ("resolve_tag", GPU_IMAGE_TAG),
        ("get_manifest", GPU_IMAGE_DIGEST),
        (
            "get_config_blob",
            GPU_IMAGE_DIGEST.split("@")[0],
            GPU_CONFIG_DIGEST,
        ),
    ]
    assert harness.ecr.calls == one_resolution * 2


@pytest.mark.parametrize(
    "mode",
    [
        "batch-failed",
        "child-nonzero",
        "artifact-upload",
        "marker-missing",
        "marker-tamper",
        "aim-failed",
        "aim-timeout",
        "aim-nonfinite",
    ],
)
def test_every_runtime_failure_stops_future_rounds_without_retry(
    tmp_path: Path,
    mode: str,
) -> None:
    harness = _make_harness(tmp_path, mode)

    with pytest.raises(ExperimentRunError) as raised:
        _run(harness)

    batch = harness.batches[0]
    report = raised.value.report
    assert report.status == "failed"
    assert report.experiment_name == "infra-brax-ppo-acceptance"
    assert report.experiment_metadata == {"purpose": "infra-acceptance"}
    assert len(batch.submitted) == 2
    assert report.submitted_job_ids == ("batch-1", "batch-2")
    assert batch.resubmit_calls == batch.cancel_calls == batch.retry_calls == 0
    assert all(
        sum(trial.state == TrialState.FAIL for trial in study.trials) == 2
        for study in harness.studies
    )
    assert not any(
        trial.state in {TrialState.RUNNING, TrialState.COMPLETE}
        for study in harness.studies
        for trial in study.trials
    )
    persisted = harness.stores[0].get_json(
        "s3://bucket/experiments/task6-exp-001/report.json"
    )
    assert persisted["experiment_name"] == "infra-brax-ppo-acceptance"
    assert persisted["experiment_metadata"] == {"purpose": "infra-acceptance"}


@pytest.mark.parametrize(
    ("mode", "submitted_ids", "submit_attempts", "cause_type", "cause_match"),
    [
        (
            "batch-timeout",
            ("batch-1", "batch-2"),
            2,
            TimeoutError,
            "timed out waiting for Batch jobs",
        ),
        ("input-upload", (), 0, OSError, "run input upload failed"),
        ("job-upload", (), 0, OSError, "job bundle upload failed"),
        (
            "partial-submit",
            ("batch-1",),
            2,
            RuntimeError,
            "second Batch submit failed",
        ),
    ],
)
def test_submission_failures_preserve_accepted_ids_and_fail_all_pending_trials(
    tmp_path: Path,
    mode: str,
    submitted_ids: tuple[str, ...],
    submit_attempts: int,
    cause_type: type[BaseException],
    cause_match: str,
) -> None:
    harness = _make_harness(tmp_path, mode)

    with pytest.raises(ExperimentRunError) as raised:
        _run(harness)

    batch = harness.batches[0]
    assert isinstance(raised.value.__cause__, cause_type)
    assert cause_match in str(raised.value.__cause__)
    assert raised.value.report.submitted_job_ids == submitted_ids
    assert raised.value.report.experiment_name == "infra-brax-ppo-acceptance"
    assert raised.value.report.experiment_metadata == {"purpose": "infra-acceptance"}
    assert batch.submit_attempts == submit_attempts
    assert len(batch.submitted) == len(submitted_ids)
    assert batch.resubmit_calls == batch.cancel_calls == batch.retry_calls == 0
    assert all(
        sum(trial.state == TrialState.FAIL for trial in study.trials) == 2
        for study in harness.studies
    )
    assert not any(
        trial.state in {TrialState.RUNNING, TrialState.COMPLETE}
        for study in harness.studies
        for trial in study.trials
    )


@pytest.mark.parametrize(
    ("mode", "error_types"),
    [
        ("persist-state", (OSError,)),
        ("persist-report", (PermissionError,)),
        ("persist-both", (OSError, PermissionError)),
    ],
)
def test_persistence_failures_are_one_shot_and_keep_all_submitted_ids(
    tmp_path: Path,
    mode: str,
    error_types: tuple[type[BaseException], ...],
) -> None:
    harness = _make_harness(tmp_path, mode)

    with pytest.raises(ExperimentRunError) as raised:
        _run(harness)

    batch = harness.batches[0]
    store = harness.stores[0]
    assert raised.value.report.submitted_job_ids == tuple(
        f"batch-{index}" for index in range(1, 7)
    )
    assert raised.value.report.experiment_name == "infra-brax-ppo-acceptance"
    assert raised.value.report.experiment_metadata == {"purpose": "infra-acceptance"}
    assert tuple(type(error) for error in raised.value.persistence_errors) == error_types
    assert sum(uri.endswith("state/final.json") for uri in store.put_json_calls) == 1
    assert sum(uri.endswith("report.json") for uri in store.put_json_calls) == 1
    assert batch.resubmit_calls == batch.cancel_calls == batch.retry_calls == 0


def test_finite_spaces_exit_before_nominal_budget_without_extra_submission(
    tmp_path: Path,
) -> None:
    harness = _make_harness(tmp_path)
    experiment = harness.experiment
    payload = yaml.safe_load(experiment.read_text())
    payload["defaults"]["hpo"]["total_trials"] = 8
    payload["groups"]["cpu"]["parameters"]["learning_rate"]["values"] = [
        0.0002,
        0.0003,
        0.0004,
    ]
    payload["groups"]["gpu"]["parameters"]["learning_rate"]["values"] = [
        0.0002,
        0.0003,
        0.0004,
    ]
    experiment.write_text(yaml.safe_dump(payload), encoding="utf-8")

    report = _run(harness)

    batch = harness.batches[0]
    assert report.completed_runs == 6
    assert len(batch.submitted) == 4
    assert [len(call) for call in batch.query_calls] == [2, 2]
    assert all(
        trial.state != TrialState.RUNNING
        for study in harness.studies
        for trial in study.trials
    )
