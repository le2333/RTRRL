from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
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
MEMO_ROOT = REPOSITORY_ROOT / "memo"
WORKER_PATH = REPOSITORY_ROOT / "rtrrl" / "infra" / "worker" / "worker.py"
IMAGE_TAG = "123456789012.dkr.ecr.eu-north-1.amazonaws.com/memo:task6"
IMAGE_DIGEST = (
    "123456789012.dkr.ecr.eu-north-1.amazonaws.com/memo@sha256:" + "a" * 64
)
CONFIG_DIGEST = "sha256:" + "b" * 64

WORKER_SPEC = importlib.util.spec_from_file_location("task6_trainer_worker", WORKER_PATH)
assert WORKER_SPEC is not None and WORKER_SPEC.loader is not None
worker = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(worker)


FIXTURE_SOURCE = r"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import types

memo_root = Path("__MEMO_ROOT__")
experiments = memo_root / "experiments"
for item in (memo_root, experiments):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from base.facility import FacilityInput, build_rtrrl_config, build_stream_ac_config
from training_sdk import Episode, bootstrap_from_environment
from training_sdk.spool import AimUnavailable


@dataclass
class StreamConfig:
    experiment: str = "task6_stream"
    agent_type: str = "rtu_rtrl"
    seed: int = 0
    hidden_dim: int = 32
    encoder_dim: int = 32
    gamma: float = 0.99
    trace_lambda: float = 0.9
    actor_lr: float = 1.0
    critic_lr: float = 1.0
    entropy_coefficient: float = 0.01
    num_envs: int = 16
    num_epochs: int = 1
    chain_length: int = 16
    num_bits: int = 1
    env_name: str = "hopper"
    mode: str = "F"
    backend: str = "spring"
    max_episode_steps: int = 16
    total_timesteps: int = 800
    patience: int = 0
    require_full_budget: bool = False


@dataclass
class RtrrlConfig:
    experiment: str = "task6_rtrrl"
    rtrrl_topology: str = "shared"
    seed: int = 0
    backbone: str = "lru"
    hidden_dim: int = 32
    gamma: float = 0.95
    lambda_pi: float = 0.97
    lambda_v: float = 0.9
    lambda_rnn: float = 0.945
    td_lr: float = 0.00003
    rnn_lr: float = 0.000002
    eta_pi: float = 0.38
    eta_f: float = 0.5
    entropy_rate: float = 0.00003
    num_envs: int = 1
    num_epochs: int = 1
    env_name: str = "hopper"
    mode: str = "F"
    backend: str = "spring"
    max_episode_steps: int = 1000
    normalize_obs: bool = True
    normalize_reward: bool = True
    total_timesteps: int = 800
    patience: int = 0
    require_full_budget: bool = False


def install_launcher_fixtures():
    fixtures = {
        "stream_ac_memorychain.run": ("StreamACMemoryChainConfig", StreamConfig),
        "stream_ac_kmemorychain.run": ("StreamACKMemoryChainConfig", StreamConfig),
        "stream_ac_mujoco_masked.run": ("StreamACMujocoMaskedConfig", StreamConfig),
        "rtrrl_hopper.run": ("RTRRLHopperConfig", RtrrlConfig),
    }
    for module_name, (class_name, config_type) in fixtures.items():
        package_name, _ = module_name.rsplit(".", 1)
        sys.modules.setdefault(package_name, types.ModuleType(package_name))
        module = types.ModuleType(module_name)
        setattr(module, class_name, config_type)
        sys.modules[module_name] = module


class OfflineAim:
    def start(self, context):
        self.context = context

    def send(self, event):
        raise AimUnavailable("Task 6 fake Aim is offline in the worker")

    def fail(self, metadata):
        del metadata

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    install_launcher_fixtures()
    value = FacilityInput.load(args.config)
    if args.launcher == "memo_stream_ac":
        config = build_stream_ac_config(value)
    elif args.launcher == "memo_rtrrl":
        config = build_rtrrl_config(value)
    else:
        raise ValueError(args.launcher)

    run = bootstrap_from_environment(
        aim_factory=lambda context, environ: OfflineAim(),
    )
    assert run is not None
    if __import__("os").environ.get("TASK6_CHILD_NONZERO") == "1":
        run.fail(RuntimeError("fixture child failed"))
        return 7

    run.log_metrics(4, {"train/loss": 0.25})
    run.log_episode_summary(
        env_steps=4,
        episode_return=2.5,
        episode_length=1,
    )
    run.log_episode(
        Episode(
            number=1,
            phase="eval",
            start_env_steps=4,
            end_env_steps=4,
            observations=((0.0,), (1.0,)),
            actions=((0.5,),),
            rewards=(2.5,),
            terminals=(True,),
            truncations=(False,),
        )
    )
    checkpoint = run.context.artifact_directory / "fixture-checkpoint.bin"
    checkpoint.write_text(
        f"{args.launcher}:{config.hidden_dim}:{run.context.run_id}",
        encoding="utf-8",
    )
    run.register_checkpoint(checkpoint)
    run.finish({"eval/rewards": float(config.hidden_dim)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


class FakeEcr:
    def __init__(self, catalog: ScriptCatalog) -> None:
        self.label = encode_catalog(catalog)
        self.calls: list[tuple[Any, ...]] = []

    def resolve_tag(self, reference: str) -> str:
        self.calls.append(("resolve_tag", reference))
        return IMAGE_DIGEST

    def get_manifest(self, reference: str) -> Mapping[str, Any]:
        self.calls.append(("get_manifest", reference))
        return {"config": {"digest": CONFIG_DIGEST}}

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


class FakePreflight:
    def validate(self, resolved: Any) -> dict[str, ValidatedJobDefinition]:
        assert {group.resources.profile for group in resolved.groups} == {"c7am"}
        return {
            "c7am": ValidatedJobDefinition(
                arn=(
                    "arn:aws:batch:eu-north-1:123456789012:job-definition/"
                    f"trainer-c7am-{'a' * 64}:1"
                ),
                image_digest=IMAGE_DIGEST,
                resource_profile="c7am",
            )
        }


class FakeBatch:
    def __init__(
        self,
        prefix: str,
        store: FakeStore,
        fixture: Path,
        mode: str = "ok",
    ) -> None:
        self.prefix = prefix
        self.store = store
        self.fixture = fixture
        self.mode = mode
        self.submitted: list[Any] = []
        self.statuses: dict[str, JobQuery] = {}
        self.query_calls: list[tuple[str, ...]] = []
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
        self.submitted.append(bundle)
        if self.mode == "partial-submit" and self.submit_attempts == 2:
            raise RuntimeError("second Batch submit failed")
        job_id = f"batch-{self.submit_attempts}"
        uri = f"{self.prefix}jobs/{bundle.job_id}/bundle.json"
        previous = os.environ.get("TASK6_CHILD_NONZERO")
        previous_pythonpath = os.environ.get("PYTHONPATH")
        sdk_source = str(REPOSITORY_ROOT / "training-sdk" / "src")
        os.environ["PYTHONPATH"] = (
            sdk_source
            if previous_pythonpath is None
            else f"{sdk_source}{os.pathsep}{previous_pythonpath}"
        )
        if self.mode == "child-nonzero":
            os.environ["TASK6_CHILD_NONZERO"] = "1"
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
            if previous is None:
                os.environ.pop("TASK6_CHILD_NONZERO", None)
            else:
                os.environ["TASK6_CHILD_NONZERO"] = previous
            if previous_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pythonpath
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
        return SubmittedJob(job_id=job_id, bundle_id=bundle.job_id)

    def query(self, job_ids: list[str]) -> tuple[JobQuery, ...]:
        self.query_calls.append(tuple(job_ids))
        return tuple(self.statuses[job_id] for job_id in job_ids)

    def resubmit(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.resubmit_calls += 1

    def cancel(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.cancel_calls += 1

    def retry(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.retry_calls += 1


@dataclass
class Harness:
    controller: ExperimentController
    experiment: Path
    stores: list[FakeStore]
    batches: list[FakeBatch]
    aims: list[FakeAim]
    ecr: FakeEcr
    studies: list[optuna.Study]


def _catalog(fixture: Path) -> ScriptCatalog:
    catalog = load_catalog_index(MEMO_ROOT / "infra" / "scripts" / "index.yaml")
    scripts = {}
    for name, descriptor in catalog.scripts.items():
        scripts[name] = descriptor.model_copy(
            update={
                "argv": (
                    sys.executable,
                    str(fixture),
                    "--launcher",
                    name,
                    "--config",
                    "{config_path}",
                )
            }
        )
    return catalog.model_copy(update={"scripts": scripts})


def _write_experiment(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "task6-complete-facility"},
                "defaults": {
                    "image": IMAGE_TAG,
                    "resources": {"profile": "c7am"},
                    "training_budget": {"env_steps": 800},
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
                    "parameters": {
                        "seed": {"values": [11]},
                    },
                },
                "groups": {
                    "stream": {
                        "script": "memo_stream_ac",
                        "parameters": {
                            "hidden_dim": {"values": [64, 96, 128, 160, 192]}
                        },
                    },
                    "hopper": {
                        "script": "memo_rtrrl",
                        "parameters": {
                            "hidden_dim": {"values": [16, 24, 32, 48, 64]}
                        },
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _make_harness(tmp_path: Path, mode: str = "ok") -> Harness:
    fixture = tmp_path / "task6_launcher_fixture.py"
    fixture.write_text(
        FIXTURE_SOURCE.replace("__MEMO_ROOT__", str(MEMO_ROOT)),
        encoding="utf-8",
    )
    ecr = FakeEcr(_catalog(fixture))
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
            fixture,
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
    _write_experiment(experiment)
    # Resolve/validate creates neither store nor mutable runtime adapter.
    validation = controller.validate(experiment)
    assert validation.experiment_name == "task6-complete-facility"
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
    assert report.experiment_name == "task6-complete-facility"
    assert report.completed_runs == 10
    assert len(report.submitted_job_ids) == 5
    assert [len(call) for call in batch.query_calls] == [2, 2, 1]

    bundles = batch.submitted
    assert len(bundles) == 5
    assert all(
        {run.run_context["group"] for run in bundle.runs} == {"stream", "hopper"}
        for bundle in bundles
    )
    assert [
        sum(run.run_context["group"] == group for bundle in bundles for run in bundle.runs)
        for group in ("stream", "hopper")
    ] == [5, 5]
    assert {
        run.run_context["run_number"]
        for bundle in bundles
        for run in bundle.runs
        if run.run_context["group"] == "stream"
    } == {1, 2, 3, 4, 5}

    for bundle in bundles:
        assert bundle.image_digest == IMAGE_DIGEST
        assert bundle.resource_profile == "c7am"
        for run in bundle.runs:
            context = run.run_context
            config = yaml.safe_load(run.config_yaml)
            assert run.attempt == 0
            assert run.image_digest == IMAGE_DIGEST
            assert run.resource_profile == "c7am"
            assert context["experiment_name"] == "task6-complete-facility"
            assert context["experiment_id"] == "task6-exp-001"
            assert context["run_id"] == run.run_id
            assert context["image_digest"] == IMAGE_DIGEST
            assert context["resource_profile"] == "c7am"
            assert config["protocol_version"] == "1"
            assert config["parameters"]["runtime"]["seed"] == 11

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
                uri == f"{run_root}checkpoints/fixture-checkpoint.bin"
                for uri in artifacts
            ) == 1
            assert sum(
                uri.startswith(f"{run_root}rerun/")
                and uri.endswith("/episode-000001.rrd")
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
    assert sorted(
        trial.value
        for study in harness.studies
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
    ) == [16.0, 24.0, 32.0, 48.0, 64.0, 64.0, 96.0, 128.0, 160.0, 192.0]
    assert batch.resubmit_calls == batch.cancel_calls == batch.retry_calls == 0
    assert harness.ecr.calls == [
        ("resolve_tag", IMAGE_TAG),
        ("get_manifest", IMAGE_DIGEST),
        (
            "get_config_blob",
            IMAGE_DIGEST.split("@")[0],
            CONFIG_DIGEST,
        ),
        ("resolve_tag", IMAGE_TAG),
        ("get_manifest", IMAGE_DIGEST),
        (
            "get_config_blob",
            IMAGE_DIGEST.split("@")[0],
            CONFIG_DIGEST,
        ),
    ]


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
    assert batch.submit_attempts == submit_attempts
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
        f"batch-{index}" for index in range(1, 6)
    )
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
    payload["groups"]["stream"]["parameters"]["hidden_dim"]["values"] = [64, 96, 128]
    payload["groups"]["hopper"]["parameters"]["hidden_dim"]["values"] = [16, 24, 32]
    experiment.write_text(yaml.safe_dump(payload), encoding="utf-8")

    report = _run(harness)

    batch = harness.batches[0]
    assert report.completed_runs == 6
    assert len(batch.submitted) == 3
    assert [len(call) for call in batch.query_calls] == [2, 1]
    assert all(
        trial.state != TrialState.RUNNING
        for study in harness.studies
        for trial in study.trials
    )
