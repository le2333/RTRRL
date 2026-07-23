from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def preflight():
    return _load("facility_preflight")


@pytest.fixture
def deploy():
    return _load("deploy_facility")


class FakeSts:
    def get_caller_identity(self) -> dict[str, str]:
        return {
            "Account": "007122174918",
            "Arn": "arn:aws:sts::007122174918:assumed-role/trainer-control/session",
        }


class FakeBatch:
    meta = SimpleNamespace(region_name="eu-north-1")

    def __init__(self, preflight: Any) -> None:
        self.preflight = preflight
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def describe_compute_environments(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("describe_compute_environments", deepcopy(kwargs)))
        name = kwargs["computeEnvironments"][0]
        expected = next(
            item for item in self.preflight.COMPUTE_ENVIRONMENTS if item["name"] == name
        )
        return {
            "computeEnvironments": [
                {
                    "computeEnvironmentName": name,
                    "computeEnvironmentArn": f"arn:aws:batch:eu-north-1:007122174918:ce/{name}",
                    "state": "ENABLED",
                    "status": "VALID",
                    "computeResources": {
                        "instanceTypes": [expected["instance_type"]],
                    },
                }
            ]
        }

    def describe_job_queues(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("describe_job_queues", deepcopy(kwargs)))
        name = kwargs["jobQueues"][0]
        expected = next(item for item in self.preflight.QUEUES if item["name"] == name)
        return {
            "jobQueues": [
                {
                    "jobQueueName": name,
                    "state": "ENABLED",
                    "status": "VALID",
                    "priority": expected["priority"],
                    "computeEnvironmentOrder": [
                        {
                            "order": 1,
                            "computeEnvironment": (
                                "arn:aws:batch:eu-north-1:007122174918:ce/"
                                + expected["environment"]
                            ),
                        }
                    ],
                }
            ]
        }


class FakeS3:
    meta = SimpleNamespace(region_name="eu-north-1")

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def head_bucket(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("head_bucket", deepcopy(kwargs)))
        return {}

    def get_bucket_location(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_bucket_location", deepcopy(kwargs)))
        return {"LocationConstraint": "eu-north-1"}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_objects_v2", deepcopy(kwargs)))
        return {"KeyCount": 0}


class FakeEcr:
    meta = SimpleNamespace(region_name="eu-north-1")

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def describe_repositories(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("describe_repositories", deepcopy(kwargs)))
        return {
            "repositories": [
                {
                    "repositoryName": "rtrrl",
                    "repositoryUri": (
                        "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl"
                    ),
                }
            ]
        }

    def batch_get_image(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("batch_get_image", deepcopy(kwargs)))
        tag = kwargs["imageIds"][0]["imageTag"]
        return {
            "failures": [],
            "images": [
                {
                    "imageId": {
                        "imageDigest": (
                            "sha256:" + ("a" if tag.endswith("cpu") else "b") * 64
                        ),
                        "imageTag": tag,
                    },
                }
            ]
        }


class FakeIam:
    meta = SimpleNamespace(region_name="aws-global")

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def simulate_principal_policy(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("simulate_principal_policy", deepcopy(kwargs)))
        return {
            "EvaluationResults": [
                {"EvalActionName": action, "EvalDecision": "allowed"}
                for action in kwargs["ActionNames"]
            ],
            "IsTruncated": False,
        }


class FakeSession:
    region_name = "eu-north-1"

    def __init__(self, preflight: Any) -> None:
        self.clients = {
            "sts": FakeSts(),
            "batch": FakeBatch(preflight),
            "s3": FakeS3(),
            "ecr": FakeEcr(),
            "iam": FakeIam(),
        }

    def client(self, name: str) -> Any:
        return self.clients[name]


def _config(preflight: Any, tmp_path: Path) -> Any:
    scratch = tmp_path / "task7-scratch"
    scratch.mkdir()
    (scratch / ".aim").mkdir()
    main = tmp_path / "main"
    main.mkdir()
    return preflight.PreflightConfig(
        aim_repo=scratch,
        main_repo=main,
        aim_endpoint="aim://127.0.0.1:53801",
    )


def test_read_only_preflight_reports_stable_complete_json(
    preflight: Any, tmp_path: Path
) -> None:
    session = FakeSession(preflight)

    report = preflight.run_preflight(
        session,
        _config(preflight, tmp_path),
        aim_probe=lambda _endpoint: {"pid": 123, "healthy": True},
    )

    assert report["status"] == "ready"
    assert report["account"]["actual"] == "007122174918"
    assert [item["profile"] for item in report["profiles"]] == [
        "c7am",
        "c7al",
        "c7ax",
        "g6x",
    ]
    assert [item["run_queue"]["priority"] for item in report["profiles"]] == [
        100,
        100,
        100,
        100,
    ]
    assert [item["dev_queue"]["priority"] for item in report["profiles"]] == [
        10,
        10,
        10,
        10,
    ]
    assert report["s3"]["prefix"] == "experiments/"
    assert report["ecr"]["images"]["memorax-rtrl-facility-cpu"]["status"] == "visible"
    assert report["iam"]["status"] == "allowed"
    assert report["aim"]["isolated"] is True
    assert json.loads(preflight.stable_json(report)) == report
    assert preflight.stable_json(report) == preflight.stable_json(report)

    aws_calls = [
        name
        for client in session.clients.values()
        for name, _kwargs in getattr(client, "calls", [])
        if isinstance(name, str)
    ]
    assert set(aws_calls) <= {
        "describe_compute_environments",
        "describe_job_queues",
        "head_bucket",
        "get_bucket_location",
        "list_objects_v2",
        "describe_repositories",
        "batch_get_image",
        "simulate_principal_policy",
    }


def test_preflight_reports_iam_read_permission_block_without_mutation(
    preflight: Any, tmp_path: Path
) -> None:
    session = FakeSession(preflight)

    def denied(**_kwargs: Any) -> dict[str, Any]:
        raise PermissionError("iam:SimulatePrincipalPolicy is not authorized")

    session.clients["iam"].simulate_principal_policy = denied

    report = preflight.run_preflight(
        session,
        _config(preflight, tmp_path),
        aim_probe=lambda _endpoint: {"pid": 123, "healthy": True},
    )

    assert report["status"] == "blocked"
    assert report["iam"]["status"] == "blocked"
    assert "SimulatePrincipalPolicy" in report["iam"]["error"]


def test_preflight_rejects_wrong_account_and_nonisolated_aim(
    preflight: Any, tmp_path: Path
) -> None:
    session = FakeSession(preflight)
    session.clients["sts"].get_caller_identity = lambda: {
        "Account": "123456789012",
        "Arn": "arn:aws:iam::123456789012:user/wrong",
    }
    config = _config(preflight, tmp_path)
    config = preflight.PreflightConfig(
        aim_repo=config.main_repo,
        main_repo=config.main_repo,
        aim_endpoint=config.aim_endpoint,
    )

    report = preflight.run_preflight(
        session,
        config,
        aim_probe=lambda _endpoint: {"pid": 123, "healthy": True},
    )

    assert report["status"] == "failed"
    assert report["account"]["status"] == "mismatch"
    assert report["aim"]["status"] == "failed"
    assert report["aim"]["isolated"] is False


def test_deploy_default_is_dry_run_and_has_no_submit_or_cleanup(
    deploy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess_calls: list[list[str]] = []
    monkeypatch.setattr(
        deploy.subprocess,
        "run",
        lambda command, **_kwargs: subprocess_calls.append(command),
    )
    session_calls: list[str] = []

    class NoAws:
        def client(self, name: str) -> Any:
            session_calls.append(name)
            raise AssertionError("dry-run must not construct AWS clients")

    plan = deploy.deploy(deploy.DeployConfig(), session=NoAws())

    assert plan["mode"] == "dry-run"
    assert plan["requested"] == {"build": False, "push": False, "register": False}
    assert subprocess_calls == []
    assert session_calls == []
    source = (SCRIPTS / "deploy_facility.py").read_text()
    assert "submit_job(" not in source
    assert "terminate_job(" not in source
    assert "delete_" not in source
    assert "deregister_job_definition(" not in source


class RegisterBatch:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def register_job_definition(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(deepcopy(kwargs))
        return {
            "jobDefinitionArn": (
                "arn:aws:batch:eu-north-1:007122174918:job-definition/"
                f"{kwargs['jobDefinitionName']}:1"
            )
        }


def test_register_is_explicit_digest_bound_and_single_attempt(deploy: Any) -> None:
    batch = RegisterBatch()
    session = SimpleNamespace(client=lambda name: batch if name == "batch" else None)
    cpu = "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:" + "a" * 64
    gpu = "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:" + "b" * 64
    config = deploy.DeployConfig(
        register=True,
        cpu_digest=cpu,
        gpu_digest=gpu,
        job_role_arn="arn:aws:iam::007122174918:role/rtrrl-batch-job-role",
        execution_role_arn=(
            "arn:aws:iam::007122174918:role/rtrrl-batch-execution-role"
        ),
    )

    report = deploy.deploy(config, session=session)

    assert report["mode"] == "execute"
    assert len(batch.calls) == 4
    assert [call["jobDefinitionName"] for call in batch.calls] == [
        f"trainer-c7am-{'a' * 64}",
        f"trainer-c7al-{'a' * 64}",
        f"trainer-c7ax-{'a' * 64}",
        f"trainer-g6x-{'b' * 64}",
    ]
    assert all(call["retryStrategy"] == {"attempts": 1} for call in batch.calls)
    assert [call["containerProperties"]["image"] for call in batch.calls] == [
        cpu,
        cpu,
        cpu,
        gpu,
    ]
    assert all(
        call["containerProperties"]["command"] == ["python", "/opt/trainer/worker.py"]
        for call in batch.calls
    )


def test_push_and_register_require_their_own_explicit_inputs(deploy: Any) -> None:
    with pytest.raises(ValueError, match="--push requires --build"):
        deploy.deploy(deploy.DeployConfig(push=True), session=SimpleNamespace())
    with pytest.raises(ValueError, match="digest"):
        deploy.deploy(deploy.DeployConfig(register=True), session=SimpleNamespace())
