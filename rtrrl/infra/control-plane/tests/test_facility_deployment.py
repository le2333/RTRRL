from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from trainer_infra.queues import (
    ACCOUNT_ID,
    EXECUTION_ROLE_ARN,
    JOB_LOG_GROUP,
    JOB_ROLE_ARN,
    QUEUES,
    REGION,
)

SCRIPT = Path(__file__).parents[1] / "scripts" / "deploy_facility.py"
REGISTRY = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com"
CPU_IMAGE = f"{REGISTRY}/rtrrl@sha256:{'a' * 64}"
GPU_IMAGE = f"{REGISTRY}/rtrrl@sha256:{'b' * 64}"


@pytest.fixture(scope="module")
def script() -> Any:
    spec = importlib.util.spec_from_file_location("deploy_facility", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBatch:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def register_job_definition(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"jobDefinitionArn": f"arn:aws:batch:::job-definition/{len(self.calls)}"}


class FakeLogs:
    def __init__(self, *, exists: bool = False, create_error: str | None = None) -> None:
        self.exists = exists
        self.create_error = create_error
        self.created: list[str] = []
        self.retention: list[tuple[str, int]] = []

    def create_log_group(self, logGroupName: str) -> None:  # noqa: N803 - botocore casing
        code = self.create_error or ("ResourceAlreadyExistsException" if self.exists else None)
        if code is not None:
            raise ClientError({"Error": {"Code": code, "Message": code}}, "CreateLogGroup")
        self.created.append(logGroupName)

    def put_retention_policy(self, logGroupName: str, retentionInDays: int) -> None:  # noqa: N803
        self.retention.append((logGroupName, retentionInDays))


class FakeSts:
    def __init__(self, account: str = ACCOUNT_ID) -> None:
        self.account = account

    def get_caller_identity(self) -> dict[str, str]:
        return {"Account": self.account}


class FakeSession:
    def __init__(
        self,
        *,
        batch: FakeBatch | None = None,
        logs: FakeLogs | None = None,
        sts: FakeSts | None = None,
    ) -> None:
        self.batch = batch or FakeBatch()
        self.logs = logs or FakeLogs()
        self.sts = sts or FakeSts()

    def client(self, service: str) -> Any:
        if service == "batch":
            return self.batch
        if service == "logs":
            return self.logs
        if service == "sts":
            return self.sts
        raise AssertionError(f"unexpected client {service!r}")


def test_default_is_a_dry_run_that_touches_nothing(script: Any) -> None:
    session = FakeSession()
    report = script.deploy(session=session)

    assert report["mode"] == "dry-run"
    assert session.batch.calls == []
    assert session.logs.created == []
    assert len(report["planned_definitions"]) == len(QUEUES)


def test_register_binds_every_profile_to_a_digest(script: Any) -> None:
    session = FakeSession()

    report = script.deploy(
        register=True,
        confirm_account=ACCOUNT_ID,
        cpu_digest=CPU_IMAGE,
        gpu_digest=GPU_IMAGE,
        session=session,
    )

    assert len(session.batch.calls) == len(QUEUES)
    assert len(report["job_definitions"]) == len(QUEUES)
    by_name = {call["jobDefinitionName"]: call for call in session.batch.calls}
    for binding in QUEUES.values():
        image = GPU_IMAGE if binding.gpus_per_job else CPU_IMAGE
        digest_hex = image.rsplit(":", 1)[1]
        call = by_name[f"trainer-{binding.profile}-{digest_hex}"]
        properties = call["containerProperties"]
        assert properties["image"] == image
        assert properties["jobRoleArn"] == JOB_ROLE_ARN
        assert properties["executionRoleArn"] == EXECUTION_ROLE_ARN
        assert call["retryStrategy"] == {"attempts": 1}
        requirements = {item["type"]: item["value"] for item in properties["resourceRequirements"]}
        assert requirements["VCPU"] == str(binding.vcpus_per_job)
        assert requirements["MEMORY"] == str(binding.memory_mib)
        if binding.gpus_per_job:
            assert requirements["GPU"] == str(binding.gpus_per_job)
        else:
            assert "GPU" not in requirements


def test_the_registered_command_is_the_sdk_worker(script: Any) -> None:
    """The container must run the worker that understands the manifest contract.

    The superseded image baked a different worker at /opt/trainer/worker.py, which
    ignores TRAINER_MANIFEST entirely. Job definitions pointing there accept the
    submission and then do nothing useful, so this is asserted rather than assumed.
    """
    session = FakeSession()

    script.deploy(
        register=True,
        confirm_account=ACCOUNT_ID,
        cpu_digest=CPU_IMAGE,
        gpu_digest=GPU_IMAGE,
        session=session,
    )

    for call in session.batch.calls:
        assert call["containerProperties"]["command"] == ["python", "-m", "training_sdk.worker"]


def test_register_creates_the_retained_log_group(script: Any) -> None:
    session = FakeSession()

    script.deploy(
        register=True,
        confirm_account=ACCOUNT_ID,
        cpu_digest=CPU_IMAGE,
        gpu_digest=GPU_IMAGE,
        session=session,
    )

    assert session.logs.created == [JOB_LOG_GROUP]
    assert session.logs.retention == [(JOB_LOG_GROUP, 30)]
    for call in session.batch.calls:
        configuration = call["containerProperties"]["logConfiguration"]
        assert configuration["logDriver"] == "awslogs"
        assert configuration["options"]["awslogs-group"] == JOB_LOG_GROUP
        assert configuration["options"]["awslogs-region"] == REGION


def test_register_tolerates_an_existing_log_group(script: Any) -> None:
    session = FakeSession(logs=FakeLogs(exists=True))

    script.deploy(
        register=True,
        confirm_account=ACCOUNT_ID,
        cpu_digest=CPU_IMAGE,
        gpu_digest=GPU_IMAGE,
        session=session,
    )

    assert session.logs.created == []
    assert session.logs.retention == [(JOB_LOG_GROUP, 30)]


def test_a_denied_log_group_is_not_swallowed(script: Any) -> None:
    session = FakeSession(logs=FakeLogs(create_error="AccessDeniedException"))

    with pytest.raises(ClientError):
        script.deploy(
            register=True,
            confirm_account=ACCOUNT_ID,
            cpu_digest=CPU_IMAGE,
            gpu_digest=GPU_IMAGE,
            session=session,
        )

    assert session.batch.calls == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confirm_account": None}, "confirm-account"),
        ({"confirm_account": "000000000000"}, "confirm-account"),
        ({"cpu_digest": None}, "cpu-digest"),
        ({"gpu_digest": None}, "gpu-digest"),
        ({"cpu_digest": f"{REGISTRY}/rtrrl:latest"}, "a tag is not accepted"),
        ({"cpu_digest": GPU_IMAGE}, "different digests"),
    ],
)
def test_register_refuses_incomplete_or_movable_input(
    script: Any, kwargs: dict[str, Any], message: str
) -> None:
    session = FakeSession()
    arguments: dict[str, Any] = {
        "register": True,
        "confirm_account": ACCOUNT_ID,
        "cpu_digest": CPU_IMAGE,
        "gpu_digest": GPU_IMAGE,
        "session": session,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        script.deploy(**arguments)

    assert session.batch.calls == []
    assert session.logs.created == []


def test_register_refuses_the_wrong_account(script: Any) -> None:
    session = FakeSession(sts=FakeSts(account="000000000000"))

    with pytest.raises(ValueError, match="000000000000"):
        script.deploy(
            register=True,
            confirm_account=ACCOUNT_ID,
            cpu_digest=CPU_IMAGE,
            gpu_digest=GPU_IMAGE,
            session=session,
        )

    assert session.batch.calls == []


def test_the_cli_prints_stable_json(script: Any, capsys: pytest.CaptureFixture[str]) -> None:
    assert script.main([]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry-run"
    assert payload["worker_command"] == ["python", "-m", "training_sdk.worker"]
