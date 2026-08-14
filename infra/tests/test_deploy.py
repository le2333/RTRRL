from __future__ import annotations

from typing import Any

import pytest

from trainer_infra.batch import ACCOUNT_ID, PROFILES
from trainer_infra.deploy import WORKER_COMMAND, deploy

REGISTRY = f"{ACCOUNT_ID}.dkr.ecr.eu-north-1.amazonaws.com/rtrrl"
CPU_IMAGE = f"{REGISTRY}@sha256:{'a' * 64}"


class FakeClient:
    def __init__(self, service: str) -> None:
        self.service = service
        self.calls: list[dict[str, Any]] = []

    def get_caller_identity(self) -> dict[str, str]:
        return {"Account": ACCOUNT_ID}

    def create_log_group(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def put_retention_policy(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def register_job_definition(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"jobDefinitionArn": f"arn:definition:{len(self.calls)}"}


class FakeSession:
    def __init__(self) -> None:
        self.clients = {name: FakeClient(name) for name in ("sts", "logs", "batch")}

    def client(self, name: str) -> FakeClient:
        return self.clients[name]


def test_deploy_is_dry_run_by_default() -> None:
    session = FakeSession()

    report = deploy(session=session)

    assert report["mode"] == "dry-run"
    assert report["worker_command"] == WORKER_COMMAND
    assert all(not client.calls for client in session.clients.values())


def test_registers_cpu_profiles_with_the_v8_worker() -> None:
    session = FakeSession()

    report = deploy(
        register=True,
        confirm_account=ACCOUNT_ID,
        cpu_image=CPU_IMAGE,
        session=session,
    )

    cpu_profiles = [profile for profile in PROFILES.values() if not profile.gpus]
    calls = session.clients["batch"].calls
    assert report["mode"] == "register"
    assert len(calls) == len(cpu_profiles)
    for call in calls:
        assert call["containerProperties"]["command"] == ["python", "-m", "worker"]
        assert call["containerProperties"]["image"] == CPU_IMAGE
        assert "training_sdk" not in " ".join(call["containerProperties"]["command"])


@pytest.mark.parametrize(
    "image",
    [None, f"{REGISTRY}:latest", f"{REGISTRY}@sha256:{'A' * 64}"],
)
def test_registration_refuses_a_missing_or_movable_image(image: str | None) -> None:
    with pytest.raises(ValueError, match="CPU image must be"):
        deploy(
            register=True,
            confirm_account=ACCOUNT_ID,
            cpu_image=image,
            session=FakeSession(),
        )
