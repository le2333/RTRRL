from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trainer_infra.facility_control import load_facility_control


SCRIPT = Path(__file__).parents[1] / "scripts" / "deploy_facility.py"
CONTROL = Path(__file__).parents[1] / "config" / "facility.yaml"


def _load():
    spec = importlib.util.spec_from_file_location("deploy_facility_review", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sts:
    def __init__(self, account: str = "007122174918") -> None:
        self.account = account
        self.calls = 0

    def get_caller_identity(self) -> dict[str, str]:
        self.calls += 1
        return {"Account": self.account, "Arn": f"arn:aws:iam::{self.account}:role/test"}


class Session:
    def __init__(self, account: str = "007122174918", region: str = "eu-north-1") -> None:
        self.region_name = region
        self.sts = Sts(account)
        self.requested: list[str] = []

    def client(self, name: str) -> Any:
        self.requested.append(name)
        if name == "sts":
            return self.sts
        raise AssertionError(f"unexpected client before identity gate: {name}")


def test_registration_requires_exact_account_confirmation() -> None:
    deploy = _load()
    control = load_facility_control(CONTROL)
    session = Session()
    request = deploy.DeployRequest(register=True)

    with pytest.raises(ValueError, match="--confirm-account 007122174918"):
        deploy.deploy(request, control=control, session=session)

    assert session.requested == []


@pytest.mark.parametrize(
    ("account", "region", "message"),
    [
        ("123456789012", "eu-north-1", "account"),
        ("007122174918", "us-east-1", "region"),
    ],
)
def test_identity_gate_precedes_all_mutation(
    account: str, region: str, message: str
) -> None:
    deploy = _load()
    control = load_facility_control(CONTROL)
    session = Session(account, region)
    request = deploy.DeployRequest(
        register=True,
        confirm_account="007122174918",
        cpu_digest="invalid",
        gpu_digest="invalid",
    )

    with pytest.raises(ValueError, match=message):
        deploy.deploy(request, control=control, session=session)

    assert session.requested == ["sts"]


def test_registration_verifies_digests_read_only_then_uses_fixed_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deploy = _load()
    control = load_facility_control(CONTROL)
    calls: list[dict[str, Any]] = []
    batch = SimpleNamespace(
        register_job_definition=lambda **kwargs: (
            calls.append(kwargs)
            or {
                "jobDefinitionArn": (
                    "arn:aws:batch:eu-north-1:007122174918:job-definition/"
                    f"{kwargs['jobDefinitionName']}:1"
                )
            }
        )
    )
    sts = Sts()
    ecr = object()
    verified: list[str] = []

    class Reader:
        def __init__(self, client: object, **_kwargs: Any) -> None:
            assert client is ecr

        def resolve_and_fetch(self, reference: str) -> Any:
            verified.append(reference)
            return SimpleNamespace(reference=reference, catalog=SimpleNamespace())

    monkeypatch.setattr(deploy, "BotoEcrCatalogReader", Reader)
    session = SimpleNamespace(
        region_name="eu-north-1",
        client=lambda name: {"sts": sts, "ecr": ecr, "batch": batch}[name],
    )
    cpu = (
        "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:"
        + "a" * 64
    )
    gpu = (
        "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:"
        + "b" * 64
    )

    deploy.deploy(
        deploy.DeployRequest(
            register=True,
            confirm_account="007122174918",
            cpu_digest=cpu,
            gpu_digest=gpu,
        ),
        control=control,
        session=session,
    )

    assert len(calls) == 4
    assert verified == [cpu, gpu]
    assert {
        call["containerProperties"]["jobRoleArn"] for call in calls
    } == {control.job_role_arn}
    assert {
        call["containerProperties"]["executionRoleArn"] for call in calls
    } == {control.execution_role_arn}
    assert [
        call["containerProperties"]["resourceRequirements"] for call in calls
    ] == [
        [{"type": "VCPU", "value": "1"}, {"type": "MEMORY", "value": "1600"}],
        [{"type": "VCPU", "value": "2"}, {"type": "MEMORY", "value": "3200"}],
        [{"type": "VCPU", "value": "4"}, {"type": "MEMORY", "value": "7168"}],
        [
            {"type": "VCPU", "value": "4"},
            {"type": "MEMORY", "value": "12000"},
            {"type": "GPU", "value": "1"},
        ],
    ]
