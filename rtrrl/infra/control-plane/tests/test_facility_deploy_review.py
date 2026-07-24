from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from trainer_infra.facility_control import load_facility_control
from trainer_infra.image_catalog import ResolvedImage, load_catalog_index
from trainer_infra.models import ScriptCatalog


SCRIPT = Path(__file__).parents[1] / "scripts" / "deploy_facility.py"
CONTROL = Path(__file__).parents[1] / "config" / "facility.yaml"
CATALOG = Path(__file__).parents[2] / "mock-trainer" / "scripts" / "index.yaml"
CPU = "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:" + "a" * 64
GPU = "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:" + "b" * 64


def _load() -> Any:
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


class Batch:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def register_job_definition(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {
            "jobDefinitionArn": (
                "arn:aws:batch:eu-north-1:007122174918:job-definition/"
                f"{kwargs['jobDefinitionName']}:1"
            )
        }


class Ecr:
    pass


class Session:
    def __init__(
        self,
        *,
        account: str = "007122174918",
        region: str = "eu-north-1",
    ) -> None:
        self.region_name = region
        self.sts = Sts(account)
        self.ecr = Ecr()
        self.batch = Batch()
        self.requested: list[str] = []

    def client(self, name: str) -> Any:
        self.requested.append(name)
        return {"sts": self.sts, "ecr": self.ecr, "batch": self.batch}[name]


class Reader:
    catalogs: dict[str, ScriptCatalog] = {}
    clients: list[Ecr] = []
    references: list[str] = []

    def __init__(self, client: Ecr, *, account_id: str, region: str) -> None:
        assert account_id == "007122174918"
        assert region == "eu-north-1"
        type(self).clients.append(client)

    def resolve_and_fetch(self, reference: str) -> ResolvedImage:
        type(self).references.append(reference)
        repository, digest = reference.rsplit("@", 1)
        return ResolvedImage(
            reference=reference,
            repository=repository,
            digest=digest,
            catalog=type(self).catalogs[reference],
        )


@pytest.fixture(autouse=True)
def reset_reader() -> None:
    Reader.catalogs = {}
    Reader.clients = []
    Reader.references = []


def _request(deploy: Any) -> Any:
    return deploy.DeployRequest(
        register=True,
        confirm_account="007122174918",
        cpu_digest=CPU,
        gpu_digest=GPU,
    )


def _catalogs(catalog: ScriptCatalog) -> dict[str, ScriptCatalog]:
    return {CPU: catalog, GPU: catalog}


def test_registration_requires_exact_account_confirmation() -> None:
    deploy = _load()
    session = Session()
    with pytest.raises(ValueError, match="--confirm-account 007122174918"):
        deploy.deploy(
            deploy.DeployRequest(register=True),
            control=load_facility_control(CONTROL),
            session=session,
        )
    assert session.requested == []


@pytest.mark.parametrize(
    ("account", "region", "message"),
    [
        ("123456789012", "eu-north-1", "account"),
        ("007122174918", "us-east-1", "region"),
    ],
)
def test_identity_gate_precedes_all_mutation(
    account: str,
    region: str,
    message: str,
) -> None:
    deploy = _load()
    session = Session(account=account, region=region)
    with pytest.raises(ValueError, match=message):
        deploy.deploy(
            _request(deploy),
            control=load_facility_control(CONTROL),
            session=session,
        )
    assert session.requested == ["sts"]


def test_registration_requires_two_distinct_digests() -> None:
    deploy = _load()
    request = deploy.DeployRequest(
        register=True,
        confirm_account="007122174918",
        cpu_digest=CPU,
        gpu_digest=CPU,
    )
    session = Session()

    with pytest.raises(ValueError, match="distinct"):
        deploy.deploy(request, control=load_facility_control(CONTROL), session=session)
    assert session.requested == ["sts"]


def test_registration_verifies_complete_expected_catalog_before_fixed_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deploy = _load()
    expected = load_catalog_index(CATALOG)
    Reader.catalogs = _catalogs(expected)
    monkeypatch.setattr(deploy, "BotoEcrCatalogReader", Reader)
    session = Session()

    report = deploy.deploy(
        _request(deploy),
        control=load_facility_control(CONTROL),
        session=session,
    )

    assert report["mode"] == "execute"
    assert Reader.clients == [session.ecr]
    assert Reader.references == [CPU, GPU]
    assert len(session.batch.calls) == 4
    assert {
        call["containerProperties"]["jobRoleArn"] for call in session.batch.calls
    } == {"arn:aws:iam::007122174918:role/rtrrl-batch-job-role"}
    assert {
        call["containerProperties"]["executionRoleArn"] for call in session.batch.calls
    } == {"arn:aws:iam::007122174918:role/rtrrl-batch-execution-role"}
    assert [
        call["containerProperties"]["resourceRequirements"] for call in session.batch.calls
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


@pytest.mark.parametrize("variant", ["cpu", "gpu"])
def test_registration_rejects_any_complete_catalog_byte_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    deploy = _load()
    expected = load_catalog_index(CATALOG)
    mismatched = expected.model_copy(update={"protocol_version": "2"})
    Reader.catalogs = _catalogs(expected)
    Reader.catalogs[CPU if variant == "cpu" else GPU] = mismatched
    monkeypatch.setattr(deploy, "BotoEcrCatalogReader", Reader)
    session = Session()

    with pytest.raises(ValueError, match="catalog"):
        deploy.deploy(
            _request(deploy),
            control=load_facility_control(CONTROL),
            session=session,
        )
    assert session.batch.calls == []


def test_local_expected_catalog_contract_is_exact() -> None:
    expected = load_catalog_index(CATALOG)
    assert expected.protocol_version == "1"
    assert set(expected.scripts) == {"brax_ppo_acceptance"}
    descriptor = expected.scripts["brax_ppo_acceptance"]
    assert descriptor.sdk_protocol_version == "1"
    assert descriptor.objective.metric == "eval/episode_return"
    assert descriptor.environments == ("inverted_pendulum",)
    assert set(descriptor.fields) == {
        "seed",
        "learning_rate",
        "num_envs",
        "episode_length",
        "failure_mode",
    }
