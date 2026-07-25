from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from tests.test_aws_batch import FakePreflightBatch
from tests.test_facility_deployment import FakeEcr, FakeS3, FakeSts
from trainer_infra.facility_control import load_facility_control


SCRIPT = Path(__file__).parents[1] / "scripts" / "facility_preflight.py"
CONTROL = Path(__file__).parents[1] / "config" / "facility.yaml"
DEPLOY = Path(__file__).parents[1] / "scripts" / "deploy_facility.py"


def _load():
    spec = importlib.util.spec_from_file_location("facility_preflight_review", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeniedIam:
    def simulate_principal_policy(self, **_kwargs: Any) -> dict[str, Any]:
        raise ClientError(
            {
                "Error": {
                    "Code": "AccessDenied",
                    "Message": "simulation unavailable",
                }
            },
            "SimulatePrincipalPolicy",
        )


class ActualBatch(FakePreflightBatch):
    def describe_compute_environments(self, **kwargs: Any) -> dict[str, Any]:
        response = super().describe_compute_environments(**kwargs)
        resources = response["computeEnvironments"][0]["computeResources"]
        resources["subnets"] = [
            "subnet-08127d1c5d4de6ac2",
            "subnet-0b8c68ea0a9784758",
            "subnet-01a2aa195678f8411",
        ]
        resources["securityGroupIds"] = ["sg-0c0ed6b927c5113dc"]
        resources["instanceRole"] = (
            "arn:aws:iam::007122174918:"
            "instance-profile/rtrrl-ecs-instance-role"
        )
        return response


class Session:
    region_name = "eu-north-1"

    def __init__(self) -> None:
        self.clients = {
            "sts": FakeSts(),
            "batch": ActualBatch(),
            "s3": FakeS3(),
            "ecr": FakeEcr(),
            "iam": DeniedIam(),
        }

    def client(self, name: str) -> Any:
        return self.clients[name]


def test_iam_simulation_unavailable_is_unknown_and_preflight_passes() -> None:
    preflight = _load()
    report = preflight.run_preflight(
        Session(),
        load_facility_control(CONTROL),
        aim_validator=lambda _control: {"status": "ready", "pid": 321},
    )

    assert report["status"] == "pass"
    assert report["iam"]["status"] == "unknown"
    assert report["iam"]["availability"] == "unavailable"
    assert report["iam"]["blocking"] is False
    assert report["caller"]["arn"].endswith("assumed-role/trainer-control/session")
    assert [item["profile"] for item in report["profiles"]] == [
        "c7am",
        "c7al",
        "c7ax",
        "g6x",
    ]
    assert [
        (item["vcpus"], item["memory_mib"], item["gpus"])
        for item in report["profiles"]
    ] == [(1, 1600, 0), (2, 3200, 0), (4, 7168, 0), (4, 12000, 1)]
    assert [item["dev_queue"] for item in report["profiles"]] == [
        {"name": "dev-cpu-c7am-queue", "priority": 10},
        {"name": "dev-cpu-c7al-queue", "priority": 10},
        {"name": "dev-cpu-c7ax-queue", "priority": 10},
        {"name": "dev-gpu-queue", "priority": 10},
    ]
    assert all(item["run_queue"]["priority"] == 100 for item in report["profiles"])


def test_authoritative_profile_drift_blocks_preflight() -> None:
    preflight = _load()
    session = Session()
    original = session.clients["batch"].describe_compute_environments

    def drift(**kwargs: Any) -> dict[str, Any]:
        response = original(**kwargs)
        response["computeEnvironments"][0]["computeResources"]["minvCpus"] = 1
        return response

    session.clients["batch"].describe_compute_environments = drift
    report = preflight.run_preflight(
        session,
        load_facility_control(CONTROL),
        aim_validator=lambda _control: {"status": "ready", "pid": 321},
    )

    assert report["status"] == "blocked"
    assert report["profiles"]["status"] == "failed"
    assert "minvCpus" in report["profiles"]["error"]


def test_preflight_writes_only_canonical_report_atomically(tmp_path: Path) -> None:
    preflight = _load()
    destination = tmp_path / "complete-facility-task7-phase-a-preflight.json"

    written = preflight.write_canonical_report(
        {"schema_version": 1, "status": "pass"},
        path=destination,
    )

    assert preflight.CANONICAL_REPORT == Path(
        "/tmp/complete-facility-task7-phase-a-preflight.json"
    )
    assert written == destination
    assert destination.read_text() == '{"schema_version":1,"status":"pass"}\n'
    assert list(tmp_path.glob("*.tmp")) == []


def test_preflight_is_generic_single_script_and_aws_read_only() -> None:
    preflight = SCRIPT.read_text(encoding="utf-8")
    deploy = DEPLOY.read_text(encoding="utf-8")
    control = load_facility_control(CONTROL)

    assert control.cpu_image_tag == "infra-acceptance-brax-ppo-cpu-20260723"
    assert control.gpu_image_tag == "infra-acceptance-brax-ppo-gpu-20260723"
    assert "memo/" not in preflight
    assert "memo_" not in preflight
    assert "memo/" not in deploy
    assert "memo_" not in deploy
    for forbidden_action in (
        "batch:RegisterJobDefinition",
        "batch:SubmitJob",
        "ecr:CompleteLayerUpload",
        "ecr:GetAuthorizationToken",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart",
        "iam:PassRole",
        "s3:PutObject",
    ):
        assert forbidden_action not in preflight
