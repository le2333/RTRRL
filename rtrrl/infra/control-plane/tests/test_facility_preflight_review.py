from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from test_aws_batch import FakePreflightBatch
from test_facility_deployment import FakeEcr, FakeS3, FakeSts
from trainer_infra.facility_control import load_facility_control


SCRIPT = Path(__file__).parents[1] / "scripts" / "facility_preflight.py"
CONTROL = Path(__file__).parents[1] / "config" / "facility.yaml"


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
