from pathlib import Path

import pytest
import yaml

from trainer_infra.facility_control import (
    EXPECTED_ACCOUNT_ID,
    EXPECTED_EXECUTION_ROLE_ARN,
    EXPECTED_JOB_ROLE_ARN,
    FacilityControl,
    load_facility_control,
)


CONTROL = Path(__file__).parents[1] / "config" / "facility.yaml"


def test_committed_facility_control_fixes_account_network_roles_and_aim() -> None:
    control = load_facility_control(CONTROL)

    assert control.account_id == EXPECTED_ACCOUNT_ID == "007122174918"
    assert control.region == "eu-north-1"
    assert control.job_role_arn == EXPECTED_JOB_ROLE_ARN
    assert control.execution_role_arn == EXPECTED_EXECUTION_ROLE_ARN
    assert control.subnets == (
        "subnet-08127d1c5d4de6ac2",
        "subnet-0b8c68ea0a9784758",
        "subnet-01a2aa195678f8411",
    )
    assert control.security_group_ids == ("sg-0c0ed6b927c5113dc",)
    assert control.aim.port == 53801
    assert control.aim.repo == Path("/home/ubuntu/trainer/task7-aim-scratch")
    assert control.aim.metadata_file.name == "aim-server-53801.json"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "123456789012"),
        ("region", "us-east-1"),
        ("job_role_arn", "arn:aws:iam::007122174918:role/arbitrary"),
        ("execution_role_arn", "arn:aws:iam::007122174918:role/arbitrary"),
    ],
)
def test_facility_control_rejects_identity_or_role_drift(
    field: str, value: str
) -> None:
    payload = yaml.safe_load(CONTROL.read_text())
    payload[field] = value

    with pytest.raises(ValueError):
        FacilityControl.model_validate(payload)
