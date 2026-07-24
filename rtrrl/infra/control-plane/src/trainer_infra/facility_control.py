from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
import yaml


EXPECTED_ACCOUNT_ID = "007122174918"
EXPECTED_REGION = "eu-north-1"
EXPECTED_JOB_ROLE_ARN = (
    "arn:aws:iam::007122174918:role/rtrrl-batch-job-role"
)
EXPECTED_EXECUTION_ROLE_ARN = (
    "arn:aws:iam::007122174918:role/rtrrl-batch-execution-role"
)


class AimScratchControl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: Path
    main_repo: Path
    host: str
    port: Literal[53801]
    metadata_file: Path
    pid_file: Path
    log_file: Path

    @property
    def endpoint(self) -> str:
        return f"aim://{self.host}:{self.port}"


class FacilityControl(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    account_id: Literal["007122174918"]
    region: Literal["eu-north-1"]
    bucket: Literal["rtrrl-artifacts-007122174918"]
    prefix: Literal["experiments"]
    ecr_repository: Literal["rtrrl"]
    cpu_image_tag: Literal["infra-acceptance-brax-ppo-cpu-20260723"]
    gpu_image_tag: Literal["infra-acceptance-brax-ppo-gpu-20260723"]
    subnets: tuple[str, ...]
    security_group_ids: tuple[str, ...]
    instance_role: Literal[
        "arn:aws:iam::007122174918:instance-profile/rtrrl-ecs-instance-role"
    ]
    job_role_arn: Literal[
        "arn:aws:iam::007122174918:role/rtrrl-batch-job-role"
    ]
    execution_role_arn: Literal[
        "arn:aws:iam::007122174918:role/rtrrl-batch-execution-role"
    ]
    aim: AimScratchControl


def load_facility_control(path: Path) -> FacilityControl:
    with Path(path).open(encoding="utf-8") as stream:
        return FacilityControl.model_validate(yaml.safe_load(stream))
