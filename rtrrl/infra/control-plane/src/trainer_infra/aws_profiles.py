from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from trainer_infra.models import ResourceProfileName


@dataclass(frozen=True)
class AwsProfile:
    name: ResourceProfileName
    dev_queue: str
    run_queue: str
    compute_environment: str
    vcpus: int
    memory_mib: int
    gpus: int


PROFILES: Mapping[ResourceProfileName, AwsProfile] = MappingProxyType(
    {
        "c7am": AwsProfile(
            name="c7am",
            dev_queue="dev-cpu-c7am-queue",
            run_queue="run-cpu-c7am-queue",
            compute_environment="rtrrl-cpu-c7am-ce",
            vcpus=1,
            memory_mib=1600,
            gpus=0,
        ),
        "c7al": AwsProfile(
            name="c7al",
            dev_queue="dev-cpu-c7al-queue",
            run_queue="run-cpu-c7al-queue",
            compute_environment="rtrrl-cpu-c7al-ce",
            vcpus=2,
            memory_mib=3200,
            gpus=0,
        ),
        "c7ax": AwsProfile(
            name="c7ax",
            dev_queue="dev-cpu-c7ax-queue",
            run_queue="run-cpu-c7ax-queue",
            compute_environment="rtrrl-cpu-c7ax-ce",
            vcpus=4,
            memory_mib=7168,
            gpus=0,
        ),
        "g6x": AwsProfile(
            name="g6x",
            dev_queue="dev-gpu-queue",
            run_queue="run-gpu-queue",
            compute_environment="rtrrl-gpu-g6x-ce",
            vcpus=4,
            memory_mib=12000,
            gpus=1,
        ),
    }
)


def profile(name: ResourceProfileName) -> AwsProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        expected = ", ".join(PROFILES)
        raise ValueError(
            f"unknown resource profile {name!r}; expected one of: {expected}"
        ) from error
