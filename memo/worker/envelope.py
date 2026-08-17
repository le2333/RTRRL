"""The projection of a run document Worker is allowed to interpret."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from deployment.contract import ContractVersion


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunIdentity(_Frozen):
    """What Worker needs to name a run: the configuration and its repetition."""

    run_id: str
    experiment: str
    launch_id: str
    trial: int
    seed: int
    role: Literal["tuning", "formal"]
    digest: str


class Artifacts(_Frozen):
    root: str


class WorkerEnvelope(_Frozen):
    contract: ContractVersion
    identity: RunIdentity
    entry: str
    artifacts: Artifacts
    algorithm: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    logging: dict[str, Any]
