"""The projection of a run document Worker is allowed to interpret."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from deployment.contract import ContractVersion


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunIdentity(_Frozen):
    run_id: str
    experiment: str
    launch_id: str
    trial: int
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
