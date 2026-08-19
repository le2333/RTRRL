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
    # Declared so a document carrying them is accepted, and left as JSON so
    # Worker does not acquire an opinion about them. A branch reads its parent
    # checkpoint out of an artifact root Worker already knows how to fill; that
    # it is a branch at all is the Entry's business.
    checkpoint: dict[str, Any] | None = None
    fork: dict[str, Any] | None = None
