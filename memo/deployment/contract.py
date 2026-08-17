"""The versioned catalog shared by an image build and its Worker."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, model_validator

# The version is written once, as the type. ``Literal`` will not take a name,
# so the constant is read back out of it rather than spelled a second time
# where the two could drift apart.
ContractVersion: TypeAlias = Literal[11]
CONTRACT_VERSION: ContractVersion = get_args(ContractVersion)[0]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EntryDescriptor(_Frozen):
    command: tuple[str, ...]
    metrics: tuple[str, ...]
    parameters: dict[str, Any]

    @model_validator(mode="after")
    def _non_empty(self) -> "EntryDescriptor":
        if not self.command:
            raise ValueError("command must not be empty")
        if not self.metrics:
            raise ValueError("metrics must not be empty")
        return self


class Catalog(_Frozen):
    contract: ContractVersion
    entries: dict[str, EntryDescriptor]
