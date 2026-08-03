"""Name a backbone and get the components it contributes to a sequence.

Which wrapper a cell needs and which keyword it takes is knowledge about the
cells: RTU goes in an ``RNN``, LRU in a ``Memoroid``, and only the second reads
``output_dim``. Kept here so a caller does not hold a copy of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
from training_sdk.parameters import param

from memorax.networks.sequence_models import (
    RNN,
    LRUCell,
    LRUConfig,
    Memoroid,
    Memoryless,
    MemorylessConfig,
    PublishedLRUCell,
    RewrittenLRUCell,
    RTUCell,
    RTUConfig,
)

RECURRENT_BACKBONES = ("lru", "rtu")
BACKBONES = (*RECURRENT_BACKBONES, "mlp")

# Outside ``BACKBONES`` so only the entry that asks for them by name gets them.
# They take the same config and build the same parameter tree as ``lru``.
UPSTREAM_BACKBONES = ("lru_published", "lru_rewritten")


@dataclass(frozen=True)
class Rtu:
    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), placeholder=192)
    feature_dim: int = param(valid=(1, 4096), search=(16, 256), placeholder=64)


@dataclass(frozen=True)
class Lru:
    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), placeholder=128)
    feature_dim: int = param(valid=(1, 4096), search=(16, 256), placeholder=32)


@dataclass(frozen=True)
class Mlp:
    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), placeholder=128)


_LRU_CELLS = {
    "lru": LRUCell,
    "lru_published": PublishedLRUCell,
    "lru_rewritten": RewrittenLRUCell,
}


def backbone(
    name: str, *, features: int, hidden_dim: int, output_dim: int | None = None
) -> tuple[nn.Module, ...]:
    """The components this backbone puts into a sequence, in order."""

    if name == "rtu":
        return (
            RNN(
                cell=RTUCell(config=RTUConfig(features=features, hidden_dim=hidden_dim))
            ),
        )
    if name in _LRU_CELLS:
        return (
            Memoroid(
                cell=_LRU_CELLS[name](
                    config=LRUConfig(
                        features=features,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                    )
                )
            ),
        )
    if name == "mlp":
        return (
            Memoryless(
                config=MemorylessConfig(features=features, hidden_dim=hidden_dim)
            ),
        )
    registered = ", ".join((*BACKBONES, *UPSTREAM_BACKBONES))
    raise ValueError(f"unknown backbone {name!r}; registered: {registered}")
