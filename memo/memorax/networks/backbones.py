"""Name a backbone and get the components it contributes to a sequence.

Which wrapper a cell needs, and which keyword it takes, is knowledge about the
cells and belongs beside them. A caller that had to know RTU goes in an ``RNN``
while LRU goes in a ``Memoroid``, and that only one of the two reads
``output_dim``, would be maintaining a copy of this file.

``mlp`` is here so that an algorithm which takes a backbone can be run without
memory, as the ablation of every recurrent one.
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

# The RTRRL authors' two revisions of the LRU, each ours with one line put back
# the way theirs has it. Deliberately outside ``BACKBONES``: they are not
# backbones to train with, they are arms to compare a reproduction against, and
# only the entry running that comparison should offer them. What makes them
# usable as arms is that they take the same config and build the same parameter
# tree as ``lru``, which is visible below rather than asserted here.
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
