"""Name a torso and get one, without knowing how it is wrapped.

Which wrapper a cell needs, and which keyword it takes, is knowledge about the
cells and belongs beside them. A caller that had to know RTU goes in an ``RNN``
while LRU goes in a ``Memoroid``, and that only one of the two reads
``output_dim``, would be maintaining a copy of this file.

``mlp`` is here so that an algorithm which takes a torso can be run without
memory, as the ablation of every recurrent one. Algorithms that are about the
recurrence itself should offer ``RECURRENT_TORSOS`` instead of ``TORSOS``.
"""

from __future__ import annotations

from dataclasses import dataclass

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

RECURRENT_TORSOS = ("lru", "rtu")
TORSOS = (*RECURRENT_TORSOS, "mlp")

# The RTRRL authors' two revisions of the LRU, each ours with one line put back
# the way theirs has it. Deliberately outside ``TORSOS``: they are not torsos to
# train with, they are arms to compare a reproduction against, and only the entry
# running that comparison should offer them. What makes them usable as arms is
# that they take the same config and build the same parameter tree as ``lru``,
# which is visible below rather than asserted here.
UPSTREAM_TORSOS = ("lru_published", "lru_rewritten")


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


def make_torso(name: str, *, features: int, hidden_dim: int, output_dim: int | None):
    """Build the torso a ``Network`` takes."""

    if name == "rtu":
        return RNN(
            cell=RTUCell(config=RTUConfig(features=features, hidden_dim=hidden_dim))
        )
    if name in _LRU_CELLS:
        return Memoroid(
            cell=_LRU_CELLS[name](
                config=LRUConfig(
                    features=features,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                )
            )
        )
    if name == "mlp":
        return Memoryless(
            config=MemorylessConfig(features=features, hidden_dim=hidden_dim)
        )
    registered = ", ".join((*TORSOS, *UPSTREAM_TORSOS))
    raise ValueError(f"unknown torso {name!r}; registered: {registered}")
