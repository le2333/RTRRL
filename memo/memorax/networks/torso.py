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

from memorax.networks.sequence_models import (
    RNN,
    LRUCell,
    LRUConfig,
    Memoroid,
    Memoryless,
    MemorylessConfig,
    RTUCell,
    RTUConfig,
)

RECURRENT_TORSOS = ("lru", "rtu")
TORSOS = (*RECURRENT_TORSOS, "mlp")


def make_torso(name: str, *, features: int, hidden_dim: int, output_dim: int | None):
    """Build the torso a ``Network`` takes."""

    if name == "rtu":
        return RNN(
            cell=RTUCell(config=RTUConfig(features=features, hidden_dim=hidden_dim))
        )
    if name == "lru":
        return Memoroid(
            cell=LRUCell(
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
    raise ValueError(f"unknown torso {name!r}; registered: {', '.join(TORSOS)}")
