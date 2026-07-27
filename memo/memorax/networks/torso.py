"""Name a recurrent torso and get one, without knowing how it is wrapped.

Which wrapper a cell needs, and which keyword it takes, is knowledge about the
cells and belongs beside them. A caller that had to know RTU goes in an ``RNN``
while LRU goes in a ``Memoroid``, and that only one of the two reads
``output_dim``, would be maintaining a copy of this file.
"""

from __future__ import annotations

from memorax.networks.sequence_models import (
    RNN,
    LRUCell,
    LRUConfig,
    Memoroid,
    RTUCell,
    RTUConfig,
)

TORSOS = ("lru", "rtu")


def make_torso(name: str, *, features: int, hidden_dim: int, output_dim: int | None):
    """Build the recurrent torso a ``Network`` takes."""

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
    raise ValueError(f"unknown torso {name!r}; registered: {', '.join(TORSOS)}")
