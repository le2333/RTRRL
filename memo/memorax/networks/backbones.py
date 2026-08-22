"""Name a backbone and get the components it contributes to a sequence.

Which wrapper a cell needs and which keyword it takes is knowledge about the
cells: RTU and LSTM go in an ``RNN``, LRU in a ``Memoroid``, and only the LRU
reads ``output_dim``. Kept here so a caller does not hold a copy of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn

from memorax.building import ComponentFamily
from memorax.networks.components import FFN, LayerNorm, LeakyReLU
from memorax.networks.differentiation import TruncatedBPTT
from memorax.networks.initialization import INITIALIZER_FAMILY, initialization
from memorax.networks.sequence_models import (
    RNN,
    LRUCell,
    LRUConfig,
    Memoroid,
    PublishedLRUCell,
    RewrittenLRUCell,
    RTUCell,
    RTUConfig,
)
from memorax.networks.sequence_models.lru import LRU_DIFFERENTIATION_FAMILY
from memorax.networks.sequence_models.rtu import RTU_DIFFERENTIATION_FAMILY
from memorax.parameters import param, structure

RECURRENT_BACKBONES = ("lru", "rtu")
BACKBONES = (*RECURRENT_BACKBONES, "mlp")

# Outside ``BACKBONES`` so only the entry that asks for them by name gets them.
# They take the same config and build the same parameter tree as ``lru``.
UPSTREAM_BACKBONES = ("lru_published", "lru_rewritten")

# Outside it for a stronger reason. A dense-gated cell has no structure for the
# online arm to carry exact recurrent sensitivity through, so a family offering
# one would offer a core only half of a matched comparison could be run on. Only
# a learner that differentiates by backpropagation may name it, which today is
# DRQN reproducing its own published network.
BPTT_BACKBONES = ("lstm",)


@dataclass(frozen=True)
class Rtu:
    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), static=True)
    differentiation: str = structure(branches=RTU_DIFFERENTIATION_FAMILY.branches)


@dataclass(frozen=True)
class Lru:
    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), static=True)
    feature_dim: int = param(valid=(1, 4096), search=(16, 256), static=True)
    differentiation: str = structure(branches=LRU_DIFFERENTIATION_FAMILY.branches)


@dataclass(frozen=True)
class Mlp:
    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), static=True)
    initialization: str = initialization()


@dataclass(frozen=True)
class BuiltBackbone:
    """The components and recurrent differentiation selected for one backbone."""

    components: tuple[nn.Module, ...]
    differentiation: object


_LRU_CELLS = {
    "lru": LRUCell,
    "lru_published": PublishedLRUCell,
    "lru_rewritten": RewrittenLRUCell,
}

BACKBONE_BRANCHES = {
    "rtu": Rtu,
    "lru": Lru,
    "mlp": Mlp,
    "lru_published": Lru,
    "lru_rewritten": Lru,
}


def backbone(
    name: str,
    *,
    features: int,
    hidden_dim: int,
    output_dim: int | None = None,
    kernel_init=None,
) -> tuple[nn.Module, ...]:
    """The components this backbone puts into a sequence, in order.

    The head that follows is the caller's; nothing here appends one.
    """

    drawn = {} if kernel_init is None else {"kernel_init": kernel_init}

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
    if name == "lstm":
        # The published DRQN's cell, wrapped as the RTU is: one cell, whose
        # hidden state is its output, so there is no readout width to size.
        # ``OptimizedLSTMCell`` is Flax's own LSTM with the four input matmuls
        # fused into one; the equations, the parameter names and the carry are
        # ``LSTMCell``'s, so nothing about the network follows from the choice.
        return (RNN(cell=nn.OptimizedLSTMCell(features=hidden_dim)),)
    if name == "mlp":
        return (
            FFN(features=hidden_dim, **drawn),
            LayerNorm(),
            LeakyReLU(),
            FFN(features=hidden_dim, **drawn),
            LayerNorm(),
            LeakyReLU(),
        )
    registered = ", ".join((*BACKBONES, *UPSTREAM_BACKBONES, *BPTT_BACKBONES))
    raise ValueError(f"unknown backbone {name!r}; registered: {registered}")


def _construct_backbone(
    selection,
    builder,
    *,
    features: int,
    output_dim: int | None = None,
):
    kernel_init = (
        builder.build(INITIALIZER_FAMILY, f"{selection.path}.initialization")
        if isinstance(selection.parameters, Mlp)
        else None
    )
    components = backbone(
        selection.kind,
        features=features,
        hidden_dim=selection.parameters.hidden_dim,
        output_dim=output_dim,
        kernel_init=kernel_init,
    )
    if selection.kind == "rtu":
        differentiation = builder.build(
            RTU_DIFFERENTIATION_FAMILY,
            f"{selection.path}.differentiation",
            core=components[0],
        )
    elif selection.kind in _LRU_CELLS:
        differentiation = builder.build(
            LRU_DIFFERENTIATION_FAMILY,
            f"{selection.path}.differentiation",
            core=components[0],
        )
    else:
        differentiation = TruncatedBPTT(None)
    return BuiltBackbone(components, differentiation)


BACKBONE_FAMILY = ComponentFamily(
    branches=BACKBONE_BRANCHES,
    construct=_construct_backbone,
)
