"""How a weight is drawn, declared by whatever has weights to draw.

``lecun`` is flax's own default for ``nn.Dense``. ``sparse`` is the one both
streaming-drl and memorax's MinAtar example choose, at 0.9 in each. Biases are
zero either way, so only the kernel branches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from memorax.networks.initializers import Initializer, lecun_normal, sparse
from memorax.parameters import param, read_branch, structure


@dataclass(frozen=True)
class Sparse:
    sparsity: float = param(valid=(0.0, 1.0), search=[0.9], placeholder=0.9)


INITIALIZATION_BRANCHES = {"lecun": (), "sparse": Sparse}


def initialization() -> Any:
    """The declaration a component with kernels puts among its own fields."""

    return structure(placeholder="lecun", branches=INITIALIZATION_BRANCHES)


def declared_initializer(component) -> Initializer:
    """The initialiser a declared branch asks for; ``None`` is the dense one."""

    if component is None:
        return lecun_normal()
    return sparse(component.sparsity)


def initializer_at(params: Mapping[str, Any], prefix: str) -> Initializer | None:
    """The initialiser declared at this path, or none if nothing declares one."""

    if f"{prefix}initialization" not in params:
        return None
    _, component = read_branch(
        params, "initialization", INITIALIZATION_BRANCHES, prefix=prefix
    )
    return declared_initializer(component)
