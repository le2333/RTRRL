"""How a weight is drawn, as a structure with a branch per way.

Biases are zero either way, which is what both sources do, so only the kernel
has a branch here.
"""

from __future__ import annotations

from dataclasses import dataclass

from training_sdk.parameters import param

from memorax.networks.initializers import Initializer, lecun_normal, sparse


@dataclass(frozen=True)
class Sparse:
    sparsity: float = param(valid=(0.0, 1.0), search=[0.9], placeholder=0.9)


INITIALIZATION_BRANCHES = {"lecun": (), "sparse": Sparse}


def declared_initializer(component) -> Initializer:
    """The initialiser a declared branch asks for; ``None`` is the dense one."""

    if component is None:
        return lecun_normal()
    return sparse(component.sparsity)
