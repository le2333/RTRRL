"""Which head each role gets, as a structure per role.

The three policy heads differ in where the scale comes from: a learnable
parameter no observation reaches, a second projection of the observation, or
that projection squashed into an interval. The critic has one. It is declared
anyway -- a role with one choice and a role with none are different things, and
this is where a second one goes.

A head has kernels, so it declares how they are drawn, for itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn

from memorax.building import ComponentFamily
from memorax.networks import heads
from memorax.networks.initialization import INITIALIZER_FAMILY, initialization

ACTOR_HEADS = {
    "global_std": heads.Gaussian,
    "state_std": heads.StateStdGaussian,
    "bounded": heads.BoundedGaussian,
}

CRITIC_HEADS = {"value": heads.VNetwork}


@dataclass(frozen=True)
class Head:
    initialization: str = initialization()


ACTOR_HEAD_BRANCHES = {name: Head for name in ACTOR_HEADS}
CRITIC_HEAD_BRANCHES = {name: Head for name in CRITIC_HEADS}


def _build(registered, role, name, kernel_init, **arguments) -> nn.Module:
    if name not in registered:
        listed = ", ".join(sorted(registered))
        raise ValueError(f"unknown {role} head {name!r}; registered: {listed}")
    drawn = {} if kernel_init is None else {"kernel_init": kernel_init}
    return registered[name](**arguments, **drawn)


def actor_head(name: str, *, action_dim: int, kernel_init=None) -> nn.Module:
    return _build(ACTOR_HEADS, "actor", name, kernel_init, action_dim=action_dim)


def critic_head(name: str, *, kernel_init=None) -> nn.Module:
    return _build(CRITIC_HEADS, "critic", name, kernel_init)


def _construct_actor_head(selection, builder, *, action_dim: int):
    kernel_init = builder.build(INITIALIZER_FAMILY, f"{selection.path}.initialization")
    return actor_head(selection.kind, action_dim=action_dim, kernel_init=kernel_init)


def _construct_critic_head(selection, builder):
    kernel_init = builder.build(INITIALIZER_FAMILY, f"{selection.path}.initialization")
    return critic_head(selection.kind, kernel_init=kernel_init)


ACTOR_HEAD_FAMILY = ComponentFamily(
    branches=ACTOR_HEAD_BRANCHES,
    construct=_construct_actor_head,
)
CRITIC_HEAD_FAMILY = ComponentFamily(
    branches=CRITIC_HEAD_BRANCHES,
    construct=_construct_critic_head,
)
