"""Which policy head, as a structure with a branch per parameterisation.

The three differ in where the scale comes from: a learnable parameter no
observation reaches, a second projection of the observation, or that projection
squashed into an interval.
"""

from __future__ import annotations

import flax.linen as nn

from memorax.networks import heads

ACTOR_HEADS = {
    "global_std": heads.Gaussian,
    "state_std": heads.StateStdGaussian,
    "bounded": heads.BoundedGaussian,
}

ACTOR_HEAD_BRANCHES = {name: () for name in ACTOR_HEADS}


def actor_head(name: str, *, action_dim: int, kernel_init=None) -> nn.Module:
    """Build the policy head this branch names."""

    if name not in ACTOR_HEADS:
        registered = ", ".join(sorted(ACTOR_HEADS))
        raise ValueError(f"unknown actor head {name!r}; registered: {registered}")
    drawn = {} if kernel_init is None else {"kernel_init": kernel_init}
    return ACTOR_HEADS[name](action_dim=action_dim, **drawn)
