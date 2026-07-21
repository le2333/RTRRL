"""Pure slow-subtree targeting and post-update Polyak sequencing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import optax
from flax import core, struct

_FORWARD_VIEWS = frozenset({"acting", "bootstrap", "differentiation"})


@dataclass(frozen=True)
class GradientDestination:
    """A gradient paired with the parameter tree it must update."""

    destination: Any
    gradient: Any


@dataclass(frozen=True)
class TargetViews:
    """All forward parameter views and the explicit update-routing contract."""

    acting: Any
    bootstrap: Any
    differentiation: Any
    update_destination: Any
    gradient_to_destination: Any


@struct.dataclass
class TargetUpdate:
    """Fast parameters, next slow torso, and unchanged online sensitivity."""

    fast_params: Any
    slow_subtree: Any
    sensitivity: Any


class SlowSubtreeTarget:
    """Use a slow torso for forwards while updates remain fast-param updates."""

    def __init__(self, update_period):
        self.update_period = update_period

    def forward_params(self, *, fast_params, slow_subtree, view):
        if view not in _FORWARD_VIEWS:
            raise ValueError(f"unknown forward view: {view}")
        if isinstance(fast_params, core.FrozenDict):
            return fast_params.copy(add_or_replace={"torso": slow_subtree})
        return {**fast_params, "torso": slow_subtree}

    def views(self, *, fast_params, slow_subtree):
        """Bind all slow forward views and route their gradients to fast params."""

        forward_views = {
            view: self.forward_params(
                fast_params=fast_params,
                slow_subtree=slow_subtree,
                view=view,
            )
            for view in _FORWARD_VIEWS
        }
        destination_structure = jax.tree_util.tree_structure(fast_params)

        def gradient_to_destination(gradient):
            if jax.tree_util.tree_structure(gradient) != destination_structure:
                raise ValueError("gradient tree must match the fast update destination")
            return GradientDestination(
                destination=fast_params,
                gradient=gradient,
            )

        return TargetViews(
            acting=forward_views["acting"],
            bootstrap=forward_views["bootstrap"],
            differentiation=forward_views["differentiation"],
            update_destination=fast_params,
            gradient_to_destination=gradient_to_destination,
        )

    def finish_update(
        self,
        *,
        fast_params,
        previous_slow_subtree,
        sensitivity,
    ):
        fast_torso = fast_params["torso"]
        slow_subtree = (
            fast_torso
            if self.update_period == 1.0
            else optax.incremental_update(
                fast_torso,
                previous_slow_subtree,
                self.update_period,
            )
        )
        return TargetUpdate(
            fast_params=fast_params,
            slow_subtree=slow_subtree,
            sensitivity=sensitivity,
        )


def make_slow_subtree_target(config):
    """Build the RTRRL slow-torso target kernel."""

    return SlowSubtreeTarget(config.update_period)
