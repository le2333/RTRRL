"""Reinforcement-learning primitives shared across algorithms.

Only computation lives here. How an algorithm allocates traces, routes
objectives, or drives its own loop stays with the algorithm.
"""

from .interaction import (
    EnvironmentStreams,
    InteractionNormalization,
    NormalizationState,
    broadcast_stream,
    select_ended,
)
from .normalization import (
    COLD_STARTS,
    VARIANCES,
    NormalizationConfig,
    NormalizationMetrics,
    Normalizer,
    Statistics,
    declared_normalizer,
    environment_owns_normalization,
    make_normalizer,
    normalization_metrics,
)
from .targets import delayed_update, periodic_incremental_update
from .td import make_td0
from .updates import (
    ObjectiveDirections,
    RuleOutput,
    UpdateRule,
    make_bounded_rule,
    make_optax_rule,
)

__all__ = [
    "EnvironmentStreams",
    "InteractionNormalization",
    "NormalizationConfig",
    "NormalizationMetrics",
    "NormalizationState",
    "Normalizer",
    "ObjectiveDirections",
    "RuleOutput",
    "Statistics",
    "COLD_STARTS",
    "VARIANCES",
    "UpdateRule",
    "delayed_update",
    "broadcast_stream",
    "declared_normalizer",
    "environment_owns_normalization",
    "make_normalizer",
    "make_bounded_rule",
    "make_optax_rule",
    "make_td0",
    "normalization_metrics",
    "periodic_incremental_update",
    "select_ended",
]
