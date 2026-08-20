"""Reinforcement-learning primitives shared across algorithms.

Only computation lives here. How an algorithm allocates traces, routes
objectives, or drives its own loop stays with the algorithm.
"""

from .intentional import (
    ADVANTAGE,
    TD,
    IntentionalOptimizer,
    IntentionalReading,
    IntentionalState,
    IntentionalUpdate,
)
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
from .spaces import action_classes, action_dim, encode_feedback
from .targets import delayed_update, periodic_incremental_update
from .td import make_td0
from .traces import CARRIED, CURRENT, Trace
from .updates import (
    DRTRRL,
    ObjectiveDirections,
    RuleOutput,
    UpdateRule,
    make_bounded_rule,
    make_d_rtrrl_rule,
    make_intentional_rule,
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
    "CARRIED",
    "COLD_STARTS",
    "CURRENT",
    "VARIANCES",
    "ADVANTAGE",
    "DRTRRL",
    "IntentionalOptimizer",
    "IntentionalReading",
    "IntentionalState",
    "IntentionalUpdate",
    "TD",
    "Trace",
    "UpdateRule",
    "action_classes",
    "action_dim",
    "delayed_update",
    "broadcast_stream",
    "encode_feedback",
    "declared_normalizer",
    "environment_owns_normalization",
    "make_normalizer",
    "make_bounded_rule",
    "make_optax_rule",
    "make_td0",
    "make_d_rtrrl_rule",
    "make_intentional_rule",
    "normalization_metrics",
    "periodic_incremental_update",
    "select_ended",
]
