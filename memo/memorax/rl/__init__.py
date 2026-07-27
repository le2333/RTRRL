"""Reinforcement-learning primitives shared across algorithms.

Only computation lives here. How an algorithm allocates traces, routes
objectives, or drives its own loop stays with the algorithm.
"""

from .credit import CREDITS, make_credit, make_exact_rtrl_credit
from .normalization import (
    NormalizationConfig,
    NormalizationMetrics,
    NormalizedStep,
    Normalizer,
    NormalizerState,
    RewardStatistics,
    RunningStatistics,
    environment_owns_normalization,
    make_normalizer,
    normalization_metrics,
)
from .targets import delayed_update, periodic_incremental_update
from .td import make_td0
from .updates import (
    BOUNDED_RULES,
    ObjectiveDirections,
    RuleOutput,
    UpdateRule,
    make_obgd_rule,
    make_optax_rule,
)

__all__ = [
    "BOUNDED_RULES",
    "CREDITS",
    "NormalizationConfig",
    "NormalizationMetrics",
    "NormalizedStep",
    "Normalizer",
    "NormalizerState",
    "ObjectiveDirections",
    "RewardStatistics",
    "RuleOutput",
    "RunningStatistics",
    "UpdateRule",
    "delayed_update",
    "environment_owns_normalization",
    "make_credit",
    "make_exact_rtrl_credit",
    "make_normalizer",
    "make_obgd_rule",
    "make_optax_rule",
    "make_td0",
    "normalization_metrics",
    "periodic_incremental_update",
]
