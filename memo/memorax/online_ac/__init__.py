"""Composable online actor-critic building blocks."""

from .build import (
    LegacyProgram,
    build_meta_program,
    build_standard_program,
    legacy_env_adapter,
    legacy_normalization_config,
)
from .credit import make_exact_rtrl_credit
from .meta import MetaState, MetaStepMetrics, make_meta_program
from .normalization import (
    NormalizationConfig,
    NormalizedStep,
    NormalizerState,
    RewardStatistics,
    RunningStatistics,
    make_normalizer,
)
from .objectives import (
    ObjectiveDirections,
    make_rtrrl_objective,
    make_stream_ac_objective,
)
from .standard import (
    NetworkState,
    StandardState,
    StandardStepMetrics,
    make_standard_program,
)
from .targets import (
    GradientDestination,
    TargetUpdate,
    TargetViews,
    make_slow_subtree_target,
)
from .td import make_td0
from .traces import TraceDirections, make_rtrrl_trace, make_stream_ac_trace
from .types import (
    ActionDecision,
    AgentProgram,
    EvalSummary,
    EvaluationConfig,
    ExactRTRLConfig,
    JAXEnvAdapter,
    MetaProgramConfig,
    SlowSubtreeTargetConfig,
    StandardProgramConfig,
    Transition,
    WholeTreeOBGDConfig,
)
from .updates import make_grouped_adam, make_whole_tree_obgd

__all__ = [
    "ActionDecision",
    "AgentProgram",
    "EvalSummary",
    "EvaluationConfig",
    "ExactRTRLConfig",
    "GradientDestination",
    "JAXEnvAdapter",
    "LegacyProgram",
    "MetaState",
    "MetaStepMetrics",
    "MetaProgramConfig",
    "NetworkState",
    "NormalizationConfig",
    "NormalizedStep",
    "NormalizerState",
    "ObjectiveDirections",
    "StandardProgramConfig",
    "StandardState",
    "StandardStepMetrics",
    "RewardStatistics",
    "RunningStatistics",
    "SlowSubtreeTargetConfig",
    "TargetUpdate",
    "TargetViews",
    "TraceDirections",
    "Transition",
    "WholeTreeOBGDConfig",
    "build_meta_program",
    "build_standard_program",
    "legacy_env_adapter",
    "legacy_normalization_config",
    "make_exact_rtrl_credit",
    "make_grouped_adam",
    "make_meta_program",
    "make_normalizer",
    "make_rtrrl_objective",
    "make_rtrrl_trace",
    "make_slow_subtree_target",
    "make_stream_ac_objective",
    "make_stream_ac_trace",
    "make_standard_program",
    "make_td0",
    "make_whole_tree_obgd",
]
