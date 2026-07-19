"""Host-side contracts shared by composable online actor-critic programs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from flax import struct


@dataclass(frozen=True)
class AgentProgram:
    """A built program and its stable host-side schemas."""

    init_fn: Callable[..., Any]
    train_epoch_fn: Callable[..., Any]
    evaluate_fn: Callable[..., Any]
    state_schema: Any
    metric_schema: Any


@struct.dataclass
class ActionDecision:
    """All action representations chosen during one acting forward pass."""

    sampled_action: Any = None
    logprob_action: Any = None
    env_action: Any = None
    bootstrap_feedback_action: Any = None
    persisted_feedback_action: Any = None


@struct.dataclass
class Transition:
    """JAX-pytree data for one online environment transition."""

    observation: Any = None
    action_decision: ActionDecision | None = None
    reward: Any = None
    done: Any = None
    next_observation: Any = None
    bootstrap_discount: Any = None
    info: Any = None


@struct.dataclass
class EvalSummary:
    """Fixed JAX-pytree evaluation outputs consumed by host-side façades."""

    info: Any = None
    normalization: Any = None


@dataclass(frozen=True)
class JAXEnvAdapter:
    """The explicit JAX boundary of an environment integration."""

    reset_fn: Callable[..., Any]
    step_fn: Callable[..., Any]
    env_params: Any
    build_context: Mapping[str, Any]


@dataclass(frozen=True)
class ExactRTRLConfig:
    """Build-time tag selecting the exact local-Jacobian credit path."""


@dataclass(frozen=True)
class SlowSubtreeTargetConfig:
    """Build-time slow-target routing for shared-torso RTRRL."""

    subtree: str | None = "torso"
    gradient_domain: str = "fast"


@dataclass(frozen=True)
class WholeTreeOBGDConfig:
    """Build-time update routing for StreamAC's whole parameter trees."""

    domain: str = "whole_tree"


@dataclass(frozen=True)
class EvaluationConfig:
    """Host-side evaluation choices, validated before a program is closed."""

    reset_on_start: bool = True
    update_during_eval: bool = True

    def replace(self, **updates):
        return replace(self, **updates)


@dataclass(frozen=True)
class MetaProgramConfig:
    """Complete host-side recipe for a shared-torso meta program."""

    static_config: Any = None
    feature_extractor: Any = None
    torso: Any = None
    actor_head: Any = None
    critic_head: Any = None
    pred_head: Any = None
    activation: Callable[[Any], Any] | None = None
    credit: Any = field(default_factory=ExactRTRLConfig)
    target: Any = field(default_factory=SlowSubtreeTargetConfig)
    normalization: Any = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_legacy_parts(cls, parts, *, normalization=None):
        return cls(
            static_config=parts.cfg,
            feature_extractor=parts.feature_extractor,
            torso=parts.torso,
            actor_head=parts.actor_head,
            critic_head=parts.critic_head,
            pred_head=getattr(parts, "pred_head", None),
            activation=getattr(parts, "activation", None),
            normalization=normalization,
        )

    def replace(self, **updates):
        return replace(self, **updates)


@dataclass(frozen=True)
class StandardProgramConfig:
    """Complete host-side recipe for independent actor/critic networks."""

    static_config: Any = None
    actor_network: Any = None
    critic_network: Any = None
    credit: Any = field(default_factory=ExactRTRLConfig)
    update: Any = field(default_factory=WholeTreeOBGDConfig)
    normalization: Any = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_legacy_parts(cls, parts, *, normalization=None):
        return cls(
            static_config=parts.cfg,
            actor_network=parts.actor_network,
            critic_network=parts.critic_network,
            normalization=normalization,
        )

    def replace(self, **updates):
        return replace(self, **updates)
