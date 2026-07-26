"""The surface every algorithm presents to whatever drives it.

An algorithm owns its own loop. What it agrees to is narrow: hand back a
program that can be initialised, trained for an epoch, and evaluated, plus
pytrees with a fixed shape so a caller can read results without knowing which
algorithm produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
class EvaluationConfig:
    """Host-side evaluation choices, validated before a program is closed."""

    reset_on_start: bool = True
    update_during_eval: bool = True

    def replace(self, **updates):
        return replace(self, **updates)
