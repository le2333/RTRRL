"""The surface every algorithm presents to whatever drives it.

An algorithm owns its own loop. What it agrees to is narrow: hand back a
program that can be initialised, trained for an epoch, and evaluated, plus
pytrees with a fixed shape so a caller can read results without knowing which
algorithm produced them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

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


def terminal_of(info, done):
    """The failure ending, from an environment that tells the two apart.

    One that does not says every ending was a failure, which is what it knew and
    what a single flag always meant. Reading it that way is also the safe one: it
    never bootstraps past an ending the environment could not explain.
    """

    return info.get("terminal", done)


@struct.dataclass
class EvalSummary:
    """One evaluation step, stacked by the caller's scan.

    The transition is spelled out rather than left inside ``info`` because a
    viewer needs to replay it, and reconstructing episodes from a bag of
    environment-specific keys would make the host care which environment ran.
    """

    info: Any = None
    normalization: Any = None
    observation: Any = None
    next_observation: Any = None
    action: Any = None
    reward: Any = None
    done: Any = None
    terminal: Any = None


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
