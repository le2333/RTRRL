"""Explicit pytrees used by the strict RTRRL online state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flax import struct


@dataclass(frozen=True)
class RTRRLComponents:
    """Static numerical components captured by lifecycle factories."""

    recurrent: Any
    head: Any


@struct.dataclass
class InitializationKeys:
    """Historical split outputs that remain explicit dynamic values."""

    step: Any
    outer: Any
    model: Any
    carry: Any
    environment: Any


@struct.dataclass
class RTRRLState:
    """Complete persisted state for one strict online transition."""

    parameters: Any
    slow_parameters: Any
    optimizer_state: Any
    environment_state: Any
    action: Any
    recurrent_state: Any
    traces: Any
    value: Any
    average_reward: Any
    emphasis: Any
    observation_statistics: Any
    reward_statistics: Any
    model_input: Any
    initial_recurrent_state: Any
    step_count: Any


@struct.dataclass
class TrainStepMetrics:
    """Scalar/event-only production metrics."""

    reward: Any
    done: Any
    td_error_mean: Any
    value_mean: Any
    value_target_mean: Any
    entropy_mean: Any
    actor_loss_mean: Any
    magnitude_loss_mean: Any = None


@struct.dataclass
class DebugStepMetrics:
    """Full eager diagnostics, restricted to the first three steps."""

    train: TrainStepMetrics
    environment_action: Any
    model_input: Any
    sampled_next_action: Any
    value_target: Any
    td_error: Any
    value: Any
    actor_loss: Any
    entropy: Any
    gradients: Any
    direct_gradients: Any
    incoming_traces: Any
    carried_traces: Any
    mean_directions: Any
    optimizer_updates: Any
    fast_parameters: Any
    slow_parameters: Any
    emphasis: Any
    average_reward: Any
    debug_active: Any


__all__ = [
    "DebugStepMetrics",
    "InitializationKeys",
    "RTRRLComponents",
    "RTRRLState",
    "TrainStepMetrics",
]
