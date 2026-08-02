"""The surface every algorithm presents to whatever drives it.

An algorithm owns its own loop. What it agrees to is narrow: hand back a
program that can be initialised, trained for an epoch, and evaluated, plus
pytrees with a fixed shape so a caller can read results without knowing which
algorithm produced them.

What one step observed is grouped by what produced it rather than by when the
step ran. Training and evaluation used to have a container each, so the fields
they share -- the transition, the endings -- were written twice and gated once,
and the quantities only one of them declared were unreachable from the other
even when it had already computed them. Grouped this way, absence means
something: no update ran, so there are no update quantities.
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


@struct.dataclass
class InteractionMetrics:
    """One transition, as the environment and its preprocessing produced it.

    The transition is spelled out rather than left inside ``info`` because a
    viewer needs to replay it, and reconstructing episodes from a bag of
    environment-specific keys would make the host care which environment ran.

    ``observation`` is what the agent acted on and ``next_observation`` is where
    the transition ended -- not where the next episode starts, because nothing
    below the algorithm resets. ``done`` ends an episode either way; ``terminal``
    is the ending that says the future is worth nothing.
    """

    observation: Any = None
    next_observation: Any = None
    action: Any = None
    action_decision: ActionDecision | None = None
    reward: Any = None
    done: Any = None
    terminal: Any = None
    info: Any = None
    normalization: Any = None
    raw_episode_return: Any = None


@struct.dataclass
class StepMetrics:
    """Everything one step observed, by whose doing it was.

    ``forward`` and ``update`` are each algorithm's own: what a network answers
    and what an update produces differ between algorithms, while a transition
    does not. ``update`` is absent during evaluation, where nothing is updated.
    """

    interaction: InteractionMetrics = InteractionMetrics()
    forward: Any = None
    update: Any = None


def terminal_of(info, done):
    """The failure ending, from an environment that tells the two apart.

    One that does not says every ending was a failure, which is what it knew and
    what a single flag always meant. Reading it that way is also the safe one: it
    never bootstraps past an ending the environment could not explain.
    """

    return info.get("terminal", done)


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
