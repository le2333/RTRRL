"""Train, evaluate, until the budget is spent, reporting when an episode ends.

The arrows only. What is being trained, on what, under which parameter names,
and which step-level quantities are worth reducing are all decided by the caller
and passed in already resolved -- this file names none of them, which is why one
copy can drive every entry without becoming the place they all have to agree.

A completed episode is the only reporting occasion. Its boundary comes from the
environment; a chunk's does not, and an epoch's says only how often to evaluate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

import jax
from training_sdk.episode import Episode
from training_sdk.rollout import complete_episodes


class Destination(Protocol):
    """All of the reporter this file touches."""

    def log_episode(self, episode: Episode) -> None: ...


# Where a kernel groups the transition, and what it has to put there before any
# of this can find an episode at all. A kernel that gates these behind a switch
# reports nothing and says nothing about it, so an entry adds them to whatever
# its own metrics need.
TRANSITIONS = "interaction"
EPISODE_FIELDS: tuple[str, ...] = tuple(
    f"{TRANSITIONS}.{name}" for name in ("reward", "done", "terminal")
)


def whole_epochs(*, total_steps: int, epoch_steps: int, num_envs: int) -> range:
    """The step count at each epoch boundary, refusing a budget that is ragged.

    A budget that does not divide has to be rounded, and either direction is a
    lie: down reports a step count the run never reached, up spends money
    nobody asked for.
    """

    if epoch_steps % num_envs:
        raise ValueError(f"epoch_steps {epoch_steps} is not {num_envs} streams' worth")
    if total_steps % epoch_steps:
        raise ValueError(
            f"total_steps {total_steps} is not whole epochs of {epoch_steps}"
        )
    return range(epoch_steps, total_steps + 1, epoch_steps)


def drive(
    reporter: Destination,
    *,
    init_fn: Callable[..., Any],
    train_fn: Callable[..., Any],
    evaluate_fn: Callable[..., Any],
    total_steps: int,
    epoch_steps: int,
    eval_steps: int,
    num_envs: int,
    seed: int,
    series: Iterable[str] = (),
    reward: str = f"{TRANSITIONS}.reward",
) -> None:
    """Run the algorithm to its budget, reporting on every episode it finishes.

    Three functions rather than an object, so that an algorithm can be a class
    with methods or a built program holding closures without either having to
    become the other to be driven.

    ``series`` names the step-level quantities to reduce over each episode. A
    name the run never produced is left out rather than reported as nothing, so
    the declaration says what this algorithm can measure rather than what this
    configuration happened to.
    """

    epochs = whole_epochs(
        total_steps=total_steps, epoch_steps=epoch_steps, num_envs=num_envs
    )

    train = jax.jit(train_fn, static_argnums=2)
    evaluate = jax.jit(evaluate_fn, static_argnums=2)

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    state = jax.jit(init_fn)(init_key)

    numbers = {"train": 1, "eval": 1}
    for done in epochs:
        key, epoch_key = jax.random.split(key)
        state, metrics = train(epoch_key, state, epoch_steps)
        numbers["train"] = _report(
            reporter,
            metrics,
            phase="train",
            start_env_steps=done - epoch_steps,
            num_envs=num_envs,
            stride=num_envs,
            first_number=numbers["train"],
            transitions=TRANSITIONS,
            reward=reward,
            terminal=f"{TRANSITIONS}.terminal",
            series=series,
        )

        if not eval_steps:
            continue
        key, eval_key = jax.random.split(key)
        _, summary = evaluate(eval_key, state, eval_steps)
        numbers["eval"] = _report(
            reporter,
            summary,
            phase="eval",
            start_env_steps=done,
            num_envs=num_envs,
            stride=0,
            first_number=numbers["eval"],
            transitions=TRANSITIONS,
            reward=reward,
            terminal=f"{TRANSITIONS}.terminal",
            series=series,
        )


def _report(reporter: Destination, chunk, **cut) -> int:
    number = cut["first_number"]
    for episode in complete_episodes(chunk, **cut):
        reporter.log_episode(episode)
        number = episode.number + 1
    return number
