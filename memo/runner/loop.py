"""Train, report, evaluate, report, until the budget is spent.

The arrows only. What is being trained, on what, under which parameter names,
and which scalars are worth reporting are all decided by the caller and passed
in already resolved -- this file names none of them, which is why one copy can
drive every entry without becoming the place they all have to agree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from training_sdk.episode import Episode

from .episodes import complete_episodes


class Destination(Protocol):
    """All of the reporter this file touches."""

    def report(self, step: int, metrics: Mapping[str, float]) -> None: ...

    def log_episode(self, episode: Episode) -> None: ...


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


def report_evaluation(
    reporter: Destination,
    summary,
    *,
    done: int,
    num_envs: int,
    number: int,
    rewards=None,
) -> int:
    """Report the score, and hand every whole episode to the viewer."""

    paid = summary.reward if rewards is None else rewards
    returns: list[float] = []
    lengths: list[int] = []
    for episode in complete_episodes(
        summary,
        phase="eval",
        start_env_steps=done,
        num_envs=num_envs,
        first_number=number,
        rewards=paid,
    ):
        reporter.log_episode(episode)
        number = episode.number + 1
        returns.append(float(sum(episode.rewards)))
        lengths.append(len(episode.actions))

    report = {"eval/reward": float(jnp.nanmean(paid))}
    # A mean over no episodes would be reported as zero and read as a score.
    if returns:
        report["eval/episode_return"] = sum(returns) / len(returns)
        report["eval/episode_length"] = sum(lengths) / len(lengths)
    reporter.report(done, report)
    return number


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
    training_report: Callable[[Any], Mapping[str, float]],
    eval_reward: Callable[[Any], Any] | None = None,
) -> None:
    """Run the algorithm to its budget, reporting once per epoch.

    Three functions rather than an object, so that an algorithm can be a class
    with methods or a built program holding closures without either having to
    become the other to be driven.

    ``eval_reward`` picks what an episode was worth out of the summary, for the
    entry whose environment hands its algorithm a rescaled reward. Left out, the
    summary's own reward is what it was worth.
    """

    epochs = whole_epochs(
        total_steps=total_steps, epoch_steps=epoch_steps, num_envs=num_envs
    )

    train = jax.jit(train_fn, static_argnums=2)
    evaluate = jax.jit(evaluate_fn, static_argnums=2)

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    state = jax.jit(init_fn)(init_key)

    number = 1
    for done in epochs:
        key, epoch_key = jax.random.split(key)
        state, metrics = train(epoch_key, state, epoch_steps)
        reporter.report(done, training_report(metrics))

        if not eval_steps:
            continue
        key, eval_key = jax.random.split(key)
        _, summary = evaluate(eval_key, state, eval_steps)
        number = report_evaluation(
            reporter,
            summary,
            done=done,
            num_envs=num_envs,
            number=number,
            rewards=None if eval_reward is None else eval_reward(summary),
        )


def named_scalars(metrics, names, *, prefix: str) -> dict[str, float]:
    """Average the fields the caller named, skipping any the run left absent.

    A field is absent when the configuration that produces it was not chosen,
    so its name is still worth declaring: it says what this algorithm can
    report, not what this run happened to.
    """

    report = {}
    for name in names:
        value = _reading(metrics, name)
        if value is None:
            continue
        report[f"{prefix}{name}"] = float(jnp.nanmean(value))
    return report


def _reading(metrics, name: str):
    """A named field, or a named key of one, which is how a family is read.

    A kernel that reports one number per part of a network cannot know what the
    parts are called, so it hands back a mapping and the entry names the parts
    it built.
    """

    found = metrics
    for step in name.split("/"):
        if isinstance(found, Mapping):
            found = found.get(step)
        else:
            found = getattr(found, step, None)
        if found is None:
            return None
    return found
