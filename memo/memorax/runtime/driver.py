"""Train, evaluate, until the budget is spent, reporting when an episode ends.

The arrows only. What is being trained, on what, under which parameter names,
and which step-level quantities are worth reducing are all decided by the caller
and passed in already resolved -- this file names none of them, which is why one
copy can drive every entry without becoming the place they all have to agree.

A completed episode is the only reporting occasion. Its boundary comes from the
environment; a chunk's does not, and an epoch's says only how often to evaluate.

Assembled once and started once: a run is a :class:`Runtime` built from an
algorithm and a budget, and then run. The container's worker holds the process
this is assembled in, not the object -- it starts an entry, and the entry builds
the algorithm and hands it here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import jax

from memorax.runtime.episode import Episode
from memorax.runtime.rollout import complete_episodes

from .program import AgentProgram, program_of


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
DEFAULT_REWARD = f"{TRANSITIONS}.reward"


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


@dataclass(frozen=True)
class Runtime:
    """One run: an algorithm's three arrows, a budget, and where episodes go.

    ``series`` names the step-level quantities to reduce over each episode. A
    name the run never produced is left out rather than reported as nothing, so
    the declaration says what this algorithm can measure rather than what this
    configuration happened to.
    """

    program: AgentProgram
    total_steps: int
    epoch_steps: int
    eval_steps: int
    num_envs: int
    seed: int
    series: tuple[str, ...] = ()
    reward: str = DEFAULT_REWARD

    @classmethod
    def from_config(
        cls,
        algorithm: Any,
        config: Any,
        *,
        series: Iterable[str] = (),
        reward: str = DEFAULT_REWARD,
    ) -> Runtime:
        """Assemble from a resolved run configuration, without naming its type.

        The fields are read, not the class: what a run configuration is belongs
        to whatever shipped it, and a runtime that imported that definition
        would make the library depend on the deployment contract to run a
        kernel on a laptop.
        """

        return cls(
            program=program_of(algorithm),
            total_steps=config.training.total_steps,
            epoch_steps=config.training.epoch_steps,
            eval_steps=config.evaluation.steps,
            num_envs=config.training.num_envs,
            seed=config.environment.seed,
            series=tuple(series),
            reward=reward,
        )

    def run(self, reporter: Destination) -> None:
        """Run the algorithm to its budget, reporting every episode it finishes."""

        epochs = whole_epochs(
            total_steps=self.total_steps,
            epoch_steps=self.epoch_steps,
            num_envs=self.num_envs,
        )

        train = jax.jit(self.program.train_epoch_fn, static_argnums=2)
        evaluate = jax.jit(self.program.evaluate_fn, static_argnums=2)

        key = jax.random.key(self.seed)
        key, init_key = jax.random.split(key)
        state = jax.jit(self.program.init_fn)(init_key)

        numbers = {"train": 1, "eval": 1}
        for done in epochs:
            key, epoch_key = jax.random.split(key)
            state, metrics = train(epoch_key, state, self.epoch_steps)
            numbers["train"] = self._report(
                reporter,
                metrics,
                phase="train",
                start_env_steps=done - self.epoch_steps,
                stride=self.num_envs,
                first_number=numbers["train"],
            )

            if not self.eval_steps:
                continue
            key, eval_key = jax.random.split(key)
            _, summary = evaluate(eval_key, state, self.eval_steps)
            numbers["eval"] = self._report(
                reporter,
                summary,
                phase="eval",
                start_env_steps=done,
                stride=0,
                first_number=numbers["eval"],
            )

    def _report(self, reporter: Destination, chunk, **cut) -> int:
        number = cut["first_number"]
        for episode in complete_episodes(
            chunk,
            num_envs=self.num_envs,
            transitions=TRANSITIONS,
            reward=self.reward,
            terminal=f"{TRANSITIONS}.terminal",
            series=self.series,
            **cut,
        ):
            reporter.log_episode(episode)
            number = episode.number + 1
        return number


def drive(
    reporter: Destination,
    *,
    init_fn: Any,
    train_fn: Any,
    evaluate_fn: Any,
    total_steps: int,
    epoch_steps: int,
    eval_steps: int,
    num_envs: int,
    seed: int,
    series: Iterable[str] = (),
    reward: str = DEFAULT_REWARD,
) -> None:
    """The function form, for a caller holding three arrows and no algorithm.

    Three functions rather than an object, so that an algorithm can be a class
    with methods or a built program holding closures without either having to
    become the other to be driven.
    """

    Runtime(
        program=AgentProgram(
            init_fn=init_fn,
            train_epoch_fn=train_fn,
            evaluate_fn=evaluate_fn,
        ),
        total_steps=total_steps,
        epoch_steps=epoch_steps,
        eval_steps=eval_steps,
        num_envs=num_envs,
        seed=seed,
        series=tuple(series),
        reward=reward,
    ).run(reporter)
