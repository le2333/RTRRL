"""A round that was interrupted, against the round that never was.

A round is one graph, so it is one snapshot: the members' parameters are
leaves of the same arrays and their keys are rows of the same key array. What
has to survive that is each member's separateness -- its own open episodes,
its own destination, its own numbering -- because the whole point of the
ensemble is that a member gets what it would have got on its own.

The failure this rules out is the one a shared snapshot invites: a resume that
restores the arrays correctly and then hands member 1's half-finished episode
to member 0. Nothing downstream could detect that. Every member is therefore
compared against its own uninterrupted run, not against the round's total.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import pytest

from memorax.runtime import (
    BuiltAlgorithm,
    FileSnapshotStore,
    ObservationSchema,
    Program,
    RuntimeConfig,
)
from memorax.runtime.ensemble import EnsembleRuntime
from tests.support.fakes import ResumableRecorder

NUM_ENVS = 2
TOTAL_STEPS = 64
EVERY_STEPS = 16
SEEDS = (3, 4, 5)

OBSERVATIONS = ObservationSchema(
    reward="interaction.reward",
    done="interaction.done",
    terminal="interaction.terminal",
)


class Interaction(NamedTuple):
    reward: Any
    done: Any
    terminal: Any


class Metrics(NamedTuple):
    interaction: Interaction


def train(key, state, num_steps):
    """Episodes that end where each member's own key says they do."""

    rows = num_steps // NUM_ENVS
    draws = jax.random.uniform(key, (rows, NUM_ENVS))
    ending = draws > 0.5
    return state + jnp.sum(draws), Metrics(
        interaction=Interaction(reward=draws + state, done=ending, terminal=ending)
    )


def evaluate(key, state, num_steps):
    """One episode per stream, reporting the member's own state."""

    del key
    rows = num_steps // NUM_ENVS
    ending = jnp.broadcast_to(jnp.arange(rows)[:, None] == 0, (rows, NUM_ENVS))
    return state, Metrics(
        interaction=Interaction(
            reward=jnp.full((rows, NUM_ENVS), state), done=ending, terminal=ending
        )
    )


def algorithm() -> BuiltAlgorithm:
    return BuiltAlgorithm(
        program=Program(
            init=lambda key: jax.random.uniform(key),
            train=train,
            open_evaluation=lambda key, state: state,
            evaluate=evaluate,
            interact=lambda key, state: None,
        ),
        observations=OBSERVATIONS,
    )


def config() -> RuntimeConfig:
    return RuntimeConfig(
        total_steps=TOTAL_STEPS,
        chunk_steps=8,
        max_episode_steps=TOTAL_STEPS,
        evaluate_every_steps=EVERY_STEPS,
        evaluation_episodes=2,
        evaluation_chunk_steps=4,
        evaluation_seed=1000,
        num_envs=NUM_ENVS,
        seed=-1,
        snapshot_every_steps=EVERY_STEPS,
    )


class Interruption(RuntimeError):
    """The machine going away, at a moment of the test's choosing."""


@dataclass
class Interrupted(ResumableRecorder):
    after: int = 0

    def log_episode(self, episode: Any) -> None:
        if len(self.episodes) >= self.after:
            raise Interruption(f"stopped after {self.after} episodes")
        super().log_episode(episode)


def reduced(recorder: ResumableRecorder) -> list[tuple[Any, ...]]:
    return [
        (
            episode.number,
            episode.phase,
            episode.stream,
            episode.start_env_steps,
            episode.end_env_steps,
            tuple(round(float(reward), 10) for reward in episode.rewards),
        )
        for episode in recorder.episodes
    ]


def run(store, recorders) -> None:
    EnsembleRuntime(
        algorithm=algorithm(), config=config(), seeds=SEEDS, snapshots=store
    ).run(recorders)


def test_every_member_of_a_resumed_round_reports_what_it_would_have(
    tmp_path: Path,
) -> None:
    """Each member against itself, so a swapped member is a failure."""

    whole = [ResumableRecorder() for _ in SEEDS]
    run(FileSnapshotStore(tmp_path / "whole"), whole)

    store = FileSnapshotStore(tmp_path / "interrupted")
    # The last member is the one that stops the round, so the members are at
    # different points in their episodes when the interruption arrives -- which
    # is what a snapshot that mixed them up would survive.
    stopped = [
        (
            Interrupted(after=len(whole[-1].episodes) // 2)
            if index == len(SEEDS) - 1
            else ResumableRecorder()
        )
        for index in range(len(SEEDS))
    ]
    with pytest.raises(Interruption):
        run(store, stopped)

    interrupted = store.latest()
    assert interrupted is not None and 0 < interrupted.step < TOTAL_STEPS
    assert len(interrupted.trackers) == len(SEEDS)

    carried = [
        ResumableRecorder(episodes=list(recorder.episodes)) for recorder in stopped
    ]
    run(store, carried)

    for resumed, expected in zip(carried, whole, strict=True):
        assert reduced(resumed) == reduced(expected)


def test_a_round_resumed_at_a_different_width_is_refused(tmp_path: Path) -> None:
    """A snapshot of three members is not a start for two.

    The arrays it holds have a member axis, and splitting them would run a
    different graph from the one that was interrupted.
    """

    store = FileSnapshotStore(tmp_path)
    run(store, [ResumableRecorder() for _ in SEEDS])

    with pytest.raises(ValueError, match="and this round has 2"):
        EnsembleRuntime(
            algorithm=algorithm(), config=config(), seeds=SEEDS[:2], snapshots=store
        ).run([ResumableRecorder(), ResumableRecorder()])


def test_a_round_that_asks_to_be_written_down_needs_somewhere_to_write() -> None:
    """The same rule as the single-member driver, for the same reason."""

    with pytest.raises(ValueError, match="nowhere to write"):
        EnsembleRuntime(algorithm=algorithm(), config=config(), seeds=SEEDS).run(
            [ResumableRecorder() for _ in SEEDS]
        )
