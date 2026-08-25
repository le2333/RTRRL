"""A run that was interrupted, against the run that never was.

What a snapshot owes an interrupted run is equality, not similarity: the
episodes, the readings and the state a second process arrives at have to be
the ones a single process would have produced. Nothing weaker is testable --
"close enough" over a training run is what a broken key stream looks like --
and nothing weaker is usable, because a result whose value depends on whether
the machine survived is not a result.

So the arithmetic here is made to notice each of the three ways a resume can
be wrong. The state accumulates, so a run that re-initialised diverges. The
rewards are drawn from the training key, so a run that rebuilt its key stream
from the seed replays transitions and diverges. The episodes end on those same
draws, so an episode open at the boundary spans it -- and a run that dropped
what the tracker was holding loses that episode's first half, its return and
every later episode's number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.runtime import (
    BuiltAlgorithm,
    FileSnapshotStore,
    ObservationSchema,
    Program,
    Runtime,
    RuntimeConfig,
)
from tests.support.fakes import ResumableRecorder

NUM_ENVS = 2
TOTAL_STEPS = 64
CHUNK_STEPS = 8
EVERY_STEPS = 16

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
    """Rewards drawn from the key, on top of everything drawn before.

    Both dependencies are the point. ``state`` carries, so a re-initialised
    run is visible in the first episode it reports; the draws come from the
    key this call was given, so a rebuilt key stream is visible in the same
    place.
    """

    rows = num_steps // NUM_ENVS
    draws = jax.random.uniform(key, (rows, NUM_ENVS))
    ending = draws > 0.5
    return state + jnp.sum(draws), Metrics(
        interaction=Interaction(reward=draws + state, done=ending, terminal=ending)
    )


def open_evaluation(key, state):
    del key
    return state


def evaluate(key, state, num_steps):
    """A rollout that ends one episode per stream immediately.

    An evaluation reaches the training state only to read it, so what it
    reports here is that state and nothing else: a resumed run whose
    evaluation matches is a resumed run that restored the right parameters.
    """

    del key
    rows = num_steps // NUM_ENVS
    ending = jnp.arange(rows)[:, None] == 0
    return state, Metrics(
        interaction=Interaction(
            reward=jnp.full((rows, NUM_ENVS), state),
            done=jnp.broadcast_to(ending, (rows, NUM_ENVS)),
            terminal=jnp.broadcast_to(ending, (rows, NUM_ENVS)),
        )
    )


def interact(key, state):
    del key
    quiet = jnp.zeros((NUM_ENVS,), dtype=bool)
    return state, Metrics(
        interaction=Interaction(
            reward=jnp.zeros((NUM_ENVS,)), done=quiet, terminal=quiet
        )
    )


ALGORITHM = BuiltAlgorithm(
    program=Program(
        init=lambda key: jnp.asarray(0.0),
        train=train,
        open_evaluation=open_evaluation,
        evaluate=evaluate,
        interact=interact,
    ),
    observations=OBSERVATIONS,
)


def config(snapshot_every_steps: int) -> RuntimeConfig:
    return RuntimeConfig(
        total_steps=TOTAL_STEPS,
        chunk_steps=CHUNK_STEPS,
        max_episode_steps=TOTAL_STEPS,
        evaluate_every_steps=EVERY_STEPS,
        evaluation_episodes=2,
        evaluation_chunk_steps=4,
        evaluation_seed=11,
        num_envs=NUM_ENVS,
        seed=3,
        snapshot_every_steps=snapshot_every_steps,
    )


class Interruption(RuntimeError):
    """The machine going away, at a moment of the test's choosing."""


@dataclass
class Interrupted(ResumableRecorder):
    """A destination that stops the run once it has seen enough episodes."""

    after: int = 0

    def log_episode(self, episode: Any) -> None:
        if len(self.episodes) >= self.after:
            raise Interruption(f"stopped after {self.after} episodes")
        super().log_episode(episode)


def reduced(recorder: ResumableRecorder) -> list[tuple[Any, ...]]:
    """Each episode as the values a reader would compare two runs on."""

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


def run_once(store: FileSnapshotStore | None, recorder: ResumableRecorder) -> None:
    Runtime(algorithm=ALGORITHM, config=config(EVERY_STEPS), snapshots=store).run(
        recorder
    )


def test_a_resumed_run_reports_what_one_uninterrupted_run_would_have(
    tmp_path: Path,
) -> None:
    """The whole claim, end to end.

    The second process is handed the first one's reported episodes, the way a
    resumed job is handed back the metrics artifact its predecessor was
    writing, and cuts them to the snapshot before adding its own. What it ends
    up holding is compared against a run that was never stopped.
    """

    whole = ResumableRecorder()
    run_once(FileSnapshotStore(tmp_path / "whole"), whole)

    store = FileSnapshotStore(tmp_path / "interrupted")
    stopped = Interrupted(after=len(whole.episodes) // 2)
    with pytest.raises(Interruption):
        run_once(store, stopped)

    interrupted = store.latest()
    assert interrupted is not None and 0 < interrupted.step < TOTAL_STEPS

    carried = ResumableRecorder(
        episodes=list(stopped.episodes), trajectories=list(stopped.trajectories)
    )
    run_once(store, carried)

    assert reduced(carried) == reduced(whole)


def test_a_resumed_run_arrives_at_the_state_the_whole_run_did(
    tmp_path: Path,
) -> None:
    """Not only the same report: the same parameters underneath it."""

    whole_store = FileSnapshotStore(tmp_path / "whole")
    run_once(whole_store, ResumableRecorder())

    store = FileSnapshotStore(tmp_path / "interrupted")
    stopped = Interrupted(after=6)
    with pytest.raises(Interruption):
        run_once(store, stopped)
    # An interruption before the first boundary resumes from nothing, and this
    # test would then be comparing a fresh run against a fresh run.
    interrupted = store.latest()
    assert interrupted is not None and interrupted.step > 0
    run_once(store, ResumableRecorder(episodes=list(stopped.episodes)))

    whole = whole_store.latest()
    resumed = store.latest()
    assert whole is not None and resumed is not None
    assert whole.step == resumed.step
    assert np.array_equal(
        np.asarray(whole.algorithm_state()), np.asarray(resumed.algorithm_state())
    )


def test_a_run_with_nowhere_to_write_is_the_run_it_always_was(
    tmp_path: Path,
) -> None:
    """No store, no snapshots, and the same episodes as a run that has one."""

    del tmp_path
    with_store = ResumableRecorder()
    without = ResumableRecorder()
    Runtime(algorithm=ALGORITHM, config=config(0)).run(without)
    Runtime(algorithm=ALGORITHM, config=config(0), snapshots=None).run(with_store)

    assert reduced(without) == reduced(with_store)


def test_a_run_that_asks_to_be_written_down_needs_somewhere_to_write(
    tmp_path: Path,
) -> None:
    """A run that says it can be resumed and cannot is the worst of the three.

    It costs nothing while it is going and everything when it is interrupted,
    and the operator finds out at the moment they were counting on it.
    """

    del tmp_path
    with pytest.raises(ValueError, match="nowhere to write"):
        Runtime(algorithm=ALGORITHM, config=config(EVERY_STEPS)).run(
            ResumableRecorder()
        )


def test_taking_no_snapshots_writes_nothing(tmp_path: Path) -> None:
    """Zero is off, and off does not mean 'every boundary, discarded'."""

    store = FileSnapshotStore(tmp_path)
    Runtime(algorithm=ALGORITHM, config=config(0), snapshots=store).run(
        ResumableRecorder()
    )

    assert store.latest() is None


def test_the_last_boundary_is_not_written_down(tmp_path: Path) -> None:
    """A snapshot at the end of the budget resumes to a run with nothing to do.

    The store still holds the boundary before it, which is what a job killed
    during the final interval would need.
    """

    store = FileSnapshotStore(tmp_path)
    run_once(store, ResumableRecorder())

    resumed = store.latest()
    assert resumed is not None and resumed.step == TOTAL_STEPS - EVERY_STEPS


def test_an_ensemble_snapshot_is_refused_by_the_single_member_driver(
    tmp_path: Path,
) -> None:
    """One member's open episodes reported under another member's name."""

    store = FileSnapshotStore(tmp_path)
    run_once(store, ResumableRecorder())
    written = store.latest()
    assert written is not None
    store.save(
        type(written)(
            step=written.step,
            eval_number=written.eval_number,
            state=written.state,
            key_data=written.key_data,
            key_impl=written.key_impl,
            trackers=(written.trackers[0], written.trackers[0]),
            destinations=(None, None),
        )
    )

    with pytest.raises(ValueError, match="2 members' trackers"):
        run_once(store, ResumableRecorder())
