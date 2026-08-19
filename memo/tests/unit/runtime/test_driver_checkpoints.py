"""When Runtime files a whole state, and where a run that resumes one starts.

The program here is arithmetic rather than an algorithm, so what is under test
is the scheduling: that a checkpoint lands on a boundary the run also measured,
that a resumed run continues on the parent's step axis instead of starting the
same numbers over, and that a schedule which cannot satisfy either is refused
before a run rather than discovered in the metrics afterwards.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import pytest

from memorax.runtime import (
    BuiltAlgorithm,
    CheckpointDirectory,
    ObservationSchema,
    Program,
    Runtime,
    RuntimeConfig,
    checkpointed,
)
from memorax.runtime.checkpoint import read
from tests.support.fakes import EpisodeRecorder

NUM_ENVS = 2

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
    """One episode per call, ending on its last row, and a state that counts."""

    del key
    rows = num_steps // NUM_ENVS
    ending = jnp.zeros((rows, NUM_ENVS), dtype=bool).at[rows - 1].set(True)
    reward = jnp.ones((rows, NUM_ENVS), dtype=jnp.float32)
    return state + num_steps, Metrics(
        interaction=Interaction(reward=reward, done=ending, terminal=ending)
    )


def runtime(directory=None, keep=None, **schedule):
    return Runtime(
        algorithm=BuiltAlgorithm(
            program=Program(
                init=lambda key: jnp.asarray(0, jnp.int32),
                train=train,
                evaluate=lambda key, state, num_steps: None,
                interact=lambda key, state: None,
            ),
            observations=OBSERVATIONS,
        ),
        config=RuntimeConfig(
            rollout_steps=0,
            num_envs=NUM_ENVS,
            seed=0,
            **schedule,
        ),
        checkpoints=(
            None if directory is None else CheckpointDirectory(directory, keep=keep)
        ),
    )


SCHEDULE = {
    "total_steps": 40,
    "chunk_steps": 20,
    "max_episode_steps": 100,
    "evaluate_every_steps": 20,
}


def test_a_run_files_one_checkpoint_at_each_declared_boundary(tmp_path):
    runtime(tmp_path, checkpoint_every_steps=20, **SCHEDULE).run(EpisodeRecorder())

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "step-000000000020.msgpack",
        "step-000000000040.msgpack",
    ]


def test_a_checkpoint_interval_wider_than_the_evaluation_is_a_subset_of_it(tmp_path):
    """Every checkpoint is a measurement; not every measurement is a checkpoint."""

    runtime(tmp_path, checkpoint_every_steps=40, **SCHEDULE).run(EpisodeRecorder())

    assert [path.name for path in tmp_path.iterdir()] == ["step-000000000040.msgpack"]


def test_a_checkpoint_that_is_not_a_measurement_is_refused():
    with pytest.raises(ValueError, match="cannot be placed on the curve"):
        checkpointed(every_steps=30, evaluate_every_steps=20)


def test_a_run_that_files_nothing_is_the_ordinary_case(tmp_path):
    runtime(tmp_path, **SCHEDULE).run(EpisodeRecorder())

    assert not list(tmp_path.iterdir())


def test_the_stored_state_is_the_state_at_that_boundary(tmp_path):
    runtime(tmp_path, checkpoint_every_steps=20, **SCHEDULE).run(EpisodeRecorder())

    for boundary in (20, 40):
        stored = read(
            tmp_path / f"step-{boundary:012d}.msgpack",
            state=jnp.asarray(0, jnp.int32),
            key=jax.random.key(0),
        )
        assert stored.env_steps == boundary
        # The program's state counts the environment steps it was given, so
        # what it holds at a boundary is that boundary. A checkpoint written
        # one chunk early or late would say so here.
        assert int(stored.state) == boundary


def test_a_resumed_run_continues_on_its_parents_step_axis(tmp_path):
    runtime(tmp_path, checkpoint_every_steps=20, **SCHEDULE).run(EpisodeRecorder())
    stored = read(
        tmp_path / "step-000000000020.msgpack",
        state=jnp.asarray(0, jnp.int32),
        key=jax.random.key(0),
    )

    recorder = EpisodeRecorder()
    runtime(**SCHEDULE).run(recorder, resume=stored)

    # The interval before the checkpoint happened and the parent reported it,
    # so the branch runs the one after it and dates its episodes accordingly.
    # The step counter numbers every stream's every step, so the two streams
    # of the same row open one apart.
    assert [episode.start_env_steps for episode in recorder.of("train")] == [20, 21]
    assert {episode.end_env_steps for episode in recorder.of("train")} == {40, 41}


def test_a_branch_from_between_two_measurements_is_refused(tmp_path):
    runtime(tmp_path, checkpoint_every_steps=20, **SCHEDULE).run(EpisodeRecorder())
    stored = read(
        tmp_path / "step-000000000020.msgpack",
        state=jnp.asarray(0, jnp.int32),
        key=jax.random.key(0),
    )

    # The same checkpoint, read by a run whose measurements fall elsewhere: its
    # boundary is no longer one of them, so the branch would be undatable.
    with pytest.raises(ValueError, match="not an evaluation boundary"):
        runtime(**{**SCHEDULE, "evaluate_every_steps": 8}).run(
            EpisodeRecorder(), resume=stored
        )
