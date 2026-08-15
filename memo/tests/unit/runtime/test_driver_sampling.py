"""What Runtime asks an algorithm for, and which episodes it keeps whole.

The programs here are arithmetic rather than algorithms, so what is under test
is the scheduling: how large a train call may be, which episode a sample point
selects when episodes cross those calls, and how the last requested episode
reaches its end after the budget is spent.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.runtime import (
    BuiltAlgorithm,
    ObservationSchema,
    Program,
    Runtime,
    RuntimeConfig,
)
from tests.support.fakes import EpisodeRecorder

NUM_ENVS = 2

OBSERVATIONS = ObservationSchema(
    reward="interaction.reward",
    done="interaction.done",
    terminal="interaction.terminal",
    series=("td_error",),
)


class Interaction(NamedTuple):
    reward: Any
    done: Any
    terminal: Any


class Metrics(NamedTuple):
    interaction: Interaction
    td_error: Any = None


def run_runtime(
    *,
    train,
    interact=None,
    recorder=None,
    sample_steps=(),
    eval_steps=0,
    initial_state=jnp.asarray(0, jnp.int32),
    **schedule,
) -> EpisodeRecorder:
    recorder = EpisodeRecorder() if recorder is None else recorder
    Runtime(
        algorithm=BuiltAlgorithm(
            program=Program(
                init=lambda key: initial_state,
                train=train,
                evaluate=lambda key, state, num_steps: None,
                interact=interact or (lambda key, state: None),
            ),
            observations=OBSERVATIONS,
        ),
        config=RuntimeConfig(
            eval_steps=eval_steps,
            num_envs=NUM_ENVS,
            seed=0,
            sample_steps=sample_steps,
            **schedule,
        ),
    ).run(recorder)
    return recorder


# ------------------------------------------------- what a train call may cost
def test_a_long_epoch_is_executed_as_bounded_train_calls():
    """A reporting interval is a schedule, not one allocation."""

    requested: list[int] = []

    def train(key, state, num_steps):
        del key
        assert num_steps <= 8
        jax.debug.callback(lambda: requested.append(num_steps))
        rows = num_steps // NUM_ENVS
        ending = jnp.ones((rows, NUM_ENVS), dtype=bool)
        return state + num_steps, Metrics(
            interaction=Interaction(
                reward=jnp.zeros((rows, NUM_ENVS)), done=ending, terminal=ending
            ),
            td_error=jnp.zeros((rows, NUM_ENVS)),
        )

    run_runtime(
        train=train,
        total_steps=80,
        epoch_steps=40,
        max_episode_steps=4,
    )

    assert requested == [8] * 10


def test_a_short_epoch_is_never_asked_for_more_than_it_has():
    requested: list[int] = []

    def train(key, state, num_steps):
        del key
        jax.debug.callback(lambda: requested.append(num_steps))
        rows = num_steps // NUM_ENVS
        ending = jnp.ones((rows, NUM_ENVS), dtype=bool)
        return state + num_steps, Metrics(
            interaction=Interaction(
                reward=jnp.zeros((rows, NUM_ENVS)), done=ending, terminal=ending
            ),
            td_error=jnp.zeros((rows, NUM_ENVS)),
        )

    run_runtime(
        train=train,
        total_steps=20,
        epoch_steps=10,
        max_episode_steps=4,
    )

    # The bound is eight, and an epoch of ten is two calls rather than one.
    assert requested == [8, 2, 8, 2]


# --------------------------------------------- which episode a sample selects
# Twelve rows of two streams. Stream zero ends at rows 1, 4, 6, 9 and 11, so an
# episode spans a chunk boundary and a sample lands where one has just ended.
DONE = jnp.asarray(
    [
        [False, True],
        [True, False],
        [False, True],
        [False, False],
        [True, True],
        [False, False],
        [True, True],
        [False, False],
        [False, True],
        [True, False],
        [False, True],
        [True, False],
    ]
)
REWARD = jnp.asarray(
    [[row * 10 + stream for stream in range(NUM_ENVS)] for row in range(12)],
    dtype=jnp.float32,
)
TD_ERROR = REWARD / 100


def sliced_train(key, state, num_steps):
    """Hand back this call's rows of one fixed table."""

    del key
    rows = num_steps // NUM_ENVS
    start = (state // NUM_ENVS).astype(jnp.int32)

    def cut(table):
        return jax.lax.dynamic_slice_in_dim(table, start, rows, axis=0)

    return state + num_steps, Metrics(
        interaction=Interaction(reward=cut(REWARD), done=cut(DONE), terminal=cut(DONE)),
        td_error=cut(TD_ERROR),
    )


def test_a_sampled_episode_is_whole_across_the_train_calls_it_crosses():
    recorder = run_runtime(
        train=sliced_train,
        total_steps=24,
        epoch_steps=12,
        max_episode_steps=3,
        sample_steps=(6, 10),
    )

    assert [one.sample_step for one in recorder.trajectories] == [6, 10]
    first, second = recorder.trajectories

    # Sample 6 is stream zero's step at row three; its episode opened at row two
    # in the previous train call and ends at row four in the next one.
    assert first.episode.stream == 0
    assert first.episode.rewards == (20.0, 30.0, 40.0)
    assert (first.episode.start_env_steps, first.episode.end_env_steps) == (4, 10)
    assert first.episode.terminals[-1]
    assert first.post_budget == (False, False, False)

    # Sample 10 falls where stream zero has just ended, so it selects the next
    # episode rather than the one that closed at the boundary.
    assert second.episode.rewards == (50.0, 60.0)
    assert (second.episode.start_env_steps, second.episode.end_env_steps) == (10, 14)
    assert second.post_budget == (False, False)


def test_a_chunk_boundary_ends_no_episode_that_the_environment_did_not_end():
    recorder = run_runtime(
        train=sliced_train,
        total_steps=24,
        epoch_steps=12,
        max_episode_steps=3,
    )

    episodes = recorder.of("train")
    # Five endings for stream zero and six for stream one; the two episodes
    # still open at the budget are not episodes yet.
    assert len(episodes) == 11
    assert [episode.number for episode in episodes] == list(range(1, 12))
    for episode in episodes:
        assert episode.terminals[-1] or episode.truncations[-1]

    spanning = next(episode for episode in episodes if episode.start_env_steps == 4)
    assert spanning.rewards == (20.0, 30.0, 40.0)


# ------------------------------------------ finishing the last asked-for walk
class Counters(NamedTuple):
    """Two counters, so an accidental update is visible rather than inferred."""

    updates: Any
    interactions: Any


FINAL_DONE = jnp.asarray(
    [
        [False, True],
        [True, False],
        [False, True],
        [False, False],
    ]
)
FINAL_REWARD = jnp.asarray(
    [[row * 10 + stream for stream in range(NUM_ENVS)] for row in range(4)],
    dtype=jnp.float32,
)


def final_sample_program():
    """A four-row budget whose stream-zero episode is unfinished when it ends."""

    trained: list[int] = []
    interactions: list[tuple[int, int]] = []

    def train(key, state, num_steps):
        del key
        rows = num_steps // NUM_ENVS
        updates = state.updates + rows
        jax.debug.callback(
            lambda steps, count: trained.append((int(steps), int(count))),
            num_steps,
            updates,
        )
        return Counters(updates, state.interactions), Metrics(
            interaction=Interaction(
                reward=FINAL_REWARD, done=FINAL_DONE, terminal=FINAL_DONE
            ),
            td_error=FINAL_REWARD / 100,
        )

    def interact(key, state):
        del key
        jax.debug.callback(
            lambda updates, count: interactions.append((int(updates), int(count))),
            state.updates,
            state.interactions,
        )
        ending = jnp.full((NUM_ENVS,), False) | (state.interactions >= 1)
        return Counters(state.updates, state.interactions + 1), Metrics(
            interaction=Interaction(
                reward=jnp.full((NUM_ENVS,), 100.0) + state.interactions,
                done=ending,
                terminal=ending,
            ),
        )

    return train, interact, trained, interactions


def test_the_final_sample_is_finished_without_another_update():
    train, interact, trained, interactions = final_sample_program()

    recorder = run_runtime(
        train=train,
        interact=interact,
        initial_state=Counters(jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32)),
        total_steps=8,
        epoch_steps=8,
        max_episode_steps=4,
        sample_steps=(8,),
    )

    # One train call for the whole budget, and nothing asked for after it.
    assert trained == [(8, 4)]
    # Both continuations saw the trained parameter count and neither raised it.
    assert interactions == [(4, 0), (4, 1)]

    sampled = recorder.trajectories[-1]
    assert len(recorder.trajectories) == 1
    assert sampled.sample_step == 8
    assert sampled.post_budget == (False, False, True, True)
    assert sampled.episode.terminals[-1]
    assert sampled.episode.rewards == (20.0, 30.0, 100.0, 101.0)


def test_a_post_budget_transition_reports_no_update_reading():
    train, interact, _, _ = final_sample_program()

    recorder = run_runtime(
        train=train,
        interact=interact,
        initial_state=Counters(jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32)),
        total_steps=8,
        epoch_steps=8,
        max_episode_steps=4,
        sample_steps=(8,),
    )

    measured = recorder.trajectories[0].episode.series["td_error"]
    assert measured[:2] == pytest.approx((0.2, 0.3))
    assert all(math.isnan(value) for value in measured[2:])


def test_no_continuation_runs_when_nothing_was_sampled():
    train, interact, _, interactions = final_sample_program()

    recorder = run_runtime(
        train=train,
        interact=interact,
        initial_state=Counters(jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32)),
        total_steps=8,
        epoch_steps=8,
        max_episode_steps=4,
    )

    assert interactions == []
    assert recorder.trajectories == []


def test_a_sample_that_cannot_end_within_the_declared_limit_is_an_error():
    def train(key, state, num_steps):
        del key, num_steps
        never = jnp.zeros((2, NUM_ENVS), dtype=bool)
        return state + 4, Metrics(
            interaction=Interaction(
                reward=jnp.zeros((2, NUM_ENVS)), done=never, terminal=never
            ),
            td_error=jnp.zeros((2, NUM_ENVS)),
        )

    def interact(key, state):
        del key
        never = jnp.zeros((NUM_ENVS,), dtype=bool)
        return state, Metrics(
            interaction=Interaction(
                reward=jnp.zeros((NUM_ENVS,)), done=never, terminal=never
            ),
        )

    with pytest.raises(ValueError, match="maximum episode length"):
        run_runtime(
            train=train,
            interact=interact,
            total_steps=4,
            epoch_steps=4,
            max_episode_steps=3,
            sample_steps=(4,),
        )


def test_evaluation_still_reports_at_the_epoch_boundary_it_measured():
    ending = jnp.asarray([[False, False], [True, True]])

    def train(key, state, num_steps):
        del key
        rows = num_steps // NUM_ENVS
        done = jnp.ones((rows, NUM_ENVS), dtype=bool)
        return state + num_steps, Metrics(
            interaction=Interaction(
                reward=jnp.zeros((rows, NUM_ENVS)), done=done, terminal=done
            ),
            td_error=jnp.zeros((rows, NUM_ENVS)),
        )

    def evaluate(key, state, num_steps):
        del key, state, num_steps
        return Metrics(
            interaction=Interaction(
                reward=jnp.ones((2, NUM_ENVS)), done=ending, terminal=ending
            )
        )

    recorder = EpisodeRecorder()
    Runtime(
        algorithm=BuiltAlgorithm(
            program=Program(
                init=lambda key: jnp.asarray(0, jnp.int32),
                train=train,
                evaluate=evaluate,
                interact=lambda key, state: None,
            ),
            observations=OBSERVATIONS,
        ),
        config=RuntimeConfig(
            total_steps=8,
            epoch_steps=4,
            eval_steps=4,
            num_envs=NUM_ENVS,
            seed=0,
            max_episode_steps=2,
        ),
    ).run(recorder)

    # An evaluation measures the policy as it stood at the boundary, so both of
    # its episodes are dated there rather than spread forward.
    assert [episode.end_env_steps for episode in recorder.of("eval")] == [4, 4, 8, 8]
    assert np.allclose([episode.rewards for episode in recorder.of("eval")], 1.0)
