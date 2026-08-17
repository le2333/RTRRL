"""What a checkpoint is scored on, and what it costs the run to measure it.

The protocol scores a checkpoint on an exact number of complete episodes. No
number of steps contains that, so Runtime keeps advancing an opened rollout
until the episodes it named have ended -- and the arithmetic here is arranged
so that the streams disagree about when they end, because a rule that only
holds when they agree is not the rule.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import pytest

from memorax.runtime import (
    BuiltAlgorithm,
    ObservationSchema,
    Program,
    Runtime,
    RuntimeConfig,
)
from memorax.runtime.driver import evaluation_quota
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
    """Training that ends nothing, so only evaluation reports."""

    del key
    rows = num_steps // NUM_ENVS
    quiet = jnp.zeros((rows, NUM_ENVS), dtype=bool)
    return state + num_steps, Metrics(
        interaction=Interaction(
            reward=jnp.zeros((rows, NUM_ENVS)), done=quiet, terminal=quiet
        )
    )


def ending_every(period: tuple[int, int]):
    """An evaluation whose streams end an episode at different rates.

    ``state`` counts the rows this rollout has already produced, so an episode
    that the previous call cut in half is finished by the next one rather than
    restarted -- which is the whole reason the rollout is carried rather than
    reopened.
    """

    def evaluate(key, state, num_steps):
        del key
        rows = num_steps // NUM_ENVS
        rank = state + jnp.arange(1, rows + 1, dtype=jnp.int32)[:, None]
        done = jnp.stack(
            [rank[:, 0] % period[stream] == 0 for stream in range(NUM_ENVS)], axis=1
        )
        return state + rows, Metrics(
            interaction=Interaction(
                reward=jnp.ones((rows, NUM_ENVS)), done=done, terminal=done
            )
        )

    return evaluate


def measure(
    *,
    evaluate,
    episodes,
    chunk_steps,
    max_episode_steps=8,
    seed=0,
    learner=train,
    total_steps=4,
):
    recorder = EpisodeRecorder()
    Runtime(
        algorithm=BuiltAlgorithm(
            program=Program(
                init=lambda key: jnp.asarray(0, jnp.int32),
                train=learner,
                open_evaluation=lambda key, state: jnp.asarray(0, jnp.int32),
                evaluate=evaluate,
                interact=lambda key, state: None,
            ),
            observations=OBSERVATIONS,
        ),
        config=RuntimeConfig(
            total_steps=total_steps,
            chunk_steps=4,
            max_episode_steps=max_episode_steps,
            evaluate_every_steps=4,
            evaluation_episodes=episodes,
            evaluation_chunk_steps=chunk_steps,
            evaluation_seed=seed,
            num_envs=NUM_ENVS,
            seed=0,
        ),
    ).run(recorder)
    return recorder


@pytest.mark.parametrize(
    "episodes, num_envs, expected",
    [
        (10, 1, (10,)),
        (10, 2, (5, 5)),
        (10, 4, (3, 3, 2, 2)),
        (5, 4, (2, 1, 1, 1)),
        (5, 8, (1, 1, 1, 1, 1, 0, 0, 0)),
    ],
)
def test_the_quota_names_the_slots_before_the_rollout_runs(
    episodes, num_envs, expected
):
    """Which episodes are scored is decided by index, never by finishing time.

    Nothing here requires the count to divide the streams: the lower indices
    simply carry one more, and the total is the number that was asked for.
    """

    quota = evaluation_quota(episodes=episodes, num_envs=num_envs)

    assert quota == expected
    assert sum(quota) == episodes


def test_a_checkpoint_reports_exactly_the_episodes_it_asked_for():
    """One stream ends every row, the other every fourth. Neither decides."""

    recorder = measure(
        evaluate=ending_every((1, 4)), episodes=4, chunk_steps=2 * NUM_ENVS
    )
    scored = recorder.of("eval")

    assert len(scored) == 4
    # Slot j * n + i is stream i's j-th episode, so the four asked for are two
    # from each stream -- not the four the fast stream produced first.
    assert [episode.stream for episode in scored] == [0, 1, 0, 1]
    assert [episode.number for episode in scored] == [1, 2, 3, 4]
    # A call holds two rows and the slow stream's episodes run to four, so the
    # rollout was advanced rather than run once, and reopening it between
    # calls would have restarted those episodes instead of finishing them.
    assert [len(episode.rewards) for episode in scored] == [1, 4, 1, 4]


def test_the_episodes_a_stream_ran_past_its_quota_are_not_scored():
    """An extra episode changes the count as much as a missing one does."""

    recorder = measure(
        evaluate=ending_every((1, 4)), episodes=2, chunk_steps=4 * NUM_ENVS
    )
    scored = recorder.of("eval")

    # The fast stream ended four episodes inside the first call. One slot was
    # its, so three of them are dropped.
    assert len(scored) == 2
    assert [episode.stream for episode in scored] == [0, 1]


def test_an_episode_that_crosses_a_call_is_one_episode():
    """The rollout is carried between calls, so a call boundary ends nothing."""

    together = measure(
        evaluate=ending_every((4, 4)), episodes=2, chunk_steps=8 * NUM_ENVS
    )
    apart = measure(evaluate=ending_every((4, 4)), episodes=2, chunk_steps=1 * NUM_ENVS)

    assert [len(episode.rewards) for episode in together.of("eval")] == [4, 4]
    assert [len(episode.rewards) for episode in apart.of("eval")] == [4, 4]


def test_a_rollout_whose_episodes_never_end_says_so():
    """A measurement that cannot finish is worse than one that fails."""

    def never(key, state, num_steps):
        del key
        rows = num_steps // NUM_ENVS
        quiet = jnp.zeros((rows, NUM_ENVS), dtype=bool)
        return state + rows, Metrics(
            interaction=Interaction(
                reward=jnp.zeros((rows, NUM_ENVS)), done=quiet, terminal=quiet
            )
        )

    with pytest.raises(ValueError, match="did not complete 1 episodes"):
        measure(
            evaluate=never,
            episodes=1,
            chunk_steps=2 * NUM_ENVS,
            max_episode_steps=4,
        )


def keyed_train(key, state, num_steps):
    """Training whose reported transitions are a function of the key it got.

    Two runs that received the same keys report the same numbers, so comparing
    what training reported is comparing the key stream it ran on.
    """

    rows = num_steps // NUM_ENVS
    ending = jnp.ones((rows, NUM_ENVS), dtype=bool)
    return state + num_steps, Metrics(
        interaction=Interaction(
            reward=jax.random.uniform(key, (rows, NUM_ENVS)),
            done=ending,
            terminal=ending,
        )
    )


def test_measuring_the_policy_does_not_move_the_training_stream():
    """Whether a run was watched cannot change what it then learned.

    Training keys come from ``seed`` and evaluation keys from
    ``evaluation_seed``. Were evaluation to split from the training chain
    instead -- as it did while there was only one -- everything after the
    first checkpoint would be different transitions, and a run measured every
    ten thousand steps would not be the run measured every hundred thousand.
    """

    def trained(episodes, seed):
        recorder = measure(
            evaluate=ending_every((1, 1)),
            episodes=episodes,
            chunk_steps=2 * NUM_ENVS,
            seed=seed,
            learner=keyed_train,
            total_steps=8,
        )
        return [episode.rewards for episode in recorder.of("train")]

    unmeasured = trained(0, 0)

    # Two train calls of two rows, each row ending both streams.
    assert len(unmeasured) == 8
    assert trained(2, 0) == unmeasured
    assert trained(2, 99) == unmeasured
