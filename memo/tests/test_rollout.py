"""Cutting a stacked rollout into episodes, which is nobody's private business.

Every trainer that runs a fixed number of steps across a fixed number of streams
produces the same shape and needs the same cut. Written once here, a second
trainer inherits the boundary rule rather than inventing one beside it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from memorax.runtime import ObservationSchema
from memorax.runtime.rollout import complete_episodes, read


class Chunk:
    def __init__(self, **fields: object) -> None:
        self.__dict__.update(fields)


def chunk(done, **extra) -> Chunk:
    done = np.asarray(done, dtype=bool)
    steps, envs = done.shape
    grid = np.arange(steps * envs, dtype=float).reshape(steps, envs)
    return Chunk(
        observation=grid[..., None],
        next_observation=grid[..., None] + 100,
        action=np.zeros((steps, envs, 1)),
        reward=np.ones((steps, envs)),
        done=done,
        **extra,
    )


OBSERVATIONS = ObservationSchema(
    reward="reward",
    done="done",
    terminal="terminal",
    observation="observation",
    next_observation="next_observation",
    action="action",
)


def cut(summary, *, observations=OBSERVATIONS, **overrides):
    settings = {"phase": "eval", "start_env_steps": 0, "num_envs": 2}
    return list(
        complete_episodes(summary, observations=observations, **(settings | overrides))
    )


def test_a_partial_episode_at_either_end_is_left_out():
    #                   env 0: done at 1 and 4     env 1: never done
    episodes = cut(chunk([[0, 0], [1, 0], [0, 0], [0, 0], [1, 0]]))

    # Steps 2 and 3 of env 0 run past the end without terminating, and env 1
    # never terminates at all; neither is a return anyone can report.
    assert [len(episode.actions) for episode in episodes] == [2, 3]
    assert [episode.number for episode in episodes] == [1, 2]
    assert episodes[0].observations[-1] == [102.0]


def test_each_stream_is_cut_on_its_own_boundaries():
    episodes = cut(chunk([[0, 0], [1, 0], [0, 1], [0, 0]]))

    assert [len(episode.rewards) for episode in episodes] == [2, 3]
    assert [episode.end_env_steps for episode in episodes] == [4, 6]


def test_a_declared_series_is_carried_alongside_the_rewards():
    steps = np.arange(8, dtype=float).reshape(4, 2)
    episodes = cut(
        chunk(
            [[0, 0], [1, 0], [0, 1], [0, 0]], td_error=steps, by_part={"torso": steps}
        ),
        observations=replace(
            OBSERVATIONS,
            series=("td_error", "by_part.torso", "never_produced"),
        ),
    )

    assert set(episodes[0].series) == {"td_error", "by_part.torso"}
    assert episodes[0].series["td_error"] == (0.0, 2.0)


def test_a_quantity_the_kernel_reduced_over_the_batch_belongs_to_every_stream():
    """One number per step rather than one per stream is still a series.

    A kernel that averaged the streams before handing a scalar back has said the
    same thing about each of them, which is not a reason to drop it.
    """

    episodes = cut(
        chunk([[0, 0], [1, 0], [0, 1], [0, 0]], entropy=np.arange(4, dtype=float)),
        observations=replace(OBSERVATIONS, series=("entropy",)),
    )

    assert episodes[0].series["entropy"] == (0.0, 1.0)
    assert episodes[1].series["entropy"] == (0.0, 1.0, 2.0)


def test_the_environments_own_reward_can_live_somewhere_else():
    """An entry that normalised in a wrapper knows where the original went."""

    paid = np.full((4, 2), 7.0)
    episodes = cut(
        chunk([[0, 0], [1, 0], [0, 0], [0, 0]], info={"environment_reward": paid}),
        observations=replace(OBSERVATIONS, reward="info.environment_reward"),
    )

    assert episodes[0].rewards == (7.0, 7.0)


def test_a_stride_of_nothing_dates_every_episode_at_one_step():
    """Evaluation does not advance the axis it is plotted against."""

    episodes = cut(
        chunk([[0, 0], [1, 0], [0, 1], [0, 0]]), start_env_steps=64, stride=0
    )

    for episode in episodes:
        assert (episode.start_env_steps, episode.end_env_steps) == (64, 64)


def test_a_chunk_without_a_trajectory_still_cuts():
    """Recording every observation of training costs what it is worth."""

    steps = np.arange(8, dtype=float).reshape(4, 2)
    summary = Chunk(reward=np.ones((4, 2)), done=np.asarray([[0, 0], [1, 0]] * 2, bool))
    del steps
    episodes = cut(summary)

    assert episodes[0].observations is None
    assert episodes[0].actions is None
    assert episodes[0].rewards == (1.0, 1.0)


def test_an_episode_says_which_of_the_two_endings_it_reached():
    """A run cut off at its step limit did not fail, and the record says so.

    Both endings end an episode, so both cut here; only one of them says the
    future was worth nothing, and writing every ending down as a termination
    loses the difference before anyone can read it.
    """

    episodes = cut(
        chunk(
            [[0, 0], [1, 0], [0, 1], [0, 0]],
            terminal=[[0, 0], [1, 0], [0, 0], [0, 0]],
        ),
        observations=replace(OBSERVATIONS, terminal="terminal"),
    )

    assert episodes[0].terminals[-1] and not episodes[0].truncations[-1]
    assert episodes[1].truncations[-1] and not episodes[1].terminals[-1]


def test_a_kernel_that_tells_nobody_says_every_ending_was_a_termination():
    """Which is what it knew. The record reflects the kernel, not a guess."""

    episodes = cut(chunk([[0, 0], [1, 0], [0, 1], [0, 0]]))

    assert all(episode.terminals[-1] for episode in episodes)
    assert not any(episode.truncations[-1] for episode in episodes)


def test_every_episode_says_which_stream_it_came_from():
    """An episode belongs to one stream, so it has to be able to say which.

    Without it a sink cannot tell two streams' episodes apart, and a sample
    step -- which names one stream -- has no way to pick out its own.
    """

    episodes = cut(chunk([[0, 0], [1, 0], [0, 1], [0, 0]]))

    assert [episode.stream for episode in episodes] == [0, 1]


def test_numbering_continues_where_the_last_chunk_stopped():
    episodes = cut(chunk([[0, 0], [1, 0], [0, 1], [0, 0]]), first_number=7)

    assert [episode.number for episode in episodes] == [7, 8]


@pytest.mark.parametrize(
    "path, found",
    [
        ("reward", True),
        ("info.environment_reward", True),
        ("info.absent", False),
        ("absent.x", False),
    ],
)
def test_a_family_is_read_one_member_at_a_time(path, found):
    summary = chunk([[0, 0], [1, 0]], info={"environment_reward": np.zeros((2, 2))})
    assert (read(summary, path) is not None) is found
