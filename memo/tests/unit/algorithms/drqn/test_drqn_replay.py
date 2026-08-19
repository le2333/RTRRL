"""Which window the published random update draws, and what it hands the loss.

A truncation sweep measures how far back the gradient has to reach, so the
number of transitions a window actually carries is the experiment's independent
variable. These tests hold it: a window lies inside one completed episode, it is
exactly as long as the truncation names, and which episode it came from does not
depend on how long that episode was.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms.drqn import ReplayTransition, SelectedLearning, learner_sequence
from memorax.buffers import make_episode_buffer
from memorax.rl.recurrent_replay import completed_episode_starts, episode_window_starts

EPISODE_LENGTH = 8
TRUNCATION = 2
# Two lengths, so that "uniform over episodes" and "uniform over positions" are
# different distributions and a test can tell them apart.
LENGTHS = (4, 8, 4, 8)


class _Sample:
    def __init__(self, drawn):
        self.experience = drawn


def stream(lengths):
    """Episodes laid end to end, each transition labelled with where it came from.

    The observation carries ``(episode, index within episode)`` so that a drawn
    window can be traced back to the episode it was taken from, which is what
    the sampling claims are about.
    """

    observation, episode_start, done = [], [], []
    for episode, length in enumerate(lengths):
        for index in range(length):
            observation.append([float(episode), float(index)])
            episode_start.append(index == 0)
            done.append(index == length - 1)
    return observation, episode_start, done


def transition(observation, episode_start, done):
    return ReplayTransition(
        observation=jnp.asarray([observation]),
        episode_start=jnp.asarray([episode_start]),
        action=jnp.zeros((1,), dtype=jnp.int32),
        reward=jnp.zeros((1,)),
        next_observation=jnp.asarray([observation]),
        done=jnp.asarray([done]),
        terminal=jnp.asarray([done]),
    )


def experience(observation, episode_start, done):
    """The same stream as one stored block, for calling a rule directly."""

    return ReplayTransition(
        observation=jnp.asarray([observation]),
        episode_start=jnp.asarray([episode_start]),
        action=jnp.zeros((1, len(done)), dtype=jnp.int32),
        reward=jnp.zeros((1, len(done))),
        next_observation=jnp.asarray([observation]),
        done=jnp.asarray([done]),
        terminal=jnp.asarray([done]),
    )


def stored(lengths=LENGTHS, truncation=TRUNCATION):
    buffer = make_episode_buffer(
        max_length=64,
        min_length=4,
        sample_batch_size=8,
        sample_sequence_length=truncation,
        get_start_flags=partial(
            episode_window_starts,
            truncation=truncation,
            episode_length=EPISODE_LENGTH,
        ),
        add_sequences=False,
        add_batch_size=1,
    )
    observation, episode_start, done = stream(lengths)
    state = buffer.init(
        jax.tree.map(lambda value: value[0], transition(observation[0], True, False))
    )
    for values in zip(observation, episode_start, done):
        state = buffer.add(state, transition(*values))
    return buffer, state


def drawn(buffer, state, keys=32):
    """Many drawn windows, as one stacked block."""

    return jax.tree.map(
        lambda *blocks: jnp.concatenate(blocks, axis=0),
        *[
            buffer.sample(state, jax.random.key(seed)).experience
            for seed in range(keys)
        ],
    )


# --------------------------------------------------- where a window may begin
@pytest.mark.parametrize("truncation", [1, 2, 3, 4])
def test_the_legal_starts_are_exactly_those_a_whole_window_fits_in(truncation):
    """For an episode of length L, starts 0..L-t and no others."""

    length = 4
    weights = episode_window_starts(
        experience(*stream((length,))),
        truncation=truncation,
        episode_length=EPISODE_LENGTH,
    )

    admitted = np.flatnonzero(np.asarray(weights)[0] > 0.0).tolist()
    assert admitted == list(range(length - truncation + 1))


def test_an_episode_shorter_than_the_truncation_admits_nothing():
    """Not a shortened window: no window at all, which is the honest answer."""

    weights = episode_window_starts(
        experience(*stream((3,))), truncation=4, episode_length=EPISODE_LENGTH
    )

    assert not np.any(np.asarray(weights) > 0.0)


def test_an_episode_still_being_played_is_not_drawn_from():
    """A completed episode is one whose ending is in the buffer."""

    observation, episode_start, _ = stream((4,))
    unfinished = experience(observation, episode_start, [False] * 4)

    weights = episode_window_starts(
        unfinished, truncation=2, episode_length=EPISODE_LENGTH
    )

    assert not np.any(np.asarray(weights) > 0.0)


def test_a_window_longer_than_the_horizon_is_refused():
    with pytest.raises(ValueError, match="cannot fit inside an episode"):
        episode_window_starts(
            experience(*stream((4,))), truncation=9, episode_length=EPISODE_LENGTH
        )


# ------------------------------------------------------- what the buffer draws
def test_every_drawn_window_lies_inside_one_episode():
    """No window crosses an ending, so none is cut short by the validity mask."""

    buffer, state = stored()

    windows = drawn(buffer, state)

    episodes = np.asarray(windows.observation)[..., 0]
    indices = np.asarray(windows.observation)[..., 1]
    assert np.all(episodes[:, 0][:, None] == episodes)
    assert np.all(np.diff(indices, axis=1) == 1.0)
    # An ending may fall on the window's last transition and nowhere earlier.
    assert not np.any(np.asarray(windows.done)[:, :-1])


def test_every_drawn_window_carries_the_full_truncation():
    """The nominal t and the number of transitions the gradient crosses agree.

    This is what the sweep's independent variable has to be. Cutting a window
    at an ending would make the effective truncation a function of where in an
    episode the window landed, and t=64 would in places be t=5.
    """

    buffer, state = stored()

    windows = drawn(buffer, state)
    sequence = learner_sequence(_Sample(windows), transition_count=TRUNCATION)

    assert sequence.valid.shape == (windows.done.shape[0], TRUNCATION)
    assert np.all(np.asarray(sequence.valid))


def test_an_episode_is_not_drawn_more_often_for_being_longer():
    """Uniform over episodes, which is what the published update draws.

    Uniform over stored positions would weight an episode by how many starts it
    offers -- here seven against three -- and on a task whose episodes grow as
    the policy improves that is a drift in what gets replayed rather than a
    fixed sampling rule.
    """

    buffer, state = stored()

    windows = drawn(buffer, state)

    episodes = np.asarray(windows.observation)[:, 0, 0].astype(int)
    long_episodes = np.isin(episodes, [1, 3]).mean()
    # Two long and two short, so half. Weighting by starts would give
    # 14/20 = 0.7, which is far outside this band.
    assert 0.4 < long_episodes < 0.6, long_episodes


def test_uniform_replay_keeps_no_priority_to_update():
    buffer, state = stored()

    assert not hasattr(state, "priorities")
    assert not hasattr(buffer, "set_priorities")
    assert not hasattr(buffer.sample(state, jax.random.key(1)), "probabilities")


def test_each_branch_reads_the_window_it_names():
    truncated = SelectedLearning("truncated", 3)
    full = SelectedLearning("full_bptt", 0)

    assert truncated.window(EPISODE_LENGTH) == 3
    assert full.window(EPISODE_LENGTH) == EPISODE_LENGTH

    inside = truncated.start_flags(EPISODE_LENGTH)
    assert inside.func is episode_window_starts
    assert inside.keywords == {"truncation": 3, "episode_length": EPISODE_LENGTH}
    # A full episode offers exactly one start, so drawing uniformly over starts
    # already draws uniformly over episodes and needs no weighting.
    assert full.start_flags(EPISODE_LENGTH).func is completed_episode_starts
    assert full.start_flags(EPISODE_LENGTH).keywords == {
        "transition_count": EPISODE_LENGTH
    }


# ------------------------------------------------------ what the window hands on
def test_the_window_is_run_over_one_more_input_than_it_has_transitions():
    drawn_window = ReplayTransition(
        observation=jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2),
        episode_start=jnp.asarray([[True, False, False]]),
        action=jnp.asarray([[0, 1, 0]]),
        reward=jnp.asarray([[1.0, 2.0, 3.0]]),
        next_observation=jnp.arange(6, 12, dtype=jnp.float32).reshape(1, 3, 2),
        done=jnp.asarray([[False, False, False]]),
        terminal=jnp.zeros((1, 3), dtype=jnp.bool_),
    )

    sequence = learner_sequence(_Sample(drawn_window), transition_count=3)

    assert sequence.inputs.observation.shape == (1, 4, 2)
    assert sequence.actions.shape == (1, 3)
    # The extra input is the last transition's own successor, so that the last
    # target bootstraps from a state the unroll reached.
    np.testing.assert_array_equal(
        np.asarray(sequence.inputs.observation[:, -1]),
        np.asarray(drawn_window.next_observation[:, -1]),
    )
    np.testing.assert_array_equal(
        np.asarray(sequence.bootstrap_inputs.observation),
        np.asarray(drawn_window.next_observation),
    )
    assert not bool(jnp.any(sequence.inputs.episode_start[:, -1]))


def test_validity_stops_at_the_first_ending_and_keeps_the_ending_itself():
    """Still the rule for the full-episode branch, whose window is padded."""

    drawn_window = ReplayTransition(
        observation=jnp.zeros((1, 4, 2)),
        episode_start=jnp.asarray([[True, False, False, True]]),
        action=jnp.zeros((1, 4), dtype=jnp.int32),
        reward=jnp.zeros((1, 4)),
        next_observation=jnp.zeros((1, 4, 2)),
        done=jnp.asarray([[False, True, False, False]]),
        terminal=jnp.asarray([[False, True, False, False]]),
    )

    sequence = learner_sequence(_Sample(drawn_window), transition_count=4)

    np.testing.assert_array_equal(
        np.asarray(sequence.valid), np.asarray([[True, True, False, False]])
    )
