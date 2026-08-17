"""Which stored positions a window may begin at, and what it hands the loss."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from memorax.algorithms.drqn import (
    ReplayTransition,
    SelectedLearning,
    any_position_starts,
    learner_sequence,
)
from memorax.buffers import make_episode_buffer
from memorax.rl.recurrent_replay import completed_episode_starts

EPISODE_LENGTH = 4


def experience(dones, episode_starts):
    """One stored stream, with only the fields a start rule reads filled in."""

    return ReplayTransition(
        observation=jnp.zeros((1, len(dones), 2)),
        episode_start=jnp.asarray([episode_starts]),
        action=jnp.zeros((1, len(dones)), dtype=jnp.int32),
        reward=jnp.zeros((1, len(dones))),
        next_observation=jnp.zeros((1, len(dones), 2)),
        done=jnp.asarray([dones]),
        terminal=jnp.asarray([dones]),
    )


def stored(transitions, rule=any_position_starts):
    """A buffer holding ``transitions`` steps of four-step episodes.

    One position in four begins an episode, so a rule that admits only episode
    starts and one that admits every position draw visibly different windows.
    """

    buffer = make_episode_buffer(
        max_length=64,
        min_length=4,
        sample_batch_size=8,
        sample_sequence_length=2,
        get_start_flags=rule,
        add_sequences=False,
        add_batch_size=1,
    )
    state = buffer.init(
        jax.tree.map(lambda value: value[0, 0], experience([False], [True]))
    )
    for index in range(transitions):
        ended = index % EPISODE_LENGTH == EPISODE_LENGTH - 1
        state = buffer.add(
            state,
            jax.tree.map(
                lambda value: value[:, 0],
                experience([ended], [index % EPISODE_LENGTH == 0]),
            ),
        )
    return buffer, state


def drawn_starts(buffer, state, keys=16):
    """Whether each drawn window began on an episode start, over many draws."""

    return np.concatenate(
        [
            np.asarray(
                buffer.sample(state, jax.random.key(seed)).experience.episode_start[
                    :, 0
                ]
            )
            for seed in range(keys)
        ]
    )


def test_a_random_update_point_is_every_stored_position():
    """The paper draws a point inside an episode, not the episode's beginning."""

    drawn = experience([False, False, True, False], [True, False, False, True])

    flags = any_position_starts(drawn)

    assert flags.shape == drawn.done.shape
    assert flags.dtype == jnp.bool_
    assert bool(jnp.all(flags))
    # The episode-start rule is the other branch's, and it is stricter.
    assert not bool(
        jnp.all(completed_episode_starts(drawn, transition_count=EPISODE_LENGTH))
    )


def test_the_buffer_draws_from_inside_episodes_under_this_rule_and_not_the_other():
    """Read off the sampler, because a rule is only what the buffer applies.

    The comparison is the evidence: the same stream sampled under the
    full-episode rule begins every window at an episode start, and under this
    one it does not, so the difference is the rule rather than the draw.
    """

    inside = stored(24)
    at_starts = stored(
        24, rule=partial(completed_episode_starts, transition_count=EPISODE_LENGTH)
    )

    assert inside[0].sample(inside[1], jax.random.key(0)).experience.done.shape == (
        8,
        2,
    )
    assert bool(np.all(drawn_starts(*at_starts)))
    assert not bool(np.all(drawn_starts(*inside)))


def test_uniform_replay_keeps_no_priority_to_update():
    buffer, state = stored(8)

    assert not hasattr(state, "priorities")
    assert not hasattr(buffer, "set_priorities")
    assert not hasattr(buffer.sample(state, jax.random.key(1)), "probabilities")


def test_each_branch_reads_the_window_it_names():
    truncated = SelectedLearning("truncated", 3)
    full = SelectedLearning("full_bptt", 0)

    assert truncated.window(EPISODE_LENGTH) == 3
    assert full.window(EPISODE_LENGTH) == EPISODE_LENGTH
    assert truncated.start_flags(EPISODE_LENGTH) is any_position_starts
    # Bound to the configured horizon, since a full episode has to fit.
    assert full.start_flags(EPISODE_LENGTH).func is completed_episode_starts
    assert full.start_flags(EPISODE_LENGTH).keywords == {
        "transition_count": EPISODE_LENGTH
    }


class _Sample:
    def __init__(self, drawn):
        self.experience = drawn


def test_the_window_is_run_over_one_more_input_than_it_has_transitions():
    drawn = ReplayTransition(
        observation=jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2),
        episode_start=jnp.asarray([[True, False, False]]),
        action=jnp.asarray([[0, 1, 0]]),
        reward=jnp.asarray([[1.0, 2.0, 3.0]]),
        next_observation=jnp.arange(6, 12, dtype=jnp.float32).reshape(1, 3, 2),
        done=jnp.asarray([[False, False, False]]),
        terminal=jnp.zeros((1, 3), dtype=jnp.bool_),
    )

    sequence = learner_sequence(_Sample(drawn), transition_count=3)

    assert sequence.inputs.observation.shape == (1, 4, 2)
    assert sequence.actions.shape == (1, 3)
    # The extra input is the last transition's own successor, so that the last
    # target bootstraps from a state the unroll reached.
    np.testing.assert_array_equal(
        np.asarray(sequence.inputs.observation[:, -1]),
        np.asarray(drawn.next_observation[:, -1]),
    )
    np.testing.assert_array_equal(
        np.asarray(sequence.bootstrap_inputs.observation),
        np.asarray(drawn.next_observation),
    )
    assert not bool(jnp.any(sequence.inputs.episode_start[:, -1]))


def test_validity_stops_at_the_first_ending_and_keeps_the_ending_itself():
    drawn = ReplayTransition(
        observation=jnp.zeros((1, 4, 2)),
        episode_start=jnp.asarray([[True, False, False, True]]),
        action=jnp.zeros((1, 4), dtype=jnp.int32),
        reward=jnp.zeros((1, 4)),
        next_observation=jnp.zeros((1, 4, 2)),
        done=jnp.asarray([[False, True, False, False]]),
        terminal=jnp.asarray([[False, True, False, False]]),
    )

    sequence = learner_sequence(_Sample(drawn), transition_count=4)

    # The transition that ends the episode is scored; what follows it belongs
    # to another episode the window has no business learning from.
    np.testing.assert_array_equal(
        np.asarray(sequence.valid), np.asarray([[True, True, False, False]])
    )
