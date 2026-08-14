from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from memorax.algorithms.r2d2 import (
    ReplayTransition,
    completed_episode_starts,
    learner_sequence,
    tbptt_starts,
)
from memorax.buffers.prioritised_episode_buffer import (
    PrioritisedEpisodeBufferSample,
    make_prioritised_episode_buffer,
)


def _transition(*, done, terminal=None, episode_start=None):
    transition_count = len(done)
    if terminal is None:
        terminal = done
    if episode_start is None:
        episode_start = [True] + [False] * (transition_count - 1)
    observation = jnp.arange(transition_count * 2).reshape(1, transition_count, 2)
    return ReplayTransition(
        observation=observation,
        previous_action=jnp.arange(10, 10 + transition_count)[None, :],
        previous_reward=jnp.arange(20, 20 + transition_count)[None, :],
        episode_start=jnp.asarray([episode_start]),
        action=jnp.arange(30, 30 + transition_count)[None, :],
        reward=jnp.arange(40, 40 + transition_count)[None, :],
        next_observation=observation + 100,
        done=jnp.asarray([done]),
        terminal=jnp.asarray([terminal]),
        actor_recurrence={
            "hidden": jnp.arange(transition_count * 3).reshape(
                1, transition_count, 3
            )
        },
    )


def _sample(experience):
    return PrioritisedEpisodeBufferSample(
        experience=experience,
        indices=jnp.asarray([5]),
        probabilities=jnp.asarray([0.25]),
        buffer_size=jnp.asarray(17),
    )


def test_tbptt_sequence_has_chronological_and_bootstrap_inputs():
    experience = _transition(done=[False, False, False, True])
    sequence = learner_sequence(
        _sample(experience), transition_count=4, full_episode=False
    )

    np.testing.assert_array_equal(
        sequence.inputs.observation[:, :4], experience.observation
    )
    np.testing.assert_array_equal(
        sequence.inputs.observation[:, 4], experience.next_observation[:, 3]
    )
    np.testing.assert_array_equal(
        sequence.inputs.previous_action, [[10, 11, 12, 13, 33]]
    )
    np.testing.assert_array_equal(
        sequence.inputs.previous_reward, [[20, 21, 22, 23, 43]]
    )
    np.testing.assert_array_equal(
        sequence.inputs.episode_start, [[True, False, False, False, False]]
    )
    np.testing.assert_array_equal(
        sequence.bootstrap_inputs.observation, experience.next_observation
    )
    np.testing.assert_array_equal(
        sequence.bootstrap_inputs.previous_action, experience.action
    )
    np.testing.assert_array_equal(
        sequence.bootstrap_inputs.previous_reward, experience.reward
    )
    np.testing.assert_array_equal(
        sequence.bootstrap_inputs.episode_start, jnp.zeros_like(experience.done)
    )
    np.testing.assert_array_equal(sequence.actions, experience.action)
    np.testing.assert_array_equal(sequence.rewards, experience.reward)
    np.testing.assert_array_equal(sequence.dones, experience.done)
    np.testing.assert_array_equal(sequence.terminals, experience.terminal)
    np.testing.assert_array_equal(sequence.valid, [[True, True, True, True]])
    np.testing.assert_array_equal(
        sequence.initial_recurrence["hidden"],
        experience.actor_recurrence["hidden"][:, 0],
    )
    np.testing.assert_array_equal(sequence.indices, [5])
    np.testing.assert_allclose(sequence.probabilities, [0.25])
    assert int(sequence.buffer_size) == 17


def test_time_limit_keeps_pre_reset_bootstrap_input():
    experience = _transition(
        done=[False, True, False, False],
        terminal=[False, False, False, False],
        episode_start=[True, False, True, False],
    ).replace(
        observation=_transition(done=[False] * 4).observation.at[:, 2].set(
            jnp.asarray([-9, -9])
        ),
        next_observation=_transition(done=[False] * 4).next_observation.at[:, 1].set(
            jnp.asarray([9, 9])
        ),
    )
    sequence = learner_sequence(
        _sample(experience), transition_count=4, full_episode=False
    )

    np.testing.assert_array_equal(sequence.inputs.observation[:, 2], [[-9, -9]])
    np.testing.assert_array_equal(
        sequence.bootstrap_inputs.observation[:, 1], [[9, 9]]
    )
    np.testing.assert_array_equal(
        sequence.bootstrap_inputs.episode_start[:, 1], [False]
    )


def test_completed_episode_starts_and_full_episode_padding():
    experience = _transition(
        done=[False, False, True, False, False, False, False, False],
        episode_start=[True, False, False, False, True, False, False, False],
    )
    starts = completed_episode_starts(experience, transition_count=5)
    sequence = learner_sequence(
        _sample(experience), transition_count=5, full_episode=True
    )

    np.testing.assert_array_equal(
        starts, [[True, False, False, False, False, False, False, False]]
    )
    np.testing.assert_array_equal(sequence.valid, [[True, True, True, False, False]])
    assert sequence.initial_recurrence is None


def test_partial_episode_is_not_a_completed_start():
    experience = _transition(
        done=[False, False, False, False, False],
        episode_start=[True, False, False, False, False],
    )
    np.testing.assert_array_equal(
        completed_episode_starts(experience, transition_count=5),
        [[False, False, False, False, False]],
    )


def test_tbptt_starts_require_one_learning_transition_after_burn_in():
    experience = _transition(
        done=[False, True, False, False, True, False],
        episode_start=[True, False, True, False, True, False],
    )
    np.testing.assert_array_equal(
        tbptt_starts(experience, burn_in_length=2),
        [[False, False, True, False, False, True]],
    )


def _fill_streaming_replay(experience):
    transition_count = 5
    buffer = make_prioritised_episode_buffer(
        max_length=8,
        min_length=transition_count,
        sample_batch_size=1,
        sample_sequence_length=transition_count,
        get_start_flags=partial(
            completed_episode_starts, transition_count=transition_count
        ),
        add_sequences=False,
        add_batch_size=1,
    )
    state = buffer.init(jax.tree.map(lambda value: value[0, 0], experience))
    for index in range(transition_count):
        state = buffer.add(
            state, jax.tree.map(lambda value: value[:, index], experience)
        )
    return buffer, state


def test_buffer_rejects_partial_episode_and_accepts_completed_episode():
    partial_experience = _transition(
        done=[False, False, False, False, False],
        episode_start=[True, False, False, False, False],
    )
    completed_experience = _transition(
        done=[False, False, True, False, False],
        episode_start=[True, False, False, False, False],
    )

    partial_buffer, partial_state = _fill_streaming_replay(partial_experience)
    completed_buffer, completed_state = _fill_streaming_replay(completed_experience)

    assert not bool(partial_buffer.can_sample(partial_state))
    assert bool(completed_buffer.can_sample(completed_state))
