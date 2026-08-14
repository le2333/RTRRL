import jax
import jax.numpy as jnp
import numpy as np

from memorax.buffers.prioritised_episode_buffer import make_prioritised_episode_buffer


def _buffer(*, sample_batch_size=2, sample_sequence_length=4):
    return make_prioritised_episode_buffer(
        max_length=64,
        min_length=sample_sequence_length,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=sample_sequence_length,
        get_start_flags=lambda experience: experience["episode_start"],
        add_sequences=True,
        add_batch_size=1,
    )


def _initialise_and_add(buffer, observations, episode_starts):
    state = buffer.init(
        {
            "observation": jnp.asarray(0, dtype=jnp.int32),
            "episode_start": jnp.asarray(False),
        }
    )
    return buffer.add(
        state,
        {
            "observation": jnp.asarray([observations], dtype=jnp.int32),
            "episode_start": jnp.asarray([episode_starts]),
        },
    )


def test_can_sample_requires_an_eligible_positive_priority_start():
    buffer = _buffer()
    no_start = _initialise_and_add(
        buffer, [10, 11, 12, 13], [False, False, False, False]
    )
    eligible_start = _initialise_and_add(
        buffer, [10, 11, 12, 13], [True, False, False, False]
    )

    assert not bool(buffer.can_sample(no_start))
    assert bool(buffer.can_sample(eligible_start))


def test_sampling_uses_only_nonzero_eligible_start_positions():
    buffer = _buffer(sample_batch_size=32, sample_sequence_length=2)
    state = _initialise_and_add(
        buffer,
        [0, 10, 20, 30, 40, 50],
        [False, False, True, False, True, False],
    )

    sample = buffer.sample(state, jax.random.key(7))
    sampled_starts = np.asarray(sample.experience["observation"][:, 0])

    assert set(sampled_starts.tolist()) == {20, 40}
    np.testing.assert_allclose(sample.probabilities, np.full(32, 0.5))
    assert int(sample.buffer_size) == 6
