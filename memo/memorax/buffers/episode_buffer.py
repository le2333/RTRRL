from typing import TYPE_CHECKING, Callable

from memorax.utils.typing import Array, Key

if TYPE_CHECKING:
    pass
else:
    pass

import chex
import jax
import jax.numpy as jnp
from flashbax import utils
from flashbax.buffers.trajectory_buffer import (
    Experience,
    TrajectoryBuffer,
    TrajectoryBufferSample,
    TrajectoryBufferState,
    make_trajectory_buffer,
)
from flashbax.utils import add_dim_to_args


def get_full_start_flags(experience: Experience) -> Array:
    return jnp.ones_like(experience.first.done)


def get_start_flags_from_done(experience: Experience) -> Array:
    return experience.first.done


def validate_sample_batch_size(sample_batch_size: int, max_length: int):
    if sample_batch_size > max_length:
        raise ValueError("sample_batch_size must be less than or equal to max_length")


def validate_min_length(min_length: int, add_batch_size: int, max_length: int):
    used_min_length = min_length // add_batch_size
    max_length_time_axis = max_length // add_batch_size
    if used_min_length > max_length_time_axis:
        raise ValueError("min_length used is too large for the buffer size.")


def validate_max_length_add_batch_size(max_length: int, add_batch_size: int):
    if max_length // add_batch_size < 2:
        raise ValueError(
            f"""max_length//add_batch_size must be greater than 2. It is currently
            {max_length}//{add_batch_size} = {max_length // add_batch_size}"""
        )


def validate_sample_sequence_length(
    sample_sequence_length: int, max_length: int, add_batch_size: int
):
    if sample_sequence_length > max_length // add_batch_size:
        raise ValueError(
            "sample_sequence_length must be <= max_length // add_batch_size"
        )


def validate_seam_margin(
    seam_margin: int, sample_sequence_length: int, max_length: int, add_batch_size: int
):
    """A ring that keeps nothing once it wraps is worse than one with a seam.

    When the margin and the window between them cover the whole ring, no
    position is admissible once it fills, and the sampler falls back to drawing
    from the beginning -- which looks like sampling and is not. Refused here,
    because the buffer that would do it is built long before it wraps.
    """

    if seam_margin < 0:
        raise ValueError("seam_margin must not be negative")
    max_length_time_axis = max_length // add_batch_size
    room = max_length_time_axis - sample_sequence_length - 2 * seam_margin
    if seam_margin and room < 0:
        raise ValueError(
            f"a ring of {max_length_time_axis} per stream has no position left to "
            f"start a window of {sample_sequence_length} once a margin of "
            f"{seam_margin} is kept clear at each end of it; give the buffer at "
            f"least {sample_sequence_length + 2 * seam_margin} per stream"
        )


def validate_episode_buffer_args(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    add_batch_size: int,
    seam_margin: int = 0,
):
    validate_sample_batch_size(sample_batch_size, max_length)
    validate_min_length(min_length, add_batch_size, max_length)
    validate_max_length_add_batch_size(max_length, add_batch_size)
    validate_sample_sequence_length(sample_sequence_length, max_length, add_batch_size)
    validate_seam_margin(
        seam_margin, sample_sequence_length, max_length, add_batch_size
    )


def _valid_start_mask(
    state: TrajectoryBufferState[Experience],
    sample_sequence_length: int,
    seam_margin: int = 0,
) -> Array:
    """Which stored positions a window may begin at, given where the head is.

    Once the ring is full the write head is a seam: the row before it is the
    newest transition and the row at it is the oldest, and they never followed
    one another. Reading a window across it splices two moments together into
    something that looks like a trajectory and is not, so the positions whose
    window would span it are excluded.

    ``seam_margin`` excludes that much further on both sides, for a caller whose
    start rule reads beyond the window itself -- looking ahead for the episode's
    ending, or back for its beginning. Such a rule is answering a question about
    a trajectory too, and the seam is not one there either.
    """

    _, max_length_time_axis = utils.get_tree_shape_prefix(state.experience, n_axes=2)
    time_indices = jnp.arange(max_length_time_axis)

    def _not_full() -> Array:
        last_valid = jnp.maximum(state.current_index - sample_sequence_length, -1)
        return (time_indices >= 0) & (time_indices <= last_valid)

    def _full() -> Array:
        # Distance forward from the oldest row, so that the seam is the two ends
        # of this range rather than somewhere in the middle of it.
        from_oldest = (time_indices - state.current_index) % max_length_time_axis
        return (from_oldest >= seam_margin) & (
            from_oldest <= max_length_time_axis - sample_sequence_length - seam_margin
        )

    return jax.lax.cond(state.is_full, _full, _not_full)


def make_episode_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    get_start_flags: Callable[[Experience], Array] = get_start_flags_from_done,
    add_sequences: bool = False,
    add_batch_size: int | None = None,
    min_length_time_axis: int | None = None,
    seam_margin: int = 0,
) -> TrajectoryBuffer:
    if add_batch_size is None:
        add_batch_size = 1
        add_batches = False
    else:
        add_batches = True

    validate_episode_buffer_args(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=sample_sequence_length,
        add_batch_size=add_batch_size,
        seam_margin=seam_margin,
    )

    buffer = make_trajectory_buffer(
        max_length_time_axis=max_length // add_batch_size,
        min_length_time_axis=max(min_length // add_batch_size, sample_sequence_length),
        add_batch_size=add_batch_size,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=1,
        max_size=None,
    )

    add_fn = buffer.add
    if not add_batches:
        add_fn = add_dim_to_args(
            add_fn, axis=0, starting_arg_index=1, ending_arg_index=2
        )
    if not add_sequences:
        axis = 1 - int(not add_batches)
        add_fn = add_dim_to_args(
            add_fn, axis=axis, starting_arg_index=1, ending_arg_index=2
        )

    def sample_fn(
        state: TrajectoryBufferState[Experience], rng_key: Key
    ) -> TrajectoryBufferSample[Experience]:
        add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
            state.experience, n_axes=2
        )

        start_flags = get_start_flags(state.experience)
        chex.assert_shape(start_flags, (add_batch_size, max_length_time_axis))
        start_flags = start_flags.astype(jnp.float32)

        valid_mask = _valid_start_mask(
            state, sample_sequence_length, seam_margin
        ).astype(jnp.float32)
        start_mask = start_flags * valid_mask[None, :]

        per_row = jnp.sum(start_mask, axis=1)
        total = jnp.sum(per_row)

        def _sample_from_starts(key: Key) -> tuple[Array, Array]:
            p_rows = per_row / total
            key_rows, key_starts = jax.random.split(key)

            rows = jax.random.choice(
                key_rows,
                a=add_batch_size,
                shape=(sample_batch_size,),
                p=p_rows,
                replace=True,
            )

            row_start_probs = start_mask[rows]
            row_start_probs = row_start_probs / jnp.maximum(
                jnp.sum(row_start_probs, axis=1, keepdims=True), 1.0
            )
            keys = jax.random.split(key_starts, sample_batch_size)
            starts = jax.vmap(
                lambda k, p: jax.random.choice(
                    k, a=max_length_time_axis, shape=(), p=p, replace=True
                )
            )(keys, row_start_probs)
            return rows, starts

        def _fallback_beginning(key: Key) -> tuple[Array, Array]:
            rows = jax.random.randint(key, (sample_batch_size,), 0, add_batch_size)
            starts = jnp.zeros((sample_batch_size,), dtype=jnp.int32)
            return rows, starts

        rows, starts = jax.lax.cond(
            total > 0, _sample_from_starts, _fallback_beginning, rng_key
        )

        time_idx = (
            starts[:, None] + jnp.arange(sample_sequence_length)
        ) % max_length_time_axis
        experience = jax.tree.map(
            lambda x: x[rows[:, None], time_idx], state.experience
        )
        return TrajectoryBufferSample(experience=experience)

    return buffer.replace(add=add_fn, sample=sample_fn)
