"""What a replayed recurrent window means, for the learners that replay one.

Two questions come up identically wherever a sequence is drawn from replay and
scored: which stored positions a full-episode window may begin at, and how a
per-timestep error becomes one number once the positions past an ending stop
counting. Neither answer mentions a target, a priority, or a truncation, which
is why they are here and the rest of each learner is not.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from memorax.utils.typing import Array


def completed_episode_starts(experience: Any, *, transition_count: int) -> Array:
    """Episode starts whose episode also ends inside a window this long.

    A full-episode learner may only begin where the whole episode is there to
    be read. A start whose ending falls past the window would be scored on a
    prefix and called an episode.
    """

    ending_within_window = jnp.zeros_like(experience.done, dtype=jnp.bool_)
    for offset in range(transition_count):
        ending_within_window = ending_within_window | jnp.roll(
            experience.done, -offset, axis=1
        )
    return experience.episode_start & ending_within_window


def masked_sequence_loss(
    td_error: Array, valid: Array, weights: Array | None = None
) -> Array:
    """Half the squared error, averaged within a sequence and then across them.

    Averaging inside the sequence first is what makes sequences of different
    valid lengths weigh the same, and ``weights`` is the caller's opportunity
    to say they should not: a learner that draws non-uniformly hands its
    correction here, and one that draws uniformly hands nothing, because
    uniform weights are the absence of a correction rather than a choice of
    one.
    """

    mask = valid.astype(td_error.dtype)
    per_sequence = 0.5 * jnp.sum(jnp.square(td_error) * mask, axis=1)
    per_sequence = per_sequence / jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    if weights is None:
        return jnp.mean(per_sequence)
    return jnp.mean(per_sequence * weights)
