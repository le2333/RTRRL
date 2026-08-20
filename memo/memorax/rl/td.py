"""Pure temporal-difference target primitives."""

from __future__ import annotations

import jax.numpy as jnp

from memorax.utils.typing import Array


def make_td0():
    """Build TD(0) over the ending that decides whether there is a future.

    The mask is the whole of what TD says about an ending, so it is formed here
    rather than by the caller: an episode cut off at its step limit was about to
    go on earning and the value of where it stopped is the best statement anyone
    has about the rest, while one that failed has nothing after it. Handed a
    discount already multiplied by something, this could not tell the two apart
    and neither could its callers.
    """

    def td0(*, reward, value, next_value, terminal, gamma):
        return reward + gamma * (1 - terminal) * next_value - value

    return td0


def masked_sequence_loss(
    td_error: Array,
    valid: Array,
    weights: Array | None = None,
    batch_valid: Array | None = None,
) -> Array:
    """Half the squared error, averaged within a sequence and then across them.

    Averaging inside the sequence first is what makes sequences of different
    valid lengths weigh the same, and ``weights`` is the caller's opportunity
    to say they should not: a learner that draws non-uniformly hands its
    correction here, and one that draws uniformly hands nothing, because
    uniform weights are the absence of a correction rather than a choice of
    one.

    ``batch_valid`` says how many sequences there really are, for a sampler
    that could not fill the minibatch. Dividing by the declared batch size
    instead would shrink the loss, and with it the gradient, in exactly the
    situation where replay is thinnest -- early training -- which is a change
    to the learning rate schedule disguised as a padding convention.
    """

    # Selected rather than multiplied by a zero: a masked position is one
    # nobody drew, and if anything ever put a non-finite value there,
    # multiplying would carry it into the sum instead of dropping it.
    kept = jnp.where(valid, 0.5 * jnp.square(td_error), 0.0)
    counted = valid.astype(td_error.dtype)
    per_sequence = jnp.sum(kept, axis=1) / jnp.maximum(jnp.sum(counted, axis=1), 1.0)
    if weights is not None:
        per_sequence = per_sequence * weights
    if batch_valid is None:
        return jnp.mean(per_sequence)
    drawn = batch_valid.astype(per_sequence.dtype)
    return jnp.sum(per_sequence * drawn) / jnp.maximum(jnp.sum(drawn), 1.0)
