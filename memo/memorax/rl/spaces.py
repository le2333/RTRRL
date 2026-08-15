"""What one action is, read off the space an environment declares.

Two questions get asked of an action space and only one of them was ever
answered here. A policy head asks how wide its own output is. A recurrent input
asks how wide the *previous* action is once it is carried beside the
observation. A ``Box`` answers both with ``shape[0]``, so the two questions
looked like one question and only ``shape[0]`` was written down. A ``Discrete``
answers the first with ``n`` and the second with ``n`` as well -- but only
because the integer it hands out is widened into a one-hot on the way in, and
nothing widened it.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from memorax.utils.typing import Array


def action_classes(space: Any) -> int | None:
    """How many actions a discrete space names, or ``None`` if it names none.

    ``None`` is the continuous answer rather than a failure: a ``Box`` has no
    count of actions to give, and callers branch on that.
    """

    count = getattr(space, "n", None)
    return None if count is None else int(count)


def action_dim(space: Any) -> int:
    """How wide one action is, both as a head output and as a network input."""

    classes = action_classes(space)
    if classes is not None:
        return classes
    shape = tuple(getattr(space, "shape", ()))
    if len(shape) != 1:
        raise ValueError(
            f"action space {space!r} is neither discrete nor a vector Box: "
            f"its shape is {shape}"
        )
    return int(shape[0])


def encode_feedback(
    action: Array, *, classes: int | None, dtype: Any = jnp.float32
) -> Array:
    """The previous action, as the vector a recurrent input can carry.

    A continuous action already is that vector and is handed back untouched, so
    a Gaussian graph reads exactly what it read before discrete spaces reached
    this far. A discrete action is one integer with no feature axis to
    concatenate on and no metric meaning if it had one, so it is widened into
    its one-hot -- the same encoding R2D2 gives its own previous action.

    The one-hot is drawn in ``float32``, which is what the reward it is
    concatenated with is already pinned to.
    """

    if classes is None:
        return action
    return jax.nn.one_hot(action, classes, dtype=dtype)
