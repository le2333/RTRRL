"""What a replayed recurrent window means, for the learners that replay one.

Three questions come up identically wherever a sequence is drawn from replay and
scored: which stored positions a full-episode window may begin at, which ones a
window of a fixed length may begin at, and how a per-timestep error becomes one
number once the positions past an ending stop counting. None of the answers
mentions a target or a priority, which is why they are here and the rest of each
learner is not.

A start rule answers in weights rather than in flags. The buffer normalises
what it is handed, so a rule that returns ones is saying every admissible
position is equally likely, and one that returns something else is saying which
population it means to draw uniformly from -- which, for a rule written per
position, is the only place the difference can be said.
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


def _nearest_offset(flags: Array, horizon: int, *, backwards: bool = False) -> Array:
    """For each position, how far to the nearest flagged position, or ``horizon``.

    Written as a bounded unrolled scan because the horizon is a static shape
    fact: an episode cannot be longer than the declared limit, so a flag that
    has not appeared within that many steps is not going to.
    """

    direction = 1 if backwards else -1
    found = jnp.full(flags.shape, horizon, dtype=jnp.int32)
    # Descending, so that a nearer flag overwrites a farther one and what is
    # left is the nearest.
    for offset in reversed(range(horizon)):
        found = jnp.where(jnp.roll(flags, direction * offset, axis=1), offset, found)
    return found


def episode_window_starts(
    experience: Any, *, truncation: int, episode_length: int
) -> Array:
    """Where a window of exactly ``truncation`` transitions may begin.

    The published random update draws an episode, then a point inside it, then
    unrolls a whole window from there. Both halves of that matter and neither
    is what drawing a stored position gives you.

    *A window lies inside one completed episode.* A window that ran over an
    ending would be cut short by the validity mask, and a learner whose windows
    are cut short is not training at the truncation it declares: near the end of
    an episode it would reach back five or ten steps while claiming sixty-four.
    That is the quantity a truncation sweep exists to measure, so it may not be
    a function of where in an episode a window happened to land. Here a position
    admits a window only if its episode runs at least ``truncation`` transitions
    further and ends within the horizon -- which also excludes an episode still
    being played, and excludes every episode shorter than ``truncation``.

    *Each episode is drawn equally often.* Every admissible position in an
    episode is weighted by one over how many that episode has, so a long episode
    does not collect probability by offering more places to start. Uniform
    weights would draw an episode in proportion to its length, and on a task
    where the policy's episodes grow as it improves that is a drift in what is
    replayed rather than a fixed sampling rule.
    """

    if truncation < 1:
        raise ValueError("a window holds at least one transition")
    if truncation > episode_length:
        raise ValueError(
            f"a window of {truncation} transitions cannot fit inside an episode of "
            f"at most {episode_length}; no start would admit one"
        )
    # With no ending closer than the window's last transition, the window holds
    # ``truncation`` transitions of one episode. Finding an ending at all is
    # what makes that episode a completed one.
    ahead = _nearest_offset(experience.done, episode_length)
    admits = (ahead >= truncation - 1) & (ahead < episode_length)

    behind = _nearest_offset(experience.episode_start, episode_length, backwards=True)
    # An episode spanning ``behind + ahead + 1`` transitions offers this many
    # places a whole window can start, and the count is the same at every
    # position in it.
    per_episode = behind + ahead + 2 - truncation
    return jnp.where(admits, 1.0 / jnp.maximum(per_episode, 1).astype(jnp.float32), 0.0)


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
