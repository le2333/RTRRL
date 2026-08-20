"""Replay that draws an episode first and a window inside it second.

The buffers next to this one answer a question about *positions*: given a mask
over stored rows, which one shall the window begin at. That is the right
question when the unit of replay is a transition, as in DQN, or a fixed-length
sequence, as in R2D2, because there the position and the item are the same
thing. It is the wrong question for the update DRQN publishes. There the unit
is an episode, drawn without replacement, and only then a point inside it:

    e_1..e_B ~ without replacement over completed episodes
    s_i      ~ U{0 .. L_{e_i} - t}
    window_i  = the t transitions from s_i

A per-position weight can reproduce the marginal of that -- weight each
position by one over how many its episode offers, and each episode is drawn
equally often -- but it cannot reproduce the joint, because "without
replacement over episodes" is a statement about B draws together and a weight
vector is a statement about one draw. Nor can it say what should happen when
fewer than B episodes are eligible; the published loop shrinks the minibatch,
and a position sampler has no way to shrink anything. Trying to express the
one in the other is what produces the long tail of masks, fallbacks and seam
margins that this module exists to not need.

So storage and sampling policy are separated. Flashbax keeps the transitions,
unchanged and per-stream, and a small index alongside it records where each
completed episode lives. The index is written in *logical* time -- a counter
that only ever increases -- and the physical slot is `logical % capacity`.
Two things fall out of that for free:

*The ring seam stops being a special case.* An episode's transitions are all
still stored exactly when its first one is, `start >= written - capacity`, and
a window taken from inside it can never splice across the write head, because
every logical index between its start and `written` is present by definition.
No margin, no exclusion zone, no arithmetic about which side of the head a
position sits on.

*"Enough data" and "something to sample" become the same question.* The index
knows how many episodes are eligible, so `can_sample` can answer honestly
instead of reporting on the buffer's length and leaving the sampler to
discover there is nothing to draw -- the failure that a silent fall back to
position zero turns into a run that trains, reports, and means nothing.

The index is also what makes replay episode-atomic without a staging buffer,
and there are two halves to that. The easy half is that transitions of an
episode still being played are in the ring but no record describes them, so
nothing can draw from them and the warmup does not count them.

The half that is easy to get wrong is *eviction*. An open episode's writes
advance the ring's head, so left alone they overwrite the oldest committed
episodes and those episodes stop being drawable -- which changes not the depth
of storage but **which episodes an update can draw while the current one is
being played**, and that is the replay distribution. A learner that stages an
episode and commits it whole leaves replay untouched until the ending.

So the ring is allocated a `max_episode_length` slack **on top of** the
capacity it was asked for, and an episode counts as stored while

    start >= open_start[stream] - committed_capacity

rather than while `start >= written - time_capacity`. That threshold moves only
when an episode commits, so the drawable set is fixed for the duration of an
episode, and it is strictly inside physical presence, because an open episode
runs at most `max_episode_length` past `open_start`.

`max_length` therefore means what it says: the transitions of *finished*
episodes replay keeps. The slack is this module's cost and is paid out of its
own allocation, not out of the caller's number -- a buffer asked for 8192 keeps
8192 and stores 8192 + `max_episode_length` per stream, where charging the
slack to the caller would quietly make it 8192 - `max_episode_length`.

Readiness is measured the same way, against what replay is holding rather than
against what it has ever held. A lifetime count only rises, so it would keep
reporting ready on a buffer that had since evicted most of what it counted.

Flashbax is storage here and nothing else. Its own `can_sample` is not called,
because it answers for its own trajectory sampler: it will not report ready
until a whole `sample_sequence_length` has been written, which is the right
contract for drawing a fixed-length slice off the head and the wrong one for
drawing an episode. Left in, it would make the warmup `max(min_length, t)`
rather than `min_length`, so a learner asking for a long window would begin
learning later than one asking for a short window at the same declared replay
settings -- the window length reaching out of the sampler and into the schedule.
A replay warmup is how much experience has been collected before learning
starts, and that is the same quantity whatever the caller intends to draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flashbax.buffers.trajectory_buffer import (
    TrajectoryBufferState,
    make_trajectory_buffer,
)
from flashbax.utils import add_dim_to_args
from flax import struct

from memorax.utils.typing import Array, Key


class EpisodeIndexState(struct.PyTreeNode):
    """Where each completed episode is, in the logical time it was written in.

    A ring of fixed-size records, because everything here has to hold a static
    shape. ``live`` marks a slot that has ever been written; whether what it
    describes is still *stored* is a separate question, asked against the
    transition ring at sample time rather than maintained here, since it
    changes with every add and depends on nothing this state knows.
    """

    stream: Array
    """Which parallel environment's row the episode was written to. ``[E]``"""

    start: Array
    """Logical index of the episode's first transition. ``[E]``"""

    length: Array
    """Transitions in the episode, ending included. ``[E]``"""

    live: Array
    """Whether this slot holds a record at all. ``[E]``"""

    write_index: Array
    """Where the next completed episode is recorded. ``[]``"""

    open_start: Array
    """Logical index each stream's unfinished episode began at. ``[streams]``"""


class EpisodeWindowBufferState(struct.PyTreeNode):
    trajectory: TrajectoryBufferState
    episodes: EpisodeIndexState
    written: Array
    """Transitions written to each stream so far. Logical, so it never wraps."""


class EpisodeWindowSample(struct.PyTreeNode):
    """A minibatch of windows, and how much of it is a minibatch.

    ``batch_valid`` is how the published loop's shrinking minibatch survives
    a static shape: with fewer than ``sample_batch_size`` episodes eligible,
    the rows that could not be filled are marked here rather than filled with
    a repeat. A learner that averages over the marked rows only is averaging
    over ``min(eligible, B)`` windows, which is what the published loop does.
    """

    experience: Any
    """``[B, window, ...]``, gathered from the transition ring."""

    valid: Array
    """``[B, window]``: which steps are inside the drawn episode."""

    batch_valid: Array
    """``[B]``: which rows are a drawn episode at all."""


@dataclass(frozen=True)
class EpisodeWindowBuffer:
    init: Callable[[Any], EpisodeWindowBufferState]
    add: Callable[[EpisodeWindowBufferState, Any], EpisodeWindowBufferState]
    sample: Callable[[EpisodeWindowBufferState, Key], EpisodeWindowSample]
    can_sample: Callable[[EpisodeWindowBufferState], Array]
    retained: Callable[[EpisodeWindowBufferState], Array]
    """Transitions of finished episodes replay is currently holding.

    The quantity the warmup is a threshold on, exposed because a caller
    checking whether replay is full should not have to infer it from
    ``can_sample`` saying no.
    """


def make_uniform_episode_window_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    add_batch_size: int,
    max_episode_length: int,
    minimum_episode_length: int | None = None,
) -> EpisodeWindowBuffer:
    """The published bootstrapped random update, as a buffer.

    One timestep per stream per ``add``; the index has to see each ending as it
    arrives to know where the next episode begins, and DRQN steps its streams
    in lockstep, so there is nothing to gain from a sequence form that would
    need a scan to stay correct.

    ``minimum_episode_length`` is the length below which an episode is not
    worth drawing. A caller wanting exactly ``t`` transitions per window sets
    it to ``t``, because ``U{0 .. L - t}`` has no start to draw from in a
    shorter episode and a window cut off at an ending would not be ``t``
    transitions. A caller wanting whole episodes sets it to one and takes
    whatever length there is, padding to the declared limit and masking the
    rest -- which is the same code path, because an episode that cannot fill
    the window simply has no start to choose between.

    ``max_length`` is the transitions of *finished* episodes replay keeps. The
    physical ring is that plus ``max_episode_length`` per stream, so the episode
    being played has somewhere to go that is not somebody else's.

    ``max_episode_length`` is the longest an episode can run. It is a declared
    limit rather than an observed maximum: an episode that overran it would not
    corrupt a sample -- the eligibility test takes whichever of the two bounds
    is stricter -- but replay would start moving under the open episode again,
    which is the thing the slack exists to prevent.
    """

    if add_batch_size < 1:
        raise ValueError("add_batch_size must be at least one")
    if min_length < 1:
        raise ValueError("min_length must be at least one transition")
    if max_episode_length < 1:
        raise ValueError("an episode runs at least one transition")
    committed_capacity = max_length // add_batch_size
    if committed_capacity < 2:
        raise ValueError(
            f"max_length//add_batch_size must be at least 2; it is "
            f"{max_length}//{add_batch_size} = {committed_capacity}"
        )
    # The slack the open episode writes into, on top of what the caller asked
    # replay to keep rather than out of it.
    time_capacity = committed_capacity + max_episode_length
    if sample_sequence_length > committed_capacity:
        raise ValueError(
            f"a window of {sample_sequence_length} does not fit in the "
            f"{committed_capacity} per stream this buffer keeps finished episodes in"
        )
    # An episode straddling the oldest kept position is dropped whole, so what
    # replay holds sits just above `committed_capacity - max_episode_length`
    # once the ring has wrapped. A warmup above that could be met and then
    # unmet, and a learner that stops learning because an episode fell off a
    # boundary is a failure nobody would look for.
    steady_state = add_batch_size * (committed_capacity - max_episode_length)
    if min_length > steady_state:
        raise ValueError(
            f"a warmup of {min_length} transitions is not one this buffer stays "
            f"above: with episodes of up to {max_episode_length} it keeps more "
            f"than {steady_state} finished transitions and not reliably more"
        )
    if minimum_episode_length is None:
        minimum_episode_length = sample_sequence_length
    if minimum_episode_length < 1:
        raise ValueError("an episode worth drawing holds at least one transition")

    # An episode occupies at least one transition, so the transition ring
    # cannot hold more episodes than transitions. Sizing the index at that
    # bound is what keeps it from ever being the binding constraint -- an
    # index that wrapped first would silently drop episodes that are still
    # stored, and the sampler would have no way to know it had.
    episode_capacity = add_batch_size * time_capacity
    if sample_batch_size > episode_capacity:
        raise ValueError(
            f"a minibatch of {sample_batch_size} episodes cannot be drawn without "
            f"replacement from a buffer that holds at most {episode_capacity}"
        )

    trajectory = make_trajectory_buffer(
        max_length_time_axis=time_capacity,
        # Flashbax's own readiness rule, declared only because it validates the
        # argument. ``can_sample`` below does not consult it: this buffer's
        # warmup is ``min_length`` transitions and nothing to do with how long
        # a window is.
        min_length_time_axis=sample_sequence_length,
        add_batch_size=add_batch_size,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=1,
        max_size=None,
    )
    # Flashbax's add takes a batch of sequences; this one takes a batch of
    # single timesteps, which is that with a length-one time axis.
    add_timestep = add_dim_to_args(
        trajectory.add, axis=1, starting_arg_index=1, ending_arg_index=2
    )

    def init_fn(experience: Any) -> EpisodeWindowBufferState:
        return EpisodeWindowBufferState(
            trajectory=trajectory.init(experience),
            episodes=EpisodeIndexState(
                stream=jnp.zeros((episode_capacity,), dtype=jnp.int32),
                start=jnp.zeros((episode_capacity,), dtype=jnp.int32),
                length=jnp.zeros((episode_capacity,), dtype=jnp.int32),
                live=jnp.zeros((episode_capacity,), dtype=jnp.bool_),
                write_index=jnp.asarray(0, dtype=jnp.int32),
                open_start=jnp.zeros((add_batch_size,), dtype=jnp.int32),
            ),
            written=jnp.asarray(0, dtype=jnp.int32),
        )

    def add_fn(
        state: EpisodeWindowBufferState, transition: Any
    ) -> EpisodeWindowBufferState:
        """Store a timestep per stream, and record every episode it ended."""

        index = state.episodes
        done = jnp.asarray(transition.done).reshape((add_batch_size,)).astype(jnp.bool_)
        # The transition being stored sits at logical ``written``, so an
        # episode ending on it runs from its start through that index.
        length = state.written + 1 - index.open_start

        # Several streams can end on the same timestep, so each ending takes
        # the next slot in order rather than all of them taking the same one.
        # Streams that did not end are aimed one past the ring, which a scatter
        # in drop mode discards -- the alternative, a scan over streams, would
        # be the same write in a loop.
        ordinal = jnp.cumsum(done.astype(jnp.int32)) - 1
        slot = jnp.where(
            done, (index.write_index + ordinal) % episode_capacity, episode_capacity
        )
        stream = jnp.arange(add_batch_size, dtype=jnp.int32)
        recorded = EpisodeIndexState(
            stream=index.stream.at[slot].set(stream, mode="drop"),
            start=index.start.at[slot].set(index.open_start, mode="drop"),
            length=index.length.at[slot].set(length, mode="drop"),
            live=index.live.at[slot].set(True, mode="drop"),
            write_index=(index.write_index + jnp.sum(done.astype(jnp.int32)))
            % episode_capacity,
            open_start=jnp.where(done, state.written + 1, index.open_start),
        )
        return EpisodeWindowBufferState(
            trajectory=add_timestep(state.trajectory, transition),
            episodes=recorded,
            written=state.written + 1,
        )

    def stored_fn(state: EpisodeWindowBufferState) -> Array:
        """Which records describe an episode replay is still holding. ``[E]``

        Measured from the stream's last *commit* rather than from its write
        head. Measuring from the head would let the episode being played evict
        the oldest committed ones as it went, so which episodes an update could
        draw would change during an episode -- a moving replay distribution,
        where a learner that commits whole episodes has a still one.

        An episode whose first transition survives is entirely there, because
        logical time is contiguous behind the head. The physical bound is kept
        as well as the reserved one, so an episode that overran the declared
        limit degrades to a moving threshold rather than to an unsound read.
        """

        index = state.episodes
        reserved = index.open_start[index.stream] - committed_capacity
        physical = state.written - time_capacity
        return index.live & (index.start >= jnp.maximum(reserved, physical))

    def eligible_fn(state: EpisodeWindowBufferState) -> Array:
        """Still stored, and long enough to draw a window from. ``[E]``"""

        return stored_fn(state) & (state.episodes.length >= minimum_episode_length)

    def retained_fn(state: EpisodeWindowBufferState) -> Array:
        """Transitions of finished episodes replay is currently holding."""

        return jnp.sum(jnp.where(stored_fn(state), state.episodes.length, 0))

    def can_sample_fn(state: EpisodeWindowBufferState) -> Array:
        """Whether a draw would return anything, not whether the buffer is big.

        Both halves are needed and neither implies the other: a buffer past its
        minimum length can be entirely one unfinished episode, and a buffer
        holding one short completed episode is not yet worth learning from.

        The first half counts the transitions of finished episodes replay is
        *currently holding*. Not transitions written, because an episode still
        being played is not experience replay has and counting it would start
        learning early by however much of one happened to be in progress. Not a
        lifetime total either, because that only rises: it would go on
        reporting ready against a buffer that had since evicted most of what it
        counted. And it is the same number whatever window the caller intends
        to draw, where deferring to Flashbax would fold the window length into
        the warmup instead.
        """

        return (retained_fn(state) >= min_length) & jnp.any(eligible_fn(state))

    def sample_fn(state: EpisodeWindowBufferState, key: Key) -> EpisodeWindowSample:
        episode_key, offset_key = jax.random.split(key)
        index = state.episodes

        # Gumbel top-k: perturbing each eligible episode's (equal) log-weight
        # by an independent Gumbel and taking the largest B draws a uniform
        # subset of size B without replacement, in one fixed-shape operation.
        # Ineligible episodes score negative infinity, which both keeps them
        # out and marks the rows that could not be filled.
        perturbed = jax.random.gumbel(episode_key, (episode_capacity,))
        scores = jnp.where(eligible_fn(state), perturbed, -jnp.inf)
        scores, drawn = jax.lax.top_k(scores, sample_batch_size)
        batch_valid = jnp.isfinite(scores)

        stream = index.stream[drawn]
        start = index.start[drawn]
        length = index.length[drawn]

        # How many places a window can begin. An episode shorter than the
        # window offers exactly one -- its beginning -- and the steps past its
        # ending are masked out below rather than drawn from.
        places = jnp.maximum(length - sample_sequence_length + 1, 1)
        offset = jax.random.randint(
            offset_key, (sample_batch_size,), 0, places, dtype=jnp.int32
        )

        step = jnp.arange(sample_sequence_length, dtype=jnp.int32)
        logical = start[:, None] + offset[:, None] + step[None, :]
        # A row that stands for no episode reads stream zero from the
        # beginning; it is masked everywhere it is used, and the read has to
        # land somewhere in bounds.
        stream = jnp.where(batch_valid, stream, 0)
        drawn_window = jax.tree.map(
            lambda value: value[stream[:, None], logical % time_capacity],
            state.trajectory.experience,
        )
        valid = (step[None, :] < (length - offset)[:, None]) & batch_valid[:, None]
        # A window longer than its episode reads slots belonging to some other
        # episode, or never written at all. Those steps are masked out of the
        # loss, but they are still unrolled through the recurrent cell, so what
        # they hold is not nothing: it decides what the padding computes and,
        # for an unwritten slot, whether that is even a defined number. Zeroing
        # them makes the padding one fixed thing rather than a reading of
        # whatever the ring happened to hold.
        experience = jax.tree.map(
            lambda value: jnp.where(
                valid.reshape(valid.shape + (1,) * (value.ndim - 2)),
                value,
                jnp.zeros((), dtype=value.dtype),
            ),
            drawn_window,
        )
        return EpisodeWindowSample(
            experience=experience, valid=valid, batch_valid=batch_valid
        )

    return EpisodeWindowBuffer(
        init=init_fn,
        add=add_fn,
        sample=sample_fn,
        can_sample=can_sample_fn,
        retained=retained_fn,
    )
