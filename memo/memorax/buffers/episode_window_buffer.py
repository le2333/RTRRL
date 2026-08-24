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
slack to the caller would quietly make it 8192 - `max_episode_length`. The
bound is strict, as the published `RememberEpisode` has it: it pushes and then
pops while `size >= capacity`, so an episode that would bring the total to
exactly the capacity is the one that goes.

Readiness is measured the same way, against what replay is holding rather than
against what it has ever held. A lifetime count only rises, so it would keep
reporting ready on a buffer that had since evicted most of what it counted.

**Every question this module answers is answered in the size of the answer and
not in the size of the buffer.** A learner that updates once per transition
asks all of them once per transition, so a pass over the index is a cost the
run pays as many times as it takes steps: replay sized for a long task is
where the agent's throughput goes, and it goes there without appearing in any
per-step number, because how long a step took is not what a training curve
records. Two properties of the index make those passes avoidable, and both
come from writing it in logical time:

*The records of one stream are sorted by `start`, because they are appended in
the order the episodes ended.* Eviction is a threshold on `start`, so the
episodes a stream still holds are a contiguous run ending at its most recent
commit, and where that run begins is a binary search -- `log2` of the index
rather than a pass over it. This is why the index is `[streams, capacity]`
with a counter per stream rather than one shared ring: the threshold moves per
stream, and in a shared ring it would cut the records into as many runs as
there are streams, which is a mask again and not a range.

*Whether an episode is long enough to draw from is settled at its ending.*
`minimum_episode_length` does not depend on anything that changes afterwards,
so the episodes that pass it are recorded a second time, in an index of their
own, where they are contiguous for the same reason. Eligibility is then a
range per stream and a count of them, and the minibatch is drawn out of
`[0, N)` by Floyd's algorithm -- B draws for B episodes, without replacement,
and without materialising anything the size of N.

What is *not* kept is which records are still stored. That changes with every
add, depends on nothing the index knows, and is a binary search when asked;
maintaining it would spend a write per step to save a search per step.

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

import math
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

    A ring per stream of fixed-size records, because everything here has to
    hold a static shape. A stream's ``n``-th completed episode is recorded at
    column ``n % capacity``, so its records are sorted by ``start`` in the
    logical position they were written at, and a threshold on ``start`` is a
    binary search over positions rather than a pass over rows.

    Whether a record describes an episode that is still *stored* is not held
    here: it changes with every add and depends on nothing this state knows,
    so it is asked against the transition ring at sample time.

    The episodes long enough to draw a window from are recorded twice -- once
    in ``start``, which is every episode and is what "how much is replay
    holding" is counted from, and once in ``drawable_start`` and
    ``drawable_length``, which are the ones an update may draw. Being long
    enough is settled at the ending and never revisited, so the second index
    is contiguous in exactly the way the first one is, and eligibility is a
    range instead of a mask.
    """

    start: Array
    """Logical index of each completed episode's first transition. ``[S, E]``"""

    committed: Array
    """Episodes each stream has completed. Logical, so it never wraps. ``[S]``"""

    drawable_start: Array
    """``start``, for the episodes worth drawing a window from. ``[S, E]``"""

    drawable_length: Array
    """Transitions in those episodes, ending included. ``[S, E]``"""

    drawable: Array
    """How many of a stream's episodes were worth drawing from. ``[S]``"""

    open_start: Array
    """Logical index each stream's unfinished episode began at. ``[S]``"""


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
    # replay holds sits at or above `committed_capacity - max_episode_length`
    # once the ring has wrapped, and never reaches `committed_capacity` itself.
    # A warmup this buffer does not stay strictly above could be met and then
    # unmet, and a learner that stops learning because an episode fell off a
    # boundary is a failure nobody would look for.
    steady_state = add_batch_size * (committed_capacity - max_episode_length)
    if min_length >= steady_state:
        raise ValueError(
            f"a warmup of {min_length} transitions is not one this buffer stays "
            f"above: with episodes of up to {max_episode_length} it holds at "
            f"least {steady_state} finished transitions and not dependably more"
        )
    if minimum_episode_length is None:
        minimum_episode_length = sample_sequence_length
    if minimum_episode_length < 1:
        raise ValueError("an episode worth drawing holds at least one transition")

    # An episode occupies at least one transition, so a stream's transitions
    # cannot describe more episodes than there are transitions. Sizing the
    # index at that bound is what keeps it from ever being the binding
    # constraint -- an index that wrapped first would silently drop episodes
    # that are still stored, and the sampler would have no way to know it had.
    episodes_per_stream = time_capacity
    episode_capacity = add_batch_size * episodes_per_stream
    if sample_batch_size > episode_capacity:
        raise ValueError(
            f"a minibatch of {sample_batch_size} episodes cannot be drawn without "
            f"replacement from a buffer that holds at most {episode_capacity}"
        )
    # Halvings that close a range of `episodes_per_stream` positions.
    halvings = int(math.ceil(math.log2(episodes_per_stream + 1)))

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
        def records() -> Array:
            return jnp.zeros((add_batch_size, episodes_per_stream), dtype=jnp.int32)

        def counts() -> Array:
            return jnp.zeros((add_batch_size,), dtype=jnp.int32)

        return EpisodeWindowBufferState(
            trajectory=trajectory.init(experience),
            episodes=EpisodeIndexState(
                start=records(),
                committed=counts(),
                drawable_start=records(),
                drawable_length=records(),
                drawable=counts(),
                open_start=counts(),
            ),
            written=jnp.asarray(0, dtype=jnp.int32),
        )

    def add_fn(
        state: EpisodeWindowBufferState, transition: Any
    ) -> EpisodeWindowBufferState:
        """Store a timestep per stream, and record every episode it ended."""

        index = state.episodes
        stream = jnp.arange(add_batch_size, dtype=jnp.int32)
        done = jnp.asarray(transition.done).reshape((add_batch_size,)).astype(jnp.bool_)
        # The transition being stored sits at logical ``written``, so an
        # episode ending on it runs from its start through that index.
        length = state.written + 1 - index.open_start
        # An ending takes the next position in its own stream's ring, so
        # several streams ending on the same timestep cannot contend for a
        # slot. A stream that did not end is aimed one past the ring, which a
        # scatter in drop mode discards -- one *past* it and not at -1, which
        # is a valid index in JAX as it is in Python and would land on the
        # last record of every stream that did not end an episode.
        missing = episodes_per_stream
        slot = jnp.where(done, index.committed % episodes_per_stream, missing)
        # Long enough to draw a window from is a property of the episode,
        # settled here and never asked again.
        drawable = done & (length >= minimum_episode_length)
        drawable_slot = jnp.where(
            drawable, index.drawable % episodes_per_stream, missing
        )
        recorded = EpisodeIndexState(
            start=index.start.at[stream, slot].set(index.open_start, mode="drop"),
            committed=index.committed + done.astype(jnp.int32),
            drawable_start=index.drawable_start.at[stream, drawable_slot].set(
                index.open_start, mode="drop"
            ),
            drawable_length=index.drawable_length.at[stream, drawable_slot].set(
                length, mode="drop"
            ),
            drawable=index.drawable + drawable.astype(jnp.int32),
            open_start=jnp.where(done, state.written + 1, index.open_start),
        )
        return EpisodeWindowBufferState(
            trajectory=add_timestep(state.trajectory, transition),
            episodes=recorded,
            written=state.written + 1,
        )

    def oldest_stored_fn(state: EpisodeWindowBufferState) -> Array:
        """The logical time an episode has to begin at to still be replay's.

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

        # Strictly inside the capacity, not up to it. The published
        # `RememberEpisode` pushes and then pops while `size >= capacity`, so
        # what it holds after a commit is always *fewer* than `capacity`
        # transitions: an episode that would make the total exactly the
        # capacity is the one that goes.
        reserved = state.episodes.open_start - committed_capacity + 1
        physical = state.written - time_capacity
        return jnp.maximum(reserved, physical)

    def first_stored_fn(records: Array, count: Array, threshold: Array) -> Array:
        """Where one stream's run of still-stored records begins.

        The records of a stream are appended in the order its episodes ended,
        so their starts increase with logical position and a threshold on the
        start is answered by halving the range. ``count`` comes back when the
        stream holds nothing, which reads as an empty run.

        The search is over logical positions and reads the ring at
        ``position % capacity``. Positions below ``count - capacity`` describe
        episodes the index has since overwritten; they are excluded from the
        range rather than probed, and no episode still stored is among them,
        because a stream stores no more episodes than it has transitions.
        """

        def halve(_: Any, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
            low, high = bounds
            middle = low + (high - low) // 2
            reaches = records[middle % episodes_per_stream] >= threshold
            # A range that has already closed must stay closed: at
            # ``low == high`` the probe reads a record outside the range, and
            # acting on its answer would move one bound past the other.
            searching = low < high
            return (
                jnp.where(searching & ~reaches, middle + 1, low),
                jnp.where(searching & reaches, middle, high),
            )

        low = jnp.maximum(count - episodes_per_stream, 0)
        return jax.lax.fori_loop(0, halvings, halve, (low, count))[0]

    first_stored = jax.vmap(first_stored_fn)

    def retained_fn(state: EpisodeWindowBufferState) -> Array:
        """Transitions of finished episodes replay is currently holding.

        Read off the boundary rather than summed over lengths: logical time is
        contiguous, so a stream's stored episodes tile the interval from the
        oldest one's start up to where its open episode began, and what replay
        holds is the width of that interval.
        """

        index = state.episodes
        stream = jnp.arange(add_batch_size, dtype=jnp.int32)
        first = first_stored(index.start, index.committed, oldest_stored_fn(state))
        oldest = index.start[stream, first % episodes_per_stream]
        return jnp.sum(jnp.where(first < index.committed, index.open_start - oldest, 0))

    def eligible_fn(state: EpisodeWindowBufferState) -> tuple[Array, Array]:
        """Per stream, where the episodes an update may draw begin, and how many.

        Still stored, and long enough to draw a window from -- the second half
        settled when the episode ended, so all that is left to ask is where the
        first half begins.
        """

        index = state.episodes
        first = first_stored(
            index.drawable_start, index.drawable, oldest_stored_fn(state)
        )
        return first, index.drawable - first

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

        _, available = eligible_fn(state)
        # Strictly greater, which is how the published loop spells it:
        # `memory_size() > memory_threshold`, so a buffer holding exactly the
        # threshold is still warming up.
        return (retained_fn(state) > min_length) & jnp.any(available > 0)

    def draw_episodes_fn(key: Key, total: Array) -> tuple[Array, Array]:
        """``B`` of the first ``total`` ranks, uniform and without replacement.

        Floyd's algorithm: for each ``j`` from ``total - B`` up to
        ``total - 1``, draw a rank in ``[0, j]`` and keep ``j`` instead if that
        rank is already held. Every subset of size ``B`` comes out equally
        likely, in ``B`` draws, without a score vector, a permutation, or
        anything else the size of the buffer -- which is the point, since this
        runs once per environment transition and ``total`` is on the order of
        replay's capacity.

        Fewer eligible episodes than ``B`` is the published loop's shrinking
        minibatch, and needs no second branch: the steps whose ``j`` would be
        negative are exactly the rows that go unfilled, and the rest run
        Floyd's for ``total`` out of ``total``, which is every episode there is.
        """

        keys = jax.random.split(key, sample_batch_size + 1)

        def take(carry: tuple[Array, Array], step: Any) -> tuple[Any, None]:
            held, filled = carry
            position, step_key = step
            bound = total - sample_batch_size + position
            drawing = bound >= 0
            candidate = jax.random.randint(
                step_key, (), 0, jnp.maximum(bound + 1, 1), dtype=jnp.int32
            )
            seen = jnp.any(filled & (held == candidate))
            return (
                held.at[position].set(
                    jnp.where(drawing, jnp.where(seen, bound, candidate), 0)
                ),
                filled.at[position].set(drawing),
            ), None

        (held, filled), _ = jax.lax.scan(
            take,
            (
                jnp.zeros((sample_batch_size,), dtype=jnp.int32),
                jnp.zeros((sample_batch_size,), dtype=jnp.bool_),
            ),
            (jnp.arange(sample_batch_size, dtype=jnp.int32), keys[:sample_batch_size]),
        )
        # Floyd's fills the batch in an order that carries information -- the
        # last row is the likeliest to hold the last rank -- so which row a
        # drawn episode lands in is decided here rather than left to the
        # algorithm's bookkeeping. Unfilled rows sort to the end, where a
        # scoring sampler's rejected rows also sat.
        place = jax.random.uniform(keys[sample_batch_size], (sample_batch_size,))
        order = jnp.argsort(jnp.where(filled, place, place + 2.0))
        return held[order], filled[order]

    def sample_fn(state: EpisodeWindowBufferState, key: Key) -> EpisodeWindowSample:
        episode_key, offset_key = jax.random.split(key)
        index = state.episodes

        first, available = eligible_fn(state)
        rank, batch_valid = draw_episodes_fn(episode_key, jnp.sum(available))
        # A rank names an episode among the eligible ones of every stream laid
        # end to end, so which stream it belongs to is where the running totals
        # put it, and its position in that stream is what is left of the rank.
        boundary = jnp.cumsum(available)
        stream = jnp.searchsorted(boundary, rank, side="right")
        # A row that stands for no episode reads stream zero, wherever that
        # stream's records happen to begin; it is masked everywhere it is used,
        # and the read has to land somewhere in bounds.
        stream = jnp.where(batch_valid, jnp.clip(stream, 0, add_batch_size - 1), 0)
        position = first[stream] + rank - (boundary[stream] - available[stream])
        slot = position % episodes_per_stream
        start = index.drawable_start[stream, slot]
        length = index.drawable_length[stream, slot]

        # How many places a window can begin. An episode shorter than the
        # window offers exactly one -- its beginning -- and the steps past its
        # ending are masked out below rather than drawn from.
        places = jnp.maximum(length - sample_sequence_length + 1, 1)
        offset = jax.random.randint(
            offset_key, (sample_batch_size,), 0, places, dtype=jnp.int32
        )

        step = jnp.arange(sample_sequence_length, dtype=jnp.int32)
        logical = start[:, None] + offset[:, None] + step[None, :]
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
