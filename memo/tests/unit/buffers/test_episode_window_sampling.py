"""What the episode sampler costs, and that the cost bought nothing away.

``test_drqn_replay.py`` says what a draw *means* -- uniform over episodes,
without replacement, a window inside one episode, a minibatch that shrinks
rather than repeats. This file is about the two things that are not visible in
a single draw.

The first is price. A learner that updates once per environment transition
runs this sampler five million times in a 5M run, so anything in it that grows
with replay's capacity is a cost the run pays five million times over, and
pays silently: no metric records how long a step took, so a sampler that scans
a 400k index shows up as a benchmark that never reaches its first evaluation
rather than as a number anyone can read. The claim is therefore made against
the jaxpr, where "work that grows with the buffer" is something a test can see
directly, and not against a wall clock, which would be measuring this machine.

The second is that the price was not paid out of the distribution. A faster
sampler that draws a slightly different subset is not a faster sampler; it is
a different algorithm with the old one's name. So the draws are compared
against the scoring sampler this replaces -- Gumbel top-k over the eligible
set, which is exactly uniform-without-replacement -- and against the uniform
distribution over subsets that both are supposed to be.
"""

from __future__ import annotations

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from memorax.buffers import make_uniform_episode_window_buffer
from memorax.utils.typing import Array, Key

DRAWS = 4000


class Step(struct.PyTreeNode):
    """The least a stored transition can be: a label, and whether it ended one.

    The buffer keeps an opaque tree and reads ``done`` off it, so a test of the
    index needs nothing that belongs to a learner.
    """

    mark: Array
    """Which episode the transition came from, so a window can be traced back."""

    done: Array


def buffer_of(
    *,
    capacity: int = 64,
    window: int = 2,
    batch: int = 4,
    horizon: int = 8,
    warmup: int = 4,
    streams: int = 1,
    minimum_episode_length: int | None = None,
):
    return make_uniform_episode_window_buffer(
        max_length=capacity,
        min_length=warmup,
        sample_batch_size=batch,
        sample_sequence_length=window,
        add_batch_size=streams,
        max_episode_length=horizon,
        minimum_episode_length=minimum_episode_length,
    )


def played(buffer, mark: np.ndarray, done: np.ndarray):
    """Transitions added one at a time, as the learner adds them.

    ``mark`` and ``done`` are ``[transitions, streams]``; the buffer sees one
    timestep per stream per call, which is the only shape it takes.
    """

    state = buffer.init(
        Step(mark=jnp.asarray(0, dtype=jnp.int32), done=jnp.asarray(False))
    )
    add = jax.jit(buffer.add)
    for step in range(len(mark)):
        state = add(
            state,
            Step(
                mark=jnp.asarray(mark[step], dtype=jnp.int32),
                done=jnp.asarray(done[step]),
            ),
        )
    return state


def filled(buffer, lengths, streams: int = 1):
    """Episodes of the given lengths, laid end to end on every stream."""

    mark = np.concatenate(
        [np.full(length, episode) for episode, length in enumerate(lengths)]
    )
    done = np.concatenate([np.arange(length) == length - 1 for length in lengths])
    return played(
        buffer, np.tile(mark[:, None], streams), np.tile(done[:, None], streams)
    )


def minibatches(buffer, state, keys: int = DRAWS, seed: int = 0):
    """Many draws in one compilation, because a per-draw trace is the slow part."""

    return jax.jit(jax.vmap(buffer.sample, in_axes=(None, 0)))(
        state, jax.random.split(jax.random.key(seed), keys)
    )


def episodes_drawn(sample) -> np.ndarray:
    """``[draws, B]``: the episode in each row, or -1 where there is none."""

    mark = np.asarray(sample.experience.mark)[..., 0]
    return np.where(np.asarray(sample.batch_valid), mark, -1)


def subset_frequencies(rows: np.ndarray, over: int, batch: int) -> np.ndarray:
    """How often each B-subset of the eligible episodes came up."""

    order = {
        subset: place for place, subset in enumerate(combinations(range(over), batch))
    }
    counts = np.zeros(len(order))
    for row in rows:
        counts[order[tuple(sorted(row.tolist()))]] += 1
    return counts / len(rows)


def gumbel_top_k(key: Key, total: int, batch: int, over: int) -> Array:
    """The sampler this replaces, kept as the oracle it was correct as.

    Perturbing every eligible episode's equal log-weight by an independent
    Gumbel and taking the largest B is a uniform subset without replacement.
    It is also a pass over ``over`` slots per draw, which is the whole reason
    it is here as a reference and not as the implementation.
    """

    scores = jnp.where(
        jnp.arange(over) < total, jax.random.gumbel(key, (over,)), -jnp.inf
    )
    return jax.lax.top_k(scores, batch)[1]


# ------------------------------------------------------------------- the price
def equation_sizes(jaxpr) -> list[int]:
    """Every value produced anywhere in a jaxpr, including inside its loops.

    A scan or a ``fori_loop`` is one equation at the top level and the body is
    where the work is, so a walk that stopped at the outer equation would
    report that a sampler which scans the whole index costs nothing.
    """

    sizes = []
    for equation in jaxpr.eqns:
        sizes += [getattr(var.aval, "size", 0) for var in equation.outvars]
        for parameter in equation.params.values():
            for nested in parameter if isinstance(parameter, tuple) else (parameter,):
                inner = getattr(nested, "jaxpr", nested)
                if hasattr(inner, "eqns"):
                    sizes += equation_sizes(inner)
    return sizes


def largest_intermediate(function, *arguments) -> int:
    return max(equation_sizes(jax.make_jaxpr(function)(*arguments).jaxpr))


def test_a_draw_costs_the_minibatch_and_not_the_buffer():
    """The claim issue 62 asks for, stated where it can be checked.

    Two buffers differing only in capacity, by a factor of sixty-four. If any
    step of the draw were a pass over the index -- a score per episode, a
    permutation, a mask -- the larger buffer's sampler would allocate sixty-
    four times as much, and it is the *largest* value produced that says so,
    because that is the one whose size the capacity would be in.
    """

    small = buffer_of(capacity=64, horizon=8, batch=4, window=2)
    large = buffer_of(capacity=4096, horizon=8, batch=4, window=2)
    key = jax.random.key(0)

    cost = [
        largest_intermediate(buffer.sample, filled(buffer, (4,) * 6), key)
        for buffer in (small, large)
    ]

    assert cost[0] == cost[1], cost
    # And not merely equal: small enough that no reading of it is a pass over
    # even the smaller of the two indices.
    assert cost[0] < 64, cost


def test_asking_whether_a_draw_is_possible_costs_no_more():
    """``can_sample`` runs every step too, on the branch that does not learn.

    It is read once per transition through the ``lax.cond`` that decides
    whether to update, so a pass over the index here is the same cost in the
    same place -- and it is also in the first trace, where it lands on
    compilation before it lands on any step.
    """

    small = buffer_of(capacity=64, horizon=8)
    large = buffer_of(capacity=4096, horizon=8)

    cost = [
        largest_intermediate(buffer.can_sample, filled(buffer, (4,) * 6))
        for buffer in (small, large)
    ]

    assert cost[0] == cost[1], cost
    assert cost[0] < 64, cost


# ------------------------------------------------------ and what it bought away
def test_a_minibatch_is_a_uniform_subset_of_the_eligible_episodes():
    """Every subset equally likely, which is what "without replacement" means.

    Five episodes and a minibatch of three is ten subsets, few enough to count
    all of them. Floyd's algorithm has the property exactly, so this is a test
    that it was implemented rather than approximated -- a sampler that drew
    with replacement and repaired collisions afterwards would over-represent
    whichever episodes it repaired towards.
    """

    buffer = buffer_of(batch=3)
    rows = episodes_drawn(minibatches(buffer, filled(buffer, (2,) * 5)))

    assert not np.any(rows < 0)
    seen = subset_frequencies(rows, over=5, batch=3)
    assert abs(seen - 0.1).max() < 0.02, seen


def test_the_draws_agree_with_the_sampler_they_replace():
    """Same distribution, not merely a defensible one.

    The Gumbel top-k this replaces was uniform over subsets, and the test
    above says the new draw is too, so this compares the two empirical
    distributions directly: if the replacement had traded exactness for speed
    anywhere -- a bounded retry, a shortcut when the buffer is nearly empty --
    the two would separate here.
    """

    buffer = buffer_of(batch=3)
    rows = episodes_drawn(minibatches(buffer, filled(buffer, (2,) * 5)))
    reference = np.asarray(
        jax.jit(jax.vmap(lambda key: gumbel_top_k(key, total=5, batch=3, over=12)))(
            jax.random.split(jax.random.key(1), DRAWS)
        )
    )

    mine = subset_frequencies(rows, over=5, batch=3)
    theirs = subset_frequencies(reference, over=5, batch=3)

    # Total variation between two four-thousand-draw histograms over ten
    # cells; sampling noise alone is a little under one percent.
    assert 0.5 * np.abs(mine - theirs).sum() < 0.04, (mine, theirs)


def test_which_row_an_episode_lands_in_carries_nothing():
    """A row is a slot in a minibatch, not a rank.

    Floyd's fills its output in an order that says something about what it
    drew -- the last row is the likeliest to hold the last rank -- and a
    learner that reduces over rows would not notice, right up until one that
    weights them, splits them across devices, or reads row zero does. The
    scoring sampler's rows were exchangeable, so these are too.
    """

    buffer = buffer_of(batch=3)
    rows = episodes_drawn(minibatches(buffer, filled(buffer, (2,) * 5)))

    for row in range(3):
        _, counts = np.unique(rows[:, row], return_counts=True)
        assert abs(counts / len(rows) - 0.2).max() < 0.03, counts


def test_no_episode_is_drawn_twice_in_one_minibatch():
    """Over thousands of minibatches, not the handful a spot check affords."""

    buffer = buffer_of(batch=4)
    rows = episodes_drawn(minibatches(buffer, filled(buffer, (2,) * 6)))

    assert not np.any(rows < 0)
    assert np.all(np.diff(np.sort(rows, axis=1), axis=1) > 0)


def test_fewer_eligible_episodes_than_rows_fills_the_rows_there_are():
    """The shrinking minibatch, at the two edges the fast path has to handle.

    Floyd's is stated for ``B`` out of ``N >= B``; here it is run for the rows
    whose draw exists and the rest are marked unfilled, so the cases worth
    naming are ``N`` just under ``B`` and ``N`` at zero -- the second reachable
    whenever a learner samples without asking ``can_sample`` first.
    """

    buffer = buffer_of(batch=4)

    two = episodes_drawn(minibatches(buffer, filled(buffer, (2,) * 2), keys=64))
    assert np.all(np.sort(two, axis=1) == np.asarray([-1, -1, 0, 1]))

    none = episodes_drawn(minibatches(buffer, filled(buffer, (1,)), keys=64))
    assert np.all(none < 0)


def test_the_unfilled_rows_are_the_last_ones():
    """Where a scoring sampler's rejected rows sat, so a reader sees no gap."""

    buffer = buffer_of(batch=4)
    valid = np.asarray(
        minibatches(buffer, filled(buffer, (2,) * 2), keys=64).batch_valid
    )

    assert np.all(valid == np.asarray([True, True, False, False]))


def test_the_same_key_draws_the_same_minibatch():
    """A seeded run is reproducible, including through the scan Floyd's runs in."""

    buffer = buffer_of(batch=3)
    state = filled(buffer, (2,) * 5)

    once = buffer.sample(state, jax.random.key(7))
    again = buffer.sample(state, jax.random.key(7))
    other = buffer.sample(state, jax.random.key(8))

    np.testing.assert_array_equal(
        np.asarray(once.experience.mark), np.asarray(again.experience.mark)
    )
    assert not np.array_equal(
        np.asarray(once.experience.mark), np.asarray(other.experience.mark)
    )


# ----------------------------------------------- membership, over a long enough run
def test_an_episode_too_short_to_draw_is_still_experience_replay_holds():
    """Two indices, one meaning each, and the difference is observable.

    An episode below ``minimum_episode_length`` offers no window, so it is
    never drawn -- but its transitions are in the ring, they are what the ring
    evicts to make room for, and the warmup is a count of what replay holds.
    Recording only the drawable episodes would make ``retained`` a count of
    the long ones and start learning late by however many short episodes the
    task produced; a policy that ends episodes early at the beginning of
    training produces a great many.
    """

    buffer = buffer_of(window=4, batch=6, capacity=64, horizon=8)
    state = filled(buffer, (1, 4) * 6)

    assert int(state.written) == 30
    assert int(buffer.retained(state)) == 30

    rows = episodes_drawn(minibatches(buffer, state, keys=256))
    assert set(rows.ravel().tolist()) == {1, 3, 5, 7, 9, 11}


def test_the_index_holds_up_once_it_has_wrapped_itself():
    """Forty episodes through an index sized for twelve.

    The index rings wrap far sooner than the transition ring when episodes are
    short, and a search that reached past its own capacity would read a record
    describing an episode that has been overwritten twice over. The range
    searched is bounded by what the index still holds, and what it still holds
    is a superset of what replay does, which is what makes the bound safe.
    """

    buffer = buffer_of(
        capacity=8, horizon=4, window=1, batch=4, warmup=3, minimum_episode_length=1
    )
    state = filled(buffer, (1,) * 40)

    # Eight kept, strictly: an episode that would bring the total to exactly
    # eight is the one that goes, so replay holds the last seven.
    assert int(buffer.retained(state)) == 7
    rows = episodes_drawn(minibatches(buffer, state, keys=256))
    assert set(rows.ravel().tolist()) == set(range(33, 40))


def test_a_stream_that_ends_nothing_holds_up_the_others():
    """Streams evict on their own commits, so their runs begin in different places.

    Two streams whose episodes end on different timesteps have different
    thresholds and therefore different eligible runs, which is the case a
    single shared index answers with a mask and a per-stream index answers
    with two ranges. Nothing here may draw from the stream that is still
    playing its first episode.
    """

    buffer = buffer_of(capacity=64, horizon=8, window=2, batch=8, streams=2)
    state = buffer.init(
        Step(mark=jnp.asarray(0, dtype=jnp.int32), done=jnp.asarray(False))
    )
    for index in range(12):
        state = buffer.add(
            state,
            Step(
                mark=jnp.asarray([index // 4, 100], dtype=jnp.int32),
                # The first stream ends an episode every fourth transition;
                # the second never ends one at all.
                done=jnp.asarray([index % 4 == 3, False]),
            ),
        )

    rows = episodes_drawn(minibatches(buffer, state, keys=128))

    assert set(rows.ravel().tolist()) == {-1, 0, 1, 2}
    assert int(jnp.sum(minibatches(buffer, state, keys=1).batch_valid)) == 3


def scripted(rng, streams: int, horizon: int, transitions: int):
    """Episodes of random lengths on each stream, ended wherever they end.

    Streams step in lockstep and their episodes do not, so the thresholds the
    index is searched against sit in different places on each one, and the
    last episode of each is still being played.
    """

    mark = np.zeros((transitions, streams), dtype=np.int32)
    done = np.zeros((transitions, streams), dtype=bool)
    for stream in range(streams):
        step, episode = 0, 0
        while step < transitions:
            length = int(rng.integers(1, horizon + 1))
            # A label that is unique across streams, so a drawn window names
            # exactly one episode of the run.
            mark[step : step + length, stream] = 100 * stream + episode
            if step + length <= transitions:
                done[step + length - 1, stream] = True
            step, episode = step + length, episode + 1
    return mark, done


def expected(mark, done, *, capacity, streams, horizon, minimum):
    """What the published rule holds and offers, worked out in numpy.

    The buffer answers both by searching an index; this answers them by
    walking the run, which is the same statement with none of the structure
    that could be wrong in the same way.
    """

    committed_capacity = capacity // streams
    written = len(mark)
    retained, eligible = 0, set()
    for stream in range(streams):
        endings = np.flatnonzero(done[:, stream])
        starts = np.concatenate(([0], endings[:-1] + 1))
        lengths = endings - starts + 1
        open_start = int(endings[-1]) + 1 if len(endings) else 0
        # The reserved bound, and the physical one underneath it.
        threshold = max(
            open_start - committed_capacity + 1,
            written - committed_capacity - horizon,
        )
        for start, length, ending in zip(starts, lengths, endings):
            if start < threshold:
                continue
            retained += int(length)
            if length >= minimum:
                eligible.add(int(mark[ending, stream]))
    return retained, eligible


def test_membership_matches_the_rule_worked_out_by_hand():
    """Completion, eviction and the length bar, over runs nobody chose.

    The tests above name the case each is about, which is what makes them
    readable and also what makes them a set of cases someone thought of. This
    one plays random episodes through several shapes of buffer -- one stream
    and two, wrapped several times over, short episodes among long ones -- and
    checks the two questions the index exists to answer against a model of the
    published rule: how much replay holds, and which episodes it will offer.
    """

    rng = np.random.default_rng(0)
    shapes = (
        dict(capacity=32, horizon=6, window=2, batch=4, warmup=2, streams=1),
        dict(capacity=64, horizon=6, window=3, batch=5, warmup=4, streams=2),
        dict(
            capacity=24,
            horizon=4,
            window=1,
            batch=3,
            warmup=2,
            streams=1,
            minimum_episode_length=1,
        ),
        dict(capacity=48, horizon=8, window=4, batch=8, warmup=3, streams=2),
    )

    for shape in shapes:
        buffer = buffer_of(**shape)
        mark, done = scripted(
            rng, shape["streams"], shape["horizon"], 3 * shape["capacity"]
        )
        state = played(buffer, mark, done)

        held, offered = expected(
            mark,
            done,
            capacity=shape["capacity"],
            streams=shape["streams"],
            horizon=shape["horizon"],
            minimum=shape.get("minimum_episode_length") or shape["window"],
        )

        assert int(buffer.retained(state)) == held, shape
        assert bool(buffer.can_sample(state)) is (
            held > shape["warmup"] and bool(offered)
        ), shape

        sample = minibatches(buffer, state, keys=512)
        rows = episodes_drawn(sample)
        # Every minibatch is as large as it can be, and never larger.
        assert np.all(
            np.asarray(sample.batch_valid).sum(axis=1)
            == min(shape["batch"], len(offered))
        ), shape
        # Nothing outside the eligible set is ever offered, and given enough
        # draws nothing inside it is withheld.
        assert set(rows.ravel().tolist()) - {-1} == offered, shape
