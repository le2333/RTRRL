"""Which windows the published random update draws, and what it hands the loss.

    e_1..e_B ~ without replacement over completed episodes
    s_i      ~ U{0 .. L_{e_i} - t}
    window_i  = the t transitions from s_i

Each line of that is a separate claim and gets a separate test. The episode is
the unit, so a long one is not drawn more often and the same one is not drawn
twice in a minibatch; the window lies inside it, so the gradient crosses the t
the learner declares rather than however much of an episode happened to be
left; and when fewer than B episodes are eligible the minibatch shrinks, rather
than being padded with repeats or -- worse -- with position zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms.drqn import ReplayTransition, SelectedLearning, learner_sequence
from memorax.buffers import make_uniform_episode_window_buffer

EPISODE_LENGTH = 8
TRUNCATION = 2
# Two lengths, so that "uniform over episodes" and "uniform over positions" are
# different distributions and a test can tell them apart.
LENGTHS = (4, 8, 4, 8)


def stream(lengths):
    """Episodes laid end to end, each transition labelled with where it came from.

    The observation carries ``(episode, index within episode)`` so that a drawn
    window can be traced back to the episode it was taken from, which is what
    the sampling claims are about.
    """

    observation, episode_start, done = [], [], []
    for episode, length in enumerate(lengths):
        for index in range(length):
            observation.append([float(episode), float(index)])
            episode_start.append(index == 0)
            done.append(index == length - 1)
    return observation, episode_start, done


def transition(observation, episode_start, done, streams=1):
    return ReplayTransition(
        observation=jnp.tile(jnp.asarray([observation]), (streams, 1)),
        episode_start=jnp.full((streams,), episode_start),
        action=jnp.zeros((streams,), dtype=jnp.int32),
        reward=jnp.zeros((streams,)),
        next_observation=jnp.tile(jnp.asarray([observation]), (streams, 1)),
        done=jnp.full((streams,), done),
        terminal=jnp.full((streams,), done),
    )


def filled(buffer, lengths, streams=1):
    observation, episode_start, done = stream(lengths)
    state = buffer.init(
        jax.tree.map(
            lambda value: value[0], transition(observation[0], True, False, streams)
        )
    )
    for values in zip(observation, episode_start, done):
        state = buffer.add(state, transition(*values, streams=streams))
    return state


def stored(
    lengths=LENGTHS,
    truncation=TRUNCATION,
    capacity=64,
    batch_size=8,
    minimum_episode_length=None,
    window=None,
    horizon=EPISODE_LENGTH,
):
    buffer = make_uniform_episode_window_buffer(
        max_length=capacity,
        min_length=4,
        sample_batch_size=batch_size,
        sample_sequence_length=truncation if window is None else window,
        add_batch_size=1,
        max_episode_length=horizon,
        minimum_episode_length=minimum_episode_length,
    )
    return buffer, filled(buffer, lengths)


def draws(buffer, state, keys=32):
    """Many minibatches, kept as minibatches so a within-batch claim can be made."""

    return [buffer.sample(state, jax.random.key(seed)) for seed in range(keys)]


def stacked(samples):
    """The same draws flattened, for claims about one window at a time."""

    return jax.tree.map(lambda *blocks: jnp.concatenate(blocks, axis=0), *samples)


def labels(sample):
    """``(episode, index)`` for every window in a batch, as integers."""

    marks = np.asarray(sample.experience.observation)
    return marks[..., 0].astype(int), marks[..., 1].astype(int)


def only_valid(sample):
    """The windows a batch actually drew, dropping the rows it could not fill."""

    keep = np.asarray(sample.batch_valid)
    episode, index = labels(sample)
    return episode[keep], index[keep]


# --------------------------------------------- the window lies inside one episode
def test_every_drawn_window_lies_inside_one_episode():
    """No window crosses an ending, so none is cut short by the validity mask."""

    buffer, state = stored()

    windows = stacked(draws(buffer, state))
    episode, index = only_valid(windows)

    assert np.all(episode[:, 0][:, None] == episode)
    assert np.all(np.diff(index, axis=1) == 1)
    # An ending may fall on the window's last transition and nowhere earlier.
    assert not np.any(np.asarray(windows.experience.done)[:, :-1])


def test_every_drawn_window_carries_the_full_truncation():
    """The nominal t and the number of transitions the gradient crosses agree.

    Cutting a window at an ending would make the effective truncation a
    function of where in the episode the window landed, so a learner declaring
    TBPTT(64) would in places be performing TBPTT(5).
    """

    buffer, state = stored()

    windows = stacked(draws(buffer, state))
    keep = np.asarray(windows.batch_valid)

    assert windows.valid.shape == (keep.shape[0], TRUNCATION)
    assert np.all(np.asarray(windows.valid)[keep])


def test_an_episode_shorter_than_the_truncation_is_never_drawn():
    """Not a shortened window: no window at all, which is the honest answer."""

    buffer, state = stored(lengths=(3, 6, 3), truncation=4)

    episode, _ = only_valid(stacked(draws(buffer, state)))

    assert set(episode.ravel().tolist()) == {1}


def test_a_start_is_uniform_over_the_places_a_window_fits():
    """L - t + 1 places, each as likely as any other."""

    buffer, state = stored(lengths=(6,), truncation=2, batch_size=1)

    _, index = only_valid(stacked(draws(buffer, state, keys=400)))

    seen, counts = np.unique(index[:, 0], return_counts=True)
    assert seen.tolist() == [0, 1, 2, 3, 4]
    assert counts.min() / counts.max() > 0.5, counts


# ------------------------------------------------- the episode is the unit drawn
def test_an_episode_is_not_drawn_more_often_for_being_longer():
    """Uniform over episodes, which is what the published update draws.

    Uniform over stored positions would weight an episode by how many starts it
    offers -- here seven against three -- and on a task whose episodes grow as
    the policy improves that is a drift in what gets replayed rather than a
    fixed sampling rule.
    """

    buffer, state = stored(batch_size=1)

    episode, _ = only_valid(stacked(draws(buffer, state, keys=400)))

    long_episodes = np.isin(episode[:, 0], [1, 3]).mean()
    # Two long and two short, so half. Weighting by starts would give
    # 14/20 = 0.7, which is far outside this band.
    assert 0.4 < long_episodes < 0.6, long_episodes


def test_a_minibatch_draws_each_episode_at_most_once():
    """Without replacement, which is the half a per-position weight cannot say.

    Weighting positions can make each episode equally likely on one draw. It
    cannot make B draws a subset, because that is a statement about the draws
    taken together. With four episodes and a minibatch of four, drawing with
    replacement would repeat one in almost every batch.
    """

    buffer, state = stored(batch_size=4)

    for sample in draws(buffer, state):
        episode, _ = only_valid(sample)
        first = episode[:, 0]
        assert len(first) == 4
        assert len(set(first.tolist())) == 4, first


def test_the_minibatch_shrinks_to_the_episodes_there_are():
    """Fewer eligible than B is a smaller minibatch, not a padded one.

    The published loop draws ``min(episodes, B)``. Filling the rest with
    repeats would quietly reweight whatever it repeated; filling them with
    position zero would train on a window nobody drew.
    """

    buffer, state = stored(lengths=(4, 4), batch_size=8)

    sample = buffer.sample(state, jax.random.key(0))
    episode, _ = only_valid(sample)

    assert int(jnp.sum(sample.batch_valid)) == 2
    assert not np.any(np.asarray(sample.valid)[~np.asarray(sample.batch_valid)])
    assert sorted(episode[:, 0].tolist()) == [0, 1]


def test_episodes_are_drawn_across_streams_without_the_stream_being_sampled():
    """A stream is an episode's address, not a second thing to draw."""

    buffer = make_uniform_episode_window_buffer(
        max_length=64,
        min_length=4,
        sample_batch_size=8,
        sample_sequence_length=TRUNCATION,
        add_batch_size=4,
        max_episode_length=EPISODE_LENGTH,
    )
    state = filled(buffer, (4, 4), streams=4)

    # Two episodes on each of four streams is eight, and a minibatch of eight
    # is all of them: every stream contributes, and none contributes twice as
    # much for being a stream.
    sample = buffer.sample(state, jax.random.key(0))
    assert int(jnp.sum(sample.batch_valid)) == 8
    streams = np.asarray(sample.experience.observation)[..., 0]
    episode, _ = only_valid(sample)
    assert sorted(episode[:, 0].tolist()) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert streams.shape == (8, TRUNCATION)


# ------------------------------------------- nothing to draw is said, not faked
def test_a_buffer_holding_no_finished_episode_says_it_cannot_sample():
    """Length and eligibility are different questions and both get asked.

    A buffer past its minimum length can be entirely one unfinished episode.
    Reporting on length alone is what lets a sampler be called with nothing to
    draw, and a sampler with nothing to draw has to invent something.
    """

    buffer = make_uniform_episode_window_buffer(
        max_length=64,
        min_length=4,
        sample_batch_size=4,
        sample_sequence_length=TRUNCATION,
        add_batch_size=1,
        max_episode_length=EPISODE_LENGTH,
    )
    observation, episode_start, _ = stream((16,))
    state = buffer.init(
        jax.tree.map(lambda value: value[0], transition(observation[0], True, False))
    )
    for mark, opening in zip(observation, episode_start):
        state = buffer.add(state, transition(mark, opening, False))

    assert bool(buffer.can_sample(state)) is False


def test_the_warmup_counts_finished_episodes_and_not_written_transitions():
    """An episode still being played is not experience replay has.

    Its transitions are in the ring and nothing can draw them, so counting them
    towards the warmup would start learning early by however much of an episode
    happened to be in progress -- the same class of error as letting the window
    length decide when learning starts, one step further in.
    """

    buffer = make_uniform_episode_window_buffer(
        max_length=64,
        min_length=8,
        sample_batch_size=2,
        sample_sequence_length=TRUNCATION,
        add_batch_size=1,
        max_episode_length=EPISODE_LENGTH,
        minimum_episode_length=1,
    )

    # Four transitions ended, four still being played: eight written, four
    # replay can do anything with.
    state = filled(buffer, (4,))
    for index in range(4):
        state = buffer.add(state, transition([9.0, float(index)], index == 0, False))

    assert int(state.written) == 8
    assert int(buffer.retained(state)) == 4
    assert bool(buffer.can_sample(state)) is False

    # The ending is what makes those four transitions replay.
    state = buffer.add(state, transition([9.0, 4.0], False, True))
    assert int(buffer.retained(state)) == 9
    assert bool(buffer.can_sample(state)) is True


def test_the_warmup_is_what_replay_holds_and_not_what_it_has_held():
    """A lifetime count only rises, and replay does not.

    Counting every episode ever finished would keep reporting ready against a
    buffer that had since evicted most of what it counted. Here what replay
    holds stays at one episode however many hundreds have gone through it.
    """

    buffer = make_uniform_episode_window_buffer(
        max_length=8,
        min_length=3,
        sample_batch_size=2,
        sample_sequence_length=2,
        add_batch_size=1,
        max_episode_length=4,
        minimum_episode_length=1,
    )
    state = filled(buffer, (4,) * 10)

    # Forty transitions have been finished; replay is holding four.
    assert int(state.written) == 40
    assert int(buffer.retained(state)) == 4
    assert bool(buffer.can_sample(state)) is True


def test_a_buffer_with_only_a_short_episode_waits_for_a_long_enough_one():
    buffer, state = stored(lengths=(3,), truncation=4)

    assert bool(buffer.can_sample(state)) is False


def test_only_completed_episodes_are_drawn_from():
    """The tail still being played is not a shorter episode; it is not one yet."""

    buffer = make_uniform_episode_window_buffer(
        max_length=64,
        min_length=4,
        sample_batch_size=4,
        sample_sequence_length=TRUNCATION,
        add_batch_size=1,
        max_episode_length=EPISODE_LENGTH,
    )
    observation, episode_start, done = stream((4, 4))
    observation += [[9.0, float(index)] for index in range(4)]
    episode_start += [True, False, False, False]
    done += [False] * 4
    state = buffer.init(
        jax.tree.map(lambda value: value[0], transition(observation[0], True, False))
    )
    for values in zip(observation, episode_start, done):
        state = buffer.add(state, transition(*values))

    episode, _ = only_valid(stacked(draws(buffer, state)))

    assert 9 not in set(episode.ravel().tolist())


def test_the_warmup_is_the_declared_minimum_and_not_the_window_length():
    """When learning may start cannot be a function of how long a window is.

    Flashbax's own readiness rule will not report ready until a whole
    ``sample_sequence_length`` has been written, which is right for drawing a
    fixed-length slice and wrong for drawing an episode. Deferring to it would
    make the warmup ``max(min_length, t)``: two learners with identical replay
    settings would start learning at different times because one asked for a
    longer window, which is the window length reaching out of the sampler and
    into the schedule.
    """

    ready = {}
    for window in (2, 16):
        buffer = make_uniform_episode_window_buffer(
            max_length=64,
            min_length=4,
            sample_batch_size=2,
            sample_sequence_length=window,
            add_batch_size=1,
            max_episode_length=EPISODE_LENGTH,
            minimum_episode_length=1,
        )
        # Two episodes of four: past the declared minimum, and both complete
        # and eligible under either window.
        ready[window] = bool(buffer.can_sample(filled(buffer, (4, 4))))

    assert ready == {2: True, 16: True}


def test_the_padding_past_an_episode_is_zeros_and_not_the_next_episode():
    """A masked step is still unrolled, so what it holds is not nothing.

    The slots after a short episode physically hold the episode that followed
    it. Those steps do not enter the loss, but they do pass through the
    recurrent cell, so leaving them as stored data would make the padding a
    reading of whatever the ring happened to hold -- and, for a slot never
    written, of whatever the buffer was initialised with.
    """

    buffer, state = stored(lengths=(4, 4), minimum_episode_length=1, window=8)

    sample = buffer.sample(state, jax.random.key(0))
    marks = np.asarray(sample.experience.observation)
    valid = np.asarray(sample.valid)

    drawn = valid[np.asarray(sample.batch_valid)]
    assert drawn.shape[0] == 2 and np.all(drawn.sum(axis=1) == 4)
    # Every masked position, in every row, is zero across every stored field.
    for field in jax.tree.leaves(sample.experience):
        blanked = np.asarray(field)[~valid]
        assert not np.any(blanked), field.shape
    # And the episode the padding covers really is a different one, so this is
    # not a test that passes because there was nothing there.
    assert set(marks[valid][:, 0].astype(int).tolist()) == {0, 1}


# ----------------------------------------------------------- once the ring wraps
def wrapped(capacity=40, truncation=TRUNCATION, rounds=15):
    """A ring overwritten several times, so its write head is a seam.

    Episodes of four written round a ring of forty-four -- forty the buffer was
    asked to keep plus four of slack -- more than once, so the newest row and
    the oldest sit next to each other somewhere in the middle of the array and
    an unguarded window could span them.
    """

    return stored(
        lengths=(4,) * rounds,
        truncation=truncation,
        capacity=capacity,
        batch_size=4,
        horizon=4,
    )


def test_the_ring_is_actually_full_and_wrapped_in_this_fixture():
    """Otherwise the tests below would be checking nothing."""

    _, state = wrapped()

    assert bool(state.trajectory.is_full)
    # Forty kept plus four of slack, so the head is somewhere inside forty-four.
    assert 0 < int(state.trajectory.current_index) < 44
    assert int(state.written) == 60


def test_no_window_is_spliced_across_the_write_head():
    """The seam needs no exclusion zone, because logical time has no seam.

    A window comes from inside an episode that is still stored, and every
    logical index from that episode's start up to the head is present. Nothing
    here does arithmetic about which side of the head a position falls on.
    """

    buffer, state = wrapped()

    episode, index = only_valid(stacked(draws(buffer, state)))

    assert np.all(episode[:, 0][:, None] == episode)
    assert np.all(np.diff(index, axis=1) == 1)


def test_an_episode_the_ring_has_overwritten_is_no_longer_drawn():
    """Named exactly, because "roughly the recent ones" is not a guarantee.

    Fifteen episodes of four are sixty transitions and the buffer was asked to
    keep forty, but it keeps *fewer* than forty: the published `RememberEpisode`
    pushes and then pops while `size >= capacity`, so the episode that would
    bring the total to exactly forty is the one that goes. Episode five begins
    at twenty and would make it exactly forty, so it is out; episode six begins
    at twenty-four and is the oldest kept.

    Episodes four and five are both still *physically* there -- the ring is
    forty-four, forty for what replay was asked to keep and four held back for
    whatever is played next -- and neither is offered.
    """

    buffer, state = wrapped()

    episode, _ = only_valid(stacked(draws(buffer, state, keys=64)))

    assert set(episode.ravel().tolist()) == set(range(6, 15))


def test_replay_keeps_fewer_than_its_capacity_and_not_exactly_it():
    """The published pop runs while `size >= capacity`, so the bound is strict.

    Spelled out on the smallest case that shows it: a capacity of eight and
    episodes of four. One episode fits. The second brings the total to exactly
    eight, so the first is dropped and replay holds four again -- never the
    eight a non-strict bound would have kept.
    """

    buffer = make_uniform_episode_window_buffer(
        max_length=8,
        min_length=3,
        sample_batch_size=2,
        sample_sequence_length=2,
        add_batch_size=1,
        max_episode_length=4,
        minimum_episode_length=1,
    )

    one = filled(buffer, (4,))
    assert int(buffer.retained(one)) == 4

    two = filled(buffer, (4, 4))
    assert int(buffer.retained(two)) == 4
    episode, _ = only_valid(stacked(draws(buffer, two, keys=16)))
    assert set(episode.ravel().tolist()) == {1}


def test_the_warmup_is_passed_and_not_merely_reached():
    """`memory_size() > memory_threshold`, so sitting on the threshold is not it."""

    buffer = make_uniform_episode_window_buffer(
        max_length=64,
        min_length=8,
        sample_batch_size=2,
        sample_sequence_length=TRUNCATION,
        add_batch_size=1,
        max_episode_length=EPISODE_LENGTH,
        minimum_episode_length=1,
    )

    exactly = filled(buffer, (4, 4))
    assert int(buffer.retained(exactly)) == 8
    assert bool(buffer.can_sample(exactly)) is False

    one_more = buffer.add(filled(buffer, (4, 4)), transition([9.0, 0.0], True, True))
    assert int(buffer.retained(one_more)) == 9
    assert bool(buffer.can_sample(one_more)) is True


def test_an_episode_being_played_evicts_nothing_that_could_be_drawn():
    """Replay holds still for the duration of an episode, which is the point.

    An open episode's transitions advance the ring's head. Measuring "still
    stored" from the head would let them push the oldest finished episodes out
    one at a time as the episode went on, so *which* episodes an update could
    draw would depend on how far into the current episode it happened to be --
    a replay distribution that moves under the learner, where an agent that
    commits whole episodes has a still one. It is not a question of how much
    storage there is; it is a question of what gets sampled.

    Measured from the last commit instead, so the reserved slack absorbs
    exactly what the open episode writes.
    """

    buffer, state = wrapped()
    before = set(only_valid(stacked(draws(buffer, state, keys=64)))[0].ravel().tolist())

    # Four transitions of a new episode, none of them an ending: the head moves
    # the whole of the reserved slack, and nothing else may move with it.
    for index in range(4):
        state = buffer.add(state, transition([99.0, float(index)], index == 0, False))

    assert int(state.written) == 64
    after = set(only_valid(stacked(draws(buffer, state, keys=64)))[0].ravel().tolist())
    assert after == before


# --------------------------------------------- the branch that takes the episode
def test_full_bptt_draws_every_completed_episode_whatever_its_length():
    """No bar on length: the window is the episode, padded out and masked."""

    buffer, state = stored(minimum_episode_length=1, window=EPISODE_LENGTH)

    windows = stacked(draws(buffer, state))
    episode, index = only_valid(windows)
    valid = np.asarray(windows.valid)[np.asarray(windows.batch_valid)]

    assert set(episode[:, 0].tolist()) == {0, 1, 2, 3}
    lengths = np.where(np.isin(episode[:, 0], [0, 2]), 4, 8)
    step = np.arange(EPISODE_LENGTH)
    np.testing.assert_array_equal(valid, step[None, :] < lengths[:, None])
    # The window is the episode from its beginning, so the valid part is
    # exactly the episode, in order, starting where the episode starts.
    assert np.all(index[valid] == np.broadcast_to(step, index.shape)[valid])


def test_each_branch_reads_the_window_it_names():
    truncated = SelectedLearning("truncated", 3)
    full = SelectedLearning("full_bptt", 0)

    assert truncated.window(EPISODE_LENGTH) == 3
    assert full.window(EPISODE_LENGTH) == EPISODE_LENGTH
    # A truncated window has to fit whole, or the gradient reaches back however
    # far the episode had left; a full-episode window takes what there is.
    assert truncated.minimum_episode_length(EPISODE_LENGTH) == 3
    assert full.minimum_episode_length(EPISODE_LENGTH) == 1


def test_a_window_longer_than_an_episode_can_run_is_refused_at_build_time():
    with pytest.raises(ValueError, match="no window that long fits"):
        SelectedLearning("truncated", 9).minimum_episode_length(EPISODE_LENGTH)


def test_a_warmup_the_buffer_could_never_hold_is_refused():
    """A threshold replay can never be at is a misconfiguration, not a long wait."""

    with pytest.raises(ValueError, match="not one this buffer stays above"):
        make_uniform_episode_window_buffer(
            max_length=64,
            min_length=100,
            sample_batch_size=2,
            sample_sequence_length=TRUNCATION,
            add_batch_size=1,
            max_episode_length=EPISODE_LENGTH,
        )


def test_a_warmup_at_the_buffer_s_capacity_is_refused_too():
    """Because replay dips below capacity, and a learner that stops is silent.

    An episode straddling the oldest kept position is dropped whole, so what
    replay holds sits a little under what it was asked to keep. A warmup set at
    capacity would be met and then unmet as episodes fell off the boundary, and
    a run that quietly stopped learning is not a failure anyone would look for.
    """

    with pytest.raises(ValueError, match="not one this buffer stays above"):
        make_uniform_episode_window_buffer(
            max_length=64,
            min_length=64,
            sample_batch_size=2,
            sample_sequence_length=TRUNCATION,
            add_batch_size=1,
            max_episode_length=EPISODE_LENGTH,
        )


def test_a_minibatch_larger_than_the_buffer_could_hold_is_refused():
    """Without replacement is a promise, so B has to be one the buffer can keep."""

    with pytest.raises(ValueError, match="without replacement"):
        make_uniform_episode_window_buffer(
            max_length=8,
            min_length=4,
            sample_batch_size=16,
            sample_sequence_length=TRUNCATION,
            add_batch_size=1,
            max_episode_length=2,
        )


def test_uniform_replay_keeps_no_priority_to_update():
    buffer, state = stored()

    assert not hasattr(state, "priorities")
    assert not hasattr(buffer, "set_priorities")
    assert not hasattr(buffer.sample(state, jax.random.key(1)), "probabilities")


# ------------------------------------------------------ what the window hands on
class _Sample:
    experience = ReplayTransition(
        observation=jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2),
        episode_start=jnp.asarray([[True, False, False]]),
        action=jnp.asarray([[0, 1, 0]]),
        reward=jnp.asarray([[1.0, 2.0, 3.0]]),
        next_observation=jnp.arange(6, 12, dtype=jnp.float32).reshape(1, 3, 2),
        done=jnp.asarray([[False, False, False]]),
        terminal=jnp.zeros((1, 3), dtype=jnp.bool_),
    )
    valid = jnp.ones((1, 3), dtype=jnp.bool_)
    batch_valid = jnp.ones((1,), dtype=jnp.bool_)


def test_the_two_sequences_are_the_states_and_the_states_arrived_at():
    sequence = learner_sequence(_Sample())

    # Two passes of the same length: the online network reads the states, the
    # target network reads the states they arrived at.
    assert sequence.inputs.observation.shape == (1, 3, 2)
    assert sequence.bootstrap_inputs.observation.shape == (1, 3, 2)
    assert sequence.actions.shape == (1, 3)
    np.testing.assert_array_equal(
        np.asarray(sequence.inputs.observation),
        np.asarray(_Sample.experience.observation),
    )
    np.testing.assert_array_equal(
        np.asarray(sequence.bootstrap_inputs.observation),
        np.asarray(_Sample.experience.next_observation),
    )
    # A successor sequence is read as one run, so it declares no episode start.
    assert not bool(jnp.any(sequence.bootstrap_inputs.episode_start))


def test_the_masks_are_the_sampler_s_and_are_not_rebuilt_from_the_window():
    """Whoever drew the episode knows what is in it; the stored flags do not.

    Rederiving validity from the window's own ``done`` flags would call a
    window valid to its end whenever it happens to contain no ending -- which
    is true of a padded window and of a spliced one alike. The claim is about
    the draw, so it comes from the draw.
    """

    buffer, state = stored(lengths=(4, 8), minimum_episode_length=1, window=8)

    sample = buffer.sample(state, jax.random.key(0))
    sequence = learner_sequence(sample)

    np.testing.assert_array_equal(np.asarray(sequence.valid), np.asarray(sample.valid))
    np.testing.assert_array_equal(
        np.asarray(sequence.batch_valid), np.asarray(sample.batch_valid)
    )
    # And it is not the same answer: the short episode's padding carries no
    # ending, so the window's own flags would have called all eight steps good.
    assert not np.all(np.asarray(sample.valid))
