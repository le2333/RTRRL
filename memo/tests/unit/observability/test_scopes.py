"""What each scope selects, and what its selection is and is not biased by.

The bias these scopes replace: a step mark chose the episode spanning it, and
a long episode spans more of the axis. The runs below are built so that the
old rule and each new one disagree -- long episodes pay more than short ones,
which is what made the old sampled mean an over-estimate.
"""

from __future__ import annotations

import pytest

from memorax.observability.scopes import EpisodeScope, StepScope, WindowScope
from memorax.runtime.episode import Episode


def episode(number, *, start, length, reward=1.0, width=1, phase="train"):
    """One stream's episode of ``length`` transitions, each ``width`` steps."""

    return Episode(
        number=number,
        phase=phase,
        start_env_steps=start,
        end_env_steps=start + length * width,
        rewards=[reward] * length,
        terminals=[False] * (length - 1) + [True],
        truncations=[False] * length,
        series={"td_error": [float(index) for index in range(length)]},
    )


def alternating(count, *, short=2, long=8, width=1):
    """Episodes of two lengths, where the long one pays what its length is."""

    start, number = 0, 1
    for index in range(count):
        length = short if index % 2 else long
        yield episode(
            number, start=start, length=length, reward=float(length), width=width
        )
        start += length * width
        number += 1


def test_a_step_mark_reports_the_transition_it_lands_in():
    scope = StepScope(every_steps=4)

    taken = scope.take(episode(1, start=0, length=6))

    assert [step for step, _ in taken] == [4]
    assert taken[0][1] == {
        "train/step/return_per_step": 1.0,
        "train/step/td_error": 4.0,
    }


def test_a_step_mark_is_unmoved_by_how_long_its_episode_ran():
    """The reading at step 8 is the same whichever episode was running.

    This is the whole difference from the sampling it replaces: that one asked
    which episode spanned the mark and reported the episode, so the answer
    depended on a length the reading has nothing to do with.
    """

    scope = StepScope(every_steps=8)

    inside_a_long_one = scope.take(episode(1, start=0, length=20))
    inside_a_short_one = scope.take(episode(2, start=6, length=4))

    assert inside_a_long_one[0][1]["train/step/return_per_step"] == 1.0
    assert inside_a_short_one[0][1]["train/step/return_per_step"] == 1.0


def test_a_mark_belongs_to_the_one_stream_whose_step_it_numbers():
    """The step counter numbers every stream's every step, so a mark names one.

    Two streams of a four-wide run, both spanning step 8. Only the stream that
    owns step 8 answers for it; the other reports its own marks and not this.
    """

    scope = StepScope(every_steps=8)
    owner = episode(1, start=0, length=4, width=4)
    neighbour = episode(2, start=2, length=4, width=4)

    assert [step for step, _ in scope.take(owner)] == [8]
    assert scope.take(neighbour) == ()


def test_no_mark_is_owed_before_the_first_interval_has_passed():
    scope = StepScope(every_steps=10)

    assert scope.take(episode(1, start=0, length=4)) == ()


def test_a_long_episode_answers_every_mark_it_covers():
    scope = StepScope(every_steps=4)

    taken = scope.take(episode(1, start=0, length=20))

    assert [step for step, _ in taken] == [4, 8, 12, 16]


@pytest.mark.parametrize(
    "lengths",
    [
        (2, 9, 3, 7, 2, 8, 4, 5, 6),
        (6, 5, 4, 8, 2, 7, 3, 9, 2),
    ],
)
def test_every_nth_episode_is_chosen_by_number_and_never_by_length(lengths):
    """Uniform in episode space: the same episodes are chosen either way.

    Two runs whose episodes end at wildly different places, and the choice is
    the same one in three. Nothing about how far an episode reached along the
    step axis -- which is what the old sampling was reading -- takes part.
    """

    scope = EpisodeScope(3)
    start = 0
    chosen = []
    for number, length in enumerate(lengths, start=1):
        if scope.take(episode(number, start=start, length=length)):
            chosen.append(number)
        start += length

    assert chosen == [3, 6, 9]


def test_an_episode_reading_is_stamped_where_the_episode_ended():
    scope = EpisodeScope(1)

    [(step, _)] = scope.take(episode(1, start=6, length=4))

    assert step == 10


def test_a_window_closes_on_its_mark_and_counts_the_episodes_ending_in_it():
    scope = WindowScope(every_steps=10)

    taken = [
        reading
        for one in (
            episode(1, start=0, length=4),
            episode(2, start=4, length=4),
            episode(3, start=8, length=6),  # ends at 14, so the next window
        )
        for reading in scope.take(one)
    ]

    assert [step for step, _ in taken] == [10]
    assert taken[0][1]["train/window/length"] == 4.0


def test_an_episode_is_counted_in_the_window_it_ends_in_and_only_there():
    """Ending partitions the episodes; spanning would count one twice."""

    scope = WindowScope(every_steps=10)
    crossing = episode(1, start=5, length=10)  # spans two window closes

    scope.take(crossing)
    scope.take(episode(2, start=15, length=2))
    taken = scope.close()

    assert [step for step, _ in taken] == [20]
    assert taken[0][1]["train/window/length"] == 6.0


def test_a_window_mean_is_the_length_weighted_one_the_old_sampling_missed():
    """Every episode in the stretch, so the long ones cannot crowd it.

    The old rule picked the episode spanning a step mark, which favoured long
    ones; here the window holds all ten and its ``return`` is their plain mean.
    """

    scope = WindowScope(every_steps=50)
    for one in alternating(10):
        scope.take(one)

    [(_, values)] = scope.close()

    assert values["train/window/return"] == pytest.approx(34.0)
    assert values["train/window/length"] == pytest.approx(5.0)


def test_a_shorter_length_keeps_the_stretch_before_the_close():
    scope = WindowScope(every_steps=20, length_steps=5)

    scope.take(episode(1, start=0, length=4))  # ends at 4, outside the stretch
    scope.take(episode(2, start=14, length=4))  # ends at 18, inside it
    taken = scope.close()

    assert [step for step, _ in taken] == [20]
    assert taken[0][1]["train/window/return"] == 4.0


def test_a_window_no_episode_ended_in_reports_nothing():
    scope = WindowScope(every_steps=10, length_steps=2)

    scope.take(episode(1, start=0, length=4))  # ends at 4, outside the stretch

    assert scope.close() == ()


def test_the_run_ending_reports_the_window_still_open():
    scope = WindowScope(every_steps=100)

    scope.take(episode(1, start=0, length=4))
    taken = scope.close()

    # Cut short by the budget, and reported at the close it was scheduled for
    # rather than at whichever episode happened to be the last one.
    assert [step for step, _ in taken] == [100]
    assert scope.close() == ()


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: StepScope(0), "every_steps"),
        (lambda: EpisodeScope(0), "every_episodes"),
        (lambda: WindowScope(0), "every_steps"),
        (lambda: WindowScope(10, 0), "length_steps"),
        (lambda: WindowScope(10, 11), "length_steps"),
    ],
)
def test_an_interval_that_cannot_be_met_is_refused(build, message):
    with pytest.raises(ValueError, match=message):
        build()
