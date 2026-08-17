import re

import pytest

from memorax.observability import check_names, metric_names, statistics
from memorax.observability.metrics import WindowStatistics, step_statistics
from memorax.runtime.episode import Episode
from tests.support.observability import completed_episode


def test_episode_statistics_keep_the_existing_scalar_contract():
    assert statistics(completed_episode()) == {
        "train/episode/length": 2.0,
        "train/episode/return": 4.0,
        "train/episode/return_per_step": 2.0,
        "train/episode/return_per_step_variance": 1.0,
        "train/episode/td_error": 1.0,
        "train/episode/td_error_variance": 1.0,
    }


def test_declared_metric_names_equal_the_statistics_they_describe():
    names = metric_names("train", ("td_error",))

    check_names(names)
    assert names == tuple(statistics(completed_episode()))


def test_a_reading_at_one_moment_has_no_total_and_no_spread():
    """Both need a stretch: one value is neither accumulated nor scattered."""

    names = metric_names("train", ("td_error",), scope="step")

    check_names(names)
    assert names == ("train/step/return_per_step", "train/step/td_error")
    assert names == tuple(step_statistics(completed_episode(), 0))


def test_one_moment_is_that_transition_and_not_the_episode_around_it():
    episode = completed_episode()

    assert step_statistics(episode, 0) == {
        "train/step/return_per_step": 1.0,
        "train/step/td_error": 0.0,
    }
    assert step_statistics(episode, 1) == {
        "train/step/return_per_step": 3.0,
        "train/step/td_error": 2.0,
    }


def test_a_window_declares_the_same_quantities_as_an_episode():
    window = WindowStatistics()
    window.add(completed_episode())

    names = metric_names("train", ("td_error",), scope="window")

    check_names(names)
    assert names == tuple(window.statistics())


def episode_with_rewards(rewards, number=1):
    return Episode(
        number=number,
        phase="train",
        start_env_steps=0,
        end_env_steps=8,
        rewards=rewards,
        terminals=[False] * (len(rewards) - 1) + [True],
        truncations=[False] * len(rewards),
    )


def test_variance_distinguishes_steady_and_spiky_episode_returns():
    steady = statistics(episode_with_rewards([2.0, 2.0]))
    spiky = statistics(episode_with_rewards([0.0, 4.0]))

    assert steady["train/episode/return_per_step"] == 2.0
    assert spiky["train/episode/return_per_step"] == 2.0
    assert steady["train/episode/return_per_step_variance"] == 0.0
    assert spiky["train/episode/return_per_step_variance"] == 4.0


def test_a_window_pools_transitions_rather_than_averaging_episode_means():
    """A mean of means is not the mean when the episodes are unequal.

    Eight transitions, one of which paid 4: the mean transition paid 0.5.
    Averaging the two episodes' own means -- 0 and 2 -- answers 1, because the
    two-transition episode is given the weight of the six-transition one. The
    variance is worse still: neither episode's variance knows how far its
    transitions sit from the other's mean, so no combination of them is it.
    """

    window = WindowStatistics()
    window.add(episode_with_rewards([0.0] * 6))
    window.add(episode_with_rewards([0.0, 4.0], number=2))

    values = window.statistics()

    assert values["train/window/return_per_step"] == pytest.approx(0.5)
    assert values["train/window/return_per_step_variance"] == pytest.approx(1.75)
    # Per-episode quantities average per episode: both levels are real here.
    assert values["train/window/length"] == 4.0
    assert values["train/window/return"] == 2.0


def test_a_window_reports_nothing_when_no_episode_ended_in_it():
    with pytest.raises(ValueError, match="empty window"):
        WindowStatistics().statistics()


def test_phase_is_the_first_metric_axis():
    episode = completed_episode(phase="eval")

    assert all(name.startswith("eval/") for name in metric_names("eval"))
    assert statistics(episode)["eval/episode/return"] == 4.0


def test_component_family_stays_inside_the_quantity_axis():
    episode = Episode(
        number=1,
        phase="train",
        start_env_steps=0,
        end_env_steps=2,
        rewards=[1.0, 1.0],
        terminals=[False, True],
        truncations=[False, False],
        series={"actor_grad_norm.torso": [1.0, 1.0]},
    )

    names = statistics(episode)
    assert "train/episode/actor_grad_norm.torso" in names
    assert all(name.count("/") == 2 for name in names)


@pytest.mark.parametrize(
    "name",
    [
        "episode_return",
        "train/return",
        "train/episode/return/torso",
        "train/episodes/return",
        "train/chunk/return",
        "train//return",
    ],
)
def test_names_without_a_known_scope_are_rejected(name):
    with pytest.raises(ValueError, match=re.escape(name)):
        check_names([name])


@pytest.mark.parametrize("scope", ["step", "episode", "window"])
def test_every_scope_a_reduction_exists_for_is_a_name_a_run_may_use(scope):
    check_names(metric_names("train", ("td_error",), scope=scope))


def test_a_scope_nothing_reduces_over_cannot_be_asked_for():
    with pytest.raises(ValueError, match="chunk"):
        metric_names("train", scope="chunk")
