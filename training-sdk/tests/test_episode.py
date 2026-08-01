"""What a completed episode is worth, and what those numbers are called."""

from __future__ import annotations

import pytest

from training_sdk.episode import Episode, metric_names, statistics


def make(**overrides) -> Episode:
    fields = {
        "number": 1,
        "phase": "train",
        "start_env_steps": 0,
        "end_env_steps": 8,
        "rewards": [1.0, 3.0],
        "terminals": [False, True],
        "truncations": [False, False],
    }
    return Episode(**(fields | overrides))


def test_an_episode_is_worth_its_length_its_return_and_the_spread():
    assert statistics(make()) == {
        "train/episode/length": 2.0,
        "train/episode/return": 4.0,
        "train/episode/return_per_step": 2.0,
        "train/episode/return_per_step_variance": 1.0,
    }


def test_a_mean_alone_cannot_tell_a_steady_episode_from_a_spiky_one():
    steady = statistics(make(rewards=[2.0, 2.0]))
    spiky = statistics(make(rewards=[0.0, 4.0]))

    assert (
        steady["train/episode/return_per_step"] == spiky["train/episode/return_per_step"]
    )
    assert steady["train/episode/return_per_step_variance"] == 0.0
    assert spiky["train/episode/return_per_step_variance"] == 4.0


def test_a_declared_series_is_reduced_over_the_same_window():
    stats = statistics(make(series={"td_error": [0.0, 2.0]}))

    assert stats["train/episode/td_error"] == 1.0
    assert stats["train/episode/td_error_variance"] == 1.0


def test_every_name_has_three_parts_and_the_middle_one_is_the_window():
    for name in statistics(make(series={"td_error": [0.0, 2.0]})):
        phase, window, quantity = name.split("/")
        assert (phase, window) == ("train", "episode")
        assert quantity


def test_the_phase_is_the_first_part():
    assert all(name.startswith("eval/") for name in metric_names("eval"))
    assert statistics(make(phase="eval"))["eval/episode/return"] == 4.0


def test_what_an_episode_reports_is_what_the_names_declare():
    series = ("td_error", "value")
    episode = make(series={name: [1.0, 2.0] for name in series})

    assert tuple(statistics(episode)) == metric_names("train", series)


def test_a_family_within_a_quantity_keeps_the_name_three_parts():
    """A kernel that measures parts of a network reports one number per part.

    The three slashes are the axes of the name; a part inside a quantity is not
    a fourth axis, so it is spelled the way a parameter inside a structure is.
    """

    stats = statistics(make(series={"actor_grad_norm.torso": [1.0, 1.0]}))

    assert "train/episode/actor_grad_norm.torso" in stats
    assert all(name.count("/") == 2 for name in stats)


def test_an_episode_without_a_trajectory_is_still_an_episode():
    """Recording every observation of training costs what the episode is worth.

    The statistics need the reward and the ending and nothing else, so an
    episode that carries only those is complete.
    """

    episode = make()

    assert episode.observations is None
    assert episode.actions is None
    assert statistics(episode)["train/episode/return"] == 4.0


def test_a_series_that_does_not_span_the_episode_is_refused():
    with pytest.raises(ValueError, match="series"):
        make(series={"td_error": [1.0]})


def test_a_trajectory_that_does_not_span_the_episode_is_refused():
    with pytest.raises(ValueError, match="observations"):
        make(observations=[[0.0], [1.0]], actions=[[0.0], [0.0]])


def test_an_unfinished_episode_is_refused():
    with pytest.raises(ValueError, match="complete"):
        make(terminals=[False, False])
