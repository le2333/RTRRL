"""Runtime owns episode identity and completeness, not its interpretation."""

import pytest

from memorax.runtime.episode import Episode


def make_episode(**overrides) -> Episode:
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


def test_episode_need_not_retain_a_trajectory():
    episode = make_episode()

    assert episode.observations is None
    assert episode.actions is None


def test_series_must_span_the_completed_episode():
    with pytest.raises(ValueError, match="series"):
        make_episode(series={"td_error": [1.0]})


def test_trajectory_must_have_one_more_observation_than_actions():
    with pytest.raises(ValueError, match="observations"):
        make_episode(observations=[[0.0], [1.0]], actions=[[0.0], [0.0]])


def test_unfinished_episode_is_not_an_episode():
    with pytest.raises(ValueError, match="complete"):
        make_episode(terminals=[False, False])
