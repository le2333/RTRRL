"""Runtime owns episode identity and completeness, not its interpretation."""

import pytest

from memorax.runtime.episode import Episode, SampledTrajectory


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


def test_a_long_run_may_end_more_episodes_than_a_six_digit_name_held():
    """How many episodes a run ends is the environment's business.

    A masked Hopper ends one every thirty-odd steps before it learns to stand,
    so a fifty million step run ends over a million of them. The number used to
    stop at 999999 because it named a recording; recordings are named after the
    sample that asked for them now.
    """

    assert make_episode(number=1_380_000).number == 1_380_000


def test_an_episode_number_still_starts_at_one():
    with pytest.raises(ValueError, match="positive"):
        make_episode(number=0)


def test_trajectory_must_have_one_more_observation_than_actions():
    with pytest.raises(ValueError, match="observations"):
        make_episode(observations=[[0.0], [1.0]], actions=[[0.0], [0.0]])


def test_unfinished_episode_is_not_an_episode():
    with pytest.raises(ValueError, match="complete"):
        make_episode(terminals=[False, False])


def test_sampled_trajectory_marks_every_transition_budget_side():
    episode = make_episode()
    sampled = SampledTrajectory(
        episode=episode,
        sample_step=1,
        post_budget=(False, True),
    )

    assert sampled.sample_step == 1
    assert sampled.post_budget == (False, True)


def test_sampled_trajectory_requires_one_budget_mark_per_transition():
    with pytest.raises(ValueError, match="post_budget"):
        SampledTrajectory(
            episode=make_episode(),
            sample_step=1,
            post_budget=(False,),
        )
