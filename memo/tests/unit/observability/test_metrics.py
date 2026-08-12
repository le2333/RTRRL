import re

import pytest

from memorax.observability import check_names, metric_names, statistics
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


def episode_with_rewards(rewards):
    return Episode(
        number=1,
        phase="train",
        start_env_steps=0,
        end_env_steps=8,
        rewards=rewards,
        terminals=[False, True],
        truncations=[False, False],
    )


def test_variance_distinguishes_steady_and_spiky_episode_returns():
    steady = statistics(episode_with_rewards([2.0, 2.0]))
    spiky = statistics(episode_with_rewards([0.0, 4.0]))

    assert steady["train/episode/return_per_step"] == 2.0
    assert spiky["train/episode/return_per_step"] == 2.0
    assert steady["train/episode/return_per_step_variance"] == 0.0
    assert spiky["train/episode/return_per_step_variance"] == 4.0


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
        "train/step/return",
        "train/chunk/return",
        "train//return",
    ],
)
def test_names_without_the_episode_window_are_rejected(name):
    with pytest.raises(ValueError, match=re.escape(name)):
        check_names([name])
