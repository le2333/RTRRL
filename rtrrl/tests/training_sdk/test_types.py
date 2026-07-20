import pytest

from training_sdk import Episode


def make_episode(**overrides):
    values = {
        "number": 4,
        "phase": "eval",
        "start_env_steps": 100,
        "end_env_steps": 102,
        "observations": [1, 2, 3],
        "actions": [0, 1],
        "rewards": [1.0, 2.0],
        "terminals": [False, True],
        "truncations": [False, False],
    }
    values.update(overrides)
    return Episode(**values)


def test_episode_accepts_n_plus_one_observations():
    episode = make_episode()

    assert len(episode.observations) == len(episode.actions) + 1


def test_episode_rejects_one_observation_per_transition():
    with pytest.raises(ValueError, match="observations"):
        make_episode(observations=[1, 2])


@pytest.mark.parametrize(
    "field,value",
    [
        ("rewards", [1.0]),
        ("terminals", [False]),
        ("truncations", [False]),
    ],
)
def test_episode_transition_arrays_must_have_equal_lengths(field, value):
    with pytest.raises(ValueError, match="transition"):
        make_episode(**{field: value})


def test_episode_observations_must_match_n_plus_one_transitions():
    with pytest.raises(ValueError, match="observations"):
        make_episode(observations=[1, 2, 3, 4])


def test_episode_must_be_complete():
    with pytest.raises(ValueError, match="complete"):
        make_episode(terminals=[False, False])


def test_episode_must_complete_on_its_last_transition():
    with pytest.raises(ValueError, match="complete"):
        make_episode(terminals=[True, False])


def test_episode_can_end_by_truncation():
    episode = make_episode(
        terminals=[False, False],
        truncations=[False, True],
    )

    assert episode.truncations[-1] is True


def test_episode_end_steps_cannot_precede_start_steps():
    with pytest.raises(ValueError, match="end_env_steps"):
        make_episode(end_env_steps=99)


def test_episode_copies_transition_sequences():
    rewards = [1.0, 2.0]
    episode = make_episode(rewards=rewards)
    rewards.append(3.0)

    assert episode.rewards == (1.0, 2.0)
