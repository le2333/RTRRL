"""Stateful episode reconstruction across bounded rollout chunks."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from memorax.runtime import ObservationSchema
from memorax.runtime.tracker import EpisodeTracker


OBSERVATIONS = ObservationSchema(
    reward="reward",
    done="done",
    terminal="terminal",
    series=("td_error",),
    observation="observation",
    next_observation="next_observation",
    action="action",
)

EXPECTED_EPISODE_AT_FOUR = ([0.0], [2.0], [4.0], [6.0], [106.0])
EXPECTED_EPISODE_AFTER_DONE_AT_TEN = ([10.0], [12.0], [14.0], [114.0])
FULL_PREFIX_AND_SUFFIX = ([1.0], [3.0], [5.0], [7.0], [107.0])


def summary(start_env_steps: int, done: list[list[int]]) -> SimpleNamespace:
    endings = np.asarray(done, dtype=bool)
    steps, num_envs = endings.shape
    positions = np.arange(
        start_env_steps,
        start_env_steps + steps * num_envs,
        dtype=float,
    ).reshape(steps, num_envs)
    return SimpleNamespace(
        observation=positions[..., None],
        next_observation=(positions + 100.0)[..., None],
        action=(positions + 200.0)[..., None],
        reward=positions + 1.0,
        done=endings,
        terminal=endings.copy(),
        td_error=positions + 300.0,
    )


def first_chunk() -> SimpleNamespace:
    return summary(0, [[0, 0], [0, 0], [0, 0]])


def second_chunk() -> SimpleNamespace:
    return summary(6, [[1, 0], [1, 0], [0, 1]])


def third_chunk() -> SimpleNamespace:
    return summary(12, [[0, 0], [1, 1]])


def make_tracker(**overrides: object) -> EpisodeTracker:
    settings = {
        "observations": OBSERVATIONS,
        "num_envs": 2,
        "max_episode_steps": 4,
    }
    return EpisodeTracker(**(settings | overrides))


def chunk_with_two_env_zero_endings() -> SimpleNamespace:
    return summary(0, [[0, 0], [1, 0], [0, 0], [1, 1]])


def prefix_without_done() -> SimpleNamespace:
    return summary(0, [[0, 0], [0, 0]])


def suffix_with_done() -> SimpleNamespace:
    return summary(4, [[0, 0], [1, 1]])


def three_steps_without_done() -> SimpleNamespace:
    return summary(0, [[0, 0], [0, 0], [0, 0]])


def test_sampled_episode_survives_chunks_and_done_boundary_selects_next():
    tracker = EpisodeTracker(
        observations=OBSERVATIONS,
        num_envs=2,
        max_episode_steps=6,
        sample_steps=(4, 10),
    )

    first = tracker.consume(first_chunk(), start_env_steps=0)
    second = tracker.consume(second_chunk(), start_env_steps=6)
    third = tracker.consume(third_chunk(), start_env_steps=12)

    sampled = first.sampled + second.sampled + third.sampled
    assert [one.sample_step for one in sampled] == [4, 10]
    assert [one.episode.stream for one in sampled] == [0, 0]
    assert sampled[0].episode.observations == EXPECTED_EPISODE_AT_FOUR
    assert sampled[1].episode.observations == EXPECTED_EPISODE_AFTER_DONE_AT_TEN
    assert all(one.episode.terminals[-1] for one in sampled)


def test_one_stream_reuses_its_slot_for_multiple_episodes_in_one_chunk():
    tracker = make_tracker(sample_steps=())
    result = tracker.consume(chunk_with_two_env_zero_endings(), start_env_steps=0)

    assert [episode.stream for episode in result.completed] == [0, 0, 1]
    assert [len(episode.rewards) for episode in result.completed] == [2, 2, 4]


def test_unfinished_episode_is_retained_without_being_reported():
    tracker = make_tracker(sample_steps=(3,))
    first = tracker.consume(prefix_without_done(), start_env_steps=0)
    second = tracker.consume(suffix_with_done(), start_env_steps=4)

    assert first.completed == first.sampled == ()
    assert second.sampled[0].episode.observations == FULL_PREFIX_AND_SUFFIX


def test_sample_at_chunk_ending_boundary_is_pending_for_continuation():
    tracker = make_tracker(sample_steps=(4,))

    first = tracker.consume(summary(0, [[0, 0], [1, 0]]), start_env_steps=0)

    assert first.sampled == ()
    assert tracker.pending_sample_steps == (4,)

    second = tracker.consume(summary(4, [[1, 0]]), start_env_steps=4)
    assert len(second.sampled) == 1
    assert second.sampled[0].sample_step == 4
    assert second.sampled[0].episode.observations == ([4.0], [104.0])


def test_every_configured_series_path_must_exist():
    tracker = make_tracker(
        observations=replace(OBSERVATIONS, series=("td_error", "missing_series"))
    )

    with pytest.raises(
        ValueError,
        match="configured series.*missing_series.*missing",
    ):
        tracker.consume(summary(0, [[1, 1]]), start_env_steps=0)


def test_all_unset_trajectory_paths_are_valid_no_trajectory_mode():
    tracker = make_tracker(
        observations=replace(
            OBSERVATIONS,
            observation=None,
            next_observation=None,
            action=None,
        )
    )

    result = tracker.consume(summary(0, [[1, 1]]), start_env_steps=0)

    assert all(episode.observations is None for episode in result.completed)
    assert all(episode.actions is None for episode in result.completed)


def test_partially_configured_trajectory_schema_raises():
    tracker = make_tracker(observations=replace(OBSERVATIONS, action=None))

    with pytest.raises(
        ValueError,
        match="trajectory schema.*all configured or all unset",
    ):
        tracker.consume(summary(0, [[1, 1]]), start_env_steps=0)


def test_missing_configured_trajectory_field_raises():
    tracker = make_tracker()
    chunk = summary(0, [[1, 1]])
    del chunk.next_observation

    with pytest.raises(
        ValueError,
        match="configured trajectory.*next_observation.*missing",
    ):
        tracker.consume(chunk, start_env_steps=0)


def test_report_completed_false_still_returns_sample_and_advances_number():
    tracker = make_tracker(sample_steps=(0,))

    result = tracker.consume(
        summary(0, [[1, 0]]),
        start_env_steps=0,
        report_completed=False,
    )

    assert result.completed == ()
    assert [sample.sample_step for sample in result.sampled] == [0]
    assert result.sampled[0].episode.observations == ([0.0], [100.0])
    assert tracker.next_number == 2


def test_sample_at_non_aligned_start_uses_current_transition_stream():
    tracker = make_tracker(sample_steps=(1,))

    result = tracker.consume(summary(1, [[1, 1]]), start_env_steps=1)

    assert [sample.episode.stream for sample in result.sampled] == [0]


def test_episode_longer_than_declared_limit_fails_instead_of_truncating():
    tracker = make_tracker(max_episode_steps=2)
    with pytest.raises(ValueError, match="maximum episode length"):
        tracker.consume(three_steps_without_done(), start_env_steps=0)
