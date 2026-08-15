"""Small observability values shared by unit and backend integration tests."""

from memorax.runtime.episode import Episode, SampledTrajectory


def completed_episode(
    number: int = 1,
    phase: str = "train",
    *,
    span: tuple[int, int] = (0, 8),
    stream: int = 0,
) -> Episode:
    return Episode(
        number=number,
        phase=phase,
        stream=stream,
        start_env_steps=span[0],
        end_env_steps=span[1],
        observations=[[0.0], [1.0], [2.0]],
        actions=[[0.0], [1.0]],
        rewards=[1.0, 3.0],
        terminals=[False, True],
        truncations=[False, False],
        series={"td_error": [0.0, 2.0]},
    )


def completed_trajectory(
    *,
    sample_step: int = 1,
    post_budget: tuple[bool, ...] = (False, False),
    stream: int = 0,
) -> SampledTrajectory:
    return SampledTrajectory(
        episode=completed_episode(stream=stream),
        sample_step=sample_step,
        post_budget=post_budget,
    )
