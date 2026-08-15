"""Project a run document's logging requests onto what Runtime and assembly need.

Whether a run keeps trajectories is a deployment decision: a fixed reference
run asks for them and a search does not.  Neither Runtime nor an algorithm
infers it -- both are told, here, once per entry.
"""

from __future__ import annotations

from memorax.runtime import ObservationSchema


def trajectory_at_steps(config) -> tuple[int, ...]:
    """The environment steps whose training episode is to be kept whole.

    The schedule begins at the first interval rather than at zero: a run that
    has trained nothing has no training episode to show.
    """

    rerun = config.logging.rerun
    if rerun is None:
        return ()
    return tuple(
        range(
            rerun.log_every_steps,
            config.training.total_steps + 1,
            rerun.log_every_steps,
        )
    )


def trajectory_record(config, observations: ObservationSchema) -> frozenset[str]:
    """The heavy per-step fields this run's graph must retain, if any."""

    if config.logging.rerun is None:
        return frozenset()
    return observations.trajectory_fields


def training_every_steps(config) -> int:
    """How often a training episode reaches the dashboard; zero for never."""

    training = config.logging.aim.training
    return 0 if training is None else training.log_every_steps
