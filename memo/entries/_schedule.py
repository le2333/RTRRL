"""Project a run document's Rerun request onto what Runtime and assembly need.

Whether a run keeps trajectories is a deployment decision: a fixed reference
run asks for them and a search does not.  Neither Runtime nor an algorithm
infers it -- both are told, here, once per entry.
"""

from __future__ import annotations

from memorax.runtime import ObservationSchema


def sample_steps(config) -> tuple[int, ...]:
    """The environment steps whose training episode is to be kept whole.

    The schedule starts at the first interval rather than at zero: a run that
    has trained nothing has no training episode to show.
    """

    rerun = config.logging.rerun
    if rerun is None:
        return ()
    return tuple(
        range(rerun.every_steps, config.runtime.total_steps + 1, rerun.every_steps)
    )


def trajectory_record(config, observations: ObservationSchema) -> frozenset[str]:
    """The heavy per-step fields this run's graph must retain, if any."""

    if config.logging.rerun is None:
        return frozenset()
    return observations.trajectory_fields
