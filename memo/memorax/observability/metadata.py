"""Run identity projected for observation backends."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunMetadata:
    """Enough to tell one run from another on a dashboard.

    ``seed`` and ``role`` are here because without them a formal launch is ten
    runs of one trial that a dashboard cannot separate, and a tuning curve and
    a reportable one look alike.
    """

    run_id: str
    experiment: str
    launch_id: str
    trial: int
    seed: int
    role: str
    entry: str
    digest: str
