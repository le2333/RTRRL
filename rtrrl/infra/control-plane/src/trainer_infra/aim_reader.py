from __future__ import annotations

from collections.abc import Callable
import hashlib
import math
import time
from typing import Any


class AimResultError(RuntimeError):
    """An exact Aim run reached an invalid terminal state."""


class AimResultTimeout(TimeoutError):
    """An exact Aim run did not finalize before its deadline."""


def _open_aim_run(**kwargs: Any) -> Any:
    from aim import Run

    return Run(**kwargs)


def _prepare_aim_read_only_close(run: Any) -> None:
    tracker = getattr(run, "_tracker", None)
    if (
        getattr(run, "read_only", None) is True
        and tracker is not None
        and not hasattr(tracker, "sequence_infos")
    ):
        # Aim 3.28.0 omits this mapping for read-only trackers, but Run.close()
        # unconditionally clears it. Supply the missing state so Aim can finish cleanup.
        tracker.sequence_infos = {}


class AimReader:
    def __init__(
        self,
        repo: str | None = None,
        *,
        run_factory: Callable[..., Any] = _open_aim_run,
        replay_spool: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval: float = 1.0,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._repo = repo
        self._run_factory = run_factory
        self._replay_spool = replay_spool or (lambda _run_id: None)
        self._clock = clock
        self._sleep = sleep
        self._poll_interval = poll_interval

    def _open(self, run_id: str) -> Any:
        options: dict[str, Any] = {
            "run_hash": hashlib.sha256(run_id.encode()).hexdigest()[:24],
            "read_only": True,
        }
        if self._repo is not None:
            options["repo"] = self._repo
        return self._run_factory(**options)

    @staticmethod
    def _read_metric(run: Any, objective: str) -> float:
        from aim.storage.context import Context

        metric = run.get_metric(objective, Context({"sdk_stream": "final"}))
        if metric is None:
            raise AimResultError(f"exact objective {objective!r} is missing")
        values = metric.data.values_list()
        sequence = values[0] if values else ()
        if not sequence:
            raise AimResultError(f"exact objective {objective!r} has no values")
        value = sequence[-1]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise AimResultError("objective value must be finite numeric")
        return float(value)

    def wait_for_result(self, run_id: str, objective: str, timeout: float) -> float:
        if not run_id or not objective:
            raise ValueError("run_id and objective must be non-empty")
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        self._replay_spool(run_id)
        deadline = self._clock() + timeout

        while True:
            run = self._open(run_id)
            try:
                identity = run.get("hparams", {}).get("identity", {})
                recorded_run_id = identity.get("run_id")
                if recorded_run_id is not None and recorded_run_id != run_id:
                    raise AimResultError(f"Aim run does not contain exact run_id {run_id!r}")
                if recorded_run_id == run_id and run.get("sdk/failed", False) is True:
                    raise AimResultError(f"Aim run {run_id!r} has sdk/failed=True")
                if recorded_run_id == run_id and run.get("sdk/finalized", False) is True:
                    recorded = run.get("sdk/objective_metric")
                    if recorded != objective:
                        raise AimResultError(
                            f"Aim objective {recorded!r} does not equal exact objective "
                            f"{objective!r}"
                        )
                    return self._read_metric(run, objective)
            finally:
                close = getattr(run, "close", None)
                if callable(close):
                    _prepare_aim_read_only_close(run)
                    close()

            now = self._clock()
            if now >= deadline:
                raise AimResultTimeout(
                    f"timed out waiting for Aim run {run_id!r} objective {objective!r}"
                )
            self._sleep(min(self._poll_interval, deadline - now))
