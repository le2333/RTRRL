from __future__ import annotations

from typing import Any, Callable

from .context import RunContext
from .spool import AimUnavailable, MetricEvent


class AimAdapter:
    """Translate SDK events to the Aim API at one explicit boundary."""

    def __init__(
        self,
        *,
        run_factory: Callable[..., Any] | None = None,
        availability_errors: tuple[type[BaseException], ...] = (
            ConnectionError,
            TimeoutError,
            OSError,
        ),
        **run_options: Any,
    ) -> None:
        if run_factory is None:
            from aim import Run

            run_factory = Run
        self._run_factory = run_factory
        self._run_options = run_options
        self._availability_errors = availability_errors
        self._run: Any | None = None
        self._context: RunContext | None = None

    def start(self, context: RunContext) -> None:
        self._context = context
        try:
            run = self._run_factory(
                experiment=context.experiment_name, **self._run_options
            )
            run.name = context.run_name
            run["hparams"] = context.hparams
        except self._availability_errors as exc:
            raise AimUnavailable("Aim is temporarily unavailable") from exc
        self._run = run

    def send(self, event: MetricEvent) -> None:
        if self._run is None:
            if self._context is None:
                raise RuntimeError("AimAdapter has not been started")
            self.start(self._context)
        assert self._run is not None
        marker = f"sdk/event_ids/{event.event_id}"
        try:
            if self._run.get(marker, False):
                return
            for name, value in event.metrics.items():
                self._run.track(value, name=name, step=event.env_steps)
            if event.kind == "final":
                self._run["sdk/objective_metric"] = event.data["objective_metric"]
                self._run["sdk/finalized"] = True
            self._run[marker] = True
        except self._availability_errors as exc:
            raise AimUnavailable("Aim is temporarily unavailable") from exc
