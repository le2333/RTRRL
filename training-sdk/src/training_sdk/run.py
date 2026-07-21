from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from .context import RunContext
from .spool import AimUnavailable, MetricEvent
from .types import Episode


class AimSink(Protocol):
    def start(self, context: RunContext) -> None: ...

    def send(self, event: MetricEvent) -> None: ...

    def fail(self, metadata: Mapping[str, str]) -> None: ...

    def close(self) -> None: ...


class Spool(Protocol):
    @property
    def events(self) -> tuple[MetricEvent, ...]: ...

    @property
    def unsent_events(self) -> tuple[MetricEvent, ...]: ...

    def append_many(self, events: tuple[MetricEvent, ...]) -> None: ...

    def mark_sent(self, event_id: str) -> None: ...

    def close(self) -> None: ...


class RerunSink(Protocol):
    def log_episode(self, episode: Episode) -> Path | None: ...

    def close(self) -> None: ...


class NullRerun:
    """Task 2 placeholder; Task 3 supplies the Rerun implementation."""

    def log_episode(self, episode: Episode) -> Path | None:
        del episode
        return None

    def close(self) -> None:
        return None


class TrainingRun:
    def __init__(
        self,
        context: RunContext,
        aim: AimSink,
        rerun: RerunSink,
        spool: Spool,
        *,
        context_path: Path | None = None,
    ) -> None:
        self.context = context
        self.aim = aim
        self.rerun = rerun
        self.spool = spool
        self._context_path = (
            None if context_path is None else Path(context_path).resolve()
        )
        self._last_env_steps: int | None = None
        self._last_metric_env_steps: int | None = None
        self._terminal_state = "active"
        self._closed = False
        self._summary_sequence = max(
            (
                event.aim_step
                for event in spool.events
                if event.stream == "episode_summary"
            ),
            default=0,
        )

        interval = context.logging.get("aim_every_env_steps", 1)
        if type(interval) is not int:
            raise TypeError("logging.aim_every_env_steps must be an integer")
        if interval <= 0:
            raise ValueError("logging.aim_every_env_steps must be positive")
        self._aim_every_env_steps = interval

    @property
    def context_path(self) -> Path | None:
        return self._context_path

    def start(self) -> None:
        try:
            self.aim.start(self.context)
        except AimUnavailable:
            return

    def _validate_env_steps(self, env_steps: int) -> None:
        if type(env_steps) is not int:
            raise TypeError("env_steps must be an integer")
        if env_steps < 0:
            raise ValueError("env_steps must be non-negative")
        if self._last_env_steps is not None and env_steps < self._last_env_steps:
            raise ValueError("env_steps must be monotonic (non-decreasing)")
        self._last_env_steps = env_steps

    def _emit(self, event: MetricEvent) -> None:
        self._emit_many((event,))

    def _emit_many(self, events: tuple[MetricEvent, ...]) -> None:
        self.spool.append_many(events)
        for event in events:
            try:
                self.aim.send(event)
            except AimUnavailable:
                break
            self.spool.mark_sent(event.event_id)

    def log_metrics(
        self, env_steps: int, metrics: Mapping[str, int | float]
    ) -> None:
        events = tuple(
            MetricEvent.metrics_event(env_steps, {name: value})
            for name, value in metrics.items()
        )
        self._validate_env_steps(env_steps)
        if (
            self._last_metric_env_steps is not None
            and env_steps - self._last_metric_env_steps < self._aim_every_env_steps
        ):
            return
        self._emit_many(events)
        self._last_metric_env_steps = env_steps

    def log_episode_summary(
        self,
        *,
        env_steps: int,
        episode_return: int | float,
        episode_length: int,
    ) -> None:
        if type(episode_length) is not int:
            raise TypeError("episode_length must be an integer")
        if episode_length < 0:
            raise ValueError("episode_length must be non-negative")
        persisted_summary_sequence = max(
            (
                event.aim_step
                for event in self.spool.events
                if event.stream == "episode_summary"
            ),
            default=0,
        )
        next_summary_sequence = (
            max(self._summary_sequence, persisted_summary_sequence) + 1
        )
        events = MetricEvent.episode_summary(
            env_steps=env_steps,
            summary_sequence=next_summary_sequence,
            episode_return=episode_return,
            episode_length=episode_length,
        )
        self._validate_env_steps(env_steps)
        self._emit_many(events)
        self._summary_sequence = next_summary_sequence

    def log_episode(self, episode: Episode) -> Path | None:
        return self.rerun.log_episode(episode)

    def register_checkpoint(self, path: Path) -> None:
        del path

    def finish(self, final_metrics: Mapping[str, int | float]) -> None:
        if self._terminal_state != "active":
            return
        self._terminal_state = "finishing"
        try:
            objective_metric = self.context.objective.get("metric")
            if not self.context.objective:
                self._terminal_state = "finished"
                self._close_resources(suppress_errors=False)
                return
            if not isinstance(objective_metric, str) or not objective_metric:
                raise ValueError("objective.metric must be a non-empty string")
            if objective_metric not in final_metrics:
                raise ValueError(
                    f"final_metrics must contain objective metric {objective_metric!r}"
                )
            ordered_metrics = [
                (name, value)
                for name, value in final_metrics.items()
                if name != objective_metric
            ]
            ordered_metrics.append(
                (objective_metric, final_metrics[objective_metric])
            )
            events = tuple(
                MetricEvent.final(
                    env_steps=self._last_env_steps or 0,
                    metrics={name: value},
                    objective_metric=objective_metric,
                    finalized=name == objective_metric,
                )
                for name, value in ordered_metrics
            )
            self._emit_many(events)
        except (TypeError, ValueError) as exc:
            self._close_resources(suppress_errors=True)
            raise type(exc)(f"invalid final_metrics: {exc}") from exc
        except BaseException:
            self._close_resources(suppress_errors=True)
            raise
        self._terminal_state = "finished"
        self._close_resources(suppress_errors=False)

    def fail(self, error: BaseException) -> None:
        """Mark the run failed without publishing an objective or finalized marker."""

        if self._terminal_state != "active":
            return
        self._terminal_state = "failed"
        metadata = {"type": type(error).__name__}
        try:
            fail = getattr(self.aim, "fail", None)
            if callable(fail):
                fail(metadata)
        except BaseException:
            pass
        self._close_resources(suppress_errors=True)

    def abort(self, error: BaseException) -> None:
        self.fail(error)

    def _close_resources(self, *, suppress_errors: bool) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        for resource in (self.rerun, self.spool, self.aim):
            try:
                close = getattr(resource, "close", None)
                if callable(close):
                    close()
            except BaseException as error:
                errors.append(error)
        if errors and not suppress_errors:
            raise errors[0]
