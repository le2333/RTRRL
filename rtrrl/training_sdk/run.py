from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from .context import RunContext
from .spool import AimUnavailable, MetricEvent
from .types import Episode


class AimSink(Protocol):
    def start(self, context: RunContext) -> None: ...

    def send(self, event: MetricEvent) -> None: ...


class Spool(Protocol):
    @property
    def unsent_events(self) -> tuple[MetricEvent, ...]: ...

    def append(self, event: MetricEvent) -> None: ...

    def mark_sent(self, event_id: str) -> None: ...


class RerunSink(Protocol):
    def log_episode(self, episode: Episode) -> None: ...


class NullRerun:
    """Task 2 placeholder; Task 3 supplies the Rerun implementation."""

    def log_episode(self, episode: Episode) -> None:
        del episode


class TrainingRun:
    def __init__(
        self,
        context: RunContext,
        aim: AimSink,
        rerun: RerunSink,
        spool: Spool,
    ) -> None:
        self.context = context
        self.aim = aim
        self.rerun = rerun
        self.spool = spool
        self._last_env_steps: int | None = None
        self._last_metric_env_steps: int | None = None

        interval = context.logging.get("aim_every_env_steps", 1)
        if type(interval) is not int:
            raise TypeError("logging.aim_every_env_steps must be an integer")
        if interval <= 0:
            raise ValueError("logging.aim_every_env_steps must be positive")
        self._aim_every_env_steps = interval

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
        self.spool.append(event)
        try:
            self.aim.send(event)
        except AimUnavailable:
            return
        self.spool.mark_sent(event.event_id)

    def log_metrics(
        self, env_steps: int, metrics: Mapping[str, int | float]
    ) -> None:
        event = MetricEvent.metrics_event(env_steps, metrics)
        self._validate_env_steps(env_steps)
        if (
            self._last_metric_env_steps is not None
            and env_steps - self._last_metric_env_steps < self._aim_every_env_steps
        ):
            return
        self._emit(event)
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
        event = MetricEvent.episode_summary(
            env_steps=env_steps,
            episode_return=episode_return,
            episode_length=episode_length,
        )
        self._validate_env_steps(env_steps)
        self._emit(event)

    def log_episode(self, episode: Episode) -> None:
        self.rerun.log_episode(episode)

    def register_checkpoint(self, path: Path) -> None:
        del path

    def finish(self, final_metrics: Mapping[str, int | float]) -> None:
        objective_metric = self.context.objective.get("metric")
        if not self.context.objective:
            return
        if not isinstance(objective_metric, str) or not objective_metric:
            raise ValueError("objective.metric must be a non-empty string")
        if objective_metric not in final_metrics:
            raise ValueError(
                f"final_metrics must contain objective metric {objective_metric!r}"
            )
        try:
            event = MetricEvent.final(
                env_steps=self._last_env_steps or 0,
                metrics=final_metrics,
                objective_metric=objective_metric,
            )
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"invalid final_metrics: {exc}") from exc
        self._emit(event)
