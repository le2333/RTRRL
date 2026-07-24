from __future__ import annotations

import hashlib
from typing import Any, Callable

from .context import RunContext
from .spool import AimUnavailable, MetricEvent


def _default_run_factory(**kwargs: Any) -> Any:
    from aim import Run
    from aim.sdk.repo_utils import get_repo

    run_hash = kwargs["run_hash"]
    repo = get_repo(kwargs.get("repo"))
    if not repo.run_exists(run_hash):
        run_tree = repo.request_tree(
            "meta",
            run_hash,
            read_only=False,
            from_union=False,
            no_cache=True,
        ).subtree(("meta", "chunks", run_hash))
        run_tree["sdk_precreated"] = True
        repo._all_run_hashes.cache_clear()
    kwargs["repo"] = repo
    return Run(**kwargs)


class AimAdapter:
    """Translate SDK events to the Aim API at one explicit boundary."""

    def __init__(
        self,
        *,
        run_factory: Callable[..., Any] | None = None,
        availability_errors: tuple[type[BaseException], ...] = (
            ConnectionError,
            TimeoutError,
        ),
        **run_options: Any,
    ) -> None:
        if run_factory is None:
            run_factory = _default_run_factory
        self._run_factory = run_factory
        self._run_options = run_options
        self._availability_errors = availability_errors
        self._run: Any | None = None
        self._context: RunContext | None = None
        self._closed = False

    def start(self, context: RunContext) -> None:
        self._context = context
        run_hash = hashlib.sha256(context.run_id.encode("utf-8")).hexdigest()[:24]
        try:
            run = self._run_factory(
                experiment=context.experiment_name,
                run_hash=run_hash,
                force_resume=True,
                **self._run_options,
            )
            self._run = run
            run.name = context.run_name
            run["hparams"] = context.hparams
            run["context"] = {
                "experiment_id": context.experiment_id,
                "run_id": context.run_id,
            }
        except self._availability_errors as exc:
            raise AimUnavailable("Aim is temporarily unavailable") from exc

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
            self._run.track(
                event.metric_value,
                name=event.metric_name,
                step=event.aim_step,
                context={"sdk_stream": event.stream},
                epoch=event.env_steps,
            )
            if event.kind == "final" and event.data["finalized"]:
                self._run["sdk/objective_metric"] = event.data["objective_metric"]
                self._run["sdk/finalized"] = True
            self._run[marker] = True
        except self._availability_errors as exc:
            raise AimUnavailable("Aim is temporarily unavailable") from exc

    def fail(self, metadata: dict[str, str]) -> None:
        if self._run is None:
            return
        try:
            self._run["sdk/failed"] = True
            self._run["sdk/error"] = dict(metadata)
        except self._availability_errors as exc:
            raise AimUnavailable("Aim is temporarily unavailable") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._run is None:
            return
        close = getattr(self._run, "close", None)
        if callable(close):
            close()
