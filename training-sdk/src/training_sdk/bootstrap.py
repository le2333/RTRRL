from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path

from .aim_adapter import AimAdapter
from .context import RunContext
from .rerun_adapter import RerunAdapter
from .run import AimSink, RerunSink, TrainingRun
from .spool import EventSpool

RUN_CONTEXT_ENV = "TRAINER_RUN_CONTEXT_PATH"

_bootstrap_lock = threading.Lock()

AimFactory = Callable[[RunContext, Mapping[str, str]], AimSink]
RerunFactory = Callable[[RunContext], RerunSink]


def _default_aim_factory(
    context: RunContext, environ: Mapping[str, str]
) -> AimSink:
    del context
    return AimAdapter(repo=environ.get("AIM_REPO"))


def _default_rerun_factory(context: RunContext) -> RerunSink:
    return RerunAdapter(
        context,
        every_episodes=int(context.logging["rerun_every_episodes"]),
        root=context.artifact_directory,
    )


def bootstrap_from_environment(
    environ: Mapping[str, str] = os.environ,
    *,
    aim_factory: AimFactory | None = None,
    rerun_factory: RerunFactory | None = None,
) -> TrainingRun | None:
    value = environ.get(RUN_CONTEXT_ENV)
    if value is None:
        return None
    context_path = Path(value).resolve()

    from . import maybe_current_run, set_current_run

    with _bootstrap_lock:
        current = maybe_current_run()
        if current is not None:
            if current.context_path != context_path:
                raise RuntimeError(
                    "training SDK already uses a different run context"
                )
            return current

        context = RunContext.from_path(context_path)
        if aim_factory is None:
            aim_factory = _default_aim_factory
        if rerun_factory is None:
            rerun_factory = _default_rerun_factory

        run = TrainingRun(
            context,
            aim_factory(context, environ),
            rerun_factory(context),
            EventSpool(
                context.artifact_directory / "aim-buffer" / "events.jsonl"
            ),
            context_path=context_path,
        )
        run.start()
        set_current_run(run)
        return run
