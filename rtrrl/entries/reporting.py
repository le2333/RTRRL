"""The logger their training loop expects, writing to the reporter we score.

`train_rtrrl` was written against `logging_util.DummyLogger`, which is a `dict`
subclass with logging methods hung off it. That shape is load-bearing in two
places and neither is optional: the loop assigns `logger["best_eval_reward"]`
before evaluating and reads it back afterwards to decide what a new best is,
and it calls `finalize` twice, once from a `finally` and once after it. A
plain adapter with only a `log` method would raise a `KeyError` at the first
evaluation, which under a 2,000,000-step budget is forty minutes in.

So this subclasses `dict` as theirs does, and forwards the one call that
carries numbers. The rest exist because the loop calls them.

Two things are translated on the way through.

Their evaluation scalar is named `eval/rewards`, and the score every arm of
this comparison is read on is named `eval/episode/return`. They are the same
quantity -- the mean over the evaluation environments of the reward
accumulated up to each one's first termination -- so the rename is the whole
of the mapping, and it happens here rather than in the experiment file because
an experiment file that has to know an implementation's spelling is an
experiment file that cannot be moved between arms.

Their x-axis is not. The loop reports at `log_steps`, which counts scanned
iterations, while the environment steps those cost are `log_steps` times the
environment batch size, and the loop puts that second number in the payload
under `steps`. At the batch size of one this comparison runs at, the two
coincide; at any other they do not, and a score window measured in environment
steps would silently cover the wrong fraction of the run. The payload's own
count is therefore what is reported as the step, and the argument is the
fallback for the reports that do not carry one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol

# The scalar their loop computes at evaluation time, under the name the five
# arms are compared on. `eval/episode/best_return` is their running maximum of
# it, written only on an improvement; the score reduces with `max` over the
# window and so does not need it, but it is the reading their own runs are read
# on and it costs one line to keep.
RENAMED = {
    "eval/rewards": "eval/episode/return",
    "eval/best_eval_reward": "eval/episode/best_return",
}

# Their environment-step count, which this reports as the step rather than as a
# metric. Reporting it as both would give every plot a straight line through it.
STEP_KEY = "steps"

TRAIN_PREFIX = "train/"


class Destination(Protocol):
    """All of the reporter this file touches."""

    def report(self, step: int, metrics: Mapping[str, float]) -> None: ...


def scored_name(key: str) -> str:
    """The name the control plane knows a scalar of theirs by."""

    return RENAMED.get(key, f"{TRAIN_PREFIX}{key}")


class ReporterLogger(dict):
    """Their logger interface over our reporter."""

    def __init__(self, reporter: Destination) -> None:
        super().__init__()
        self._reporter = reporter
        self._step = 0
        # Seeded here as well as in their loop. Their assignment happens once,
        # early, and a revision that moved it below the first evaluation would
        # turn a comparison into a KeyError; this costs nothing and removes the
        # dependency on where their line sits.
        self["best_eval_reward"] = -math.inf

    def __repr__(self) -> str:
        return "ReporterLogger"

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        """Forward one report, renamed and counted in environment steps."""

        if not metrics:
            return
        payload = dict(metrics)
        counted = payload.pop(STEP_KEY, step)
        self._step = int(counted) if counted is not None else self._step
        self._reporter.report(
            self._step,
            {scored_name(name): float(value) for name, value in payload.items()},
        )

    def log_params(self, params_dict: Mapping[str, Any]) -> None:
        """Nothing to do: the run configuration is what produced these.

        The control plane wrote the parameters it sampled into the run config
        and the Aim sink recorded them from there before this process started.
        Logging the dataclass they were expanded into would record the same
        choices a second time under different names.
        """

    def finalize(self, all_param_norms: Any = None, x_vals: Any = None) -> None:
        """Called twice by their loop, from a `finally` and again after it."""

    def save_model(self, model: Any, filename: str = "model.ckpt") -> None:
        """Unreachable at `save_model=False`, which is the default they ship."""

    def log_video(
        self,
        name: str,
        frames: Any,
        step: int | None = None,
        fps: int = 30,
        **kwargs: Any,
    ) -> None:
        """Unreachable at `render=False`, which is what this arm runs at.

        Their loop guards the call with `args.env_params.render`, so a run that
        renders nothing never arrives here. It exists because the guard is
        theirs to change and a missing method would only be found by a run that
        had already paid for its rendering.
        """
