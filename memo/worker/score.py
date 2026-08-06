from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from training_sdk.contract import ScoreConfig

WORST_MAGNITUDE = 1e30


class ScoreError(ValueError):
    """The metrics file does not contain a usable value for the score window."""


def compute_score(metrics_path: Path, spec: ScoreConfig) -> float:
    low, high = spec.window_steps
    selected: list[tuple[int, float]] = []
    for line in Path(metrics_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["step"])
        if low <= step <= high and spec.metric in row["metrics"]:
            selected.append((step, float(row["metrics"][spec.metric])))
    if not selected:
        raise ScoreError(
            f"no reported value for metric {spec.metric!r} with step in "
            f"[{low}, {high}]; the run finished without covering the score window"
        )
    selected.sort()
    values = [value for _, value in selected]
    if not all(math.isfinite(value) for value in values):
        if spec.non_finite == "worst":
            return -WORST_MAGNITUDE if spec.direction == "maximize" else WORST_MAGNITUDE
        return float(spec.non_finite)
    return float(
        {
            "mean": lambda: statistics.fmean(values),
            "median": lambda: statistics.median(values),
            "min": lambda: min(values),
            "max": lambda: max(values),
            "last": lambda: values[-1],
        }[spec.reduce]()
    )
