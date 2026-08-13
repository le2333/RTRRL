from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

WORST_MAGNITUDE = 1e30

Reduce = Literal["mean", "median", "min", "max", "last"]
Direction = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class ScoreSpec:
    metric: str
    window_steps: tuple[int, int]
    reduce: Reduce
    direction: Direction
    non_finite: Literal["worst"] | float

    def __post_init__(self) -> None:
        if self.window_steps[0] > self.window_steps[1]:
            raise ValueError("window_steps must be ordered")
        if self.reduce not in {"mean", "median", "min", "max", "last"}:
            raise ValueError(f"unknown score reduce {self.reduce!r}")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError(f"unknown score direction {self.direction!r}")
        if isinstance(self.non_finite, str) and self.non_finite != "worst":
            raise ValueError(f"unknown score non_finite policy {self.non_finite!r}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ScoreSpec:
        window = cast(Sequence[object], value["window_steps"])
        low, high = window
        return cls(
            metric=str(value["metric"]),
            window_steps=(int(low), int(high)),
            reduce=cast(Reduce, value["reduce"]),
            direction=cast(Direction, value["direction"]),
            non_finite=cast(Literal["worst"] | float, value["non_finite"]),
        )


class ScoreError(ValueError):
    """The metrics file does not contain a usable value for the score window."""


def compute_score(metrics_path: Path, spec: ScoreSpec) -> float:
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
