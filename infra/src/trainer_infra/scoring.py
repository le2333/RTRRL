from __future__ import annotations

import json
import math
import statistics
from array import array
from collections.abc import Iterable, Mapping, Sequence
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
    with Path(metrics_path).open(encoding="utf-8") as lines:
        return score_lines(lines, spec)


def score_lines(lines: Iterable[str], spec: ScoreSpec) -> float:
    """Reduce the score window over a stream of metric rows.

    A finished twenty-million-step run writes several gigabytes of
    ``metrics.jsonl``, and the machine that scores it is the same micro
    instance that hosts the study. Holding the file, or one Python object per
    row, is what killed an HPO controller with its trials already finished and
    uploaded, so the reduction is folded row by row and only the fold survives
    the line it came from.
    """

    low, high = spec.window_steps
    window = _Window(spec.reduce)
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["step"])
        if low <= step <= high and spec.metric in row["metrics"]:
            window.admit(step, float(row["metrics"][spec.metric]))
    if window.empty:
        raise ScoreError(
            f"no reported value for metric {spec.metric!r} with step in "
            f"[{low}, {high}]; the run finished without covering the score window"
        )
    if not window.finite:
        if spec.non_finite == "worst":
            return -WORST_MAGNITUDE if spec.direction == "maximize" else WORST_MAGNITUDE
        return float(spec.non_finite)
    return window.reduced()


class _Window:
    """The reduction the score asks for, accumulated over the admitted rows.

    Every mode but ``median`` is a fixed number of registers. ``median`` is the
    one that needs the values themselves; it keeps them as doubles rather than
    as parsed rows, eight bytes against the hundreds a row occupied on the way
    in.
    """

    def __init__(self, reduce: Reduce) -> None:
        self._reduce = reduce
        self._count = 0
        self._finite = True
        self._total = 0.0
        self._correction = 0.0
        self._minimum = math.inf
        self._maximum = -math.inf
        self._last_step: float = -math.inf
        self._last_value = -math.inf
        self._values = array("d")

    @property
    def empty(self) -> bool:
        return self._count == 0

    @property
    def finite(self) -> bool:
        return self._finite

    def admit(self, step: int, value: float) -> None:
        self._count += 1
        if not self._finite:
            return
        if not math.isfinite(value):
            # One non-finite value decides the score by itself. The rows after
            # it still count towards the window being covered at all.
            self._finite = False
            return
        if self._reduce == "median":
            self._values.append(value)
        # Neumaier summation: a mean over millions of rows should not drift
        # with how many of them there were.
        total = self._total + value
        if abs(self._total) >= abs(value):
            self._correction += (self._total - total) + value
        else:
            self._correction += (value - total) + self._total
        self._total = total
        self._minimum = min(self._minimum, value)
        self._maximum = max(self._maximum, value)
        if (step, value) > (self._last_step, self._last_value):
            self._last_step = step
            self._last_value = value

    def reduced(self) -> float:
        if self._reduce == "mean":
            return (self._total + self._correction) / self._count
        if self._reduce == "median":
            return float(statistics.median(self._values))
        if self._reduce == "min":
            return self._minimum
        if self._reduce == "max":
            return self._maximum
        return self._last_value
