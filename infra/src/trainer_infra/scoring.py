from __future__ import annotations

import json
import math
import statistics
from array import array
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

WORST_MAGNITUDE = 1e30

Reduce = Literal["mean", "median", "min", "max", "last", "auc", "last_checkpoints"]
Direction = Literal["maximize", "minimize"]

POINT_REDUCES = frozenset({"mean", "median", "min", "max", "last"})
# What a checkpoint reduce reads is the evaluation curve: one point per
# measured step, whatever number of rows reported it. A point reduce reads
# the rows themselves, and the two answer different questions -- the mean of
# every row weights a checkpoint by how many episodes it happened to record.
TRAJECTORY_REDUCES = frozenset({"auc", "last_checkpoints"})


@dataclass(frozen=True)
class ScoreSpec:
    """What one number out of a run's metrics is, said completely.

    ``checkpoints`` is how many of the last measured steps ``last_checkpoints``
    averages -- the protocol's five, written down rather than assumed.

    ``episodes_per_checkpoint`` is the exactness the formal protocol claims: a
    checkpoint that did not report this many values for the metric did not run
    the protocol, and a score computed over it would be a different quantity
    wearing the same name. Left out, nothing is checked.
    """

    metric: str
    window_steps: tuple[int, int]
    reduce: Reduce
    direction: Direction
    non_finite: Literal["worst"] | float
    checkpoints: int | None = None
    episodes_per_checkpoint: int | None = None

    def __post_init__(self) -> None:
        if self.window_steps[0] > self.window_steps[1]:
            raise ValueError("window_steps must be ordered")
        if self.reduce not in POINT_REDUCES | TRAJECTORY_REDUCES:
            raise ValueError(f"unknown score reduce {self.reduce!r}")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError(f"unknown score direction {self.direction!r}")
        if isinstance(self.non_finite, str) and self.non_finite != "worst":
            raise ValueError(f"unknown score non_finite policy {self.non_finite!r}")
        if (self.reduce == "last_checkpoints") != (self.checkpoints is not None):
            raise ValueError(
                "checkpoints says how many of the last measured steps to average, "
                "so last_checkpoints requires it and no other reduce accepts it"
            )
        if self.checkpoints is not None and self.checkpoints < 1:
            raise ValueError("checkpoints must be positive")
        if self.episodes_per_checkpoint is not None:
            if self.reduce not in TRAJECTORY_REDUCES:
                raise ValueError(
                    "episodes_per_checkpoint counts the rows of one measured step, "
                    f"which the point reduce {self.reduce!r} never groups"
                )
            if self.episodes_per_checkpoint < 1:
                raise ValueError("episodes_per_checkpoint must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ScoreSpec:
        window = cast(Sequence[object], value["window_steps"])
        low, high = window
        checkpoints = value.get("checkpoints")
        episodes = value.get("episodes_per_checkpoint")
        return cls(
            metric=str(value["metric"]),
            window_steps=(int(low), int(high)),
            reduce=cast(Reduce, value["reduce"]),
            direction=cast(Direction, value["direction"]),
            non_finite=cast(Literal["worst"] | float, value["non_finite"]),
            checkpoints=None if checkpoints is None else int(checkpoints),
            episodes_per_checkpoint=None if episodes is None else int(episodes),
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
    window = _Window(spec.reduce) if spec.reduce in POINT_REDUCES else _Trajectory(spec)
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["step"])
        if low <= step <= high and spec.metric in row["metrics"]:
            window.admit(step, float(row["metrics"][spec.metric]))
    window.close()
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

    def close(self) -> None:
        """Nothing is held open: a point reduce finishes with its last row."""

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


class _Trajectory:
    """The evaluation curve, folded one checkpoint at a time.

    A checkpoint is a measured step, and it arrives as several rows: the
    protocol scores it on ten episodes, and each episode is reported on its
    own. So the fold is two-level -- the rows of one step become that
    checkpoint's mean, and the reduction runs over those means. Averaging the
    rows directly instead would weight each checkpoint by how many episodes it
    happened to report, which is exactly what the fixed episode count exists
    to stop varying.

    Rows for one metric arrive in step order, because a run writes them as it
    reaches them; a step that goes backwards means the file is not the
    trajectory it is being read as, and is refused rather than integrated.

    What is held is one open checkpoint, the checkpoint before it, and the
    last ``checkpoints`` means. None of that grows with the file.
    """

    def __init__(self, spec: ScoreSpec) -> None:
        self._reduce = spec.reduce
        self._expected = spec.episodes_per_checkpoint
        self._metric = spec.metric
        self._step: int | None = None
        self._total = 0.0
        self._count = 0
        self._finite = True
        self._checkpoints = 0
        self._first_step: int | None = None
        self._last_step: int | None = None
        self._previous: float | None = None
        self._area = 0.0
        self._recent: deque[float] = deque(maxlen=spec.checkpoints or 1)

    @property
    def empty(self) -> bool:
        return self._checkpoints == 0

    @property
    def finite(self) -> bool:
        return self._finite

    def admit(self, step: int, value: float) -> None:
        if self._step is not None and step < self._step:
            raise ScoreError(
                f"metric {self._metric!r} reported step {step} after step "
                f"{self._step}; a trajectory reduce reads the rows in the order "
                "the run measured them"
            )
        if self._step is not None and step != self._step:
            self.close()
        self._step = step
        self._count += 1
        if not math.isfinite(value):
            # One non-finite value decides the score by itself, the same way it
            # does for a point reduce. The rows after it still count towards
            # the checkpoint having been reported at all.
            self._finite = False
            return
        self._total += value

    def close(self) -> None:
        """Settle the open checkpoint, if a row ever opened one."""

        if self._step is None:
            return
        if self._expected is not None and self._count != self._expected:
            raise ScoreError(
                f"the checkpoint at {self._step} environment steps reported "
                f"{self._count} values for metric {self._metric!r}; the protocol "
                f"declares exactly {self._expected}"
            )
        if self._finite:
            mean = self._total / self._count
            if self._previous is not None and self._last_step is not None:
                self._area += (self._step - self._last_step) * (self._previous + mean) / 2
            if self._first_step is None:
                self._first_step = self._step
            self._previous = mean
            self._recent.append(mean)
        self._last_step = self._step
        self._checkpoints += 1
        self._step = None
        self._total = 0.0
        self._count = 0

    def reduced(self) -> float:
        if self._reduce == "auc":
            return self._integrated()
        if len(self._recent) < self._recent.maxlen:
            raise ScoreError(
                f"metric {self._metric!r} was measured at {self._checkpoints} "
                f"checkpoints in the window; averaging the last "
                f"{self._recent.maxlen} needs that many"
            )
        return sum(self._recent) / len(self._recent)

    def _integrated(self) -> float:
        """Area under the curve per environment step: a step-weighted mean.

        The endpoints are the first and last checkpoint the window admitted,
        not the window's own bounds -- extending to a bound nothing was
        measured at would extrapolate the policy into steps that were never
        evaluated. Between them the trapezoid rule spans whatever spacing the
        checkpoints have, so an interval missing its measurement is crossed by
        the line between its neighbours rather than dropped.

        Dividing by the span is what makes this readable beside the returns it
        integrates, and against a run of a different length. For one budget it
        is the raw integral times a constant, so it ranks trials identically.
        """

        if self._first_step is None or self._last_step is None:
            raise ScoreError(f"metric {self._metric!r} has no finite checkpoint")
        span = self._last_step - self._first_step
        if span <= 0:
            raise ScoreError(
                f"metric {self._metric!r} was measured at one step "
                f"({self._first_step}); an area needs an interval"
            )
        return self._area / span
