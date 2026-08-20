"""Deciding, per seed, whether a fixed-evaluation curve collapsed and when.

The R2 question is not "did the return go down". It is whether a run that had
learned something gave a definite fraction of it back and stayed there, which
is a claim with three frozen numbers in it: what counts as the floor, how much
of the distance to it counts as a decline, and how long the decline has to hold
to be more than one bad evaluation. Those three are declared in a
:class:`CollapseSpec` before any formal curve is read, and every decision this
module emits carries the spec it was decided under, so a result cannot be
quoted without the definition that produced it.

Three things this deliberately does not do:

*It does not average seeds.* A collapse is a per-seed event with a step
attached; the mean of five curves has no first qualifying collapse and no
checkpoint to fork from. Each seed gets its own decision and they are reported
side by side.

*It does not skip non-finite evaluations.* A run whose evaluation returns NaN
has diverged, which is a stronger and more interesting outcome than a collapse,
and the arithmetic that would hide it is the ordinary arithmetic: ``max`` and
``>=`` are both silently false against a NaN, so a curve that went non-finite
would report no collapse and look healthy. The scan checks for it first and the
decision says so.

*It does not choose the peak by hindsight.* The drawdown at a checkpoint is
measured against the highest value seen up to and including that checkpoint,
because a run cannot give back a return it has not earned yet.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# The five per-group quantities R2 reads around a collapse, under the names the
# RTRRL entry publishes them by. They are the training-side explanation of what
# the evaluation curve did, so an event window that could not find them is an
# analysis with a hole in it rather than a shorter one.
TELEMETRY: tuple[str, ...] = (
    "raw_update_norm",
    "abs_td_error",
    "used_trace_norm",
    "clip_fraction",
    "realized_update_norm",
)
GROUPS: tuple[str, ...] = ("torso", "actor", "critic")
NORMALIZATIONS = ("peak_to_floor",)


class CollapseError(ValueError):
    """The curve or the specification cannot support a collapse decision."""


@dataclass(frozen=True)
class CollapseSpec:
    """The definition a collapse is decided under, frozen before it is read.

    ``random_floor`` is the return of an unlearned policy on this environment,
    which is what "toward the random floor" is measured against. It is required
    and has no default: guessing it would make the whole statistic a function
    of an unrecorded choice.

    ``normalization`` names the formula rather than describing it, so that a
    decision document says which one produced it. There is one:
    ``peak_to_floor``, ``D = (peak - value) / (peak - floor)``, zero at the
    peak and one at the floor.

    ``decline`` is how much of that distance a qualifying collapse gives back,
    ``sustain`` is how many consecutive checkpoints it must hold for, and
    ``recovery`` is the drawdown at or below which the run is deemed to have
    come back.
    """

    metric: str
    random_floor: float
    normalization: str = "peak_to_floor"
    decline: float = 0.5
    sustain: int = 2
    recovery: float = 0.2

    def __post_init__(self) -> None:
        if self.normalization not in NORMALIZATIONS:
            raise CollapseError(
                f"unknown drawdown normalization {self.normalization!r}; "
                f"the formulas are {', '.join(NORMALIZATIONS)}"
            )
        if not math.isfinite(self.random_floor):
            raise CollapseError("the random floor must be a finite return")
        if not 0.0 < self.decline <= 1.0:
            raise CollapseError("a qualifying decline is a fraction of the distance")
        if self.sustain < 1:
            raise CollapseError("a collapse must be sustained for a checkpoint or more")
        if not 0.0 <= self.recovery < self.decline:
            raise CollapseError(
                "recovery must be a smaller drawdown than the decline that "
                "qualified, or every collapse recovers at the moment it begins"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CollapseSpec:
        for name in ("metric", "random_floor"):
            if value.get(name) is None:
                raise CollapseError(
                    f"the collapse specification does not say {name!r}. The "
                    "random floor is the return of an unlearned policy on this "
                    "environment: it is measured, frozen before any formal "
                    "curve is read, and there is no default for it because a "
                    "guess would make every drawdown a function of the guess"
                )
        known = {name: value[name] for name in _FIELDS if name in value}
        return cls(**known)  # type: ignore[arg-type]

    def as_mapping(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _FIELDS}


_FIELDS = (
    "metric",
    "random_floor",
    "normalization",
    "decline",
    "sustain",
    "recovery",
)


@dataclass(frozen=True)
class Collapse:
    """The first qualifying collapse of one curve, and what surrounds it."""

    step: int
    drawdown: float
    sustained: int
    peak_step: int
    peak: float
    trough_step: int
    trough: float
    recovered_step: int | None = None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "drawdown": self.drawdown,
            "sustained": self.sustained,
            "peak_step": self.peak_step,
            "peak": self.peak,
            "trough_step": self.trough_step,
            "trough": self.trough,
            "recovered_step": self.recovered_step,
        }


@dataclass(frozen=True)
class SeedDecision:
    """One seed's answer, and enough of the working to argue with it.

    ``verdict`` is one of ``collapsed``, ``steady``, ``non_finite`` or
    ``never_learned``. The last two are not variants of "no collapse": a run
    that diverged and a run that never got above the floor are different
    failures, and folding either into "steady" would report a broken run as a
    stable one.
    """

    run_id: str
    seed: int | None
    verdict: str
    reason: str
    spec: CollapseSpec
    checkpoints: int
    max_drawdown: float | None = None
    max_drawdown_step: int | None = None
    peak: float | None = None
    peak_step: int | None = None
    collapse: Collapse | None = None
    non_finite_step: int | None = None
    windows: dict[str, list[dict[str, float]]] = field(default_factory=dict)

    @property
    def collapsed(self) -> bool:
        return self.verdict == "collapsed"

    def as_mapping(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "verdict": self.verdict,
            "reason": self.reason,
            "spec": self.spec.as_mapping(),
            "checkpoints": self.checkpoints,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_step": self.max_drawdown_step,
            "peak": self.peak,
            "peak_step": self.peak_step,
            "collapse": None if self.collapse is None else self.collapse.as_mapping(),
            "non_finite_step": self.non_finite_step,
            "windows": self.windows,
        }


def drawdowns(
    curve: Sequence[tuple[int, float]], spec: CollapseSpec
) -> list[tuple[int, float]]:
    """The normalized drawdown at each checkpoint, against the running peak.

    Defined only once the run has been above the floor: before that the
    denominator is zero or negative and the ratio would be a number with no
    meaning rather than a large drawdown. Those checkpoints are omitted, which
    is why a curve that never learns produces an empty list rather than a flat
    one.
    """

    found: list[tuple[int, float]] = []
    peak = -math.inf
    for step, value in curve:
        peak = max(peak, value)
        distance = peak - spec.random_floor
        if distance <= 0.0:
            continue
        found.append((step, (peak - value) / distance))
    return found


def detect(curve: Sequence[tuple[int, float]], spec: CollapseSpec) -> SeedDecision:
    """Decide one curve, which is one seed of one method on one environment."""

    return _decide(curve, spec, run_id="", seed=None)


def analyze(
    source: Any,
    spec: CollapseSpec,
    *,
    run_id: str = "",
    seed: int | None = None,
    window_steps: int = 0,
    telemetry: Sequence[str] = (),
) -> SeedDecision:
    """Read one run's metrics artifact and decide it.

    The rows are read twice when a window is asked for, because the window is
    centred on a step that the first pass is what finds. A metrics artifact is
    gigabytes, so ``source`` is something that can be *re-read* -- a path, a
    sequence, or a factory -- rather than something that can be held. A
    one-shot iterator is refused instead of quietly yielding an empty window on
    the second pass.
    """

    rows = _rereadable(source)
    curve = evaluation_curve(rows, spec.metric)
    decision = _decide(curve, spec, run_id=run_id, seed=seed)
    if not window_steps or decision.collapse is None:
        return decision
    return replace(
        decision,
        windows=event_window(
            rows,
            around=decision.collapse.step,
            width=window_steps,
            names=tuple(telemetry) or training_series(),
        ),
    )


def training_series(
    groups: Sequence[str] = GROUPS, quantities: Sequence[str] = TELEMETRY
) -> tuple[str, ...]:
    """The metric names an event window reads, one per group per quantity."""

    return tuple(
        f"train/episode/update.{group}.{quantity}"
        for group in groups
        for quantity in quantities
    )


def evaluation_curve(rows: Iterable[str], metric: str) -> list[tuple[int, float]]:
    """One value per checkpoint: the mean over that checkpoint's episodes.

    The metrics artifact keeps one row per episode, and a fixed evaluation is
    several episodes reported at the same step. What the curve is made of is
    their mean, so a checkpoint counts once however many episodes it ran.

    A non-finite episode makes its checkpoint non-finite rather than being
    dropped from the mean. One diverged episode is the run diverging.
    """

    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for row in _parsed(rows):
        metrics = row.get("metrics") or {}
        if metric not in metrics:
            continue
        step = int(row["step"])
        value = float(metrics[metric])
        counts[step] = counts.get(step, 0) + 1
        # A non-finite episode carries through the sum, so its checkpoint is
        # non-finite too. Dropping it would report the mean of the episodes
        # that happened to stay finite, which is a healthier curve than the run
        # had.
        totals[step] = totals.get(step, 0.0) + value
    return [(step, totals[step] / counts[step]) for step in sorted(totals)]


def event_window(
    rows: Iterable[str],
    *,
    around: int,
    width: int,
    names: Sequence[str],
) -> dict[str, list[dict[str, float]]]:
    """Every reading of the named series within ``width`` steps of the event.

    Aligned by keeping the step rather than by re-basing it: a window whose
    x-axis had been shifted to zero could not be laid over the evaluation curve
    it is supposed to explain.
    """

    wanted = set(names)
    low, high = around - width, around + width
    found: dict[str, list[dict[str, float]]] = {name: [] for name in names}
    for row in _parsed(rows):
        step = int(row["step"])
        if not low <= step <= high:
            continue
        for name, value in (row.get("metrics") or {}).items():
            if name in wanted:
                found[name].append({"step": step, "value": float(value)})
    return {name: values for name, values in found.items() if values}


def decisions(document: Iterable[SeedDecision]) -> dict[str, Any]:
    """The per-seed decisions as one document, with nothing reduced away."""

    found = list(document)
    return {
        "seeds": [decision.as_mapping() for decision in found],
        "collapsed": sorted(
            decision.run_id for decision in found if decision.verdict == "collapsed"
        ),
        "non_finite": sorted(
            decision.run_id for decision in found if decision.verdict == "non_finite"
        ),
        "never_learned": sorted(
            decision.run_id for decision in found if decision.verdict == "never_learned"
        ),
    }


def _decide(
    curve: Sequence[tuple[int, float]],
    spec: CollapseSpec,
    *,
    run_id: str,
    seed: int | None,
) -> SeedDecision:
    def answer(verdict: str, reason: str, **rest: Any) -> SeedDecision:
        return SeedDecision(
            run_id=run_id,
            seed=seed,
            verdict=verdict,
            reason=reason,
            spec=spec,
            checkpoints=len(curve),
            **rest,
        )

    if not curve:
        return answer("never_learned", "the run reported no evaluation checkpoint")

    # First, and before any comparison: a NaN loses every comparison it is in,
    # so a curve scanned for a maximum first would report the diverged run as
    # the steadiest one on the page.
    for step, value in curve:
        if not math.isfinite(value):
            return answer(
                "non_finite",
                f"evaluation at {step} steps is not finite; the run diverged "
                "rather than declining, which no drawdown describes",
                non_finite_step=step,
            )

    measured = drawdowns(curve, spec)
    if not measured:
        return answer(
            "never_learned",
            f"no checkpoint rose above the random floor of {spec.random_floor:g}, "
            "so there is no learned peak to decline from",
        )

    peak_step, peak = max(curve, key=lambda point: (point[1], -point[0]))
    worst_step, worst = max(measured, key=lambda point: (point[1], -point[0]))
    common = {
        "max_drawdown": worst,
        "max_drawdown_step": worst_step,
        "peak": peak,
        "peak_step": peak_step,
    }

    collapse = _first_qualifying(curve, measured, spec)
    if collapse is None:
        return answer(
            "steady",
            f"no drawdown of {spec.decline:g} held for {spec.sustain} consecutive "
            f"checkpoints; the worst was {worst:.3f} at {worst_step} steps",
            **common,
        )
    return answer(
        "collapsed",
        f"a drawdown of {collapse.drawdown:.3f} at {collapse.step} steps held for "
        f"{collapse.sustained} checkpoints",
        collapse=collapse,
        **common,
    )


def _first_qualifying(
    curve: Sequence[tuple[int, float]],
    measured: Sequence[tuple[int, float]],
    spec: CollapseSpec,
) -> Collapse | None:
    """The first decline that qualifies, and only the first, as R2 asks."""

    values = dict(curve)
    peaks = _running_peaks(curve)
    for index, (step, drawdown) in enumerate(measured):
        held = measured[index : index + spec.sustain]
        if len(held) < spec.sustain:
            return None
        if any(value < spec.decline for _, value in held):
            continue
        after = measured[index:]
        trough_step, _ = min(after, key=lambda point: (values[point[0]], point[0]))
        sustained = 0
        for _, value in after:
            if value < spec.decline:
                break
            sustained += 1
        return Collapse(
            step=step,
            drawdown=drawdown,
            sustained=sustained,
            peak_step=peaks[step][0],
            peak=peaks[step][1],
            trough_step=trough_step,
            trough=values[trough_step],
            recovered_step=next(
                (later for later, value in after if value <= spec.recovery), None
            ),
        )
    return None


def _running_peaks(curve: Sequence[tuple[int, float]]) -> dict[int, tuple[int, float]]:
    peaks: dict[int, tuple[int, float]] = {}
    best = (curve[0][0], -math.inf)
    for step, value in curve:
        if value > best[1]:
            best = (step, value)
        peaks[step] = best
    return peaks


class _Reread:
    """A metrics file as something iterable more than once, a line at a time.

    Neither the file nor a list of its lines is held: each pass opens it again.
    A finished twenty-million-step run writes several gigabytes here, and the
    machine reading it is the micro instance that hosts the study.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def __iter__(self) -> Iterator[str]:
        with self._path.open(encoding="utf-8") as handle:
            yield from handle


def _rereadable(source: Any) -> Iterable[str]:
    """Rows that can be read twice: a path, a sequence, or a factory of them."""

    if isinstance(source, (str, Path)):
        return _Reread(Path(source))
    if callable(source):
        return _Factory(source)
    if isinstance(source, Sequence):
        return source
    raise CollapseError(
        "a collapse analysis reads its rows twice -- once for the curve and "
        "once for the window around the event it finds -- so it needs a path, "
        "a sequence or a factory rather than a one-shot iterator"
    )


class _Factory:
    def __init__(self, make: Any) -> None:
        self._make = make

    def __iter__(self) -> Iterator[str]:
        return iter(self._make())


def _parsed(rows: Iterable[str]) -> Iterator[dict[str, Any]]:
    for row in rows:
        if not row.strip():
            continue
        yield json.loads(row)
