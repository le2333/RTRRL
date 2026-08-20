"""Locating the shortest truncation that matches the untruncated gradient.

The R1 truncation question is: how far back does DRQN's gradient have to reach
before it stops mattering? Answering it means running the same configuration at
several truncations and deciding, from the formal seeds, which of them perform
the same as the untruncated arm. Two things live here, both pure decisions over
results that already exist:

**The equivalence rule.** Preregistered, so that "the same" is a rule applied to
numbers rather than a reading of a plot. Two candidates are equivalent when the
bootstrap confidence interval of their relative difference lies inside the
margin -- on the primary AUC and on the secondary final return, both.

**The bracketed search.** The grid is ordered, so the smallest equivalent
truncation can be found without running every candidate. What the search
changes is which candidates run; every candidate that does run gets the same
training and evaluation budget, and a comparison across different budgets is
refused rather than reported.

Nothing here launches anything or reads a metrics file. It is handed the scores
a formal launch already produced and says which candidate to run next, or what
the answer is.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

FULL = "full"
# A truncation is a number of transitions, or the whole episode. ``full`` is
# the longest and sorts last, which is what makes the grid ordered.
Truncation = int | str

DEFAULT_GRID: tuple[Truncation, ...] = (1, 4, 16, 64, FULL)
# Used only to refine a crossing that falls between 16 and 64, and only when
# the crossing is worth resolving more finely than the grid already does.
REFINEMENT: tuple[Truncation, ...] = (24, 32, 48)


class TruncationError(ValueError):
    """The candidates cannot be compared, or the grid cannot be searched."""


def ordered(grid: Sequence[Truncation]) -> tuple[Truncation, ...]:
    """The grid shortest-first, with ``full`` last, checked for sanity.

    The reference is the longest candidate, so a grid that does not end at
    ``full`` is a grid whose comparison has no untruncated arm to be equivalent
    to.
    """

    if not grid:
        raise TruncationError("a truncation grid names at least one candidate")
    lengths = [candidate for candidate in grid if candidate != FULL]
    if any(not isinstance(candidate, int) for candidate in lengths):
        raise TruncationError(f"a truncation is an integer or {FULL!r}: {grid!r}")
    if any(candidate < 1 for candidate in lengths):  # type: ignore[operator]
        raise TruncationError("a truncation reaches back at least one step")
    if len(set(grid)) != len(grid):
        raise TruncationError(f"the grid repeats a candidate: {grid!r}")
    if FULL not in grid:
        raise TruncationError(
            f"the grid must contain {FULL!r}: it is the untruncated arm every "
            "shorter candidate is judged equivalent to"
        )
    return (*sorted(lengths), FULL)  # type: ignore[type-var]


@dataclass(frozen=True)
class EquivalenceRule:
    """The preregistered margins, and how the interval around them is drawn.

    Two margins because the protocol declares two: the primary AUC is held to
    5% and the secondary final return to 10%. Both must pass -- a candidate
    that integrates the same curve but ends somewhere else has not performed
    the same.

    The comparison is two-sided. A candidate whose interval sits *above* the
    margin is reported as ``better`` rather than as equivalent: shortening the
    gradient is not expected to improve the result, and a candidate that
    appears to is a finding about the setup rather than a shorter answer to
    quote.

    ``resamples`` and ``seed`` fix the interval, so two people reading the same
    scores reach the same verdict.
    """

    auc_margin: float = 0.05
    final_margin: float = 0.10
    confidence: float = 0.95
    resamples: int = 10_000
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("auc_margin", "final_margin"):
            if getattr(self, name) <= 0.0:
                raise TruncationError(f"{name} must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise TruncationError("confidence is a probability strictly inside (0, 1)")
        if self.resamples < 1:
            raise TruncationError("resamples must be positive")


@dataclass(frozen=True)
class Measured:
    """One candidate's formal result: the two scores, per seed, at one budget.

    ``seeds`` is carried rather than assumed so that the pairing is checkable.
    A bootstrap over seeds only removes seed variance from the comparison if
    both candidates were measured on the same seeds, and two candidates scored
    on different seeds are a different comparison wearing this one's name.

    ``total_steps`` is the budget the candidate was given. The search changes
    which candidates run, never how much they get, so a comparison across two
    budgets is refused here rather than reported with a caveat.
    """

    truncation: Truncation
    seeds: tuple[int, ...]
    auc: tuple[float, ...]
    final: tuple[float, ...]
    total_steps: int

    def __post_init__(self) -> None:
        if not self.seeds:
            raise TruncationError(
                f"candidate {self.truncation!r} names no seeds; a formal result is "
                "measured on the protocol's fresh seeds"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise TruncationError(f"candidate {self.truncation!r} repeats a seed")
        for name in ("auc", "final"):
            values = getattr(self, name)
            if len(values) != len(self.seeds):
                raise TruncationError(
                    f"candidate {self.truncation!r} has {len(values)} {name} values "
                    f"for {len(self.seeds)} seeds"
                )
            if any(not math.isfinite(value) for value in values):
                raise TruncationError(
                    f"candidate {self.truncation!r} has a non-finite {name}; a "
                    "non-finite score is a failed run, not a low one"
                )
        if self.total_steps < 1:
            raise TruncationError("total_steps must be positive")


@dataclass(frozen=True)
class MetricVerdict:
    """One metric's relative difference, its interval, and the margin it met."""

    name: str
    relative: float
    interval: tuple[float, float]
    margin: float

    @property
    def within(self) -> bool:
        """The whole interval inside the margin, not merely the estimate.

        Holding the point estimate alone would call ten noisy seeds equivalent
        whenever their means happened to land close.
        """

        low, high = self.interval
        return low >= -self.margin and high <= self.margin

    @property
    def worse(self) -> bool:
        return self.interval[0] < -self.margin

    @property
    def better(self) -> bool:
        return self.interval[1] > self.margin


@dataclass(frozen=True)
class Verdict:
    """Whether one candidate performed the same as the reference, and how."""

    truncation: Truncation
    reference: Truncation
    auc: MetricVerdict
    final: MetricVerdict

    @property
    def equivalent(self) -> bool:
        return self.auc.within and self.final.within

    @property
    def reason(self) -> str:
        if self.equivalent:
            return "equivalent"
        failed = [reading for reading in (self.auc, self.final) if not reading.within]
        if all(reading.better for reading in failed):
            return "better"
        return "worse"


def _relative(candidate: Sequence[float], reference: Sequence[float]) -> float:
    """The candidate's shortfall as a fraction of the reference's own scale.

    Dividing by the reference's magnitude keeps the margin readable as a
    percentage on tasks whose returns are of any size. A reference mean of zero
    has no scale to be a fraction of, so the difference is reported as is.
    """

    base = fmean(reference)
    difference = fmean(candidate) - base
    return difference if base == 0.0 else difference / abs(base)


def _interval(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    rule: EquivalenceRule,
    stream: random.Random,
) -> tuple[float, float]:
    """A percentile interval for the relative difference, resampled by seed.

    One resample draws seeds, not values: the same drawn seed contributes its
    candidate score and its reference score together, so the interval is of
    the paired difference and not of two independent spreads.
    """

    count = len(candidate)
    if count == 1:
        # One seed has no spread to resample. The interval is the estimate, and
        # the caller learns nothing about it that the estimate did not say.
        estimate = _relative(candidate, reference)
        return (estimate, estimate)
    drawn = []
    for _ in range(rule.resamples):
        picks = [stream.randrange(count) for _ in range(count)]
        drawn.append(
            _relative(
                [candidate[pick] for pick in picks],
                [reference[pick] for pick in picks],
            )
        )
    drawn.sort()
    tail = (1.0 - rule.confidence) / 2.0
    return (_percentile(drawn, tail), _percentile(drawn, 1.0 - tail))


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def judge(candidate: Measured, reference: Measured, rule: EquivalenceRule | None = None) -> Verdict:
    """Apply the preregistered rule to one candidate against the reference."""

    rule = rule or EquivalenceRule()
    if candidate.seeds != reference.seeds:
        raise TruncationError(
            f"candidate {candidate.truncation!r} was measured on seeds "
            f"{list(candidate.seeds)} and the reference on {list(reference.seeds)}; "
            "the rule is a paired comparison and has no pairing here"
        )
    if candidate.total_steps != reference.total_steps:
        raise TruncationError(
            f"candidate {candidate.truncation!r} ran {candidate.total_steps} steps "
            f"against the reference's {reference.total_steps}; the search chooses "
            "which candidates run, not how much each one gets"
        )
    stream = random.Random(rule.seed)
    return Verdict(
        truncation=candidate.truncation,
        reference=reference.truncation,
        auc=MetricVerdict(
            name="auc",
            relative=_relative(candidate.auc, reference.auc),
            interval=_interval(candidate.auc, reference.auc, rule=rule, stream=stream),
            margin=rule.auc_margin,
        ),
        final=MetricVerdict(
            name="final",
            relative=_relative(candidate.final, reference.final),
            interval=_interval(candidate.final, reference.final, rule=rule, stream=stream),
            margin=rule.final_margin,
        ),
    )


@dataclass(frozen=True)
class Outcome:
    """What the measured candidates say, and what would still settle it."""

    grid: tuple[Truncation, ...]
    reference: Truncation
    verdicts: tuple[Verdict, ...]
    t_eq: Truncation | None
    verified: bool
    monotone: bool
    outstanding: tuple[Truncation, ...]

    @property
    def settled(self) -> bool:
        """A shortest equivalent truncation, with the crossing confirmed."""

        return self.t_eq is not None and self.verified and self.monotone

    def statement(self) -> str:
        """The claim this outcome licenses, with its qualification attached."""

        if not self.monotone:
            return (
                "performance is not monotone in t on this grid, so a bracketed "
                "search cannot locate a crossing; enumerate "
                f"{list(self.outstanding)}"
            )
        if self.t_eq == self.reference and self.verified:
            # The reference is equivalent to itself, so this is not a failure to
            # find an answer: the answer is that no truncation on the grid does.
            return (
                f"no truncation on {[c for c in self.grid if c != self.reference]} is "
                f"equivalent to {self.reference!r} at equal budget; the untruncated "
                "gradient is required on this grid"
            )
        if not self.verified:
            return (
                f"t_eq is at most {self.t_eq!r}, unconfirmed: the adjacent shorter "
                f"candidate {list(self.outstanding)} has not been measured"
            )
        return (
            f"t_eq = {self.t_eq!r} under the tuned configuration, at equal budget. "
            "Not a minimum truncation length independent of optimizer, learner or "
            "search space."
        )


@dataclass(frozen=True)
class Search:
    """A bracketed search for the shortest equivalent truncation.

    Written as a decision over what has been measured rather than as a loop
    that measures: a formal candidate is ten runs on a Batch queue, so the
    caller launches them and comes back. ``next_candidate`` says what to run,
    ``outcome`` says what the answer is, and both are pure.
    """

    grid: tuple[Truncation, ...] = DEFAULT_GRID
    rule: EquivalenceRule = EquivalenceRule()

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", ordered(self.grid))

    @property
    def reference(self) -> Truncation:
        return self.grid[-1]

    def _judged(self, measured: Mapping[Truncation, Measured]) -> dict[Truncation, Verdict]:
        reference = measured[self.reference]
        return {
            candidate: judge(result, reference, self.rule)
            for candidate, result in measured.items()
            if candidate != self.reference
        }

    def _unknown(self, measured: Mapping[Truncation, Measured]) -> tuple[Truncation, ...]:
        return tuple(candidate for candidate in self.grid if candidate not in measured)

    def next_candidate(self, measured: Mapping[Truncation, Measured]) -> Truncation | None:
        """The candidate to run next, or ``None`` when nothing more would help."""

        for candidate in measured:
            if candidate not in self.grid:
                raise TruncationError(
                    f"{candidate!r} was measured but is not on the grid {list(self.grid)}"
                )
        if self.reference not in measured:
            # Nothing is equivalent to an arm that has not run.
            return self.reference
        outcome = self.outcome(measured)
        return outcome.outstanding[0] if outcome.outstanding else None

    def outcome(self, measured: Mapping[Truncation, Measured]) -> Outcome:
        """Read the measured candidates as far as they determine an answer."""

        if self.reference not in measured:
            raise TruncationError(
                f"the reference {self.reference!r} has not been measured, so no "
                "candidate has anything to be equivalent to"
            )
        verdicts = self._judged(measured)
        equivalent = {candidate: verdict.equivalent for candidate, verdict in verdicts.items()}
        equivalent[self.reference] = True

        monotone, bracket = self._monotone(equivalent)
        ordered_verdicts = tuple(
            verdicts[candidate] for candidate in self.grid if candidate in verdicts
        )
        if not monotone:
            return Outcome(
                grid=self.grid,
                reference=self.reference,
                verdicts=ordered_verdicts,
                t_eq=None,
                verified=False,
                monotone=False,
                outstanding=tuple(candidate for candidate in bracket if candidate not in measured),
            )

        # What the measurements already determine: the shortest candidate known
        # to be equivalent. The reference is always one of them, so this exists.
        # Under monotonicity nothing between it and the reference needs running.
        shortest = min(
            index for index, candidate in enumerate(self.grid) if equivalent.get(candidate, False)
        )
        t_eq = self.grid[shortest]
        outstanding = self._bisected(equivalent, shortest)
        return Outcome(
            grid=self.grid,
            reference=self.reference,
            verdicts=ordered_verdicts,
            t_eq=t_eq,
            verified=not outstanding,
            monotone=True,
            outstanding=outstanding,
        )

    def _bisected(
        self, equivalent: Mapping[Truncation, bool], shortest: int
    ) -> tuple[Truncation, ...]:
        """The next candidate that would settle the crossing, bisecting for it.

        The bracket is what is still open: below the shortest candidate known
        to be equivalent, and above the longest known *not* to be. Both ends
        matter. Starting from zero regardless would keep asking for candidates
        that cannot change the answer -- with 16 measured and short of the
        margin and 64 measured and equivalent, the crossing is already between
        them, and running 4 and then 1 spends twenty formal runs to confirm
        something monotonicity already settled.
        """

        short_of_it = [
            index
            for index, candidate in enumerate(self.grid)
            if candidate in equivalent and not equivalent[candidate]
        ]
        low = max(short_of_it) + 1 if short_of_it else 0
        high = shortest
        while low < high:
            middle = (low + high) // 2
            candidate = self.grid[middle]
            if candidate not in equivalent:
                return (candidate,)
            if equivalent[candidate]:
                high = middle
            else:
                low = middle + 1
        return ()

    def _monotone(
        self, equivalent: Mapping[Truncation, bool]
    ) -> tuple[bool, tuple[Truncation, ...]]:
        """Whether a shorter candidate ever matched where a longer one did not.

        Monotonicity is what a bracketed search assumes. Where the measurements
        contradict it the search has no crossing to find, and the honest answer
        is the bracket the contradiction sits in, enumerated.
        """

        known = [
            (index, candidate)
            for index, candidate in enumerate(self.grid)
            if candidate in equivalent
        ]
        for position, (index, candidate) in enumerate(known):
            if not equivalent[candidate]:
                continue
            for later_index, later in known[position + 1 :]:
                if not equivalent[later]:
                    return False, self.grid[index : later_index + 1]
        return True, ()
