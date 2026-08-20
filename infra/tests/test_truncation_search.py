"""The equivalence rule and the bracketed search, as decisions over scores."""

from __future__ import annotations

import pytest

from trainer_infra.truncation import (
    DEFAULT_GRID,
    FULL,
    EquivalenceRule,
    Measured,
    Search,
    TruncationError,
    judge,
    ordered,
)

SEEDS = tuple(range(10))
COUNT = len(SEEDS)
BUDGET = 1_000_000


def measured(truncation, auc, final=None, *, seeds=SEEDS, total_steps=BUDGET):
    """One candidate's formal result, with the final values defaulting to AUC."""

    return Measured(
        truncation=truncation,
        seeds=seeds,
        auc=tuple(auc),
        final=tuple(final if final is not None else auc),
        total_steps=total_steps,
    )


def flat(value, count=COUNT):
    return (float(value),) * count


def spread(centre, step=1.0, count=COUNT):
    """Values around a centre, so a bootstrap has something to resample."""

    return tuple(centre + step * (index - (count - 1) / 2) for index in range(count))


# ------------------------------------------------------------------- the grid
def test_the_grid_is_ordered_shortest_first_with_the_untruncated_arm_last():
    assert ordered((64, 1, FULL, 4)) == (1, 4, 64, FULL)
    assert ordered(DEFAULT_GRID) == DEFAULT_GRID


def test_a_grid_without_the_untruncated_arm_has_no_reference():
    with pytest.raises(TruncationError, match="untruncated arm"):
        ordered((1, 4, 16))


@pytest.mark.parametrize(
    "grid,message",
    [
        ((), "at least one candidate"),
        ((0, FULL), "at least one step"),
        ((4, 4, FULL), "repeats a candidate"),
    ],
)
def test_a_grid_that_cannot_be_searched_says_so(grid, message):
    with pytest.raises(TruncationError, match=message):
        ordered(grid)


# ------------------------------------------------------------ the paired rule
def test_two_candidates_measured_on_different_seeds_are_not_a_comparison():
    with pytest.raises(TruncationError, match="no pairing"):
        judge(
            measured(4, flat(100.0), seeds=tuple(range(10, 20))),
            measured(FULL, flat(100.0)),
        )


def test_two_candidates_given_different_budgets_are_not_a_comparison():
    """Equal budget per candidate is the premise, so violating it is an error."""

    with pytest.raises(TruncationError, match="not how much each one gets"):
        judge(
            measured(4, flat(100.0), total_steps=BUDGET // 2),
            measured(FULL, flat(100.0)),
        )


def test_a_non_finite_score_is_a_failed_run_and_is_refused():
    with pytest.raises(TruncationError, match="non-finite"):
        measured(4, (float("nan"),) + flat(100.0, 9))


def test_a_result_must_carry_one_score_per_seed():
    with pytest.raises(TruncationError, match="for 10 seeds"):
        Measured(
            truncation=4,
            seeds=SEEDS,
            auc=flat(1.0, 3),
            final=flat(1.0, 3),
            total_steps=BUDGET,
        )


# ------------------------------------------------------------- the two margins
def test_an_identical_candidate_is_equivalent():
    verdict = judge(measured(4, spread(100.0)), measured(FULL, spread(100.0)))

    assert verdict.equivalent
    assert verdict.reason == "equivalent"
    assert verdict.auc.relative == pytest.approx(0.0)


def test_a_candidate_far_below_the_reference_is_not_equivalent():
    verdict = judge(measured(1, spread(70.0)), measured(FULL, spread(100.0)))

    assert not verdict.equivalent
    assert verdict.reason == "worse"
    assert verdict.auc.relative == pytest.approx(-0.30)
    assert verdict.auc.worse


def test_the_primary_margin_is_five_percent_and_the_secondary_is_ten():
    """Each margin is checked on its own metric and not on the other's."""

    rule = EquivalenceRule(resamples=200)
    reference = measured(FULL, spread(100.0), spread(100.0))

    # 8% down on AUC alone: inside the final margin, outside the primary one.
    auc_only = judge(measured(4, spread(92.0), spread(100.0)), reference, rule)
    # 8% down on the final value alone: inside its own margin of ten.
    final_only = judge(measured(4, spread(100.0), spread(92.0)), reference, rule)

    assert not auc_only.equivalent
    assert not auc_only.auc.within
    assert auc_only.final.within
    assert final_only.equivalent


def test_both_metrics_must_pass_for_a_candidate_to_be_equivalent():
    rule = EquivalenceRule(resamples=200)
    reference = measured(FULL, spread(100.0), spread(100.0))

    verdict = judge(measured(4, spread(100.0), spread(50.0)), reference, rule)

    assert verdict.auc.within
    assert not verdict.final.within
    assert not verdict.equivalent


def test_a_candidate_above_the_margin_is_reported_as_better_not_as_equivalent():
    """Shortening the gradient is not expected to help; say so rather than pass it."""

    verdict = judge(
        measured(1, spread(140.0)), measured(FULL, spread(100.0)), EquivalenceRule(resamples=200)
    )

    assert not verdict.equivalent
    assert verdict.reason == "better"
    assert verdict.auc.better


# --------------------------------------------------------------- the interval
def test_the_interval_and_not_the_estimate_decides():
    """Ten noisy seeds whose means happen to land close are not equivalent."""

    rule = EquivalenceRule(resamples=2000, seed=7)
    # Means 2% apart, which the point estimate alone would admit, but each
    # arm's own spread is enormous.
    reference = measured(FULL, spread(100.0, step=40.0))
    candidate = measured(4, spread(98.0, step=40.0))

    verdict = judge(candidate, reference, rule)

    assert verdict.auc.relative == pytest.approx(-0.02)
    low, high = verdict.auc.interval
    assert low < -rule.auc_margin or high > rule.auc_margin
    assert not verdict.auc.within


def test_the_interval_is_of_the_paired_difference():
    """Two arms that differ by a constant per seed have almost no spread to show.

    Unpaired resampling would find the arms' own spread instead, and the
    interval would widen with the task's seed variance rather than with the
    disagreement between the candidates.
    """

    rule = EquivalenceRule(resamples=2000, seed=3)
    walk = spread(100.0, step=30.0)
    reference = measured(FULL, walk)
    candidate = measured(4, tuple(value * 1.01 for value in walk))

    verdict = judge(candidate, reference, rule)

    low, high = verdict.auc.interval
    assert verdict.auc.within
    assert high - low < 0.01


def test_the_same_scores_give_the_same_interval_twice():
    rule = EquivalenceRule(resamples=500, seed=11)
    candidate = measured(4, spread(97.0, step=5.0))
    reference = measured(FULL, spread(100.0, step=5.0))

    assert (
        judge(candidate, reference, rule).auc.interval
        == judge(candidate, reference, rule).auc.interval
    )


def test_one_seed_has_no_spread_to_resample():
    verdict = judge(measured(4, (95.0,), seeds=(0,)), measured(FULL, (100.0,), seeds=(0,)))

    assert verdict.auc.interval == (pytest.approx(-0.05), pytest.approx(-0.05))


def test_a_reference_mean_of_zero_has_no_scale_to_be_a_fraction_of():
    verdict = judge(measured(4, spread(0.0, step=0.0)), measured(FULL, spread(0.0, step=0.0)))

    assert verdict.auc.relative == pytest.approx(0.0)


# ------------------------------------------------------------------ the search
def worse(truncation):
    return measured(truncation, spread(50.0, step=0.5))


def same(truncation):
    return measured(truncation, spread(100.0, step=0.5))


def test_the_search_asks_for_the_untruncated_arm_first():
    search = Search(rule=EquivalenceRule(resamples=200))

    assert search.next_candidate({}) == FULL
    with pytest.raises(TruncationError, match="has not been measured"):
        search.outcome({})


def test_the_search_bisects_the_ordered_grid():
    """Five candidates, and the crossing found without running all of them."""

    search = Search(rule=EquivalenceRule(resamples=200))
    results = {FULL: same(FULL)}

    asked = []
    while (candidate := search.next_candidate(results)) is not None:
        asked.append(candidate)
        # Equivalent from 16 upwards, so t_eq is 16 and 4 is the crossing.
        results[candidate] = same(candidate) if candidate in (16, 64) else worse(candidate)

    outcome = search.outcome(results)

    assert outcome.t_eq == 16
    assert outcome.settled
    assert 64 not in asked, "a bisection that reached 64 did not bisect"
    assert set(asked) == {16, 4}


def test_the_crossing_is_confirmed_on_the_adjacent_shorter_candidate():
    """A bisection can land on t_eq without ever running the one below it."""

    search = Search(grid=(1, 4, FULL), rule=EquivalenceRule(resamples=200))
    without = {FULL: same(FULL), 4: same(4)}

    unverified = search.outcome(without)

    assert unverified.t_eq == 4
    assert not unverified.verified
    assert unverified.outstanding == (1,)
    assert "unconfirmed" in unverified.statement()

    confirmed = search.outcome({**without, 1: worse(1)})

    assert confirmed.t_eq == 4
    assert confirmed.verified
    assert confirmed.settled
    assert search.next_candidate({**without, 1: worse(1)}) is None


def test_the_shortest_candidate_needs_no_confirmation_below_it():
    search = Search(grid=(1, 4, FULL), rule=EquivalenceRule(resamples=200))

    outcome = search.outcome({FULL: same(FULL), 4: same(4), 1: same(1)})

    assert outcome.t_eq == 1
    assert outcome.settled


def test_no_truncation_matching_is_itself_the_answer():
    """The untruncated arm is equivalent to itself, so this is a finding.

    Every truncation on the grid falling short is not a failure to search; it
    says the gradient has to reach the whole episode.
    """

    search = Search(grid=(1, 4, FULL), rule=EquivalenceRule(resamples=200))

    outcome = search.outcome({FULL: same(FULL), 4: worse(4), 1: worse(1)})

    assert outcome.t_eq == FULL
    assert outcome.settled
    statement = outcome.statement()
    assert "no truncation on [1, 4]" in statement
    assert "untruncated gradient is required" in statement


def test_a_non_monotone_grid_falls_back_to_enumerating_the_active_bracket():
    """A short candidate matching where a longer one does not has no crossing."""

    search = Search(rule=EquivalenceRule(resamples=200))
    results = {FULL: same(FULL), 4: same(4), 64: worse(64)}

    outcome = search.outcome(results)

    assert not outcome.monotone
    assert outcome.t_eq is None
    # The bracket is the span the contradiction sits in, and what is left of it
    # is what the fallback enumerates.
    assert outcome.outstanding == (16,)
    assert "not monotone" in outcome.statement()
    assert search.next_candidate(results) == 16


def test_the_settled_statement_refuses_to_overclaim():
    search = Search(grid=(1, 4, FULL), rule=EquivalenceRule(resamples=200))

    statement = search.outcome({FULL: same(FULL), 4: same(4), 1: worse(1)}).statement()

    assert "t_eq = 4" in statement
    assert "equal budget" in statement
    assert "Not a minimum truncation length" in statement


def test_a_candidate_off_the_grid_is_refused():
    search = Search(grid=(1, 4, FULL))

    with pytest.raises(TruncationError, match="not on the grid"):
        search.next_candidate({FULL: same(FULL), 7: same(7)})


def test_the_verdicts_come_back_shortest_first():
    search = Search(rule=EquivalenceRule(resamples=200))

    outcome = search.outcome({FULL: same(FULL), 16: same(16), 4: worse(4), 1: worse(1)})

    assert [verdict.truncation for verdict in outcome.verdicts] == [1, 4, 16]
    assert all(verdict.reference == FULL for verdict in outcome.verdicts)


def test_a_crossing_already_bracketed_asks_for_nothing_more():
    """16 short of the margin and 64 equivalent settles it: the crossing is there.

    Continuing to bisect downwards from zero would ask for 4, and then 1, and
    neither can move a crossing that monotonicity has already placed between
    two measured candidates. At ten formal seeds a candidate, that is twenty
    runs spent confirming what is known.
    """

    search = Search(rule=EquivalenceRule(resamples=200))
    results = {FULL: same(FULL), 64: same(64), 16: worse(16)}

    outcome = search.outcome(results)

    assert outcome.t_eq == 64
    assert outcome.verified
    assert outcome.settled
    assert outcome.outstanding == ()
    assert search.next_candidate(results) is None


def test_the_bracket_is_halved_from_whichever_end_is_still_open():
    """Both ends move, and the next candidate is the midpoint of what is left."""

    search = Search(rule=EquivalenceRule(resamples=200))
    equivalent_long = {FULL: same(FULL), 64: same(64)}

    # Open bracket [1, 4, 16]: its midpoint is 4, which splits it evenly.
    assert search.next_candidate(equivalent_long) == 4
    # 4 equivalent closes the bracket from above, leaving only 1 below it.
    assert search.next_candidate({**equivalent_long, 4: same(4)}) == 1
    # 4 short of the margin closes it from below instead, leaving 16.
    assert search.next_candidate({**equivalent_long, 4: worse(4)}) == 16
