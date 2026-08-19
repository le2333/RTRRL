"""What the collapse detector calls a collapse, and what it refuses to.

The definition has three frozen numbers in it, so most of these cases are
curves built to sit either side of one of them: a decline just short of the
threshold, a decline that reaches it for one checkpoint only, one that holds,
and one that comes back. The remaining cases are the ones where the arithmetic
would otherwise lie -- a run that diverged, a run that never learned, and a
peak chosen with hindsight.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from trainer_infra.collapse import (
    CollapseError,
    CollapseSpec,
    analyze,
    decisions,
    detect,
    drawdowns,
    evaluation_curve,
    event_window,
    training_series,
)

FLOOR = 0.0
PEAK = 100.0
EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


def spec(**overrides) -> CollapseSpec:
    return CollapseSpec(
        **{
            "metric": "eval/episode/return",
            "random_floor": FLOOR,
            **overrides,
        }
    )


def curve(*values: float, every: int = 10000) -> list[tuple[int, float]]:
    return [(every * (index + 1), value) for index, value in enumerate(values)]


def test_the_drawdown_is_the_distance_given_back_over_the_distance_earned():
    """``D = (peak - value) / (peak - floor)``: zero at the peak, one at the floor."""

    measured = dict(drawdowns(curve(PEAK, 50.0, FLOOR, 120.0), spec()))

    assert measured[10000] == 0.0
    assert measured[20000] == 0.5
    assert measured[30000] == 1.0
    # Back above the old peak: the drawdown is against the running peak, so a
    # new high is a drawdown of zero rather than a negative one.
    assert measured[40000] == 0.0


def test_a_floor_that_is_not_zero_moves_what_half_the_distance_means():
    """The floor is the frozen part of the definition, and it changes the answer.

    The same curve is a qualifying collapse against one floor and not against
    another, which is the whole reason the floor is declared before the formal
    results are read rather than chosen while looking at them.
    """

    declining = curve(PEAK, 60.0, 60.0)

    assert detect(declining, spec(random_floor=0.0)).verdict == "steady"
    assert detect(declining, spec(random_floor=20.0)).verdict == "collapsed"


def test_a_decline_that_does_not_hold_is_not_a_collapse():
    """One bad evaluation is one bad evaluation."""

    decision = detect(curve(PEAK, 10.0, 95.0, 98.0), spec())

    assert decision.verdict == "steady"
    assert decision.collapse is None
    # It is still the worst drawdown of the run, and it is still reported: the
    # statistic and the event are two readings, and only one of them qualified.
    assert decision.max_drawdown == pytest.approx(0.9)
    assert decision.max_drawdown_step == 20000


def test_a_decline_that_holds_for_two_checkpoints_qualifies():
    decision = detect(curve(PEAK, 40.0, 30.0, 35.0), spec())

    assert decision.verdict == "collapsed"
    assert decision.collapse is not None
    assert decision.collapse.step == 20000
    assert decision.collapse.drawdown == pytest.approx(0.6)
    assert decision.collapse.sustained == 3
    assert decision.collapse.peak == PEAK
    assert decision.collapse.peak_step == 10000
    assert decision.collapse.trough == 30.0
    assert decision.collapse.trough_step == 30000
    assert decision.collapse.recovered_step is None


def test_a_run_that_comes_back_says_when_it_came_back():
    """Recovery is reported, not subtracted: the collapse still happened."""

    decision = detect(curve(PEAK, 20.0, 20.0, 60.0, 90.0), spec())

    assert decision.verdict == "collapsed"
    assert decision.collapse is not None
    assert decision.collapse.step == 20000
    # A drawdown back to 0.2 or below, which the fourth point is not (0.4) and
    # the fifth is (0.1).
    assert decision.collapse.recovered_step == 50000


def test_only_the_first_qualifying_collapse_is_analysed():
    """R2 says the first one, because the second is a different regime."""

    decision = detect(curve(PEAK, 10.0, 10.0, 90.0, 5.0, 5.0), spec())

    assert decision.collapse is not None
    assert decision.collapse.step == 20000


def test_a_decline_at_the_very_end_is_not_called_sustained():
    """A curve that ends mid-decline has not shown the decline holding.

    The alternative -- counting the checkpoints that exist and calling it
    sustained -- would make the last evaluation of every run a candidate.
    """

    decision = detect(curve(PEAK, 10.0), spec())

    assert decision.verdict == "steady"
    assert decision.max_drawdown == pytest.approx(0.9)


def test_a_diverged_run_is_not_a_steady_one():
    """The failure this is written against: NaN loses every comparison.

    Scanned for a maximum first, a curve with a NaN in it reports no collapse
    and the smallest drawdown on the page, so the run that blew up is the one
    that looks most stable. The scan checks for it before comparing anything.
    """

    decision = detect(curve(PEAK, 50.0, math.nan, math.nan), spec())

    assert decision.verdict == "non_finite"
    assert decision.non_finite_step == 30000
    assert decision.collapse is None
    assert "diverged" in decision.reason


def test_a_run_that_never_rose_above_the_floor_has_no_peak_to_fall_from():
    decision = detect(curve(-5.0, -8.0, -20.0), spec())

    assert decision.verdict == "never_learned"
    assert decision.max_drawdown is None
    assert "random floor" in decision.reason


def test_a_run_with_no_evaluation_is_not_silently_steady():
    assert detect([], spec()).verdict == "never_learned"


def test_the_decision_carries_the_specification_it_was_decided_under():
    decision = detect(curve(PEAK, 20.0, 20.0), spec())

    assert decision.spec.as_mapping() == {
        "metric": "eval/episode/return",
        "random_floor": 0.0,
        "normalization": "peak_to_floor",
        "decline": 0.5,
        "sustain": 2,
        "recovery": 0.2,
    }


def test_a_specification_without_a_measured_floor_is_refused():
    """There is no default floor, because a guessed one decides every drawdown.

    The message is the instruction: the floor is the return of an unlearned
    policy on this environment under this evaluation protocol, and it is
    measured and committed before any formal curve is read.
    """

    with pytest.raises(CollapseError, match="random_floor"):
        CollapseSpec.from_mapping({"metric": "eval/episode/return"})
    with pytest.raises(CollapseError, match="random_floor"):
        CollapseSpec.from_mapping({"metric": "e", "random_floor": None})
    with pytest.raises(CollapseError, match="metric"):
        CollapseSpec.from_mapping({"random_floor": 0.0})


def test_the_committed_specifications_are_the_frozen_part_of_the_result():
    """The two files that must exist before the runs are read, and be readable.

    Their floors are absent on purpose -- the measurement has not been made --
    so what is checked is that everything else is there and that the file fails
    for the one reason it is supposed to.
    """

    import yaml

    for name in ("collapse halfcheetah.yaml", "collapse hopper.yaml"):
        declared = yaml.safe_load((EXPERIMENTS / name).read_text(encoding="utf-8"))
        assert declared["metric"] == "eval/episode/return"
        assert declared["normalization"] == "peak_to_floor"
        assert (declared["decline"], declared["sustain"], declared["recovery"]) == (
            0.5,
            2,
            0.2,
        )
        with pytest.raises(CollapseError, match="random_floor"):
            CollapseSpec.from_mapping(declared)


def test_a_specification_that_cannot_mean_anything_is_refused():
    with pytest.raises(CollapseError, match="normalization"):
        spec(normalization="whatever")
    with pytest.raises(CollapseError, match="fraction of the distance"):
        spec(decline=0.0)
    with pytest.raises(CollapseError, match="sustained"):
        spec(sustain=0)
    with pytest.raises(CollapseError, match="smaller drawdown"):
        spec(recovery=0.6)
    with pytest.raises(CollapseError, match="finite return"):
        spec(random_floor=math.nan)


# --------------------------------------------------------------- metrics files
def write(path: Path, rows: list[tuple[int, dict[str, float]]]) -> Path:
    path.write_text(
        "\n".join(json.dumps({"step": step, "metrics": values}) for step, values in rows),
        encoding="utf-8",
    )
    return path


def test_a_checkpoint_is_the_mean_of_its_episodes_not_one_of_them(tmp_path):
    """Five Brax episodes at one boundary are one point on the curve."""

    rows = [(10000, {"eval/episode/return": value}) for value in (10.0, 20.0, 30.0)]
    rows += [(20000, {"eval/episode/return": 8.0})]
    # Training rows share the file and must not reach the evaluation curve.
    rows += [(15000, {"train/episode/return": 1000.0})]

    assert evaluation_curve(
        write(tmp_path / "metrics.jsonl", rows).read_text().splitlines(),
        "eval/episode/return",
    ) == [(10000, 20.0), (20000, 8.0)]


def test_one_diverged_episode_makes_its_checkpoint_diverged(tmp_path):
    rows = [
        (10000, {"eval/episode/return": 10.0}),
        (10000, {"eval/episode/return": math.inf}),
    ]
    path = write(tmp_path / "metrics.jsonl", rows)

    curve = evaluation_curve(path.read_text().splitlines(), "eval/episode/return")

    assert len(curve) == 1
    assert not math.isfinite(curve[0][1])


def test_an_event_window_keeps_the_steps_it_found_the_readings_at(tmp_path):
    """Aligned by keeping the axis: a window rebased to zero could not be
    laid over the evaluation curve it explains."""

    name = "train/episode/update.torso.raw_update_norm"
    rows = [(step, {name: float(step)}) for step in range(0, 100000, 10000)]
    path = write(tmp_path / "metrics.jsonl", rows)

    window = event_window(
        path.read_text().splitlines(), around=50000, width=20000, names=(name,)
    )

    assert [point["step"] for point in window[name]] == [30000, 40000, 50000, 60000, 70000]


def test_the_window_reads_the_five_quantities_for_each_of_the_three_groups():
    names = training_series()

    assert len(names) == 15
    assert "train/episode/update.torso.raw_update_norm" in names
    assert "train/episode/update.critic.clip_fraction" in names
    assert "train/episode/update.actor.realized_update_norm" in names


def test_analysis_reads_the_file_twice_without_holding_it(tmp_path):
    """The curve is one pass and the window around what it found is another."""

    name = "train/episode/update.torso.realized_update_norm"
    rows: list[tuple[int, dict[str, float]]] = []
    for index, value in enumerate((PEAK, 20.0, 20.0, 20.0)):
        step = 10000 * (index + 1)
        rows.append((step, {"eval/episode/return": value}))
        rows.append((step, {name: float(index)}))
    path = write(tmp_path / "metrics.jsonl", rows)

    decision = analyze(
        path, spec(), run_id="seed-0", seed=0, window_steps=10000, telemetry=(name,)
    )

    assert decision.verdict == "collapsed"
    assert decision.collapse is not None and decision.collapse.step == 20000
    assert [point["step"] for point in decision.windows[name]] == [10000, 20000, 30000]


def test_a_one_shot_iterator_is_refused_rather_than_read_empty(tmp_path):
    rows = iter(["{}"])

    with pytest.raises(CollapseError, match="one-shot iterator"):
        analyze(rows, spec(), window_steps=1000)


def test_the_seeds_are_reported_one_by_one_and_never_averaged():
    """A collapse is a per-seed event with a step; a mean of curves has neither."""

    document = decisions(
        [
            detect(curve(PEAK, 20.0, 20.0), spec()),
            detect(curve(PEAK, 99.0, 98.0), spec()),
            detect(curve(PEAK, math.nan), spec()),
        ]
    )

    assert [seed["verdict"] for seed in document["seeds"]] == [
        "collapsed",
        "steady",
        "non_finite",
    ]
    assert document["non_finite"] == [""]
    assert all("collapse" in seed for seed in document["seeds"])
