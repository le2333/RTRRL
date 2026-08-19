"""The two reductions the formal protocol scores on.

Both read the evaluation curve rather than the rows that reported it: a
checkpoint is one measured step and arrives as one row per episode, so what is
integrated or averaged is the checkpoint's mean, never the rows themselves.
Everything here is about that distinction and about what the curve is allowed
to look like.
"""

import json
import math
import tracemalloc
from pathlib import Path

import pytest

from trainer_infra.scoring import (
    WORST_MAGNITUDE,
    ScoreError,
    ScoreSpec,
    compute_score,
)

METRIC = "eval/episode/return"


def write_checkpoints(path: Path, checkpoints: dict[int, list[float]]) -> None:
    """One row per episode, the way a run reports them as it reaches them."""

    path.write_text(
        "\n".join(
            json.dumps({"step": step, "metrics": {METRIC: value}})
            for step, values in checkpoints.items()
            for value in values
        )
        + "\n",
        encoding="utf-8",
    )


def spec(**overrides: object) -> ScoreSpec:
    payload: dict[str, object] = {
        "metric": METRIC,
        "window_steps": [0, 1000],
        "reduce": "auc",
        "direction": "maximize",
        "non_finite": "worst",
    }
    payload.update(overrides)
    return ScoreSpec.from_mapping(payload)


# --------------------------------------------------------------- the integral
def test_auc_is_the_step_weighted_mean_of_the_curve(tmp_path: Path) -> None:
    """Normalised by the span, so it reads on the scale of the returns."""

    path = tmp_path / "metrics.jsonl"
    write_checkpoints(path, {100: [1.0], 200: [2.0], 300: [3.0]})

    # Trapezoids of 1.5 and 2.5 over equal intervals.
    assert compute_score(path, spec()) == pytest.approx(2.0)


def test_a_checkpoint_is_one_point_however_many_episodes_reported_it(
    tmp_path: Path,
) -> None:
    """Otherwise a checkpoint would weigh what it happened to record.

    The two files hold the same curve. One reported each checkpoint on ten
    episodes and the other on two, and a reduction over rows would call them
    different runs.
    """

    many = tmp_path / "many.jsonl"
    few = tmp_path / "few.jsonl"
    write_checkpoints(many, {100: [0.0, 2.0] * 5, 200: [1.0, 3.0] * 5})
    write_checkpoints(few, {100: [0.0, 2.0], 200: [1.0, 3.0]})

    assert compute_score(many, spec()) == compute_score(few, spec()) == 1.5


def test_irregular_spacing_weighs_the_intervals_it_actually_spans(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "regular.jsonl"
    irregular = tmp_path / "irregular.jsonl"
    write_checkpoints(regular, {0: [0.0], 100: [10.0], 200: [10.0]})
    write_checkpoints(irregular, {0: [0.0], 100: [10.0], 400: [10.0]})

    # 500 over the first hundred and 1000 over the second: 7.5 of two hundred.
    assert compute_score(regular, spec()) == pytest.approx(7.5)
    # The same first interval, then ten held for three hundred: 3500 of four
    # hundred. The long interval counts for its length, not for one point.
    assert compute_score(irregular, spec()) == pytest.approx(8.75)


def test_a_missing_checkpoint_is_crossed_rather_than_dropped(
    tmp_path: Path,
) -> None:
    """A gap in the schedule is still time the policy spent somewhere.

    Dropping the interval would shorten the run; the trapezoid between the
    checkpoints either side is the same line a plot of the curve draws.
    """

    complete = tmp_path / "complete.jsonl"
    gapped = tmp_path / "gapped.jsonl"
    write_checkpoints(complete, {0: [0.0], 100: [5.0], 200: [10.0]})
    write_checkpoints(gapped, {0: [0.0], 200: [10.0]})

    assert compute_score(complete, spec()) == compute_score(gapped, spec()) == 5.0


def test_the_endpoints_are_the_checkpoints_the_window_admitted(
    tmp_path: Path,
) -> None:
    """Never the window's own bounds, which nothing was measured at.

    Extending to them would carry the first and last measurements into steps
    the policy was not evaluated at, which is an extrapolation the run did not
    make.
    """

    path = tmp_path / "metrics.jsonl"
    write_checkpoints(path, {0: [1.0], 100: [1.0], 200: [3.0], 900: [100.0]})

    assert compute_score(path, spec(window_steps=[0, 200])) == pytest.approx(1.5)


def test_one_checkpoint_has_no_area_and_says_so(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_checkpoints(path, {100: [1.0, 2.0]})

    with pytest.raises(ScoreError, match="measured at one step"):
        compute_score(path, spec())


def test_a_window_the_run_never_reached_is_the_ordinary_empty_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    write_checkpoints(path, {100: [1.0], 200: [2.0]})

    with pytest.raises(ScoreError, match="no reported value"):
        compute_score(path, spec(window_steps=[300, 400]))


# ------------------------------------------------------- the last checkpoints
def test_last_checkpoints_averages_the_curve_not_the_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_checkpoints(
        path,
        {
            100: [0.0],
            200: [100.0, 100.0, 100.0],
            300: [2.0, 4.0],
            400: [5.0],
            500: [6.0],
            600: [7.0],
        },
    )

    # The last five checkpoints are 100, 3, 5, 6, 7 -- the second one counted
    # once despite arriving three times.
    assert compute_score(
        path, spec(reduce="last_checkpoints", checkpoints=5)
    ) == pytest.approx(24.2)


def test_last_checkpoints_reads_the_same_trajectory_the_area_did(
    tmp_path: Path,
) -> None:
    """One curve, two reductions: the secondary metric is not a second run."""

    path = tmp_path / "metrics.jsonl"
    write_checkpoints(path, {0: [1.0], 100: [1.0], 200: [1.0]})

    assert compute_score(path, spec()) == 1.0
    assert compute_score(path, spec(reduce="last_checkpoints", checkpoints=3)) == 1.0


def test_fewer_checkpoints_than_asked_for_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_checkpoints(path, {100: [1.0], 200: [2.0]})

    with pytest.raises(ScoreError, match="measured at 2 checkpoints"):
        compute_score(path, spec(reduce="last_checkpoints", checkpoints=5))


# ----------------------------------------------------- what the rows may hold
def test_one_non_finite_value_decides_the_score_as_it_does_for_a_point(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    write_checkpoints(path, {0: [1.0], 100: [1.0, math.nan], 200: [1.0]})

    assert compute_score(path, spec()) == -WORST_MAGNITUDE
    assert compute_score(path, spec(direction="minimize")) == WORST_MAGNITUDE
    assert compute_score(path, spec(non_finite=-7.5)) == -7.5


def test_a_checkpoint_short_of_its_declared_episodes_is_refused(
    tmp_path: Path,
) -> None:
    """The exactness the protocol claims, checked where the claim is used.

    A checkpoint scored on nine episodes is not the quantity the other runs
    report, and averaging it into a curve would hide that rather than say it.
    """

    path = tmp_path / "metrics.jsonl"
    write_checkpoints(path, {0: [1.0, 2.0], 100: [1.0]})

    with pytest.raises(ScoreError, match="checkpoint at 100 .* reported 1 "):
        compute_score(path, spec(episodes_per_checkpoint=2))


def test_the_declared_episode_count_passes_when_every_checkpoint_has_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    write_checkpoints(path, {0: [1.0, 3.0], 100: [1.0, 3.0]})

    assert compute_score(path, spec(episodes_per_checkpoint=2)) == 2.0


def test_rows_that_go_backwards_are_refused_rather_than_integrated(
    tmp_path: Path,
) -> None:
    """A file out of step order is not the trajectory it is being read as."""

    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"step": 200, "metrics": {METRIC: 1.0}})
        + "\n"
        + json.dumps({"step": 100, "metrics": {METRIC: 2.0}})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ScoreError, match="reported step 100 after step 200"):
        compute_score(path, spec())


# ------------------------------------------------------------ what a spec says
def test_checkpoints_belongs_to_the_reduce_that_counts_them() -> None:
    with pytest.raises(ValueError, match="last_checkpoints requires it"):
        spec(reduce="last_checkpoints")
    with pytest.raises(ValueError, match="no other reduce accepts it"):
        spec(reduce="auc", checkpoints=5)


def test_an_episode_count_is_refused_where_nothing_groups_the_rows() -> None:
    with pytest.raises(ValueError, match="point reduce 'mean'"):
        spec(reduce="mean", episodes_per_checkpoint=10)


# --------------------------------------------------------------- what it costs
def test_a_trajectory_reduce_holds_the_curve_and_not_the_file(
    tmp_path: Path,
) -> None:
    """The same constraint the point reduces are under, for the same reason.

    A finished run's metrics file is gigabytes and the machine that scores it
    hosts the study. What a checkpoint fold keeps is the open checkpoint, the
    one before it, and the last five means.
    """

    path = tmp_path / "metrics.jsonl"
    padding = "y" * 200
    rows = 60_000
    with path.open("w", encoding="utf-8") as handle:
        for step in range(rows):
            handle.write(
                json.dumps({"step": step, "metrics": {METRIC: 1.0, "note": padding}})
                + "\n"
            )
    size = path.stat().st_size

    tracemalloc.start()
    try:
        value = compute_score(path, spec(window_steps=[0, rows]))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert value == 1.0
    assert size > 8_000_000
    assert peak < size // 16
