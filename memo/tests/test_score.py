import json
import math
from pathlib import Path

import pytest

from worker.contract import ScoreConfig
from worker.score import WORST_MAGNITUDE, ScoreError, compute_score


def write_metrics(path: Path, rows: list[tuple[int, float]]) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"step": step, "metrics": {"episode_return": value}})
            for step, value in rows
        ),
        encoding="utf-8",
    )


def spec(**overrides: object) -> ScoreConfig:
    payload = {
        "metric": "episode_return",
        "window_steps": [10, 20],
        "reduce": "mean",
        "direction": "maximize",
        "non_finite": "worst",
        "s3": "s3://bucket/score.json",
    }
    payload.update(overrides)
    return ScoreConfig.model_validate(payload)


def test_window_is_inclusive_and_mean_is_used(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(5, 100.0), (10, 1.0), (15, 2.0), (20, 3.0), (25, 100.0)])
    assert compute_score(path, spec()) == 2.0


def test_last_reduction_takes_the_highest_step(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(20, 3.0), (10, 1.0), (15, 2.0)])
    assert compute_score(path, spec(reduce="last")) == 3.0


def test_empty_window_raises_naming_metric_and_window(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(1, 1.0)])
    with pytest.raises(ScoreError, match="episode_return.*10.*20"):
        compute_score(path, spec())


def test_non_finite_becomes_worst_for_each_direction(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, math.nan)])
    assert compute_score(path, spec()) == -WORST_MAGNITUDE
    assert compute_score(path, spec(direction="minimize")) == WORST_MAGNITUDE


def test_declared_numeric_substitute_is_used_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, math.inf)])
    assert compute_score(path, spec(non_finite=-5.0)) == -5.0


def test_window_excludes_rows_outside_bounds(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(9, 100.0), (10, 1.0), (20, 2.0), (21, 100.0)])
    assert compute_score(path, spec(reduce="mean")) == 1.5


REDUCE_MODE_DISCRIMINATION_ROWS = [(10, 1.0), (12, 2.0), (15, 7.0), (20, 3.0)]


@pytest.mark.parametrize(
    ("reduce", "expected"),
    [
        ("mean", 3.25),
        ("median", 2.5),
        ("min", 1.0),
        ("max", 7.0),
        ("last", 3.0),
    ],
)
def test_each_reduce_mode_returns_distinct_value(
    tmp_path: Path, reduce: str, expected: float
) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, REDUCE_MODE_DISCRIMINATION_ROWS)
    assert compute_score(path, spec(reduce=reduce)) == expected


def test_non_finite_worst_orders_below_maximize_baseline(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, math.nan)])
    worst = compute_score(path, spec(direction="maximize"))
    write_metrics(path, [(10, 0.0), (15, 1.0)])
    baseline = compute_score(path, spec(direction="maximize"))
    assert worst < baseline


def test_non_finite_worst_orders_above_minimize_baseline(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, math.nan)])
    worst = compute_score(path, spec(direction="minimize"))
    write_metrics(path, [(10, 100.0), (15, 50.0)])
    baseline = compute_score(path, spec(direction="minimize", reduce="mean"))
    assert worst > baseline


def test_json_non_finite_tokens_are_parsed(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"step": 10, "metrics": {"episode_return": NaN}}\n'
        '{"step": 15, "metrics": {"episode_return": Infinity}}\n',
        encoding="utf-8",
    )
    assert compute_score(path, spec(reduce="mean")) == -WORST_MAGNITUDE
    assert compute_score(path, spec(reduce="max", non_finite=-99.0)) == -99.0


def test_empty_window_when_metric_missing_from_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"step": 15, "metrics": {"other_metric": 1.0}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ScoreError, match="episode_return.*10.*20"):
        compute_score(path, spec())


def test_mixed_finite_and_inf_median_substitutes_worst(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, 10.0), (15, 20.0), (20, math.inf)])
    assert compute_score(path, spec(reduce="median")) == -WORST_MAGNITUDE


def test_mixed_finite_and_inf_min_substitutes_worst(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, 10.0), (15, 20.0), (20, math.inf)])
    assert compute_score(path, spec(reduce="min")) == -WORST_MAGNITUDE


def test_nan_position_in_window_does_not_affect_score(tmp_path: Path) -> None:
    start_nan = tmp_path / "start_nan.jsonl"
    middle_nan = tmp_path / "middle_nan.jsonl"
    write_metrics(start_nan, [(10, math.nan), (15, 10.0), (20, 20.0)])
    write_metrics(middle_nan, [(10, 10.0), (15, math.nan), (20, 20.0)])
    score_config = spec(reduce="median")
    assert compute_score(start_nan, score_config) == compute_score(
        middle_nan, score_config
    )
    assert compute_score(start_nan, score_config) == -WORST_MAGNITUDE


def test_mixed_window_worst_orders_below_maximize_baseline(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, 10.0), (15, 20.0), (20, math.inf)])
    worst = compute_score(path, spec(direction="maximize", reduce="min"))
    write_metrics(path, [(10, 0.0), (15, 1.0)])
    baseline = compute_score(path, spec(direction="maximize", reduce="mean"))
    assert worst < baseline


def test_mixed_window_worst_orders_above_minimize_baseline(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, 10.0), (15, 20.0), (20, math.inf)])
    worst = compute_score(path, spec(direction="minimize", reduce="min"))
    write_metrics(path, [(10, 100.0), (15, 50.0)])
    baseline = compute_score(path, spec(direction="minimize", reduce="mean"))
    assert worst > baseline


def test_mixed_window_explicit_non_finite_substitute(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, 10.0), (15, 20.0), (20, math.inf)])
    assert compute_score(path, spec(reduce="median", non_finite=-7.5)) == -7.5


def test_json_mixed_window_non_finite_tokens_substitute_worst(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"step": 10, "metrics": {"episode_return": 10.0}}\n'
        '{"step": 15, "metrics": {"episode_return": NaN}}\n'
        '{"step": 20, "metrics": {"episode_return": 20.0}}\n',
        encoding="utf-8",
    )
    assert compute_score(path, spec(reduce="min")) == -WORST_MAGNITUDE
