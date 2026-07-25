import json
import math
from pathlib import Path

import pytest

from training_sdk.contract import ScoreConfig
from training_sdk.score import WORST_MAGNITUDE, ScoreError, compute_score


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


def test_median_min_max_reduce_differently(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, 1.0), (15, 10.0), (20, 2.0)])
    assert compute_score(path, spec(reduce="median")) == 2.0
    assert compute_score(path, spec(reduce="min")) == 1.0
    assert compute_score(path, spec(reduce="max")) == 10.0
    assert compute_score(path, spec(reduce="mean")) == pytest.approx(13 / 3)
    assert compute_score(path, spec(reduce="last")) == 2.0


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
