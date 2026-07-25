from __future__ import annotations

import json
from pathlib import Path

import pytest
from training_sdk.contract import Catalog

from tests.conftest import AimServer
from tests.helpers import EXAMPLE, write_experiment
from tests.test_preflight_offline import CATALOG, modified, write_catalog
from trainer_infra.cli import main
from trainer_infra.experiment import load_experiment
from trainer_infra.preflight import check_offline
from trainer_infra.controller import (
    ExperimentReport,
    ExperimentRunError,
    GroupValidation,
    ValidationReport,
)


class Controller:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, Path]] = []

    def validate(self, path: Path) -> ValidationReport:
        self.calls.append(("validate", path))
        if self.fail:
            raise ValueError("invalid experiment")
        return ValidationReport(
            experiment_name="test",
            groups=(
                GroupValidation(
                    name="g",
                    image_digest="repo/image@sha256:" + "a" * 64,
                    profile="g6x",
                    total_trials=1,
                    configs_per_batch=1,
                    runs_per_job=1,
                    estimated_jobs=1,
                ),
            ),
        )

    def run(self, path: Path) -> ExperimentReport:
        self.calls.append(("run", path))
        return ExperimentReport(
            status="succeeded",
            experiment_id="fresh",
            experiment_name="test",
            experiment_metadata={},
            submitted_job_ids=("aws-1",),
            completed_runs=1,
        )


def test_validate_and_run_are_foreground_and_print_stable_json(
    tmp_path: Path, capsys: object
) -> None:
    control = tmp_path / "control.yaml"
    experiment = tmp_path / "experiment.yaml"
    control.write_text("{}")
    experiment.write_text("{}")
    controller = Controller()
    factory_calls: list[Path] = []

    def factory(path: Path) -> Controller:
        factory_calls.append(path)
        return controller

    assert main(["--control", str(control), "validate", str(experiment)], factory) == 0
    first = capsys.readouterr()  # type: ignore[attr-defined]
    assert json.loads(first.out)["status"] == "valid"
    assert first.err == ""

    assert main(["--control", str(control), "run", str(experiment)], factory) == 0
    second = capsys.readouterr()  # type: ignore[attr-defined]
    assert json.loads(second.out)["experiment_id"] == "fresh"
    assert second.err == ""
    assert controller.calls == [("validate", experiment), ("run", experiment)]
    assert factory_calls == [control, control]


def test_errors_use_stderr_and_nonzero_exit(tmp_path: Path, capsys: object) -> None:
    experiment = tmp_path / "bad.yaml"
    controller = Controller(fail=True)

    code = main(["validate", str(experiment)], lambda _: controller)

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert code == 1
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {"message": "invalid experiment", "type": "ValueError"}
    }


def test_experiment_error_prints_complete_structured_failure(
    tmp_path: Path, capsys: object
) -> None:
    experiment = tmp_path / "bad.yaml"
    report = ExperimentReport(
        status="failed",
        experiment_id="fresh",
        experiment_name="test",
        experiment_metadata={},
        submitted_job_ids=("aws-1", "aws-2"),
        completed_runs=1,
        error="RuntimeError: Batch failed",
    )
    original = RuntimeError("Batch failed")
    error = ExperimentRunError(
        report,
        original_cause=original,
        persistence_errors=(OSError("state write failed"),),
    )

    class FailedController:
        def run(self, path: Path) -> ExperimentReport:
            del path
            raise error

    code = main(["run", str(experiment)], lambda _: FailedController())

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.err)
    assert code == 1
    assert captured.out == ""
    assert payload == {
        "status": "failed",
        "error": {"message": "Batch failed", "type": "RuntimeError"},
        "report": report.model_dump(mode="json"),
        "submitted_job_ids": ["aws-1", "aws-2"],
        "persistence_errors": [
            {"message": "state write failed", "type": "OSError"}
        ],
    }


def test_cli_exposes_exactly_validate_and_run(capsys: object) -> None:
    for forbidden in ("status", "resume", "history"):
        code = main([forbidden, "experiment.yaml"], lambda _: Controller())
        captured = capsys.readouterr()  # type: ignore[attr-defined]
        assert code == 2
        assert captured.out == ""
        assert "invalid choice" in captured.err


def test_validate_catalog_exits_zero_and_prints_resolved_space(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog_path = write_catalog(tmp_path)
    space = check_offline(load_experiment(EXAMPLE), CATALOG)

    code = main(["validate", str(EXAMPLE), "--catalog", str(catalog_path)])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    lines = captured.out.strip().splitlines()
    assert lines[0] == "resolved search space:"
    assert len(lines) == 1 + len(space)
    assert "  total_steps: 128" in captured.out


def test_validate_catalog_rejects_unsupported_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = Catalog.model_validate(CATALOG.model_dump() | {"contract": 99})
    catalog_path = write_catalog(tmp_path, catalog)

    code = main(["validate", str(EXAMPLE), "--catalog", str(catalog_path)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "contract 99" in captured.err
    assert "contract 2" in captured.err


def test_validate_catalog_rejects_unknown_score_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    modified(tmp_path, "metric: episode_return", "metric: reward")
    catalog_path = write_catalog(tmp_path)

    code = main(["validate", str(tmp_path / "experiment.yaml"), "--catalog", str(catalog_path)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "reward" in captured.err


def test_validate_catalog_rejects_score_window_beyond_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    modified(tmp_path, "window_steps: [0, 128]", "window_steps: [0, 129]")
    catalog_path = write_catalog(tmp_path)

    code = main(["validate", str(tmp_path / "experiment.yaml"), "--catalog", str(catalog_path)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "score window upper bound 129" in captured.err


def test_validate_catalog_rejects_unknown_space_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    modified(
        tmp_path,
        "  seed: [0]",
        "  seed: [0]\n  rogue: [1]",
    )
    catalog_path = write_catalog(tmp_path)

    code = main(["validate", str(tmp_path / "experiment.yaml"), "--catalog", str(catalog_path)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "rogue" in captured.err
    assert "does not accept" in captured.err


def test_validate_catalog_rejects_grid_sampler_with_continuous_space(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    modified(tmp_path, "sampler: tpe", "sampler: grid")
    catalog_path = write_catalog(tmp_path)

    code = main(["validate", str(tmp_path / "experiment.yaml"), "--catalog", str(catalog_path)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "grid sampler" in captured.err
    assert "learning_rate" in captured.err


def test_run_local_backend_exits_zero_on_success(
    s3_base: str,
    tmp_path: Path,
    aim_endpoint: AimServer,
    acceptance_catalog: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_path = write_experiment(tmp_path, s3_base, aim_endpoint.uri)

    code = main(
        [
            "run",
            str(experiment_path),
            "--backend",
            "local",
            "--catalog",
            str(acceptance_catalog),
            "--archive-dir",
            str(tmp_path / "archive"),
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert '"status": "succeeded"' in captured.out
    report_path = next((tmp_path / "archive").glob("**/report.json"))
    payload = json.loads(report_path.read_text())
    assert payload["status"] == "succeeded"
    assert len(payload["trials"]) == 4


def test_run_local_backend_exits_nonzero_on_failure(
    s3_base: str,
    tmp_path: Path,
    aim_endpoint: AimServer,
    failing_catalog: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    experiment_path = write_experiment(tmp_path, s3_base, aim_endpoint.uri)

    code = main(
        [
            "run",
            str(experiment_path),
            "--backend",
            "local",
            "--catalog",
            str(failing_catalog),
            "--archive-dir",
            str(tmp_path / "archive"),
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "round 0 had" in captured.err
    report_path = next((tmp_path / "archive").glob("**/report.json"))
    archived = json.loads(report_path.read_text())
    assert archived["status"] == "failed"
