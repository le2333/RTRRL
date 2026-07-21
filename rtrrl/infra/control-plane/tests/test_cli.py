from __future__ import annotations

import json
from pathlib import Path

from trainer_infra.cli import main
from trainer_infra.controller import (
    ExperimentReport,
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


def test_cli_exposes_exactly_validate_and_run(capsys: object) -> None:
    for forbidden in ("status", "resume", "history"):
        code = main([forbidden, "experiment.yaml"], lambda _: Controller())
        captured = capsys.readouterr()  # type: ignore[attr-defined]
        assert code == 2
        assert captured.out == ""
        assert "invalid choice" in captured.err
