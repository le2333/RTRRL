from __future__ import annotations

import json
from pathlib import Path

import pytest
from training_sdk.contract import Catalog

from tests.conftest import AimServer
from tests.helpers import CATALOG, EXAMPLE, write_experiment
from training_sdk import objects
from tests.test_preflight_offline import modified, write_catalog
from trainer_infra.cli import main
from trainer_infra.experiment import load_experiment
from trainer_infra.preflight import check_offline


def test_errors_use_stderr_and_nonzero_exit(tmp_path: Path, capsys: object) -> None:
    modified(tmp_path, "metric: episode_return", "metric: reward")
    experiment = tmp_path / "experiment.yaml"
    bad_catalog = write_catalog(tmp_path)

    code = main(["validate", str(experiment), "--catalog", str(bad_catalog)])

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert code == 1
    assert captured.out == ""
    assert "reward" in captured.err


def test_cli_exposes_exactly_validate_and_run(capsys: pytest.CaptureFixture[str]) -> None:
    for forbidden in ("status", "resume", "history"):
        code = main([forbidden, "experiment.yaml"])
        captured = capsys.readouterr()
        assert code == 2
        assert captured.out == ""
        assert "invalid choice" in captured.err


def test_run_requires_a_backend(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`run` with no backend must say so rather than pick one.

    There is no default: local runs a subprocess here, batch spends money on AWS,
    and guessing between them is not the CLI's decision to make.
    """
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(EXAMPLE.read_text(encoding="utf-8"))

    code = main(["run", str(experiment)])

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "--backend local or --backend batch" in captured.err


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
    assert '  total_steps: {"type":"int","low":1,"high":100000' in captured.out


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
    assert "contract 4" in captured.err


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


def test_validate_requires_exactly_one_of_catalog_or_batch_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(EXAMPLE.read_text(encoding="utf-8"))
    catalog_path = write_catalog(tmp_path)

    neither = main(["validate", str(experiment)])
    neither_captured = capsys.readouterr()
    assert neither == 2
    assert "exactly one of --catalog or --backend batch" in neither_captured.err

    both = main(
        [
            "validate",
            str(experiment),
            "--catalog",
            str(catalog_path),
            "--backend",
            "batch",
        ]
    )
    both_captured = capsys.readouterr()
    assert both == 2
    assert "exactly one of --catalog or --backend batch" in both_captured.err


def test_validate_batch_backend_never_submits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    listening_endpoint: str,
) -> None:
    from tests.test_preflight_aws import FakeBatch, FakeEcr, FakeS3, read_url

    submitted: list[dict] = []

    class TrackingBatch(FakeBatch):
        def submit_job(self, **kwargs: object) -> dict:
            submitted.append(kwargs)
            raise AssertionError("submit_job should not be called during validate")

    experiment = write_experiment(tmp_path, "s3://rtrrl-training-data", listening_endpoint)

    class Session:
        def client(self, service: str):
            if service == "ecr":
                return FakeEcr()
            if service == "batch":
                return TrackingBatch()
            if service == "s3":
                return FakeS3()
            raise AssertionError(f"unexpected client {service!r}")

    monkeypatch.setattr("trainer_infra.cli._batch_session_factory", lambda: Session())
    monkeypatch.setattr("trainer_infra.cli._read_ecr_url", read_url)

    code = main(["validate", str(experiment), "--backend", "batch"])

    captured = capsys.readouterr()
    assert code == 0
    assert submitted == []
    assert captured.err == ""
    assert captured.out.startswith("resolved search space:")


def test_validate_batch_backend_warns_for_dev_queues(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    listening_endpoint: str,
) -> None:
    from tests.test_preflight_aws import FakeBatch, FakeEcr, FakeS3, read_url

    experiment = write_experiment(tmp_path, "s3://rtrrl-training-data", listening_endpoint)

    class Session:
        def client(self, service: str):
            if service == "ecr":
                return FakeEcr()
            if service == "batch":
                return FakeBatch(queues=("dev-cpu-c7am-queue",))
            if service == "s3":
                return FakeS3()
            raise AssertionError(f"unexpected client {service!r}")

    monkeypatch.setattr("trainer_infra.cli._batch_session_factory", lambda: Session())
    monkeypatch.setattr("trainer_infra.cli._read_ecr_url", read_url)

    code = main(["validate", str(experiment), "--backend", "batch", "--queues", "dev"])

    captured = capsys.readouterr()
    assert code == 0
    assert "warning: dev queues are for infrastructure development only" in captured.err


def test_run_batch_backend_exits_zero_on_success(
    s3_base: str,
    tmp_path: Path,
    aim_endpoint: AimServer,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_batch_backend import FakeBatch as PollingBatch, FakeLogs
    from tests.test_preflight_aws import (
        DEFINITION,
        FakeBatch as PreflightBatch,
        FakeEcr,
        FakeS3,
        read_url,
    )

    experiment_path = write_experiment(tmp_path, s3_base, aim_endpoint.uri)
    submitted: list[dict] = []

    class RunBatch(PollingBatch):
        def __init__(self) -> None:
            super().__init__(["SUCCEEDED"])
            self._preflight = PreflightBatch()

        def describe_job_queues(self, **kwargs: object) -> dict:
            return self._preflight.describe_job_queues(**kwargs)

        def describe_job_definitions(self, **kwargs: object) -> dict:
            return self._preflight.describe_job_definitions(**kwargs)

        def submit_job(self, **kwargs: object) -> dict:
            submitted.append(kwargs)
            response = super().submit_job(**kwargs)
            environment = {
                item["name"]: item["value"]
                for item in kwargs["containerOverrides"]["environment"]
            }
            manifest = json.loads(objects.get_bytes(environment["TRAINER_MANIFEST"]))
            for config_uri in manifest["runs"]:
                config = json.loads(objects.get_bytes(config_uri))
                objects.put_bytes(
                    config["score"]["s3"],
                    json.dumps({"value": 42.0}).encode(),
                )
            return response

    class Session:
        def client(self, service: str):
            if service == "ecr":
                return FakeEcr()
            if service == "batch":
                return RunBatch()
            if service == "s3":
                return FakeS3()
            if service == "logs":
                return FakeLogs()
            raise AssertionError(f"unexpected client {service!r}")

    monkeypatch.setattr("trainer_infra.cli._batch_session_factory", lambda: Session())
    monkeypatch.setattr("trainer_infra.cli._read_ecr_url", read_url)

    code = main(
        [
            "run",
            str(experiment_path),
            "--backend",
            "batch",
            "--archive-dir",
            str(tmp_path / "archive"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert submitted
    assert all(request["jobQueue"] == "run-cpu-c7am-queue" for request in submitted)
    assert all(request["jobDefinition"] == DEFINITION for request in submitted)
    report = json.loads(captured.out)
    assert report["status"] == "succeeded"
    assert len(report["trials"]) == 4


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
    # stdout must be nothing but the report, so `trainerctl run > report.json` works.
    assert json.loads(captured.out)["status"] == "succeeded"
    assert "best trial" in captured.err
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
