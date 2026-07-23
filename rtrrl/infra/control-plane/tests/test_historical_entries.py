from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import yaml


REPOSITORY_ROOT = Path(__file__).parents[4]
BASELINE = "fd195c4"


def _baseline_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", BASELINE],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(result.stdout.splitlines())


def _compatibility_entries(paths: set[str]) -> set[str]:
    entries = set()
    for path in paths:
        suffix = Path(path).suffix
        if path.startswith(".github/workflows/") and suffix in {".yml", ".yaml"}:
            entries.add(path)
        if (
            path.startswith(("infra/", "memo/infra/", "rtrrl/infra/"))
            and suffix == ".sh"
        ):
            entries.add(path)
        if "/infra/scripts/" in path and suffix in {".yml", ".yaml"}:
            entries.add(path)
        if path.startswith(("infra/hpo/", "rtrrl/hpo/")) and suffix in {
            ".csv",
            ".json",
            ".md",
            ".py",
            ".toml",
            ".yaml",
            ".yml",
        }:
            entries.add(path)
    return entries


def test_every_baseline_command_descriptor_workflow_and_hpo_entry_exists() -> None:
    baseline_paths = _baseline_paths()
    entries = _compatibility_entries(baseline_paths)

    assert "infra/submit.sh" in entries
    assert "rtrrl/infra/submit.sh" not in baseline_paths
    assert ".github/workflows/build-rtrrl-image.yml" in entries
    assert ".github/workflows/build-memo-image.yml" in entries
    assert "memo/infra/scripts/index.yaml" in entries
    assert "rtrrl/infra/scripts/index.yaml" in entries
    assert any(path.startswith("rtrrl/hpo/specs/") for path in entries)
    assert any("/plan.json" in path for path in entries)
    assert len(entries) > 1_300

    missing = sorted(
        path
        for path in entries
        if not (REPOSITORY_ROOT / path).is_file()
    )
    assert missing == []


def test_baseline_workflows_and_descriptors_still_parse() -> None:
    entries = _compatibility_entries(_baseline_paths())
    structured = sorted(
        path
        for path in entries
        if (
            path.startswith(".github/workflows/")
            or "/infra/scripts/" in path
            or path.startswith("rtrrl/hpo/specs/")
        )
    )

    for relative in structured:
        value = yaml.safe_load((REPOSITORY_ROOT / relative).read_text())
        assert isinstance(value, dict), relative


def test_safe_historical_help_and_hpo_dry_run_do_not_call_aws(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    aws_marker = tmp_path / "aws-called"
    fake_aws = fake_bin / "aws"
    fake_aws.write_text(
        f"#!/bin/sh\ntouch {aws_marker}\nexit 97\n",
        encoding="utf-8",
    )
    fake_aws.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

    run_many = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "infra" / "run_many.py"), "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--fail-fast" in run_many.stdout

    legacy_submit = subprocess.run(
        ["bash", "-n", str(REPOSITORY_ROOT / "infra" / "submit.sh")],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert legacy_submit.stdout == legacy_submit.stderr == ""
    assert not aws_marker.exists()

    plans = sorted((REPOSITORY_ROOT / "rtrrl" / "hpo" / "runs").glob("**/plan.json"))
    assert plans
    scheduler = (
        REPOSITORY_ROOT
        / "infra"
        / "hpo"
        / "src"
        / "hpo_control"
        / "scheduler.py"
    )
    dry_run = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(REPOSITORY_ROOT / "infra" / "hpo"),
            "--frozen",
            "python",
            str(scheduler),
            "--project-root",
            str(REPOSITORY_ROOT / "rtrrl"),
            "submit",
            "--plan",
            str(plans[0]),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Dry run only." in dry_run.stdout
    assert not aws_marker.exists()
