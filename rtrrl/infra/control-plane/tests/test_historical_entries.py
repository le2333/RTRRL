from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).parents[4]
BASELINE = "fd195c494ff4ac3b34dff066a6ccb1efb024b16b"
MANIFEST_PATH = Path(__file__).parent / "data" / "historical-fd195c4.json"


def _git(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
    )


def _baseline_tree() -> dict[str, tuple[str, str]]:
    fields = _git("ls-tree", "-r", "--full-tree", "-z", BASELINE).split(b"\0")
    result: dict[str, tuple[str, str]] = {}
    for field in fields:
        if not field:
            continue
        metadata, raw_path = field.split(b"\t", 1)
        mode, object_type, blob = metadata.decode().split()
        assert object_type == "blob"
        result[raw_path.decode()] = (mode, blob)
    return result


def _blob(blob: str) -> bytes:
    return _git("cat-file", "blob", blob)


def _cli_registration_paths(
    tree: dict[str, tuple[str, str]],
) -> set[str]:
    registrations: set[str] = set()
    for path, (_, blob) in tree.items():
        if Path(path).name != "pyproject.toml":
            continue
        parsed = tomllib.loads(_blob(blob).decode())
        scripts = parsed.get("project", {}).get("scripts", {})
        if not scripts:
            continue
        registrations.add(path)
        root = Path(path).parent
        for target in scripts.values():
            module = str(target).split(":", 1)[0]
            relative = Path(*module.split("."))
            candidates = (
                root / "src" / relative.with_suffix(".py"),
                root / relative.with_suffix(".py"),
                root / "src" / relative / "__init__.py",
                root / relative / "__init__.py",
            )
            matches = [candidate.as_posix() for candidate in candidates if candidate.as_posix() in tree]
            assert len(matches) == 1, (path, target, matches)
            registrations.add(matches[0])
    return registrations


def _expected_manifest_entries() -> list[dict[str, Any]]:
    tree = _baseline_tree()
    cli_registrations = _cli_registration_paths(tree)
    entries: list[dict[str, Any]] = []
    for path, (mode, blob) in sorted(tree.items()):
        content = _blob(blob)
        categories: list[str] = []
        is_cli_source = (
            b'if __name__ == "__main__"' in content
            or b"if __name__ == '__main__'" in content
        )
        if (
            mode == "100755"
            or content.startswith(b"#!")
            or path in cli_registrations
            or is_cli_source
        ):
            categories.append("command")
        if path.startswith(".github/workflows/") and Path(path).suffix in {
            ".yaml",
            ".yml",
        }:
            categories.append("workflow")
        if "/infra/scripts/" in path and Path(path).suffix in {".yaml", ".yml"}:
            categories.append("descriptor")
        if path.startswith(("infra/hpo/", "rtrrl/hpo/")):
            categories.append("hpo")
        if categories:
            entries.append(
                {
                    "path": path,
                    "blob": blob,
                    "categories": categories,
                }
            )
    return entries


def _manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def test_manifest_exactly_matches_task6_baseline_selection() -> None:
    manifest = _manifest()

    assert manifest["baseline"] == BASELINE
    assert manifest["selection"] == {
        "command": [
            "git mode 100755",
            "shebang",
            "Python __main__ guard",
            "pyproject project.scripts registration and target",
        ],
        "descriptor": "**/infra/scripts/*.{yaml,yml}",
        "workflow": ".github/workflows/*.{yaml,yml}",
        "hpo": ["infra/hpo/**", "rtrrl/hpo/**"],
    }
    assert manifest["entries"] == _expected_manifest_entries()
    assert any(
        entry["path"] == "infra/hpo/uv.lock"
        and "hpo" in entry["categories"]
        for entry in manifest["entries"]
    )


def test_every_manifest_entry_exists_with_unchanged_git_blob_identity() -> None:
    entries = _manifest()["entries"]
    paths = [entry["path"] for entry in entries]
    assert paths == sorted(set(paths))

    for entry in entries:
        path = REPOSITORY_ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert _git_blob_sha(path.read_bytes()) == entry["blob"], entry["path"]


def test_manifest_workflows_descriptors_and_hpo_specs_still_parse() -> None:
    for entry in _manifest()["entries"]:
        categories = set(entry["categories"])
        path = Path(entry["path"])
        if (
            categories & {"workflow", "descriptor"}
            or "hpo" in categories
            and "/specs/" in path.as_posix()
            and path.suffix in {".yaml", ".yml"}
        ):
            value = yaml.safe_load((REPOSITORY_ROOT / path).read_text())
            assert isinstance(value, dict), path


def test_safe_historical_help_syntax_and_hpo_dry_run_do_not_call_aws(
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

    for entry in _manifest()["entries"]:
        path = REPOSITORY_ROOT / entry["path"]
        if "command" in entry["categories"] and path.suffix == ".sh":
            subprocess.run(
                ["bash", "-n", str(path)],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

    help_commands = (
        [sys.executable, str(REPOSITORY_ROOT / "infra" / "run_many.py"), "--help"],
        [
            sys.executable,
            "-m",
            "trainer_infra",
            "--help",
        ],
        [
            sys.executable,
            str(REPOSITORY_ROOT / "rtrrl" / "infra" / "worker" / "worker.py"),
            "--help",
        ],
    )
    for command in help_commands:
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

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
