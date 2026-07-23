from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "check-infra-merge-boundary.sh"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def make_test_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "memo").mkdir()
    (repo / ".github" / "workflows").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    (repo / "memo" / "existing.txt").write_text("memo baseline\n", encoding="utf-8")
    (repo / ".github" / "workflows" / "build-memo-image.yml").write_text(
        "name: build memo\n",
        encoding="utf-8",
    )
    (repo / ".github" / "workflows" / "memo-ci.yml").write_text(
        "name: memo ci\n",
        encoding="utf-8",
    )
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Boundary Test")
    git(repo, "config", "user.email", "boundary-test@example.invalid")
    git(repo, "config", "core.fileMode", "true")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    return repo, git(repo, "rev-parse", "HEAD")


def test_boundary_script_protects_memo_and_both_memo_workflows() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "memo" in source
    assert ".github/workflows/build-memo-image.yml" in source
    assert ".github/workflows/memo-ci.yml" in source
    assert "git diff --raw" in source
    assert "git diff --name-status" in source


def test_current_head_has_zero_protected_tree_diff() -> None:
    result = subprocess.run(
        [str(SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "protected tree matches" in result.stdout


@pytest.mark.parametrize("change", ["add", "delete", "content", "mode"])
def test_gate_rejects_each_protected_tree_change(
    tmp_path: Path,
    change: str,
) -> None:
    repo, base = make_test_repo(tmp_path)
    existing = repo / "memo" / "existing.txt"

    if change == "add":
        (repo / "memo" / "added.txt").write_text("added\n", encoding="utf-8")
    elif change == "delete":
        existing.unlink()
    elif change == "content":
        existing.write_text("changed blob\n", encoding="utf-8")
    else:
        existing.chmod(0o755)

    git(repo, "add", "--all")
    git(repo, "commit", "-m", f"{change} protected path")
    assert git(repo, "status", "--porcelain") == ""

    result = subprocess.run(
        [str(repo / "scripts" / SCRIPT.name), base],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "protected tree differs" in result.stderr
