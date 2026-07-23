from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "check-infra-merge-boundary.sh"


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
