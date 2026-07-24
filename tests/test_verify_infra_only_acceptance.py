from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import stat
import subprocess


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "verify-infra-only-acceptance.sh"


@dataclass(frozen=True)
class GateStep:
    timeout: int
    unset_environment: tuple[str, ...]
    environment: tuple[str, ...]
    command: tuple[str, ...]


def _logical_lines(source: str) -> tuple[str, ...]:
    lines: list[str] = []
    pending = ""
    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        lines.append(pending)
        pending = ""
    assert not pending
    return tuple(lines)


def _gate_steps(source: str) -> tuple[GateStep, ...]:
    steps: list[GateStep] = []
    for line in _logical_lines(source):
        tokens = shlex.split(line, comments=True, posix=True)
        if tokens[:2] == ["if", "run"]:
            tokens = tokens[1:]
            if tokens[-2:] == [";", "then"]:
                tokens = tokens[:-2]
            elif tokens[-1].endswith(";"):
                tokens[-1] = tokens[-1][:-1]
        if not tokens or tokens[0] != "run":
            continue

        timeout = int(tokens[1])
        command_tokens = tokens[2:]
        unset: list[str] = []
        environment: list[str] = []
        if command_tokens[0] == "env":
            index = 1
            while index < len(command_tokens):
                if command_tokens[index] == "-u":
                    unset.append(command_tokens[index + 1])
                    index += 2
                elif "=" in command_tokens[index]:
                    environment.append(command_tokens[index])
                    index += 1
                else:
                    break
            command_tokens = command_tokens[index:]
        steps.append(
            GateStep(
                timeout=timeout,
                unset_environment=tuple(unset),
                environment=tuple(environment),
                command=tuple(command_tokens),
            )
        )
    return tuple(steps)


def _source_and_steps() -> tuple[str, tuple[GateStep, ...]]:
    source = SCRIPT.read_text(encoding="utf-8")
    return source, _gate_steps(source)


def test_gate_is_executable_fail_fast_rooted_and_valid_bash() -> None:
    source, _ = _source_and_steps()

    assert source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert SCRIPT.stat().st_mode & stat.S_IXUSR
    assert 'ROOT="$(git rev-parse --show-toplevel)"' in source
    assert 'cd "$ROOT"' in source
    assert 'TIME_BIN="${VERIFY_TIME_BIN:-/usr/bin/time}"' in source
    assert 'TIMEOUT_BIN="${VERIFY_TIMEOUT_BIN:-timeout}"' in source
    assert '"$TIME_BIN" -v "$TIMEOUT_BIN"' in source
    assert "--kill-after=30s" in source
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_gate_parses_to_the_complete_required_command_order() -> None:
    _, steps = _source_and_steps()

    assert [step.command for step in steps] == [
        ("scripts/check-infra-merge-boundary.sh",),
        ("uv", "lock", "--project", "training-sdk", "--check"),
        ("uv", "run", "--directory", "training-sdk", "pytest", "-q"),
        ("uv", "run", "--directory", "training-sdk", "ruff", "check", "src", "tests"),
        ("uv", "lock", "--project", "rtrrl/infra/mock-trainer", "--check"),
        (
            "uv",
            "run",
            "--directory",
            "rtrrl/infra/mock-trainer",
            "--with-editable",
            "../control-plane",
            "pytest",
            "-q",
        ),
        (
            "uv",
            "run",
            "--directory",
            "rtrrl/infra/mock-trainer",
            "ruff",
            "check",
            "src",
            "tests",
        ),
        ("uv", "lock", "--project", "rtrrl/infra/control-plane", "--check"),
        ("uv", "run", "--directory", "rtrrl/infra/control-plane", "pytest", "-q"),
        (
            "uv",
            "run",
            "--directory",
            "rtrrl/infra/control-plane",
            "ruff",
            "check",
            "src",
            "tests",
            "scripts",
        ),
        (
            "rg",
            "-n",
            (
                "^[[:space:]]*(from[[:space:]]+(memo|trainer_infra)"
                "([.]|[[:space:]])|import[[:space:]]+(memo|trainer_infra)"
                "([.]|[[:space:],]|$))"
            ),
            "rtrrl/infra/mock-trainer/src",
        ),
        (
            "rg",
            "-n",
            "memo_stream_ac|memo_rtrrl|memo/infra",
            "rtrrl/infra/control-plane/examples",
            "rtrrl/infra/control-plane/scripts",
            "rtrrl/infra/control-plane/src",
            "rtrrl/infra/control-plane/tests/test_end_to_end.py",
            "rtrrl/infra/control-plane/tests/test_facility_concrete_contract.py",
        ),
        ("git", "diff", "--check"),
        ("scripts/check-infra-merge-boundary.sh",),
    ]
    assert all(step.timeout > 0 for step in steps)


def test_full_suites_are_isolated_cpu_runs_without_filters_or_duplicate_micro_ppo() -> None:
    _, steps = _source_and_steps()
    pytest_steps = [step for step in steps if "pytest" in step.command]

    assert [step.command[3] for step in pytest_steps] == [
        "training-sdk",
        "rtrrl/infra/mock-trainer",
        "rtrrl/infra/control-plane",
    ]
    assert all(step.command[-2:] == ("pytest", "-q") for step in pytest_steps)
    assert sum(step.command[3] == "rtrrl/infra/mock-trainer" for step in pytest_steps) == 1

    isolated = {
        "PYTHONPATH",
        "BRAX_ACCEPTANCE_TEST_MODE",
        "BRAX_ACCEPTANCE_E2E_FAST",
        "CUDA_VISIBLE_DEVICES",
    }
    assert all(isolated <= set(step.unset_environment) for step in pytest_steps)
    for step in pytest_steps[1:]:
        assert "JAX_PLATFORM_NAME=cpu" in step.environment
        assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in step.environment
        assert "OMP_NUM_THREADS=1" in step.environment
    mock_step = next(
        step
        for step in pytest_steps
        if step.command[3] == "rtrrl/infra/mock-trainer"
    )
    assert not any(value.startswith("PYTHONPATH=") for value in mock_step.environment)
    control_plane_step = next(
        step
        for step in pytest_steps
        if step.command[3] == "rtrrl/infra/control-plane"
    )
    assert control_plane_step.timeout >= 7200

    assert (ROOT / "rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py").is_file()
    assert (ROOT / "rtrrl/infra/mock-trainer/tests/test_catalog.py").is_file()
    assert (ROOT / "rtrrl/infra/mock-trainer/tests/test_image_contract.py").is_file()
    assert (ROOT / "rtrrl/infra/control-plane/tests/test_end_to_end.py").is_file()
    assert (ROOT / "rtrrl/infra/control-plane/tests/test_acceptance_image_workflow.py").is_file()


def test_gate_has_only_local_commands_and_precisely_scoped_forbidden_scans() -> None:
    _, steps = _source_and_steps()
    executables = {step.command[0] for step in steps}
    scans = [step.command for step in steps if step.command[0] == "rg"]

    assert executables.isdisjoint({"docker", "gh", "aws"})
    assert len(scans) == 2
    assert all(
        not any(
            target.startswith(("memo/", "docs/", ".superpowers/"))
            or "/tests/data/" in target
            for target in scan[3:]
        )
        for scan in scans
    )
    assert all("historical" not in target for scan in scans for target in scan[3:])
