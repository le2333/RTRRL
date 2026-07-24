from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import stat
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "verify-infra-only-acceptance.sh"
FORBIDDEN_EXECUTABLES = {
    "aws",
    "curl",
    "docker",
    "gh",
    "oras",
    "podman",
    "scp",
    "skopeo",
    "ssh",
    "wget",
}


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


def _shell_tokens(line: str) -> tuple[str, ...]:
    lexer = shlex.shlex(
        line,
        posix=True,
        punctuation_chars=";&|(){}",
    )
    lexer.commenters = "#"
    lexer.whitespace_split = True
    return tuple(lexer)


def _command_substitutions(source: str) -> tuple[str, ...]:
    substitutions: list[str] = []

    def scan(text: str) -> None:
        index = 0
        quote: str | None = None
        while index < len(text):
            char = text[index]
            if quote == "'":
                if char == "'":
                    quote = None
                index += 1
                continue
            if char == "\\":
                index += 2
                continue
            if (
                quote is None
                and char == "#"
                and (
                    index == 0
                    or text[index - 1].isspace()
                    or text[index - 1] in ";&|(){}"
                )
            ):
                newline = text.find("\n", index)
                if newline == -1:
                    return
                index = newline + 1
                continue
            if char == '"':
                quote = None if quote == '"' else '"'
                index += 1
                continue
            if quote is None and char == "'":
                quote = "'"
                index += 1
                continue
            if char == "$" and index + 1 < len(text) and text[index + 1] == "(":
                start = index + 2
                cursor = start
                depth = 1
                inner_quote: str | None = None
                while cursor < len(text):
                    inner = text[cursor]
                    if inner_quote == "'":
                        if inner == "'":
                            inner_quote = None
                        cursor += 1
                        continue
                    if inner == "\\":
                        cursor += 2
                        continue
                    if inner == '"':
                        inner_quote = None if inner_quote == '"' else '"'
                        cursor += 1
                        continue
                    if inner_quote is None and inner == "'":
                        inner_quote = "'"
                        cursor += 1
                        continue
                    if inner == "(":
                        depth += 1
                    elif inner == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    cursor += 1
                assert depth == 0, "unterminated command substitution"
                body = text[start:cursor]
                substitutions.append(body)
                scan(body)
                index = cursor + 1
                continue
            if char == "`":
                cursor = index + 1
                while cursor < len(text):
                    if text[cursor] == "\\":
                        cursor += 2
                        continue
                    if text[cursor] == "`":
                        break
                    cursor += 1
                assert cursor < len(text), "unterminated backtick substitution"
                body = text[index + 1 : cursor]
                substitutions.append(body)
                scan(body)
                index = cursor + 1
                continue
            index += 1

    scan(source)
    return tuple(substitutions)


def _direct_shell_executables(source: str) -> tuple[str, ...]:
    executables: list[str] = []
    command_openers = {"if", "elif", "while", "until", "then", "else", "do", "{"}
    command_prefixes = {"!", "command", "builtin", "exec", "nohup"}
    separators = {";", "&&", "||", "|", "&"}
    assignment = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")

    for line in _logical_lines(source):
        tokens = list(_shell_tokens(line))
        if len(tokens) >= 2 and tokens[0].isidentifier() and "(" in tokens[1]:
            while tokens and "{" not in tokens[0]:
                tokens.pop(0)
            if tokens:
                tokens.pop(0)

        expect_command = True
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in separators or any(character in token for character in ";&|"):
                expect_command = True
                index += 1
                continue
            if token in command_openers:
                expect_command = True
                index += 1
                continue
            if token in {"fi", "done", "esac", "}"}:
                expect_command = False
                index += 1
                continue
            if not expect_command:
                index += 1
                continue
            if assignment.match(token):
                index += 1
                continue
            if token in command_prefixes:
                index += 1
                continue
            executables.append(token)
            expect_command = False
            index += 1

    for substitution in _command_substitutions(source):
        executables.extend(_direct_shell_executables(substitution))
    return tuple(executables)


def _assert_no_forbidden_shell_executables(source: str) -> None:
    direct = _direct_shell_executables(source)
    wrapped = tuple(
        step.command[0]
        for step in _gate_steps(source)
        if step.command
    )
    forbidden = sorted(
        executable
        for executable in (*direct, *wrapped)
        if Path(executable).name in FORBIDDEN_EXECUTABLES
    )
    assert forbidden == [], f"forbidden shell executables: {forbidden}"


def test_gate_is_executable_fail_fast_rooted_and_valid_bash() -> None:
    source, _ = _source_and_steps()

    assert source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert SCRIPT.stat().st_mode & stat.S_IXUSR
    assert 'ROOT="$(git rev-parse --show-toplevel)"' in source
    assert 'cd "$ROOT"' in source
    assert "VERIFY_TIME_BIN" not in source
    assert "VERIFY_TIMEOUT_BIN" not in source
    assert "/usr/bin/time -v /usr/bin/timeout" in source
    assert "--kill-after=30s" in source
    assert "export UV_OFFLINE=1" in source
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_gate_parses_to_the_complete_required_command_order() -> None:
    _, steps = _source_and_steps()

    assert [step.command for step in steps] == [
        ("scripts/check-infra-merge-boundary.sh",),
        ("uv", "lock", "--offline", "--project", "training-sdk", "--check"),
        ("uv", "run", "--offline", "--directory", "training-sdk", "pytest", "-q"),
        (
            "uv",
            "run",
            "--offline",
            "--directory",
            "training-sdk",
            "ruff",
            "check",
            "src",
            "tests",
        ),
        ("uv", "lock", "--offline", "--project", "rtrrl/infra/mock-trainer", "--check"),
        (
            "uv",
            "run",
            "--offline",
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
            "--offline",
            "--directory",
            "rtrrl/infra/mock-trainer",
            "ruff",
            "check",
            "src",
            "tests",
        ),
        (
            "uv",
            "lock",
            "--offline",
            "--project",
            "rtrrl/infra/control-plane",
            "--check",
        ),
        (
            "uv",
            "run",
            "--offline",
            "--directory",
            "rtrrl/infra/control-plane",
            "pytest",
            "-q",
        ),
        (
            "uv",
            "run",
            "--offline",
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
    assert all(
        step.command[2] == "--offline"
        for step in steps
        if step.command[:2] in {("uv", "lock"), ("uv", "run")}
    )


def test_full_suites_are_isolated_cpu_runs_without_filters_or_duplicate_micro_ppo() -> None:
    _, steps = _source_and_steps()
    pytest_steps = [step for step in steps if "pytest" in step.command]
    pytest_directories = [
        step.command[step.command.index("--directory") + 1]
        for step in pytest_steps
    ]

    assert pytest_directories == [
        "training-sdk",
        "rtrrl/infra/mock-trainer",
        "rtrrl/infra/control-plane",
    ]
    assert all(step.command[-2:] == ("pytest", "-q") for step in pytest_steps)
    assert pytest_directories.count("rtrrl/infra/mock-trainer") == 1

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
        if step.command[step.command.index("--directory") + 1]
        == "rtrrl/infra/mock-trainer"
    )
    assert not any(value.startswith("PYTHONPATH=") for value in mock_step.environment)
    control_plane_step = next(
        step
        for step in pytest_steps
        if step.command[step.command.index("--directory") + 1]
        == "rtrrl/infra/control-plane"
    )
    assert control_plane_step.timeout >= 7200

    assert (ROOT / "rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py").is_file()
    assert (ROOT / "rtrrl/infra/mock-trainer/tests/test_catalog.py").is_file()
    assert (ROOT / "rtrrl/infra/mock-trainer/tests/test_image_contract.py").is_file()
    assert (ROOT / "rtrrl/infra/control-plane/tests/test_end_to_end.py").is_file()
    assert (ROOT / "rtrrl/infra/control-plane/tests/test_acceptance_image_workflow.py").is_file()


def test_gate_has_only_local_commands_and_precisely_scoped_forbidden_scans() -> None:
    source, steps = _source_and_steps()
    scans = [step.command for step in steps if step.command[0] == "rg"]

    _assert_no_forbidden_shell_executables(source)
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


@pytest.mark.parametrize("executable", sorted(FORBIDDEN_EXECUTABLES))
def test_forbidden_scanner_rejects_executable_inserted_in_run_function(
    executable: str,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    mutated = source.replace(
        '  local seconds="$1"',
        f'  {executable} --version\n  local seconds="$1"',
        1,
    )

    with pytest.raises(AssertionError, match=executable):
        _assert_no_forbidden_shell_executables(mutated)


@pytest.mark.parametrize("executable", sorted(FORBIDDEN_EXECUTABLES))
def test_forbidden_scanner_rejects_bare_external_executable(
    executable: str,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    mutated = f"{source}\n{executable} --version\n"

    with pytest.raises(AssertionError, match=executable):
        _assert_no_forbidden_shell_executables(mutated)


def test_forbidden_scanner_ignores_comments_and_quoted_argument_data() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    safe = (
        f"{source}\n"
        "# docker gh aws curl wget $(podman) `skopeo`\n"
        "printf '%s\\n' 'podman skopeo oras ssh scp $(curl) `wget`'\n"
    )

    _assert_no_forbidden_shell_executables(safe)
