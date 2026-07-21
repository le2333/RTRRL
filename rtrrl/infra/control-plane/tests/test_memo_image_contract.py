from __future__ import annotations

import os
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import tomllib

import pytest
import yaml

from trainer_infra.image_catalog import LABEL

REPOSITORY_ROOT = Path(__file__).parents[4]
MEMO_ROOT = REPOSITORY_ROOT / "memo"
WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "build-memo-image.yml"
DOCKERFILES = (
    "Dockerfile",
    "Dockerfile.gpu",
    "Dockerfile.facility",
    "Dockerfile.facility.gpu",
)


def _dockerignore_rules() -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in (REPOSITORY_ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _matches(path: str, pattern: str) -> bool:
    candidate = PurePosixPath(path)
    if pattern == "**":
        return True
    if pattern.startswith("**/"):
        tail = pattern[3:]
        if tail.endswith("/**"):
            directory = tail[:-3]
            return directory in candidate.parts
        if "/" not in tail and not any(character in tail for character in "*?["):
            return tail in candidate.parts
        return candidate.match(pattern) or candidate.match(tail)
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return path == prefix or path.startswith(f"{prefix}/")
    if "/" in pattern and path.split("/", 1)[0] != pattern.split("/", 1)[0]:
        return False
    if candidate.match(pattern):
        return True
    return False


def _included(path: str, rules: tuple[str, ...]) -> bool:
    ignored = False
    for rule in rules:
        negate = rule.startswith("!")
        pattern = rule[1:] if negate else rule
        if _matches(path, pattern):
            ignored = not negate
    return not ignored


def _context_files() -> set[str]:
    rules = _dockerignore_rules()
    included: set[str] = set()
    for root, directories, files in os.walk(REPOSITORY_ROOT):
        relative_root = Path(root).relative_to(REPOSITORY_ROOT)
        kept_directories = []
        for directory in directories:
            relative = (relative_root / directory).as_posix()
            if _included(relative, rules):
                kept_directories.append(directory)
        directories[:] = kept_directories
        for filename in files:
            relative = (relative_root / filename).as_posix()
            if _included(relative, rules):
                included.add(relative)
    return included


def _copy_sources(dockerfile: Path) -> tuple[str, ...]:
    sources: list[str] = []
    logical_lines = dockerfile.read_text().replace("\\\n", " ").splitlines()
    for line in logical_lines:
        stripped = line.strip()
        if not stripped.startswith("COPY "):
            continue
        parts = shlex.split(stripped)
        if any(part.startswith("--from=") for part in parts[1:]):
            continue
        values = [part for part in parts[1:] if not part.startswith("--")]
        sources.extend(values[:-1])
    return tuple(sources)


@pytest.mark.parametrize(
    ("filename", "sync_extras"),
    [
        ("Dockerfile.facility", "--extra brax --extra facility"),
        ("Dockerfile.facility.gpu", "--extra brax --extra facility --extra cuda12"),
    ],
)
def test_formal_memo_dockerfiles_use_root_context_and_runtime_only_venv(
    filename: str,
    sync_extras: str,
) -> None:
    contents = (MEMO_ROOT / "infra" / "docker" / filename).read_text()

    assert " AS builder" in contents
    assert " AS runtime" in contents
    assert "COPY training-sdk /workspace/training-sdk" in contents
    assert "COPY memo/pyproject.toml memo/uv.lock /workspace/memo/" in contents
    assert f"uv sync --frozen --no-dev --no-editable {sync_extras}" in contents
    assert "COPY --from=builder /opt/venv /opt/venv" in contents
    assert "COPY memo /app" in contents
    assert "COPY rtrrl/infra/worker/worker.py /opt/trainer/worker.py" in contents
    assert "COPY memo/infra/scripts /opt/trainer/scripts" in contents
    assert "COPY rtrrl/infra/control-plane" not in contents
    assert "awscli" not in contents.lower()
    assert "build-essential" not in contents.split(" AS runtime", 1)[1]


@pytest.mark.parametrize(
    ("filename", "extra"),
    [("Dockerfile", ""), ("Dockerfile.gpu", " --extra cuda12")],
)
def test_legacy_memo_dockerfiles_are_clean_buildable_from_root_context(
    filename: str,
    extra: str,
) -> None:
    contents = (MEMO_ROOT / "infra" / "docker" / filename).read_text()

    assert "COPY training-sdk /training-sdk" in contents
    assert "COPY memo/pyproject.toml memo/uv.lock ./" in contents
    assert "COPY memo /app" in contents
    assert "COPY memo/infra/docker/entrypoint.sh /usr/local/bin/entrypoint.sh" in contents
    assert (
        f"uv sync --frozen --no-dev --no-editable --extra brax{extra}" in contents
    )
    assert "uv pip install" not in contents


@pytest.mark.parametrize("filename", ["Dockerfile.facility", "Dockerfile.facility.gpu"])
def test_formal_images_require_nonempty_decodable_catalog_label(filename: str) -> None:
    contents = (MEMO_ROOT / "infra" / "docker" / filename).read_text()
    guard = 'RUN test -n "${TRAINER_SCRIPT_CATALOG}"'
    label = f'LABEL {LABEL}="${{TRAINER_SCRIPT_CATALOG}}"'

    assert "ARG TRAINER_SCRIPT_CATALOG" in contents
    assert guard in contents
    assert label in contents
    assert contents.index(guard) < contents.index(label)


def test_root_dockerignore_actual_context_is_minimal_and_complete() -> None:
    rules = _dockerignore_rules()
    context = _context_files()

    assert rules[0] == "**"
    required = {
        "memo/pyproject.toml",
        "memo/uv.lock",
        "memo/experiments/memo_stream_ac/run.py",
        "memo/experiments/memo_rtrrl/run.py",
        "memo/infra/scripts/index.yaml",
        "memo/infra/scripts/memo_stream_ac.yaml",
        "memo/infra/scripts/memo_rtrrl.yaml",
        "training-sdk/pyproject.toml",
        "training-sdk/src/training_sdk/__init__.py",
        "rtrrl/infra/worker/worker.py",
    }
    assert required <= context
    assert all(
        path.startswith(("memo/", "training-sdk/", "rtrrl/infra/worker/"))
        for path in context
    )
    forbidden_parts = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".cache",
        "history",
        "Aim",
        "aim",
        "Optuna",
        "optuna",
        "artifacts",
    }
    assert not [
        path for path in context if forbidden_parts.intersection(PurePosixPath(path).parts)
    ]


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        "memo/.venv/bin/python",
        "memo/.cache/wheel",
        "memo/history/run.json",
        "memo/Aim/repo",
        "memo/Optuna/study.db",
        "memo/artifacts/model.bin",
        "memo/.env",
        "memo/.env.production",
        "memo/aws-credentials.json",
        "memo/client-secret.json",
        "memo/private.pem",
        "memo/private.key",
        "other-worktree/memo/pyproject.toml",
    ],
)
def test_root_dockerignore_excludes_state_worktrees_and_common_secrets(path: str) -> None:
    assert not _included(path, _dockerignore_rules())


@pytest.mark.parametrize("filename", DOCKERFILES)
def test_every_root_context_copy_source_exists_and_is_included(filename: str) -> None:
    dockerfile = MEMO_ROOT / "infra" / "docker" / filename
    context = _context_files()

    for source in _copy_sources(dockerfile):
        path = REPOSITORY_ROOT / source
        assert path.exists(), f"{filename}: missing COPY source {source}"
        if path.is_file():
            assert source in context, f"{filename}: ignored COPY source {source}"
        else:
            assert any(
                item == source or item.startswith(f"{source}/") for item in context
            ), f"{filename}: empty or ignored COPY directory {source}"


@pytest.mark.parametrize("filename", DOCKERFILES)
def test_every_memo_image_copies_uv_path_dependency(filename: str) -> None:
    pyproject = tomllib.loads((MEMO_ROOT / "pyproject.toml").read_text())
    source = pyproject["tool"]["uv"]["sources"]["training-sdk"]["path"]
    assert source == "../training-sdk"

    contents = (MEMO_ROOT / "infra" / "docker" / filename).read_text()
    assert "COPY training-sdk " in contents


def test_memo_manifest_has_lock_consistent_cpu_and_cuda_runtime_extras() -> None:
    contents = (MEMO_ROOT / "pyproject.toml").read_text()

    assert 'facility = ["boto3"]' in contents
    assert 'cuda12 = ["jax[cuda12]==0.10.0"]' in contents


def test_workflow_keeps_legacy_builds_and_adds_root_context_formal_builds() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    expected = {
        "build": [
            ("memorax-rtrl-cpu", "memo/infra/docker/Dockerfile"),
            ("memorax-rtrl-gpu", "memo/infra/docker/Dockerfile.gpu"),
        ],
        "build-facility": [
            ("memorax-rtrl-facility-cpu", "memo/infra/docker/Dockerfile.facility"),
            ("memorax-rtrl-facility-gpu", "memo/infra/docker/Dockerfile.facility.gpu"),
        ],
    }

    assert set(workflow["jobs"]) == set(expected)
    for job_name, expected_matrix in expected.items():
        job = workflow["jobs"][job_name]
        matrix = job["strategy"]["matrix"]["include"]
        assert [(item["tag"], item["dockerfile"]) for item in matrix] == expected_matrix
        build_step = next(
            step for step in job["steps"] if step.get("uses") == "docker/build-push-action@v6"
        )
        assert build_step["with"]["context"] == "."
        assert build_step["with"]["file"] == "${{ matrix.dockerfile }}"
        for _, dockerfile in expected_matrix:
            assert (REPOSITORY_ROOT / dockerfile).is_file()

    formal_steps = workflow["jobs"]["build-facility"]["steps"]
    catalog_step = next(step for step in formal_steps if step.get("id") == "catalog")
    assert "trainer-image-catalog memo/infra/scripts/index.yaml" in catalog_step["run"]
    formal_build = next(
        step
        for step in formal_steps
        if step.get("uses") == "docker/build-push-action@v6"
    )
    assert (
        "TRAINER_SCRIPT_CATALOG=${{ steps.catalog.outputs.value }}"
        in formal_build["with"]["build-args"]
    )
