from pathlib import Path

import pytest

from trainer_infra.image_catalog import LABEL

REPOSITORY_ROOT = Path(__file__).parents[4]
MEMO_ROOT = REPOSITORY_ROOT / "memo"
WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "build-memo-image.yml"


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


@pytest.mark.parametrize("filename", ["Dockerfile.facility", "Dockerfile.facility.gpu"])
def test_formal_images_require_nonempty_decodable_catalog_label(filename: str) -> None:
    contents = (MEMO_ROOT / "infra" / "docker" / filename).read_text()
    guard = 'RUN test -n "${TRAINER_SCRIPT_CATALOG}"'
    label = f'LABEL {LABEL}="${{TRAINER_SCRIPT_CATALOG}}"'

    assert "ARG TRAINER_SCRIPT_CATALOG" in contents
    assert guard in contents
    assert label in contents
    assert contents.index(guard) < contents.index(label)


def test_root_dockerignore_excludes_state_but_keeps_formal_image_inputs() -> None:
    contents = (REPOSITORY_ROOT / ".dockerignore").read_text()

    for excluded in (
        ".git",
        "**/.git",
        "**/.venv",
        "**/__pycache__",
        "**/.pytest_cache",
        "**/.ruff_cache",
        "**/.mypy_cache",
        "**/.cache",
        "**/history",
        "**/aim",
        "**/optuna",
        "**/artifacts",
        "**/logs",
    ):
        assert excluded in contents
    for forbidden in (
        "memo",
        "training-sdk",
        "rtrrl/infra/worker",
        "memo/infra/scripts",
    ):
        assert forbidden not in {
            line.strip().rstrip("/")
            for line in contents.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }


def test_memo_manifest_has_lock_consistent_cpu_and_cuda_runtime_extras() -> None:
    contents = (MEMO_ROOT / "pyproject.toml").read_text()

    assert 'facility = ["boto3"]' in contents
    assert 'cuda12 = ["jax[cuda12]==0.10.0"]' in contents


def test_workflow_keeps_legacy_builds_and_adds_root_context_formal_builds() -> None:
    contents = WORKFLOW.read_text()

    for legacy in (
        "context: memo",
        "memo/infra/docker/Dockerfile",
        "memo/infra/docker/Dockerfile.gpu",
        "memorax-rtrl-cpu",
        "memorax-rtrl-gpu",
    ):
        assert legacy in contents
    for formal in (
        "context: .",
        "memo/infra/docker/Dockerfile.facility",
        "memo/infra/docker/Dockerfile.facility.gpu",
        "memorax-rtrl-facility-cpu",
        "memorax-rtrl-facility-gpu",
        "trainer-image-catalog memo/infra/scripts/index.yaml",
        "TRAINER_SCRIPT_CATALOG=${{ steps.catalog.outputs.value }}",
    ):
        assert formal in contents
    assert '"training-sdk/**"' in contents
    assert '"rtrrl/infra/worker/**"' in contents
