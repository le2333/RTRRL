import re
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[4]
DOCKER_DIR = ROOT / "rtrrl" / "infra" / "mock-trainer" / "docker"
DOCKERFILES = ("Dockerfile.cpu", "Dockerfile.gpu")
ROOT_CONTEXT_SOURCES = {
    "training-sdk",
    "rtrrl/infra/mock-trainer",
    "rtrrl/infra/worker/worker.py",
    "rtrrl/infra/mock-trainer/scripts",
}
EXPECTED_PATHS = {
    "/opt/trainer/worker.py",
    "/opt/trainer/scripts",
    "/opt/acceptance",
}


def _root_context_copies(contents: str) -> list[tuple[str, str]]:
    copies: list[tuple[str, str]] = []
    for line in contents.splitlines():
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        words = shlex.split(line)
        assert len(words) == 3, f"COPY must have one source and one destination: {line}"
        copies.append((words[1], words[2]))
    return copies


@pytest.mark.parametrize("filename", DOCKERFILES)
def test_dockerfile_has_pinned_amd64_inputs_and_catalog_label(filename: str) -> None:
    contents = (DOCKER_DIR / filename).read_text(encoding="utf-8")

    from_lines = [line for line in contents.splitlines() if line.startswith("FROM ")]
    assert len(from_lines) == 2
    assert all("latest" not in line for line in from_lines)
    assert all(re.search(r"@sha256:[0-9a-f]{64}(?:\s|$)", line) for line in from_lines)
    assert 'ARG TRAINER_SCRIPT_CATALOG' in contents
    guard = 'RUN test -n "${TRAINER_SCRIPT_CATALOG}"'
    label = 'LABEL org.rtrrl.trainer.scripts.v1="${TRAINER_SCRIPT_CATALOG}"'
    assert guard in contents
    assert label in contents
    assert contents.index(guard) < contents.index(label)


@pytest.mark.parametrize("filename", DOCKERFILES)
def test_dockerfile_copies_only_the_infra_acceptance_context(filename: str) -> None:
    contents = (DOCKER_DIR / filename).read_text(encoding="utf-8")
    copies = _root_context_copies(contents)

    assert {source for source, _ in copies} == ROOT_CONTEXT_SOURCES
    assert EXPECTED_PATHS <= {destination for _, destination in copies}
    assert "memo" not in {source.split("/", 1)[0] for source, _ in copies}
    assert all("control-plane" not in source and "trainer_infra" not in source for source, _ in copies)


def test_cpu_and_gpu_images_share_paths_catalog_and_only_differ_in_acceleration() -> None:
    cpu = (DOCKER_DIR / "Dockerfile.cpu").read_text(encoding="utf-8")
    gpu = (DOCKER_DIR / "Dockerfile.gpu").read_text(encoding="utf-8")

    assert _root_context_copies(cpu) == _root_context_copies(gpu)
    assert "--extra cuda12" not in cpu
    assert "--extra cuda12" in gpu
    assert "JAX_PLATFORM_NAME=cpu" in cpu
    assert "JAX_PLATFORM_NAME=cpu" not in gpu


def test_dockerignore_exposes_every_copy_source_and_no_control_plane() -> None:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    active = {line for line in lines if line and not line.startswith("#")}

    assert lines[0] == "**"
    assert {
        "!training-sdk",
        "!training-sdk/**",
        "!rtrrl",
        "!rtrrl/infra",
        "!rtrrl/infra/worker",
        "!rtrrl/infra/worker/worker.py",
        "!rtrrl/infra/mock-trainer",
        "!rtrrl/infra/mock-trainer/**",
    } <= active
    assert "!memo" not in active
    assert "!memo/**" not in active
    assert all("control-plane" not in line for line in active)

    for source in ROOT_CONTEXT_SOURCES:
        assert (ROOT / source).exists()
