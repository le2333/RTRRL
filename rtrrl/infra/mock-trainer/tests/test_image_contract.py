import fnmatch
import json
import os
import shlex
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).parents[4]
DOCKER_DIR = ROOT / "rtrrl" / "infra" / "mock-trainer" / "docker"
DOCKERFILES = ("Dockerfile.cpu", "Dockerfile.gpu")
PYTHON_REF = (
    "python:3.12.11-slim-bookworm"
    "@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49"
)
UV_REF = (
    "ghcr.io/astral-sh/uv:0.11.25"
    "@sha256:bb38f0ebd7d5d42c60d46a72cc8dbf2ed66c7263212f3b94a8ddfe2b60f7f8ca"
)
RECORDED_MANIFEST_INDEXES = {
    "python:3.12.11-slim-bookworm": json.loads(
        """
        {
          "schemaVersion": 2,
          "mediaType": "application/vnd.oci.image.index.v1+json",
          "manifests": [
            {
              "digest": "sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49",
              "platform": {"architecture": "amd64", "os": "linux"}
            },
            {
              "digest": "sha256:9bb659dc6d5218917236f3711e866a5634bb4c2f208de9d4533aa4863f57c1d3",
              "platform": {"architecture": "arm64", "os": "linux", "variant": "v8"}
            }
          ]
        }
        """
    ),
    "ghcr.io/astral-sh/uv:0.11.25": json.loads(
        """
        {
          "schemaVersion": 2,
          "mediaType": "application/vnd.oci.image.index.v1+json",
          "manifests": [
            {
              "digest": "sha256:bb38f0ebd7d5d42c60d46a72cc8dbf2ed66c7263212f3b94a8ddfe2b60f7f8ca",
              "platform": {"architecture": "amd64", "os": "linux"}
            },
            {
              "digest": "sha256:afb0e190853be0b7418175027d5158a3f4c2d30f3ca64312df8dac758cc2fcb2",
              "platform": {"architecture": "arm64", "os": "linux"}
            }
          ]
        }
        """
    ),
}
ROOT_CONTEXT_SOURCES = {
    "training-sdk",
    "rtrrl/infra/mock-trainer",
    "rtrrl/infra/mock-trainer/catalog.json",
    "rtrrl/infra/worker/worker.py",
}
EXPECTED_PATHS = {
    "/opt/trainer/worker.py",
    "/opt/trainer/catalog.json",
    "/opt/acceptance",
}


def _image_inputs(contents: str) -> tuple[list[str], list[str]]:
    stages: set[str] = set()
    from_images: list[str] = []
    copy_froms: list[str] = []
    for line in contents.splitlines():
        if not line.startswith(("FROM ", "COPY ")):
            continue
        words = shlex.split(line)
        if words[0] == "FROM":
            from_images.append(words[1])
            if len(words) == 4 and words[2].upper() == "AS":
                stages.add(words[3])
        elif words[0] == "COPY":
            copy_froms.extend(
                word.removeprefix("--from=")
                for word in words[1:]
                if word.startswith("--from=")
            )
    assert all(source in stages or "@" in source for source in copy_froms)
    return from_images, copy_froms


def _root_context_copies(contents: str) -> list[tuple[str, str]]:
    copies: list[tuple[str, str]] = []
    for line in contents.splitlines():
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        words = shlex.split(line)
        assert len(words) == 3, f"COPY must have one source and one destination: {line}"
        copies.append((words[1], words[2]))
    return copies


def _pinned_image(reference: str) -> tuple[str, str, str]:
    tagged, digest = reference.rsplit("@", 1)
    repository, version = tagged.rsplit(":", 1)
    return repository, version, digest


def _amd64_digest(index: dict) -> str:
    matches = [
        manifest["digest"]
        for manifest in index["manifests"]
        if manifest.get("platform") == {"architecture": "amd64", "os": "linux"}
    ]
    assert len(matches) == 1
    return matches[0]


class DockerIgnoreMatcher:
    def __init__(self, contents: str) -> None:
        self.rules = tuple(
            line.strip()
            for line in contents.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    @staticmethod
    def _matches(path: str, pattern: str, *, include_ancestors: bool) -> bool:
        pattern = pattern.strip("/")
        if pattern == "**":
            return True
        candidate = PurePosixPath(path)
        candidates = [candidate]
        if include_ancestors:
            candidates.extend(candidate.parents[:-1])
        patterns = [pattern]
        if pattern.startswith("**/"):
            patterns.append(pattern[3:])
        for candidate in candidates:
            for candidate_pattern in patterns:
                if "/" not in candidate_pattern:
                    if fnmatch.fnmatchcase(candidate.name, candidate_pattern):
                        return True
                elif candidate.match(candidate_pattern):
                    return True
        return False

    def ignored(self, path: str) -> bool:
        ignored = False
        for rule in self.rules:
            negated = rule.startswith("!")
            pattern = rule[1:] if negated else rule
            if self._matches(path, pattern, include_ancestors=not negated):
                ignored = not negated
        return ignored


def _included_context_files(matcher: DockerIgnoreMatcher) -> set[str]:
    included: set[str] = set()
    for directory, dirnames, filenames in os.walk(ROOT):
        directory_path = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if not matcher.ignored((directory_path / name).relative_to(ROOT).as_posix())
        ]
        for name in filenames:
            relative = (directory_path / name).relative_to(ROOT).as_posix()
            if not matcher.ignored(relative):
                included.add(relative)
    return included


@pytest.mark.parametrize("filename", DOCKERFILES)
def test_dockerfile_inputs_match_recorded_amd64_registry_manifests(filename: str) -> None:
    contents = (DOCKER_DIR / filename).read_text(encoding="utf-8")
    from_images, copy_froms = _image_inputs(contents)

    assert from_images == [PYTHON_REF, PYTHON_REF]
    assert copy_froms == [UV_REF, "builder"]
    external_images = {PYTHON_REF, UV_REF}
    assert {
        (repository, version)
        for repository, version, _ in map(_pinned_image, external_images)
    } == {
        ("python", "3.12.11-slim-bookworm"),
        ("ghcr.io/astral-sh/uv", "0.11.25"),
    }
    for reference in external_images:
        repository, version, digest = _pinned_image(reference)
        recorded = RECORDED_MANIFEST_INDEXES[f"{repository}:{version}"]
        assert recorded["schemaVersion"] == 2
        assert recorded["mediaType"] == "application/vnd.oci.image.index.v1+json"
        assert digest == _amd64_digest(recorded)

    assert 'ARG TRAINER_CATALOG_V2' in contents
    guard = 'RUN test -n "${TRAINER_CATALOG_V2}"'
    label = 'LABEL org.rtrrl.trainer.catalog.v2="${TRAINER_CATALOG_V2}"'
    assert guard in contents
    assert label in contents
    assert contents.index(guard) < contents.index(label)
    assert "COPY rtrrl/infra/mock-trainer/catalog.json /opt/trainer/catalog.json" in contents


@pytest.mark.parametrize("filename", DOCKERFILES)
def test_dockerfile_copies_only_the_infra_acceptance_context(filename: str) -> None:
    contents = (DOCKER_DIR / filename).read_text(encoding="utf-8")
    copies = _root_context_copies(contents)
    matcher = DockerIgnoreMatcher((ROOT / ".dockerignore").read_text(encoding="utf-8"))

    assert {source for source, _ in copies} == ROOT_CONTEXT_SOURCES
    assert EXPECTED_PATHS <= {destination for _, destination in copies}
    assert all(source != "." for source, _ in copies)
    assert "memo" not in {source.split("/", 1)[0] for source, _ in copies}
    assert all("control-plane" not in source and "trainer_infra" not in source for source, _ in copies)
    for source, _ in copies:
        source_path = ROOT / source
        assert source_path.exists()
        assert not matcher.ignored(source)


def test_cpu_and_gpu_images_share_paths_catalog_and_only_differ_in_acceleration() -> None:
    cpu = (DOCKER_DIR / "Dockerfile.cpu").read_text(encoding="utf-8")
    gpu = (DOCKER_DIR / "Dockerfile.gpu").read_text(encoding="utf-8")

    assert _root_context_copies(cpu) == _root_context_copies(gpu)
    assert "--extra cuda12" not in cpu
    assert "--extra cuda12" in gpu
    assert "JAX_PLATFORM_NAME=cpu" in cpu
    assert "JAX_PLATFORM_NAME=cpu" not in gpu


def test_dockerignore_rules_apply_allowlist_before_runtime_and_secret_exclusions() -> None:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    matcher = DockerIgnoreMatcher("\n".join(lines))
    rules = matcher.rules
    allowlist = (
        "!training-sdk",
        "!training-sdk/**",
        "!rtrrl",
        "!rtrrl/infra",
        "!rtrrl/infra/worker",
        "!rtrrl/infra/worker/worker.py",
        "!rtrrl/infra/mock-trainer",
        "!rtrrl/infra/mock-trainer/**",
    )
    exclusions = (
        ".git",
        "**/.git",
        "**/worktrees",
        "**/Aim",
        "**/artifacts",
        "**/.env",
        "**/.netrc",
        "**/.pypirc",
        "**/.npmrc",
        "**/*credentials*",
        "**/*secret*",
        "**/*.pem",
        "**/*.key",
    )

    assert rules[0] == "**"
    assert rules[1 : 1 + len(allowlist)] == allowlist
    assert all(rule in rules for rule in exclusions)
    assert max(rules.index(rule) for rule in allowlist) < min(
        rules.index(rule) for rule in exclusions
    )
    assert "!memo" not in rules
    assert "!memo/**" not in rules
    assert all("control-plane" not in rule for rule in rules)

    for source in ROOT_CONTEXT_SOURCES:
        assert (ROOT / source).exists()
        assert not matcher.ignored(source)

    excluded_examples = {
        "memo/train.py",
        "rtrrl/infra/control-plane/src/trainer_infra/cli.py",
        ".git",
        ".git/worktrees/task/HEAD",
        "worktrees/task/file.py",
        "rtrrl/infra/mock-trainer/Aim/run.db",
        "rtrrl/infra/mock-trainer/artifacts/result.json",
        "training-sdk/.env",
        "training-sdk/.env.local",
        "training-sdk/.netrc",
        "training-sdk/.pypirc",
        "training-sdk/.npmrc",
        "training-sdk/aws-credentials.json",
        "training-sdk/client-secret.txt",
        "training-sdk/client.pem",
        "training-sdk/client.key",
    }
    assert all(matcher.ignored(path) for path in excluded_examples)


def test_dockerignore_actual_root_context_contains_only_required_inputs() -> None:
    matcher = DockerIgnoreMatcher((ROOT / ".dockerignore").read_text(encoding="utf-8"))
    included = _included_context_files(matcher)

    assert included
    assert "rtrrl/infra/worker/worker.py" in included
    assert any(path.startswith("training-sdk/") for path in included)
    assert any(path.startswith("rtrrl/infra/mock-trainer/") for path in included)
    assert all(
        path == "rtrrl/infra/worker/worker.py"
        or path.startswith(("training-sdk/", "rtrrl/infra/mock-trainer/"))
        for path in included
    )
    assert not any(
        forbidden in path
        for path in included
        for forbidden in (
            "memo/",
            "control-plane/",
            "/.git/",
            "worktrees/",
            "/Aim/",
            "/artifacts/",
        )
    )
