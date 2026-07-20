from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from trainer_infra import heavy_test_cli
from trainer_infra.heavy_tests import (
    AggregateJobFailure,
    HeavyTestRunner,
    JobEvidence,
    PartialSubmissionError,
    SubmittedTestJob,
    TEST_PROFILES,
    AwsNetworkSettings,
    ProfileDriftError,
    create_c7ax_if_missing,
    validate_test_profile,
)

NETWORK_SETTINGS = AwsNetworkSettings(
    subnets=(
        "subnet-08127d1c5d4de6ac2",
        "subnet-0b8c68ea0a9784758",
        "subnet-01a2aa195678f8411",
    ),
    security_group_ids=("sg-0c0ed6b927c5113dc",),
    instance_role=(
        "arn:aws:iam::007122174918:instance-profile/rtrrl-ecs-instance-role"
    ),
)
REPOSITORY_ROOT = Path(__file__).parents[4]
HEAVY_TEST_IMAGE_DIR = REPOSITORY_ROOT / "infra" / "batch" / "heavy-tests"


class FakeBatch:
    def __init__(self, *, include_c7ax: bool = True) -> None:
        self.compute_environments = {
            name: self._compute_environment(name)
            for name in ("c7am", "g6x", *(("c7ax",) if include_c7ax else ()))
        }
        self.job_queues = {
            name: self._job_queue(name)
            for name in ("c7am", "g6x", *(("c7ax",) if include_c7ax else ()))
        }
        self.create_compute_environment_calls: list[dict[str, object]] = []
        self.create_job_queue_calls: list[dict[str, object]] = []
        self.update_calls: list[dict[str, object]] = []

    @staticmethod
    def _compute_environment(name: str) -> dict[str, object]:
        profile = TEST_PROFILES[name]
        return {
            "computeEnvironmentName": profile.compute_environment,
            "computeEnvironmentArn": (
                "arn:aws:batch:eu-north-1:123456789012:"
                f"compute-environment/{profile.compute_environment}"
            ),
            "type": "MANAGED",
            "state": "ENABLED",
            "status": "VALID",
            "computeResources": {
                "type": "EC2",
                "minvCpus": 0,
                "maxvCpus": 32,
                "desiredvCpus": 0,
                "instanceTypes": [profile.instance_type],
                "subnets": list(NETWORK_SETTINGS.subnets),
                "securityGroupIds": list(NETWORK_SETTINGS.security_group_ids),
                "instanceRole": NETWORK_SETTINGS.instance_role,
            },
        }

    def _job_queue(self, name: str) -> dict[str, object]:
        profile = TEST_PROFILES[name]
        compute_environment = self.compute_environments[name]
        return {
            "jobQueueName": profile.queue,
            "jobQueueArn": (
                f"arn:aws:batch:eu-north-1:123456789012:job-queue/{profile.queue}"
            ),
            "state": "ENABLED",
            "status": "VALID",
            "priority": 1,
            "computeEnvironmentOrder": [
                {
                    "order": 1,
                    "computeEnvironment": compute_environment["computeEnvironmentArn"],
                }
            ],
        }

    def describe_compute_environments(
        self, *, computeEnvironments: list[str]
    ) -> dict[str, object]:
        environments = [
            deepcopy(environment)
            for name, environment in self.compute_environments.items()
            if TEST_PROFILES[name].compute_environment in computeEnvironments
        ]
        return {"computeEnvironments": environments}

    def describe_job_queues(self, *, jobQueues: list[str]) -> dict[str, object]:
        queues = [
            deepcopy(queue)
            for name, queue in self.job_queues.items()
            if TEST_PROFILES[name].queue in jobQueues
        ]
        return {"jobQueues": queues}

    def create_compute_environment(self, **kwargs: object) -> dict[str, str]:
        self.create_compute_environment_calls.append(deepcopy(kwargs))
        profile = TEST_PROFILES["c7ax"]
        self.compute_environments["c7ax"] = self._compute_environment("c7ax")
        return {
            "computeEnvironmentName": profile.compute_environment,
            "computeEnvironmentArn": self.compute_environments["c7ax"][
                "computeEnvironmentArn"
            ],
        }

    def create_job_queue(self, **kwargs: object) -> dict[str, str]:
        self.create_job_queue_calls.append(deepcopy(kwargs))
        profile = TEST_PROFILES["c7ax"]
        self.job_queues["c7ax"] = self._job_queue("c7ax")
        return {
            "jobQueueName": profile.queue,
            "jobQueueArn": self.job_queues["c7ax"]["jobQueueArn"],
        }


class InvalidCreateArnBatch(FakeBatch):
    def __init__(self, response: dict[str, object]) -> None:
        super().__init__(include_c7ax=False)
        self.response = response

    def create_compute_environment(self, **kwargs: object) -> dict[str, object]:
        super().create_compute_environment(**kwargs)
        return self.response


@pytest.fixture
def fake_batch() -> FakeBatch:
    return FakeBatch()


def test_profiles_are_exact_and_immutable() -> None:
    assert set(TEST_PROFILES) == {"c7am", "c7ax", "g6x"}
    assert TEST_PROFILES["c7am"].queue == "rtrrl-cpu-c7am-queue"
    assert TEST_PROFILES["c7am"].compute_environment == "rtrrl-cpu-c7am-ce"
    assert TEST_PROFILES["c7am"].instance_type == "c7a.medium"
    assert TEST_PROFILES["c7am"].vcpus == 1
    assert TEST_PROFILES["c7am"].memory_mib == 1600
    assert TEST_PROFILES["c7am"].gpus == 0
    assert TEST_PROFILES["c7am"].gpu_model is None
    assert TEST_PROFILES["c7ax"].queue == "rtrrl-cpu-c7ax-queue"
    assert TEST_PROFILES["c7ax"].compute_environment == "rtrrl-cpu-c7ax-ce"
    assert TEST_PROFILES["c7ax"].instance_type == "c7a.xlarge"
    assert TEST_PROFILES["c7ax"].vcpus == 4
    assert TEST_PROFILES["c7ax"].memory_mib == 7168
    assert TEST_PROFILES["c7ax"].gpus == 0
    assert TEST_PROFILES["c7ax"].gpu_model is None
    assert TEST_PROFILES["g6x"].queue == "rtrrl-gpu-g6x-queue"
    assert TEST_PROFILES["g6x"].compute_environment == "rtrrl-gpu-g6x-ce"
    assert TEST_PROFILES["g6x"].instance_type == "g6.xlarge"
    assert TEST_PROFILES["g6x"].vcpus == 4
    assert TEST_PROFILES["g6x"].memory_mib == 12000
    assert TEST_PROFILES["g6x"].gpus == 1
    assert TEST_PROFILES["g6x"].gpu_model == "NVIDIA L4"

    with pytest.raises(TypeError):
        TEST_PROFILES["other"] = TEST_PROFILES["c7am"]
    with pytest.raises(FrozenInstanceError):
        TEST_PROFILES["c7am"].vcpus = 2


def test_validate_returns_exact_profile_and_resource_arns(fake_batch: FakeBatch) -> None:
    validated = validate_test_profile(fake_batch, "g6x")

    assert validated.profile is TEST_PROFILES["g6x"]
    assert validated.queue_arn.endswith("/rtrrl-gpu-g6x-queue")
    assert validated.compute_environment_arn.endswith("/rtrrl-gpu-g6x-ce")
    assert fake_batch.update_calls == []


@pytest.mark.parametrize("name", ["c7am", "c7ax", "g6x"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subnets", ["subnet-wrong"]),
        (
            "subnets",
            [
                NETWORK_SETTINGS.subnets[0],
                NETWORK_SETTINGS.subnets[0],
                *NETWORK_SETTINGS.subnets[1:],
            ],
        ),
        ("subnets", [*NETWORK_SETTINGS.subnets, 7]),
        ("securityGroupIds", ["sg-wrong"]),
        (
            "securityGroupIds",
            [
                NETWORK_SETTINGS.security_group_ids[0],
                NETWORK_SETTINGS.security_group_ids[0],
            ],
        ),
        ("securityGroupIds", [*NETWORK_SETTINGS.security_group_ids, None]),
        ("instanceRole", "arn:aws:iam::123456789012:instance-profile/wrong"),
    ],
)
def test_every_profile_network_drift_fails_closed(
    fake_batch: FakeBatch, name: str, field: str, value: object
) -> None:
    compute_resources = fake_batch.compute_environments[name]["computeResources"]
    assert isinstance(compute_resources, dict)
    compute_resources[field] = value

    with pytest.raises(ProfileDriftError, match=field):
        validate_test_profile(fake_batch, name)

    assert fake_batch.update_calls == []
    assert fake_batch.create_compute_environment_calls == []
    assert fake_batch.create_job_queue_calls == []


@pytest.mark.parametrize("field", ["subnets", "securityGroupIds"])
def test_network_list_order_is_not_profile_drift(
    fake_batch: FakeBatch, field: str
) -> None:
    settings = AwsNetworkSettings(
        subnets=("subnet-a", "subnet-b"),
        security_group_ids=("sg-a", "sg-b"),
        instance_role=NETWORK_SETTINGS.instance_role,
    )
    compute_resources = fake_batch.compute_environments["c7am"]["computeResources"]
    assert isinstance(compute_resources, dict)
    compute_resources["subnets"] = list(settings.subnets)
    compute_resources["securityGroupIds"] = list(settings.security_group_ids)
    compute_resources[field] = list(reversed(compute_resources[field]))

    validated = validate_test_profile(fake_batch, "c7am", settings=settings)

    assert validated.profile is TEST_PROFILES["c7am"]
    assert fake_batch.update_calls == []


@pytest.mark.parametrize(
    ("resource", "field", "value"),
    [
        ("compute", "computeEnvironmentName", "wrong-ce"),
        ("compute", "type", "UNMANAGED"),
        ("compute", "state", "DISABLED"),
        ("compute", "status", "INVALID"),
        ("resources", "type", "SPOT"),
        ("resources", "minvCpus", 1),
        ("resources", "maxvCpus", 16),
        ("resources", "instanceTypes", ["c7a.large"]),
        ("queue", "jobQueueName", "wrong-queue"),
        ("queue", "state", "DISABLED"),
        ("queue", "status", "INVALID"),
        ("queue", "priority", 2),
        ("queue", "computeEnvironmentOrder", []),
    ],
)
def test_every_existing_profile_drift_fails_closed(
    fake_batch: FakeBatch, resource: str, field: str, value: object
) -> None:
    if resource == "compute":
        fake_batch.compute_environments["c7am"][field] = value
    elif resource == "resources":
        compute_resources = fake_batch.compute_environments["c7am"]["computeResources"]
        assert isinstance(compute_resources, dict)
        compute_resources[field] = value
    else:
        fake_batch.job_queues["c7am"][field] = value

    with pytest.raises(ProfileDriftError, match=field):
        validate_test_profile(fake_batch, "c7am")

    assert fake_batch.update_calls == []
    assert fake_batch.create_compute_environment_calls == []
    assert fake_batch.create_job_queue_calls == []


def test_nonzero_desired_vcpus_is_not_profile_drift(fake_batch: FakeBatch) -> None:
    compute_resources = fake_batch.compute_environments["c7am"]["computeResources"]
    assert isinstance(compute_resources, dict)
    compute_resources["desiredvCpus"] = 4

    validated = validate_test_profile(fake_batch, "c7am")

    assert validated.profile is TEST_PROFILES["c7am"]
    assert fake_batch.update_calls == []


def test_queue_binding_drift_fails_closed(fake_batch: FakeBatch) -> None:
    fake_batch.job_queues["c7am"]["computeEnvironmentOrder"] = [
        {"order": 2, "computeEnvironment": "arn:aws:batch:eu-north-1:123:compute-environment/wrong"}
    ]

    with pytest.raises(ProfileDriftError, match="computeEnvironmentOrder"):
        validate_test_profile(fake_batch, "c7am")

    assert fake_batch.update_calls == []


@pytest.mark.parametrize("resource", ["compute", "queue"])
def test_missing_existing_profile_resource_fails_closed(
    fake_batch: FakeBatch, resource: str
) -> None:
    if resource == "compute":
        del fake_batch.compute_environments["g6x"]
    else:
        del fake_batch.job_queues["g6x"]

    with pytest.raises(ProfileDriftError, match="missing"):
        validate_test_profile(fake_batch, "g6x")

    assert fake_batch.update_calls == []


def test_unknown_profile_is_rejected(fake_batch: FakeBatch) -> None:
    with pytest.raises(ValueError, match="unknown test profile"):
        validate_test_profile(fake_batch, "cpu")


def test_existing_c7ax_is_validated_and_never_mutated(fake_batch: FakeBatch) -> None:
    fake_batch.compute_environments["c7ax"]["computeResources"]["instanceTypes"] = [
        "c7a.large"
    ]

    with pytest.raises(ProfileDriftError, match="instanceTypes"):
        create_c7ax_if_missing(
            fake_batch,
            NETWORK_SETTINGS,
        )

    assert fake_batch.update_calls == []
    assert fake_batch.create_compute_environment_calls == []
    assert fake_batch.create_job_queue_calls == []


def test_missing_c7ax_resources_are_created_exactly() -> None:
    fake_batch = FakeBatch(include_c7ax=False)
    settings = NETWORK_SETTINGS

    create_c7ax_if_missing(fake_batch, settings)

    assert fake_batch.create_compute_environment_calls == [
        {
            "computeEnvironmentName": "rtrrl-cpu-c7ax-ce",
            "type": "MANAGED",
            "state": "ENABLED",
            "computeResources": {
                "type": "EC2",
                "minvCpus": 0,
                "maxvCpus": 32,
                "desiredvCpus": 0,
                "instanceTypes": ["c7a.xlarge"],
                "subnets": list(NETWORK_SETTINGS.subnets),
                "securityGroupIds": list(NETWORK_SETTINGS.security_group_ids),
                "instanceRole": NETWORK_SETTINGS.instance_role,
            },
        }
    ]
    compute_environment_arn = fake_batch.compute_environments["c7ax"][
        "computeEnvironmentArn"
    ]
    assert fake_batch.create_job_queue_calls == [
        {
            "jobQueueName": "rtrrl-cpu-c7ax-queue",
            "state": "ENABLED",
            "priority": 1,
            "computeEnvironmentOrder": [
                {"order": 1, "computeEnvironment": compute_environment_arn}
            ],
        }
    ]
    assert fake_batch.update_calls == []


@pytest.mark.parametrize("response", [{}, {"computeEnvironmentArn": ""}])
def test_invalid_created_compute_environment_arn_fails_closed(
    response: dict[str, object],
) -> None:
    fake_batch = InvalidCreateArnBatch(response)

    with pytest.raises(ProfileDriftError, match="computeEnvironmentArn"):
        create_c7ax_if_missing(fake_batch, NETWORK_SETTINGS)

    assert len(fake_batch.create_compute_environment_calls) == 1
    assert fake_batch.create_job_queue_calls == []
    assert fake_batch.update_calls == []


def test_only_missing_c7ax_queue_is_created(fake_batch: FakeBatch) -> None:
    del fake_batch.job_queues["c7ax"]

    create_c7ax_if_missing(
        fake_batch,
        NETWORK_SETTINGS,
    )

    assert fake_batch.create_compute_environment_calls == []
    assert len(fake_batch.create_job_queue_calls) == 1
    assert fake_batch.update_calls == []


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_heavy_test_builder_stages_only_allowed_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    context = tmp_path / "context"
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n")
    forbidden_directories = (
        ".git",
        ".venv",
        "__pycache__",
        ".cache",
        ".pytest_cache",
        ".ruff_cache",
        "cache",
        "log",
        "logs",
    )

    for tree in ("memo", "training-sdk"):
        tree_root = source / tree
        (tree_root / "src").mkdir(parents=True)
        (tree_root / "keep.txt").write_text(f"{tree} root")
        (tree_root / "src" / "keep.py").write_text(f"{tree} nested")
        (tree_root / "src" / "debug.log").write_text("forbidden log")
        for directory in forbidden_directories:
            for parent in (tree_root, tree_root / "src"):
                forbidden = parent / directory
                forbidden.mkdir()
                (forbidden / "forbidden.txt").write_text("forbidden")

    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; stage_context "$2" "$3" "$4"',
            "bash",
            str(HEAVY_TEST_IMAGE_DIR / "build-image.sh"),
            str(source),
            str(context),
            str(dockerfile),
        ],
        check=True,
    )

    assert (context / "Dockerfile").read_text() == "FROM scratch\n"
    for tree in ("memo", "training-sdk"):
        assert (context / tree / "keep.txt").read_text() == f"{tree} root"
        assert (context / tree / "src" / "keep.py").read_text() == f"{tree} nested"
    staged_paths = tuple(context.rglob("*"))
    assert all(
        not set(path.relative_to(context).parts).intersection(forbidden_directories)
        for path in staged_paths
    )
    assert all(path.suffix != ".log" for path in staged_paths)


def test_heavy_test_overlay_installs_current_sources() -> None:
    dockerfile = (HEAVY_TEST_IMAGE_DIR / "Dockerfile").read_text()

    assert "ARG BASE_IMAGE" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "apt-get install --yes --no-install-recommends time" in dockerfile
    assert "COPY training-sdk /workspace/training-sdk" in dockerfile
    assert "COPY memo /app" in dockerfile
    assert "RUN ln -s /app /app/memo" in dockerfile
    assert dockerfile.index("/opt/venv/bin/python -m ensurepip") < dockerfile.index(
        "/opt/venv/bin/python -m pip install"
    )
    assert (
        "RUN /opt/venv/bin/python -m pip install /workspace/training-sdk pytest"
        in dockerfile
    )
    assert "WORKDIR /app" in dockerfile
    assert "PYTHONPATH=/workspace/training-sdk/src:/app" in dockerfile
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in dockerfile
    assert "MALLOC_ARENA_MAX=2" in dockerfile
    assert 'ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]' in dockerfile


def test_heavy_test_builder_stdout_is_exactly_final_json(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    digest = f"sha256:{'a' * 64}"
    _write_executable(
        fake_bin / "aws",
        f"""#!/usr/bin/env bash
set -eu
case "$*" in
  *get-login-password*) echo password ;;
  *batch-get-image*) printf '%s\\n' '{{"images":[{{"imageId":{{"imageDigest":"{digest}"}}}}],"failures":[]}}' ;;
  *) echo "unexpected aws call: $*" >&2; exit 1 ;;
esac
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -eu
case "$1" in
  info) exit 0 ;;
  login) read -r password; echo "docker login: $password" ;;
  build|push|run) echo "docker $1 diagnostic" ;;
  *) exit 1 ;;
esac
""",
    )
    env = {
        **os.environ,
        "ACCOUNT_ID": "007122174918",
        "ECR_RETRY_DELAY_SECONDS": "0",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        [str(HEAVY_TEST_IMAGE_DIR / "build-image.sh"), "--profile", "c7ax"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert len(result.stdout.splitlines()) == 1
    payload = json.loads(result.stdout)
    assert payload["digest"] == digest
    assert payload["image"] == f"007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@{digest}"
    assert "docker build diagnostic" in result.stderr
    assert "docker push diagnostic" in result.stderr
    assert "docker run diagnostic" in result.stderr


def test_heavy_test_builder_retries_and_reports_ecr_failures(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    counter = tmp_path / "calls"
    _write_executable(
        fake_bin / "aws",
        """#!/usr/bin/env bash
set -eu
count=0
[ ! -f "$COUNTER_FILE" ] || count="$(cat "$COUNTER_FILE")"
count=$((count + 1))
printf '%s' "$count" >"$COUNTER_FILE"
printf '%s\n' '{"images":[],"failures":[{"failureCode":"ImageNotFound","failureReason":"missing test image"}]}'
""",
    )
    env = {
        **os.environ,
        "ACCOUNT_ID": "007122174918",
        "COUNTER_FILE": str(counter),
        "ECR_MAX_ATTEMPTS": "3",
        "ECR_RETRY_DELAY_SECONDS": "0",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        [str(HEAVY_TEST_IMAGE_DIR / "build-image.sh"), "--profile", "c7ax"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert counter.read_text() == "3"
    assert "ImageNotFound" in result.stderr
    assert "missing test image" in result.stderr
    assert '"images":[]' in result.stderr


def test_gpu_rebase_builder_is_pinned_reproducible_and_emits_provenance(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "crane-calls"
    base_digest = f"sha256:{'a' * 64}"
    overlay_digest = f"sha256:{'b' * 64}"
    final_digest = f"sha256:{'c' * 64}"
    overlay = (
        "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@"
        f"{overlay_digest}"
    )
    _write_executable(
        fake_bin / "aws",
        f"""#!/usr/bin/env bash
set -eu
case "$*" in
  *get-login-password*) echo password ;;
  *batch-get-image*) printf '%s\\n' '{{"images":[{{"imageId":{{"imageDigest":"{base_digest}"}}}}],"failures":[]}}' ;;
  *) echo "unexpected aws call: $*" >&2; exit 1 ;;
esac
""",
    )
    _write_executable(
        fake_bin / "crane",
        f"""#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >>"$CRANE_CALLS"
case "$1" in
  version) echo 0.21.7 ;;
  auth) read -r password ;;
  rebase|mutate) echo "pushed" >&2 ;;
  config) printf '%s\\n' '{{"config":{{"Env":["PATH=/opt/venv/bin:/usr/bin","JAX_PLATFORM_NAME=gpu","XLA_FLAGS="]}}}}' ;;
  manifest) printf '%s\\n' '{{"config":{{"digest":"sha256:{'d' * 64}"}}}}' ;;
  digest)
    case "${{@: -1}}" in
      *@{overlay_digest}) echo "{overlay_digest}" ;;
      *) echo "{final_digest}" ;;
    esac
    ;;
  *) echo "unexpected crane call: $*" >&2; exit 1 ;;
esac
""",
    )
    crane_sha256 = hashlib.sha256((fake_bin / "crane").read_bytes()).hexdigest()
    env = {
        **os.environ,
        "ACCOUNT_ID": "007122174918",
        "CRANE_BIN": str(fake_bin / "crane"),
        "CRANE_BIN_SHA256": crane_sha256,
        "CRANE_CALLS": str(calls),
        "ECR_RETRY_DELAY_SECONDS": "0",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    result = subprocess.run(
        [
            str(HEAVY_TEST_IMAGE_DIR / "build-image.sh"),
            "--profile",
            "g6x",
            "--gpu-rebase-from",
            overlay,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = json.loads(result.stdout)
    assert payload["mode"] == "registry-rebase"
    assert payload["crane_version"] == "0.21.7"
    assert payload["overlay_source_image"] == overlay
    assert payload["overlay_manifest_digest"] == overlay_digest
    assert payload["overlay_config_digest"] == "sha256:" + "d" * 64
    assert payload["old_cpu_base"].endswith(f"@{base_digest}")
    assert payload["new_gpu_base"].endswith(f"@{base_digest}")
    assert payload["digest"] == final_digest
    assert payload["config_environment"] == {
        "JAX_PLATFORM_NAME": "gpu",
        "XLA_FLAGS": "",
    }
    crane_calls = calls.read_text()
    assert (
        f"rebase --platform linux/amd64 --old_base "
        f"007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@{base_digest} "
        f"--new_base 007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@{base_digest}"
        in crane_calls
    )
    assert "mutate --platform linux/amd64 --env JAX_PLATFORM_NAME=gpu" in crane_calls
    assert "--env XLA_FLAGS=" in crane_calls


def test_gpu_rebase_builder_pins_crane_release_and_checksum() -> None:
    builder = (HEAVY_TEST_IMAGE_DIR / "build-image.sh").read_text()

    assert 'CRANE_VERSION="0.21.7"' in builder
    assert (
        'CRANE_ARCHIVE_SHA256="1a57bc98207fa1c0d04bf760699099e26'
        'f8383499bfd55b99c1b919a928a7230"' in builder
    )


@pytest.mark.parametrize(
    ("provided_checksum", "expected_message"),
    [
        (None, "CRANE_BIN_SHA256 is required"),
        ("0" * 64, "checksum mismatch"),
    ],
)
def test_crane_override_requires_matching_executable_checksum(
    tmp_path: Path,
    provided_checksum: str | None,
    expected_message: str,
) -> None:
    crane = tmp_path / "crane"
    _write_executable(crane, "#!/usr/bin/env bash\necho 0.21.7\n")
    env = {**os.environ, "CRANE_BIN": str(crane)}
    if provided_checksum is not None:
        env["CRANE_BIN_SHA256"] = provided_checksum

    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {HEAVY_TEST_IMAGE_DIR / 'build-image.sh'}; ensure_crane",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert expected_message in result.stderr


def test_crane_override_accepts_checksum_of_resolved_executable(
    tmp_path: Path,
) -> None:
    real_crane = tmp_path / "real-crane"
    crane_link = tmp_path / "crane"
    _write_executable(real_crane, "#!/usr/bin/env bash\necho 0.21.7\n")
    crane_link.symlink_to(real_crane)
    checksum = hashlib.sha256(real_crane.read_bytes()).hexdigest()

    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f"source {HEAVY_TEST_IMAGE_DIR / 'build-image.sh'}; "
                'ensure_crane; printf \'%s\\n\' "$CRANE_BIN_RESOLVED"'
            ),
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CRANE_BIN": str(crane_link),
            "CRANE_BIN_SHA256": checksum,
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(real_crane)


class FakeJobBatch(FakeBatch):
    def __init__(self) -> None:
        super().__init__()
        self.job_definitions: list[dict[str, object]] = []
        self.register_job_definition_calls: list[dict[str, object]] = []
        self.submit_job_calls: list[dict[str, object]] = []
        self.describe_jobs_calls: list[list[str]] = []
        self.jobs: dict[str, dict[str, object]] = {}

    def describe_job_definitions(self, **kwargs: object) -> dict[str, object]:
        name = kwargs["jobDefinitionName"]
        return {
            "jobDefinitions": [
                deepcopy(definition)
                for definition in self.job_definitions
                if definition["jobDefinitionName"] == name
            ]
        }

    def register_job_definition(self, **kwargs: object) -> dict[str, object]:
        self.register_job_definition_calls.append(deepcopy(kwargs))
        revision = len(self.job_definitions) + 1
        name = str(kwargs["jobDefinitionName"])
        container = deepcopy(kwargs["containerProperties"])
        container["logConfiguration"] = {
            "logDriver": "awslogs",
            "options": {},
            "secretOptions": [],
        }
        for field in ("volumes", "environment", "mountPoints", "ulimits", "secrets"):
            container[field] = []
        definition = {
            **deepcopy(kwargs),
            "containerProperties": container,
            "revision": revision,
            "jobDefinitionArn": (
                f"arn:aws:batch:eu-north-1:123456789012:job-definition/{name}:{revision}"
            ),
        }
        self.job_definitions.append(definition)
        return {
            "jobDefinitionName": name,
            "jobDefinitionArn": definition["jobDefinitionArn"],
            "revision": revision,
        }

    def submit_job(self, **kwargs: object) -> dict[str, str]:
        self.submit_job_calls.append(deepcopy(kwargs))
        job_id = f"job-{len(self.submit_job_calls)}"
        self.jobs[job_id] = {
            "jobId": job_id,
            "status": "SUCCEEDED",
            "container": {"logStreamName": f"stream/{job_id}"},
        }
        return {"jobName": str(kwargs["jobName"]), "jobId": job_id}

    def describe_jobs(self, *, jobs: list[str]) -> dict[str, object]:
        self.describe_jobs_calls.append(list(jobs))
        return {"jobs": [deepcopy(self.jobs[job_id]) for job_id in jobs]}


class FakeLogs:
    def __init__(self, messages: dict[str, list[str]]) -> None:
        self.messages = messages
        self.calls: list[dict[str, object]] = []

    def get_log_events(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(deepcopy(kwargs))
        stream = str(kwargs["logStreamName"])
        return {
            "events": [{"message": message} for message in self.messages.get(stream, [])],
            "nextForwardToken": "done",
        }


IMAGE = "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:" + "a" * 64


def test_one_job_per_exact_test_file() -> None:
    batch = FakeJobBatch()
    runner = HeavyTestRunner(batch, FakeLogs({}), sleep=lambda _: None)

    jobs = runner.submit(
        profile="c7ax",
        image=IMAGE,
        tests=[
            "memo/tests/online_ac/test_eval_trace.py",
            "memo/tests/online_ac/test_jit_contract.py",
        ],
    )

    assert len(jobs) == 2
    assert len(batch.submit_job_calls) == 2
    assert all(" /usr/bin/time -v " in f" {job.command_text} " for job in jobs)
    assert all(
        call["containerOverrides"]["command"][0:2] == ["bash", "-c"]
        for call in batch.submit_job_calls
    )
    assert "test_eval_trace.py -q" in jobs[0].command_text
    assert "test_jit_contract.py -q" in jobs[1].command_text
    assert all(
        str(call["jobName"]).startswith("trainer-heavy-test-c7ax-")
        for call in batch.submit_job_calls
    )
    assert str(batch.register_job_definition_calls[0]["jobDefinitionName"]).startswith(
        "trainer-heavy-test-c7ax-"
    )


def test_long_identical_stems_keep_distinct_suffixes_within_128_chars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tests_root = tmp_path / "memo" / "tests"
    tests_root.mkdir(parents=True)
    long_name = f"test_{'x' * 180}.py"
    test_path = tests_root / long_name
    test_path.write_text("def test_ok(): pass")
    monkeypatch.setattr("trainer_infra.heavy_tests.time.time_ns", lambda: 1)
    batch = FakeJobBatch()
    runner = HeavyTestRunner(
        batch,
        FakeLogs({}),
        repository_root=tmp_path,
        sleep=lambda _: None,
    )

    runner.submit(
        profile="c7ax",
        image=IMAGE,
        tests=[f"memo/tests/{long_name}", f"memo/tests/{long_name}"],
    )

    names = [str(call["jobName"]) for call in batch.submit_job_calls]
    assert len(names) == 2
    assert names[0] != names[1]
    assert all(len(name) == 128 for name in names)
    assert all(re.search(r"-[0-9a-f]{12}$", name) for name in names)


@pytest.mark.parametrize(
    "path",
    [
        "memo/pyproject.toml",
        "../tests/x.py",
        "rtrrl/tests/x.py",
        "/memo/tests/x.py",
        "memo/tests/../x.py",
        "memo/tests/x.txt",
    ],
)
def test_rejects_non_memo_test_path(path: str) -> None:
    runner = HeavyTestRunner(FakeJobBatch(), FakeLogs({}), sleep=lambda _: None)

    with pytest.raises(ValueError, match="memo/tests"):
        runner.submit(profile="c7ax", image=IMAGE, tests=[path])


def test_rejects_missing_directory_and_wrong_suffix(
    tmp_path: Path,
) -> None:
    tests_root = tmp_path / "memo" / "tests"
    tests_root.mkdir(parents=True)
    (tests_root / "directory.py").mkdir()
    (tests_root / "wrong.txt").write_text("not python")
    runner = HeavyTestRunner(
        FakeJobBatch(),
        FakeLogs({}),
        repository_root=tmp_path,
        sleep=lambda _: None,
    )

    for path in (
        "memo/tests/missing.py",
        "memo/tests/directory.py",
        "memo/tests/wrong.txt",
    ):
        with pytest.raises(ValueError, match="regular.*\\.py"):
            runner.submit(profile="c7ax", image=IMAGE, tests=[path])


def test_rejects_test_file_and_parent_directory_symlinks(
    tmp_path: Path,
) -> None:
    tests_root = tmp_path / "memo" / "tests"
    real_parent = tests_root / "real"
    real_parent.mkdir(parents=True)
    real_test = real_parent / "test_real.py"
    real_test.write_text("def test_ok(): pass")
    (tests_root / "test_link.py").symlink_to(real_test)
    (tests_root / "linked_parent").symlink_to(real_parent, target_is_directory=True)
    runner = HeavyTestRunner(
        FakeJobBatch(),
        FakeLogs({}),
        repository_root=tmp_path,
        sleep=lambda _: None,
    )

    for path in (
        "memo/tests/test_link.py",
        "memo/tests/linked_parent/test_real.py",
    ):
        with pytest.raises(ValueError, match="symlink"):
            runner.submit(profile="c7ax", image=IMAGE, tests=[path])


def test_rejects_resolved_path_outside_memo_tests(
    tmp_path: Path,
) -> None:
    tests_root = tmp_path / "memo" / "tests"
    tests_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "test_escape.py").write_text("def test_escape(): pass")
    (tests_root / "escape").symlink_to(outside, target_is_directory=True)
    runner = HeavyTestRunner(
        FakeJobBatch(),
        FakeLogs({}),
        repository_root=tmp_path,
        sleep=lambda _: None,
    )

    with pytest.raises(ValueError, match="symlink|memo/tests"):
        runner.submit(
            profile="c7ax",
            image=IMAGE,
            tests=["memo/tests/escape/test_escape.py"],
        )


@pytest.mark.parametrize(
    "image",
    [
        "repo:latest",
        "repo@sha256:abc",
        "repo@sha256:" + "A" * 64,
        "repo@sha256:" + "a" * 63,
        "https://registry.example/repo@sha256:" + "a" * 64,
        "Registry.example/repo@sha256:" + "a" * 64,
        "registry.example/Repo@sha256:" + "a" * 64,
        "registry..example/repo@sha256:" + "a" * 64,
        "registry.example/@sha256:" + "a" * 64,
        "registry.example/repo:tag@sha256:" + "a" * 64,
    ],
)
def test_rejects_image_without_exact_digest(image: str) -> None:
    runner = HeavyTestRunner(FakeJobBatch(), FakeLogs({}), sleep=lambda _: None)

    with pytest.raises(ValueError, match="digest"):
        runner.submit(
            profile="c7ax",
            image=image,
            tests=["memo/tests/online_ac/test_eval_trace.py"],
        )


def test_partial_submission_error_retains_every_successful_job_id() -> None:
    class FailSecondSubmitBatch(FakeJobBatch):
        def submit_job(self, **kwargs: object) -> dict[str, str]:
            if self.submit_job_calls:
                raise RuntimeError("second submit rejected")
            return super().submit_job(**kwargs)

    batch = FailSecondSubmitBatch()
    runner = HeavyTestRunner(batch, FakeLogs({}), sleep=lambda _: None)

    with pytest.raises(PartialSubmissionError) as raised:
        runner.submit(
            profile="c7ax",
            image=IMAGE,
            tests=[
                "memo/tests/online_ac/test_eval_trace.py",
                "memo/tests/online_ac/test_jit_contract.py",
            ],
        )

    assert [job.job_id for job in raised.value.submitted] == ["job-1"]
    assert raised.value.failed_test == "memo/tests/online_ac/test_jit_contract.py"
    assert "second submit rejected" in raised.value.cause


def test_reuses_only_exact_digest_bound_job_definition() -> None:
    batch = FakeJobBatch()
    runner = HeavyTestRunner(batch, FakeLogs({}), sleep=lambda _: None)

    first = runner.submit(
        profile="c7ax",
        image=IMAGE,
        tests=["memo/tests/online_ac/test_eval_trace.py"],
    )
    second = runner.submit(
        profile="c7ax",
        image=IMAGE,
        tests=["memo/tests/online_ac/test_jit_contract.py"],
    )

    assert len(batch.register_job_definition_calls) == 1
    definition = batch.register_job_definition_calls[0]
    assert definition["type"] == "container"
    assert definition["platformCapabilities"] == ["EC2"]
    assert definition["containerProperties"]["image"] == IMAGE
    assert definition["containerProperties"]["resourceRequirements"] == [
        {"type": "VCPU", "value": "4"},
        {"type": "MEMORY", "value": "7168"},
    ]
    assert "tags" not in definition
    assert all("tags" not in call for call in batch.submit_job_calls)
    assert all("propagateTags" not in call for call in batch.submit_job_calls)
    assert first[0].job_definition_arn == second[0].job_definition_arn
    assert first[0].job_definition_revision == 1


def test_job_definition_search_follows_all_pages() -> None:
    batch = FakeJobBatch()
    seed_runner = HeavyTestRunner(batch, FakeLogs({}), sleep=lambda _: None)
    seed_runner.submit(
        profile="c7ax",
        image=IMAGE,
        tests=["memo/tests/online_ac/test_eval_trace.py"],
    )
    existing = deepcopy(batch.job_definitions[0])
    batch.job_definitions.clear()
    batch.register_job_definition_calls.clear()
    calls: list[dict[str, object]] = []

    def paginated_describe(**kwargs: object) -> dict[str, object]:
        calls.append(deepcopy(kwargs))
        if "nextToken" not in kwargs:
            return {"jobDefinitions": [], "nextToken": "page-2"}
        return {"jobDefinitions": [deepcopy(existing)]}

    batch.describe_job_definitions = paginated_describe  # type: ignore[method-assign]
    runner = HeavyTestRunner(batch, FakeLogs({}), sleep=lambda _: None)

    submitted = runner.submit(
        profile="c7ax",
        image=IMAGE,
        tests=["memo/tests/online_ac/test_eval_trace.py"],
    )

    assert submitted[0].job_definition_revision == 1
    assert [call.get("nextToken") for call in calls] == [None, "page-2"]
    assert batch.register_job_definition_calls == []


def test_registration_is_requeried_and_must_match() -> None:
    class InvisibleRegistrationBatch(FakeJobBatch):
        def describe_job_definitions(self, **kwargs: object) -> dict[str, object]:
            return {"jobDefinitions": []}

    batch = InvisibleRegistrationBatch()
    runner = HeavyTestRunner(batch, FakeLogs({}), sleep=lambda _: None)

    with pytest.raises(RuntimeError, match="registered.*exact"):
        runner.submit(
            profile="c7ax",
            image=IMAGE,
            tests=["memo/tests/online_ac/test_eval_trace.py"],
        )

    assert len(batch.register_job_definition_calls) == 1
    assert batch.submit_job_calls == []


def test_local_process_lock_prevents_duplicate_registration(tmp_path: Path) -> None:
    batch = FakeJobBatch()

    def submit_once() -> tuple[SubmittedTestJob, ...]:
        return HeavyTestRunner(
            batch,
            FakeLogs({}),
            definition_lock_dir=tmp_path / "locks",
            sleep=lambda _: None,
        ).submit(
            profile="c7ax",
            image=IMAGE,
            tests=["memo/tests/online_ac/test_eval_trace.py"],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: submit_once(), range(2)))

    assert len(batch.register_job_definition_calls) == 1
    assert results[0][0].job_definition_arn == results[1][0].job_definition_arn
    assert list((tmp_path / "locks").glob("trainer-heavy-test-*.lock"))


def test_gpu_job_probes_jax_and_l4_before_pytest() -> None:
    batch = FakeJobBatch()
    runner = HeavyTestRunner(batch, FakeLogs({}), sleep=lambda _: None)

    job = runner.submit(
        profile="g6x",
        image=IMAGE,
        tests=["memo/tests/online_ac/test_jit_contract.py"],
    )[0]

    command = job.command_text
    assert command.index("jax.devices()") < command.index("/usr/bin/time -v")
    assert command.index("nvidia-smi --query-gpu=name,memory.total") < command.index(
        "/usr/bin/time -v"
    )
    assert command.index("NVIDIA L4") < command.index("/usr/bin/time -v")
    definition = batch.register_job_definition_calls[0]
    assert {"type": "GPU", "value": "1"} in definition["containerProperties"][
        "resourceRequirements"
    ]


def test_wait_collects_log_stream_rss_and_gpu_evidence() -> None:
    batch = FakeJobBatch()
    batch.jobs = {
        "cpu": {
            "jobId": "cpu",
            "status": "SUCCEEDED",
            "container": {"logStreamName": "stream/cpu"},
        },
        "gpu": {
            "jobId": "gpu",
            "status": "SUCCEEDED",
            "container": {"logStreamName": "stream/gpu"},
        },
    }
    logs = FakeLogs(
        {
            "stream/cpu": ["Maximum resident set size (kbytes): 1234"],
            "stream/gpu": [
                "NVIDIA L4, 23034 MiB",
                "Maximum resident set size (kbytes): 5678",
            ],
        }
    )
    runner = HeavyTestRunner(batch, logs, sleep=lambda _: None)

    evidence = runner.wait(["cpu", "gpu"])

    assert [item.status for item in evidence] == ["SUCCEEDED", "SUCCEEDED"]
    assert evidence[0].log_stream_name == "stream/cpu"
    assert evidence[0].maximum_rss_lines == (
        "Maximum resident set size (kbytes): 1234",
    )
    assert evidence[1].gpu_lines == ("NVIDIA L4, 23034 MiB",)
    assert all(call["logGroupName"] == "/aws/batch/job" for call in logs.calls)


def test_wait_aggregates_all_terminal_failures_with_evidence() -> None:
    batch = FakeJobBatch()
    batch.jobs = {
        "failed": {
            "jobId": "failed",
            "status": "FAILED",
            "statusReason": "Essential container exited",
            "container": {
                "logStreamName": "stream/failed",
                "exitCode": 1,
                "reason": "pytest failed",
            },
        },
        "ok": {
            "jobId": "ok",
            "status": "SUCCEEDED",
            "container": {"logStreamName": "stream/ok", "exitCode": 0},
        },
    }
    logs = FakeLogs(
        {
            "stream/failed": [
                "assert False",
                "Maximum resident set size (kbytes): 9000",
            ],
            "stream/ok": ["Maximum resident set size (kbytes): 1000"],
        }
    )
    runner = HeavyTestRunner(batch, logs, sleep=lambda _: None)

    with pytest.raises(AggregateJobFailure) as raised:
        runner.wait(["failed", "ok"])

    assert [item.job_id for item in raised.value.evidence] == ["failed", "ok"]
    assert raised.value.evidence[0].status_reason == "Essential container exited"
    assert raised.value.evidence[0].container_reason == "pytest failed"
    assert raised.value.evidence[0].log_lines[0] == "assert False"


def test_wait_chunks_describe_jobs_at_aws_limit() -> None:
    batch = FakeJobBatch()
    job_ids = [f"job-{index}" for index in range(205)]
    batch.jobs = {
        job_id: {"jobId": job_id, "status": "FAILED"} for job_id in job_ids
    }
    runner = HeavyTestRunner(batch, FakeLogs({}), sleep=lambda _: None)

    with pytest.raises(AggregateJobFailure) as raised:
        runner.wait(job_ids)

    assert len(raised.value.evidence) == 205
    assert [len(call) for call in batch.describe_jobs_calls[:3]] == [100, 100, 5]
    assert all(len(call) <= 100 for call in batch.describe_jobs_calls)


def test_wait_timeout_aggregates_every_unfinished_job() -> None:
    batch = FakeJobBatch()
    batch.jobs = {
        "running": {"jobId": "running", "status": "RUNNING"},
        "pending": {"jobId": "pending", "status": "RUNNABLE"},
    }
    now = iter((0.0, 0.0, 0.0, 0.0, 2.0))
    runner = HeavyTestRunner(
        batch,
        FakeLogs({}),
        monotonic=lambda: next(now),
        sleep=lambda _: None,
        wait_timeout_seconds=1.0,
    )

    with pytest.raises(AggregateJobFailure) as raised:
        runner.wait(["running", "pending"])

    assert [item.job_id for item in raised.value.evidence] == ["running", "pending"]
    assert [item.status for item in raised.value.evidence] == ["RUNNING", "RUNNABLE"]
    assert all("timeout" in item.evidence_errors[0] for item in raised.value.evidence)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf")])
def test_wait_timeout_must_be_finite(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite.*positive"):
        HeavyTestRunner(
            FakeJobBatch(),
            FakeLogs({}),
            wait_timeout_seconds=timeout,
        )

    runner = HeavyTestRunner(FakeJobBatch(), FakeLogs({}))
    with pytest.raises(ValueError, match="finite.*positive"):
        runner.wait(["job"], timeout_seconds=timeout)


def test_poll_sleep_is_capped_by_remaining_deadline_and_no_extra_aws_call() -> None:
    batch = FakeJobBatch()
    batch.jobs = {"running": {"jobId": "running", "status": "RUNNING"}}
    now = 0.0
    sleeps: list[float] = []

    def monotonic() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    runner = HeavyTestRunner(
        batch,
        FakeLogs({}),
        monotonic=monotonic,
        sleep=sleep,
        poll_interval_seconds=15.0,
        wait_timeout_seconds=5.0,
    )

    with pytest.raises(AggregateJobFailure) as raised:
        runner.wait(["running"])

    assert sleeps == [5.0]
    assert batch.describe_jobs_calls == [["running"]]
    assert raised.value.evidence[0].status == "RUNNING"
    assert "timeout" in raised.value.evidence[0].evidence_errors[0]


def test_wait_retries_late_log_stream_without_blocking_other_jobs() -> None:
    class LateStreamBatch(FakeJobBatch):
        def __init__(self) -> None:
            super().__init__()
            self.refreshes = 0
            self.jobs = {
                "late": {
                    "jobId": "late",
                    "jobName": "trainer-heavy-test-c7ax-late",
                    "status": "SUCCEEDED",
                    "container": {},
                },
                "ready": {
                    "jobId": "ready",
                    "jobName": "trainer-heavy-test-c7ax-ready",
                    "status": "SUCCEEDED",
                    "container": {"logStreamName": "stream/ready"},
                },
            }

        def describe_jobs(self, *, jobs: list[str]) -> dict[str, object]:
            response = super().describe_jobs(jobs=jobs)
            if jobs == ["late"]:
                self.refreshes += 1
                if self.refreshes >= 2:
                    response["jobs"][0]["container"] = {
                        "logStreamName": "stream/late"
                    }
            return response

    batch = LateStreamBatch()
    logs = FakeLogs(
        {
            "stream/late": ["Maximum resident set size (kbytes): 111"],
            "stream/ready": ["Maximum resident set size (kbytes): 222"],
        }
    )
    runner = HeavyTestRunner(
        batch,
        logs,
        evidence_max_attempts=3,
        retry_delay_seconds=0,
        sleep=lambda _: None,
    )

    evidence = runner.wait(["late", "ready"])

    assert [item.maximum_rss_lines for item in evidence] == [
        ("Maximum resident set size (kbytes): 111",),
        ("Maximum resident set size (kbytes): 222",),
    ]


def test_wait_retries_empty_cloudwatch_log_until_required_evidence_arrives() -> None:
    batch = FakeJobBatch()
    batch.jobs = {
        "cpu": {
            "jobId": "cpu",
            "jobName": "trainer-heavy-test-c7ax-cpu",
            "status": "SUCCEEDED",
            "container": {"logStreamName": "stream/cpu"},
        }
    }

    class LateLogs(FakeLogs):
        def get_log_events(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(deepcopy(kwargs))
            if len(self.calls) == 1:
                return {"events": [], "nextForwardToken": "done"}
            return {
                "events": [
                    {"message": "Maximum resident set size (kbytes): 444"}
                ],
                "nextForwardToken": "done",
            }

    logs = LateLogs({})
    runner = HeavyTestRunner(
        batch,
        logs,
        evidence_max_attempts=2,
        retry_delay_seconds=0,
        sleep=lambda _: None,
    )

    evidence = runner.wait(["cpu"])

    assert evidence[0].maximum_rss_lines == (
        "Maximum resident set size (kbytes): 444",
    )
    assert len(logs.calls) >= 2


def test_wait_collects_per_job_log_errors_and_requires_success_evidence() -> None:
    batch = FakeJobBatch()
    batch.jobs = {
        "broken-log": {
            "jobId": "broken-log",
            "jobName": "trainer-heavy-test-c7ax-broken",
            "status": "SUCCEEDED",
            "container": {"logStreamName": "stream/broken"},
        },
        "gpu": {
            "jobId": "gpu",
            "jobName": "trainer-heavy-test-g6x-gpu",
            "status": "SUCCEEDED",
            "container": {"logStreamName": "stream/gpu"},
        },
    }

    class PartiallyBrokenLogs(FakeLogs):
        def get_log_events(self, **kwargs: object) -> dict[str, object]:
            if kwargs["logStreamName"] == "stream/broken":
                raise RuntimeError("logs unavailable")
            return super().get_log_events(**kwargs)

    logs = PartiallyBrokenLogs(
        {
            "stream/gpu": [
                "NVIDIA L4, 23034 MiB",
                "Maximum resident set size (kbytes): 333",
            ]
        }
    )
    runner = HeavyTestRunner(
        batch,
        logs,
        evidence_max_attempts=2,
        retry_delay_seconds=0,
        sleep=lambda _: None,
    )

    with pytest.raises(AggregateJobFailure) as raised:
        runner.wait(["broken-log", "gpu"])

    broken, gpu = raised.value.evidence
    assert any("logs unavailable" in error for error in broken.evidence_errors)
    assert any("RSS" in error for error in broken.evidence_errors)
    assert any("JAX GPU" in error for error in gpu.evidence_errors)
    assert gpu.gpu_lines == ("NVIDIA L4, 23034 MiB",)


def test_submit_cli_prints_machine_readable_job_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Runner:
        def submit(self, **kwargs: object) -> tuple[SubmittedTestJob, ...]:
            assert kwargs == {
                "profile": "c7ax",
                "image": IMAGE,
                "tests": ["memo/tests/online_ac/test_eval_trace.py"],
            }
            return (
                SubmittedTestJob(
                    job_id="job-1",
                    test_file="memo/tests/online_ac/test_eval_trace.py",
                    profile="c7ax",
                    image=IMAGE,
                    job_definition_arn="arn:definition:1",
                    job_definition_revision=1,
                    command_text="/usr/bin/time -v python -m pytest test.py",
                ),
            )

    monkeypatch.setattr(heavy_test_cli, "_runner", Runner)

    result = heavy_test_cli.main(
        [
            "submit",
            "--profile",
            "c7ax",
            "--image",
            IMAGE,
            "memo/tests/online_ac/test_eval_trace.py",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["job_id"] == "job-1"


def test_wait_cli_prints_failure_evidence_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    evidence = JobEvidence(
        job_id="job-1",
        status="FAILED",
        log_stream_name="stream/job-1",
        maximum_rss_lines=("Maximum resident set size (kbytes): 1234",),
        gpu_lines=(),
        log_lines=("assert False",),
        status_reason="Essential container exited",
        exit_code=1,
    )

    class Runner:
        def wait(self, job_ids: list[str]) -> tuple[JobEvidence, ...]:
            assert job_ids == ["job-1"]
            raise AggregateJobFailure([evidence])

    monkeypatch.setattr(heavy_test_cli, "_runner", Runner)

    result = heavy_test_cli.main(["wait", "job-1"])
    captured = capsys.readouterr()

    assert result == 1
    assert json.loads(captured.out)["log_lines"] == ["assert False"]
    error = json.loads(captured.err)
    assert error["error"] == "aggregate_job_failure"
    assert error["job_ids"] == ["job-1"]


def test_submit_cli_reports_partial_submission_as_stderr_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submitted = SubmittedTestJob(
        job_id="job-1",
        test_file="memo/tests/online_ac/test_eval_trace.py",
        profile="c7ax",
        image=IMAGE,
        job_definition_arn="arn:definition:1",
        job_definition_revision=1,
        command_text="/usr/bin/time -v python -m pytest test.py",
    )

    class Runner:
        def submit(self, **kwargs: object) -> tuple[SubmittedTestJob, ...]:
            raise PartialSubmissionError(
                submitted=[submitted],
                failed_test="memo/tests/online_ac/test_jit_contract.py",
                cause=RuntimeError("submit rejected"),
            )

    monkeypatch.setattr(heavy_test_cli, "_runner", Runner)

    result = heavy_test_cli.main(
        [
            "submit",
            "--profile",
            "c7ax",
            "--image",
            IMAGE,
            "memo/tests/online_ac/test_eval_trace.py",
            "memo/tests/online_ac/test_jit_contract.py",
        ]
    )
    captured = capsys.readouterr()

    assert result != 0
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["error"] == "partial_submission"
    assert error["submitted_job_ids"] == ["job-1"]
    assert error["failed_test"] == "memo/tests/online_ac/test_jit_contract.py"


def test_cli_expected_runner_error_is_json_and_stdout_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Runner:
        def submit(self, **kwargs: object) -> tuple[SubmittedTestJob, ...]:
            raise ValueError("invalid request")

    monkeypatch.setattr(heavy_test_cli, "_runner", Runner)

    result = heavy_test_cli.main(
        [
            "submit",
            "--profile",
            "c7ax",
            "--image",
            IMAGE,
            "memo/tests/online_ac/test_eval_trace.py",
        ]
    )
    captured = capsys.readouterr()

    assert result != 0
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": "ValueError",
        "message": "invalid request",
    }


def test_wait_cli_passes_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Runner:
        def wait(
            self, job_ids: list[str], *, timeout_seconds: float
        ) -> tuple[JobEvidence, ...]:
            assert job_ids == ["job-1"]
            assert timeout_seconds == 12.5
            return ()

    monkeypatch.setattr(heavy_test_cli, "_runner", Runner)

    assert (
        heavy_test_cli.main(
            ["wait", "--timeout-seconds", "12.5", "job-1"]
        )
        == 0
    )
