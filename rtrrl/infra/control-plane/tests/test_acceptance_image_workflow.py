import re
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[4]
WORKFLOW = ROOT / ".github" / "workflows" / "build-infra-acceptance-image.yml"
PUSH_CONDITION = "${{ inputs.push }}"
EXPECTED_ACTIONS = {
    "actions/checkout": "11d5960a326750d5838078e36cf38b85af677262",
    "astral-sh/setup-uv": "d0cc045d04ccac9d8b7881df0226f9e82c39688e",
    "docker/setup-buildx-action": "8d2750c68a42422c14e847fe6c8ac0403b4cbd6f",
    "docker/build-push-action": "10e90e3645eae34f1e60eeb005ba3a3d33f178e8",
    "aws-actions/configure-aws-credentials": "61815dcd50bd041e203e49132bacad1fd04d2708",
    "aws-actions/amazon-ecr-login": "d539f0932e70871a027e9d5a9d8fc38589180a64",
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
}
MATRIX = [
    {
        "variant": "cpu",
        "dockerfile": "rtrrl/infra/mock-trainer/docker/Dockerfile.cpu",
        "local_tag": "infra-acceptance:cpu",
        "ecr_tag": "infra-acceptance-brax-ppo-cpu-20260723",
    },
    {
        "variant": "gpu",
        "dockerfile": "rtrrl/infra/mock-trainer/docker/Dockerfile.gpu",
        "local_tag": "infra-acceptance:gpu",
        "ecr_tag": "infra-acceptance-brax-ppo-gpu-20260723",
    },
]


def _workflow() -> dict:
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _job(workflow: dict, name: str) -> dict:
    assert set(workflow["jobs"]) == {"build", "push"}
    return workflow["jobs"][name]


def _step(job: dict, identifier: str) -> dict:
    return next(step for step in job["steps"] if step.get("id") == identifier)


def test_workflow_is_manual_build_only_by_default_with_isolated_matrix_runners() -> None:
    workflow = _workflow()
    trigger = workflow.get("on", workflow.get(True))

    assert set(trigger) == {"workflow_dispatch"}
    inputs = trigger["workflow_dispatch"]["inputs"]
    assert inputs["push"] == {
        "description": "Push the two fixed acceptance test tags to the existing ECR repository",
        "required": True,
        "type": "boolean",
        "default": False,
    }
    assert inputs["confirm_account"] == {
        "description": "Required only for push: enter the exact AWS account ID",
        "required": False,
        "type": "string",
        "default": "",
    }

    assert workflow["permissions"] == {"contents": "read"}
    build = _job(workflow, "build")
    push = _job(workflow, "push")
    for job in (build, push):
        assert job["runs-on"] == "ubuntu-24.04"
        assert job["strategy"]["fail-fast"] is False
        assert job["strategy"]["matrix"]["include"] == MATRIX
    assert "permissions" not in build
    assert push["if"] == PUSH_CONDITION
    assert push["needs"] == "build"
    assert push["permissions"] == {"contents": "read", "id-token": "write"}


def test_every_third_party_action_is_locked_to_the_resolved_official_tag_sha() -> None:
    workflow = _workflow()
    action_uses = {
        step["uses"]
        for job_name in ("build", "push")
        for step in _job(workflow, job_name)["steps"]
        if "uses" in step
    }

    assert action_uses == {
        f"{repository}@{sha}" for repository, sha in EXPECTED_ACTIONS.items()
    }
    assert all(
        re.fullmatch(r"[^@\s]+/[^@\s]+@[0-9a-f]{40}", action)
        for action in action_uses
    )


def test_build_job_has_no_oidc_aws_ecr_login_or_push_path() -> None:
    job = _job(_workflow(), "build")
    image_build = _step(job, "build")

    assert image_build["with"]["context"] == "."
    assert image_build["with"]["file"] == "${{ matrix.dockerfile }}"
    assert image_build["with"]["platforms"] == "linux/amd64"
    assert image_build["with"]["load"] is True
    assert image_build["with"]["push"] is False
    assert image_build["with"]["tags"] == "${{ matrix.local_tag }}"

    for step in job["steps"]:
        uses = step.get("uses", "")
        run = step.get("run", "")
        assert not uses.startswith("aws-actions/")
        assert "aws sts " not in run
        assert "aws ecr " not in run
        assert "docker push " not in run


@pytest.mark.parametrize("job_name", ["build", "push"])
def test_each_job_rebuilds_and_runs_actual_label_bound_runtime_contracts(
    job_name: str,
) -> None:
    job = _job(_workflow(), job_name)
    catalog = _step(job, "catalog")["run"]
    image_build = _step(job, "build")
    verify = _step(job, "verify")["run"]

    assert (
        "trainer-image-catalog rtrrl/infra/mock-trainer/scripts/index.yaml"
        in catalog
    )
    assert image_build["with"]["load"] is True
    assert image_build["with"]["push"] is False
    assert "docker image inspect" in verify
    assert "EXPECTED_LABEL" in verify
    assert 'actual_bytes == expected_bytes' in verify
    assert "decode_catalog" in verify
    assert "load_catalog_index" in verify
    assert "actual_catalog == source_catalog" in verify
    assert "org.rtrrl.trainer.scripts.v1" in verify
    assert 'sh -eu -c' in verify
    assert "&&" in verify
    assert "/opt/trainer/worker.py" in verify
    assert "/opt/trainer/scripts/index.yaml" in verify
    assert "/opt/trainer/scripts/brax_ppo_acceptance.yaml" in verify
    assert "/opt/acceptance" in verify
    assert "import brax_ppo_acceptance, boto3, jax, training_sdk" in verify
    assert 'jax.default_backend() == "cpu"' in verify
    assert "jax.numpy.add(2, 3).item() == 5" in verify
    assert 'find_spec("jax_cuda12_plugin") is not None' in verify
    assert 'find_spec("memo") is None' in verify
    assert 'find_spec("trainer_infra") is None' in verify
    assert "--config" not in verify


def test_push_path_confirms_account_before_oidc_and_only_pushes_fixed_test_tag() -> None:
    job = _job(_workflow(), "push")
    steps = job["steps"]
    ordered_ids = [step["id"] for step in steps if "id" in step]
    guarded_ids = [
        "confirm",
        "credentials",
        "account",
        "repository",
        "ecr",
        "push",
    ]
    assert [ordered_ids.index(identifier) for identifier in guarded_ids] == sorted(
        ordered_ids.index(identifier) for identifier in guarded_ids
    )
    assert ordered_ids.index("verify") < ordered_ids.index("confirm")

    confirm = _step(job, "confirm")
    assert 'test "$CONFIRM_ACCOUNT" = "007122174918"' in confirm["run"]
    assert confirm["env"]["AWS_ROLE_ARN"] == (
        "${{ vars.INFRA_ACCEPTANCE_AWS_ROLE_ARN }}"
    )
    assert 'case "$AWS_ROLE_ARN" in' in confirm["run"]
    assert "arn:aws:iam::007122174918:role/*" in confirm["run"]
    assert "exit 1" in confirm["run"]

    credentials = _step(job, "credentials")
    assert credentials["uses"] == (
        "aws-actions/configure-aws-credentials@"
        "61815dcd50bd041e203e49132bacad1fd04d2708"
    )
    assert credentials["with"]["role-to-assume"] == (
        "${{ vars.INFRA_ACCEPTANCE_AWS_ROLE_ARN }}"
    )
    assert credentials["with"]["aws-region"] == "eu-north-1"

    account = _step(job, "account")["run"]
    assert "aws sts get-caller-identity" in account
    assert 'test "$ACTUAL_ACCOUNT" = "007122174918"' in account

    repository = _step(job, "repository")["run"]
    assert "aws ecr describe-repositories --repository-names rtrrl" in repository
    assert "create-repository" not in repository

    push = _step(job, "push")["run"]
    assert _step(job, "push")["env"]["ECR_TAG"] == "${{ matrix.ecr_tag }}"
    assert "docker tag" in push
    assert "docker push" in push
    assert 'REMOTE_IMAGE="$REGISTRY/rtrrl:$ECR_TAG"' in push
    assert "digest: sha256:" in push
    assert "immutable_image=" in push


@pytest.mark.parametrize("job_name", ["build", "push"])
def test_json_evidence_hashes_actual_image_label(
    job_name: str,
) -> None:
    job = _job(_workflow(), job_name)
    evidence = _step(job, "evidence")["run"]
    upload = _step(job, "upload")

    for key in (
        "variant",
        "source_commit",
        "image_id",
        "image_size_bytes",
        "base_digests",
        "catalog_sha256",
        "immutable_digest",
    ):
        assert f'"{key}"' in evidence
    assert "docker image inspect" in evidence
    assert 'os.environ["ACTUAL_LABEL"].encode("ascii")' in evidence
    assert 'os.environ["CATALOG"]' not in evidence
    assert "json.dump" in evidence
    assert upload["with"]["name"] == (
        f"infra-acceptance-{job_name}-${{{{ matrix.variant }}}}-evidence"
    )
    assert upload["with"]["path"] == "infra-acceptance-${{ matrix.variant }}-evidence.json"
    assert upload["with"]["if-no-files-found"] == "error"
