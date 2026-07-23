import re
from pathlib import Path

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


def _workflow() -> dict:
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _job(workflow: dict) -> dict:
    assert set(workflow["jobs"]) == {"build"}
    return workflow["jobs"]["build"]


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

    job = _job(workflow)
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["strategy"]["fail-fast"] is False
    assert job["strategy"]["matrix"]["include"] == [
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


def test_every_third_party_action_is_locked_to_the_resolved_official_tag_sha() -> None:
    job = _job(_workflow())
    action_uses = {step["uses"] for step in job["steps"] if "uses" in step}

    assert action_uses == {
        f"{repository}@{sha}" for repository, sha in EXPECTED_ACTIONS.items()
    }
    assert all(
        re.fullmatch(r"[^@\s]+/[^@\s]+@[0-9a-f]{40}", action)
        for action in action_uses
    )


def test_default_path_loads_one_local_image_and_has_no_aws_login_or_push() -> None:
    job = _job(_workflow())
    build = _step(job, "build")

    assert build["with"]["context"] == "."
    assert build["with"]["file"] == "${{ matrix.dockerfile }}"
    assert build["with"]["platforms"] == "linux/amd64"
    assert build["with"]["load"] is True
    assert build["with"]["push"] is False
    assert build["with"]["tags"] == "${{ matrix.local_tag }}"
    assert "if" not in build

    for step in job["steps"]:
        uses = step.get("uses", "")
        run = step.get("run", "")
        if step.get("if") != PUSH_CONDITION:
            assert not uses.startswith("aws-actions/")
            assert "aws sts " not in run
            assert "aws ecr " not in run
            assert "docker push " not in run


def test_remote_runtime_contracts_are_real_but_never_run_ppo() -> None:
    job = _job(_workflow())
    catalog = _step(job, "catalog")["run"]
    verify = _step(job, "verify")["run"]

    assert (
        "trainer-image-catalog rtrrl/infra/mock-trainer/scripts/index.yaml"
        in catalog
    )
    assert "docker image inspect" in verify
    assert "trainer_infra.image_catalog import decode_catalog" in verify
    assert "org.rtrrl.trainer.scripts.v1" in verify
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
    job = _job(_workflow())
    steps = job["steps"]
    push_steps = [step for step in steps if step.get("if") == PUSH_CONDITION]

    assert [step["id"] for step in push_steps] == [
        "confirm",
        "credentials",
        "account",
        "repository",
        "ecr",
        "push",
    ]
    confirm = _step(job, "confirm")
    assert 'test "$CONFIRM_ACCOUNT" = "007122174918"' in confirm["run"]
    assert confirm["env"]["AWS_ROLE_ARN"] == (
        "${{ vars.INFRA_ACCEPTANCE_AWS_ROLE_ARN }}"
    )
    assert 'test -n "$AWS_ROLE_ARN"' in confirm["run"]

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


def test_json_evidence_records_provenance_and_optional_immutable_digest() -> None:
    job = _job(_workflow())
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
    assert "json.dump" in evidence
    assert upload["with"]["name"] == "infra-acceptance-${{ matrix.variant }}-evidence"
    assert upload["with"]["path"] == "infra-acceptance-${{ matrix.variant }}-evidence.json"
    assert upload["with"]["if-no-files-found"] == "error"
