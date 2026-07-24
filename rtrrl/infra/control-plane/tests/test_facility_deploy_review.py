from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from trainer_infra.facility_control import load_facility_control
from trainer_infra.image_catalog import encode_catalog_file


SCRIPT = Path(__file__).parents[1] / "scripts" / "deploy_facility.py"
CONTROL = Path(__file__).parents[1] / "config" / "facility.yaml"
ROOT = Path(__file__).parents[4]
LEGACY_FACILITY_COMMIT = "fd195c494ff4ac3b34dff066a6ccb1efb024b16b"


def _load():
    spec = importlib.util.spec_from_file_location("deploy_facility_review", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sts:
    def __init__(self, account: str = "007122174918") -> None:
        self.account = account
        self.calls = 0

    def get_caller_identity(self) -> dict[str, str]:
        self.calls += 1
        return {"Account": self.account, "Arn": f"arn:aws:iam::{self.account}:role/test"}


class Session:
    def __init__(self, account: str = "007122174918", region: str = "eu-north-1") -> None:
        self.region_name = region
        self.sts = Sts(account)
        self.requested: list[str] = []

    def client(self, name: str) -> Any:
        self.requested.append(name)
        if name == "sts":
            return self.sts
        raise AssertionError(f"unexpected client before identity gate: {name}")


@pytest.mark.parametrize("phase", ["push", "register"])
def test_mutating_phase_requires_exact_account_confirmation(phase: str) -> None:
    deploy = _load()
    control = load_facility_control(CONTROL)
    session = Session()
    request = deploy.DeployRequest(**{phase: True})

    with pytest.raises(ValueError, match="--confirm-account 007122174918"):
        deploy.deploy(request, control=control, session=session)

    assert session.requested == []


@pytest.mark.parametrize(
    ("account", "region", "message"),
    [
        ("123456789012", "eu-north-1", "account"),
        ("007122174918", "us-east-1", "region"),
    ],
)
def test_identity_gate_precedes_all_mutation(
    account: str, region: str, message: str
) -> None:
    deploy = _load()
    control = load_facility_control(CONTROL)
    session = Session(account, region)
    request = deploy.DeployRequest(
        push=True,
        confirm_account="007122174918",
    )

    with pytest.raises(ValueError, match=message):
        deploy.deploy(request, control=control, session=session)

    assert session.requested == ["sts"]


def test_local_build_is_independent_and_push_does_not_require_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deploy = _load()
    control = load_facility_control(CONTROL)
    built: list[str] = []
    monkeypatch.setattr(
        deploy,
        "_build_images",
        lambda _control: built.append("build") or ["cpu", "gpu"],
    )

    report = deploy.deploy(
        deploy.DeployRequest(build=True),
        control=control,
        session=Session(account="wrong", region="wrong"),
    )

    assert report["built"] == ["cpu", "gpu"]
    assert built == ["build"]


def test_push_reverifies_both_local_tags_before_login_or_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deploy = _load()
    control = load_facility_control(CONTROL)
    session = Session()
    verified: list[tuple[str, str]] = []
    push_calls: list[object] = []

    def reject_gpu(kind: str, image: str) -> None:
        verified.append((kind, image))
        if kind == "gpu":
            raise ValueError("gpu image verification failed")

    monkeypatch.setattr(deploy, "_verify_image", reject_gpu)
    monkeypatch.setattr(
        deploy,
        "_push_images",
        lambda *_args, **_kwargs: push_calls.append(object()),
    )

    with pytest.raises(ValueError, match="gpu image verification failed"):
        deploy.deploy(
            deploy.DeployRequest(
                push=True,
                confirm_account="007122174918",
            ),
            control=control,
            session=session,
        )

    assert verified == [
        ("cpu", deploy._tagged_image(control, "cpu")),
        ("gpu", deploy._tagged_image(control, "gpu")),
    ]
    assert session.requested == ["sts"]
    assert push_calls == []


def test_registration_uses_only_fixed_control_roles_and_profile_resources() -> None:
    deploy = _load()
    control = load_facility_control(CONTROL)
    calls: list[dict[str, Any]] = []
    batch = SimpleNamespace(
        register_job_definition=lambda **kwargs: (
            calls.append(kwargs)
            or {
                "jobDefinitionArn": (
                    "arn:aws:batch:eu-north-1:007122174918:job-definition/"
                    f"{kwargs['jobDefinitionName']}:1"
                )
            }
        )
    )
    sts = Sts()
    session = SimpleNamespace(
        region_name="eu-north-1",
        client=lambda name: sts if name == "sts" else batch,
    )
    cpu = (
        "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:"
        + "a" * 64
    )
    gpu = (
        "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:"
        + "b" * 64
    )

    deploy.deploy(
        deploy.DeployRequest(
            register=True,
            confirm_account="007122174918",
            cpu_digest=cpu,
            gpu_digest=gpu,
        ),
        control=control,
        session=session,
    )

    assert len(calls) == 4
    assert {
        call["containerProperties"]["jobRoleArn"] for call in calls
    } == {control.job_role_arn}
    assert {
        call["containerProperties"]["executionRoleArn"] for call in calls
    } == {control.execution_role_arn}
    assert [
        call["containerProperties"]["resourceRequirements"] for call in calls
    ] == [
        [{"type": "VCPU", "value": "1"}, {"type": "MEMORY", "value": "1600"}],
        [{"type": "VCPU", "value": "2"}, {"type": "MEMORY", "value": "3200"}],
        [{"type": "VCPU", "value": "4"}, {"type": "MEMORY", "value": "7168"}],
        [
            {"type": "VCPU", "value": "4"},
            {"type": "MEMORY", "value": "12000"},
            {"type": "GPU", "value": "1"},
        ],
    ]


def test_push_uses_temporary_docker_config_and_verifies_digest_catalog() -> None:
    deploy = _load()
    control = load_facility_control(CONTROL)
    token = base64.b64encode(b"AWS:secret").decode()
    ecr = SimpleNamespace(
        get_authorization_token=lambda: {
            "authorizationData": [{"authorizationToken": token}]
        }
    )
    session = SimpleNamespace(client=lambda name: ecr)
    docker_configs: list[str] = []
    commands: list[list[str]] = []
    digest_by_tag = {
        control.cpu_image_tag: "a" * 64,
        control.gpu_image_tag: "b" * 64,
    }

    def run_capture(
        command: list[str],
        *,
        input_text: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        del input_text
        commands.append(command)
        assert env is not None
        docker_config = env["DOCKER_CONFIG"]
        assert Path(docker_config).is_dir()
        docker_configs.append(docker_config)
        if command[1] == "push":
            tag = command[2].rsplit(":", 1)[1]
            return f"latest: digest: sha256:{digest_by_tag[tag]} size: 1234\n"
        return ""

    verified: list[str] = []

    class Reader:
        def resolve_and_fetch(self, reference: str) -> Any:
            assert "@sha256:" in reference
            verified.append(reference)
            return SimpleNamespace(reference=reference, catalog=SimpleNamespace())

    result = deploy._push_images(
        session,
        control,
        run_capture=run_capture,
        reader_factory=lambda _client, **_kwargs: Reader(),
    )

    assert set(result) == {"cpu", "gpu"}
    assert verified == [result["cpu"], result["gpu"]]
    assert all(not Path(path).exists() for path in docker_configs)
    assert not any(command[1] == "image" and "inspect" in command for command in commands)


def test_image_verification_decodes_label_and_checks_runtime_contract(
    tmp_path: Path,
) -> None:
    deploy = _load()
    control = load_facility_control(CONTROL)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("index.yaml", "memo_stream_ac.yaml", "memo_rtrrl.yaml"):
        content = subprocess.check_output(
            [
                "git",
                "show",
                f"{LEGACY_FACILITY_COMMIT}:memo/infra/scripts/{name}",
            ],
            cwd=ROOT,
        )
        (scripts / name).write_bytes(content)
    encoded = encode_catalog_file(scripts / "index.yaml")
    commands: list[list[str]] = []

    def run_capture(
        command: list[str],
        *,
        input_text: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        del input_text, env
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return json.dumps({"org.rtrrl.trainer.scripts.v1": encoded})
        if command[1] == "run":
            return json.dumps(
                {
                    "jax_variant": "cpu",
                    "launchers": True,
                    "training_sdk": True,
                    "worker": True,
                }
            )
        raise AssertionError(command)

    deploy._verify_image(
        "cpu",
        (
            "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl:"
            + control.cpu_image_tag
        ),
        run_capture=run_capture,
    )

    assert [command[1] for command in commands] == ["image", "run"]
