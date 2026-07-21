import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from trainer_infra import image_catalog
from trainer_infra.image_catalog import (
    CATALOG_PROTOCOL_VERSION,
    LABEL,
    EcrCatalogReader,
    ResolvedImage,
    decode_catalog,
    encode_catalog,
    encode_catalog_file,
    load_catalog_index,
    resolve_image,
)
from trainer_infra.models import ScriptCatalog

DIGEST = "sha256:" + "a" * 64
CONFIG_DIGEST = "sha256:" + "b" * 64


def catalog_data(name: str = "rtrrl") -> dict[str, Any]:
    return {
        "protocol_version": CATALOG_PROTOCOL_VERSION,
        "scripts": {
            name: {
                "name": name,
                "argv": ["python", f"{name}.py", "--config_path", "{config_path}"],
                "sdk_protocol_version": "1",
                "defaults": {
                    "environment": {
                        "name": "brax-hopper",
                        "options": {
                            "backend": "brax",
                            "observation_mode": "P",
                            "max_episode_steps": 1000,
                        },
                    },
                    "training_budget": {"env_steps": 1000},
                    "logging": {
                        "aim_every_env_steps": 100,
                        "rerun_every_episodes": 10,
                    },
                },
                "objective": {
                    "metric": "eval/episode_reward",
                    "direction": "maximize",
                    "reduction": "last",
                },
                "environments": ["brax-hopper"],
                "fields": {
                    "seed": {
                        "path": "seed",
                        "type": "int",
                        "default": 0,
                        "constraints": {"ge": 0},
                    }
                },
            }
        },
    }


@pytest.fixture
def catalog() -> ScriptCatalog:
    return ScriptCatalog.model_validate(catalog_data())


class FakeEcr:
    def __init__(self, catalog: ScriptCatalog) -> None:
        self.tag_digest = DIGEST
        self.manifest: dict[str, Any] = {"config": {"digest": CONFIG_DIGEST}}
        self.config_blob: dict[str, Any] = {
            "config": {"Labels": {LABEL: encode_catalog(catalog)}}
        }
        self.resolve_calls: list[str] = []
        self.manifest_calls: list[str] = []
        self.config_calls: list[tuple[str, str]] = []

    def resolve_tag(self, reference: str) -> str:
        self.resolve_calls.append(reference)
        return self.tag_digest

    def get_manifest(self, reference: str) -> dict[str, Any]:
        self.manifest_calls.append(reference)
        return self.manifest

    def get_config_blob(self, repository: str, digest: str) -> dict[str, Any]:
        self.config_calls.append((repository, digest))
        return self.config_blob


def test_catalog_label_is_deterministic_and_round_trips(catalog: ScriptCatalog) -> None:
    first = encode_catalog(catalog)
    second = encode_catalog(catalog)

    assert first == second
    assert decode_catalog(first) == catalog


def test_decode_rejects_invalid_label_with_context() -> None:
    with pytest.raises(ValueError, match=r"catalog label.*base64"):
        decode_catalog("not base64!")


def test_resolve_image_accepts_only_immutable_digest() -> None:
    image = resolve_image(f"registry.example/repo/image@{DIGEST}")

    assert image.reference == f"registry.example/repo/image@{DIGEST}"
    assert image.repository == "registry.example/repo/image"
    assert image.digest == DIGEST

    with pytest.raises(ValueError, match="immutable digest"):
        resolve_image("registry.example/repo/image:dev")
    with pytest.raises(ValueError, match="immutable"):
        resolve_image(f"https://registry.example/repo/image@{DIGEST}")


def test_resolved_image_constructor_rejects_forged_mutable_reference() -> None:
    with pytest.raises(ValueError, match="immutable digest"):
        ResolvedImage(
            reference="registry.example/repo/image:dev",
            repository="registry.example/repo/image",
            digest=DIGEST,
        )


def test_resolve_image_removes_tag_before_digest() -> None:
    image = resolve_image(f"registry.example:5000/repo/image:dev@{DIGEST}")

    assert image.reference == f"registry.example:5000/repo/image@{DIGEST}"
    assert image.repository == "registry.example:5000/repo/image"
    assert image.digest == DIGEST


def test_tag_is_resolved_once_and_all_reads_use_digest(catalog: ScriptCatalog) -> None:
    fake = FakeEcr(catalog)

    image = EcrCatalogReader(fake).resolve_and_fetch("registry.example/repo/image:dev")

    assert image.reference == f"registry.example/repo/image@{DIGEST}"
    assert image.catalog == catalog
    assert fake.resolve_calls == ["registry.example/repo/image:dev"]
    assert fake.manifest_calls == [image.reference]
    assert fake.config_calls == [("registry.example/repo/image", CONFIG_DIGEST)]


def test_digest_reference_is_never_resolved_again(catalog: ScriptCatalog) -> None:
    fake = FakeEcr(catalog)
    reference = f"registry.example/repo/image@{DIGEST}"

    image = EcrCatalogReader(fake).resolve_and_fetch(reference)

    assert image.reference == reference
    assert fake.resolve_calls == []


def test_fetch_rejects_mutable_image_before_ecr_read(catalog: ScriptCatalog) -> None:
    fake = FakeEcr(catalog)

    with pytest.raises(ValueError, match=r"repo/image:dev.*immutable digest"):
        EcrCatalogReader(fake).fetch("repo/image:dev")  # type: ignore[arg-type]

    assert fake.manifest_calls == []


def test_fetch_reports_image_context_for_missing_label(catalog: ScriptCatalog) -> None:
    fake = FakeEcr(catalog)
    fake.config_blob = {"config": {"Labels": {}}}

    with pytest.raises(ValueError, match=rf"repo/image@{DIGEST}.*{LABEL}"):
        EcrCatalogReader(fake).resolve_and_fetch("repo/image:dev")


def test_fetch_reports_context_for_malformed_manifest(catalog: ScriptCatalog) -> None:
    fake = FakeEcr(catalog)
    fake.manifest = {"config": {}}

    with pytest.raises(ValueError, match=rf"repo/image@{DIGEST}.*config digest"):
        EcrCatalogReader(fake).resolve_and_fetch("repo/image:dev")


def test_fetch_rejects_unsupported_catalog_protocol(catalog: ScriptCatalog) -> None:
    fake = FakeEcr(catalog)
    unsupported = catalog.model_copy(update={"protocol_version": "2"})
    fake.config_blob = {"config": {"Labels": {LABEL: encode_catalog(unsupported)}}}

    with pytest.raises(ValueError, match=r"protocol_version.*2.*expected.*1"):
        EcrCatalogReader(fake).resolve_and_fetch("repo/image:dev")


@pytest.mark.parametrize(
    ("scripts", "message"),
    [
        (
            {
                "first": catalog_data("same")["scripts"]["same"],
                "second": catalog_data("same")["scripts"]["same"],
            },
            "duplicate script name.*same",
        ),
        (
            {"catalog-key": catalog_data("descriptor-name")["scripts"]["descriptor-name"]},
            "catalog key.*catalog-key.*descriptor name.*descriptor-name",
        ),
    ],
)
def test_fetch_rejects_catalog_script_identity_errors(
    catalog: ScriptCatalog,
    scripts: dict[str, Any],
    message: str,
) -> None:
    fake = FakeEcr(catalog)
    invalid = ScriptCatalog.model_validate(
        {"protocol_version": CATALOG_PROTOCOL_VERSION, "scripts": scripts}
    )
    fake.config_blob = {"config": {"Labels": {LABEL: encode_catalog(invalid)}}}

    with pytest.raises(ValueError, match=message):
        EcrCatalogReader(fake).resolve_and_fetch("repo/image:dev")


def write_descriptor(directory: Path, filename: str, name: str) -> None:
    data = catalog_data(name)["scripts"][name]
    (directory / filename).write_text(json.dumps(data), encoding="utf-8")


def test_index_loader_builds_and_encodes_complete_catalog(tmp_path: Path) -> None:
    write_descriptor(tmp_path, "rtrrl.yaml", "rtrrl")
    write_descriptor(tmp_path, "ppo.yaml", "ppo")
    index = tmp_path / "index.yaml"
    index.write_text(
        "protocol_version: '1'\nscripts:\n  - rtrrl.yaml\n  - ppo.yaml\n",
        encoding="utf-8",
    )

    loaded = load_catalog_index(index)

    assert list(loaded.scripts) == ["rtrrl", "ppo"]
    assert decode_catalog(encode_catalog_file(index)) == loaded


def test_index_loader_rejects_duplicate_descriptor_file(tmp_path: Path) -> None:
    write_descriptor(tmp_path, "rtrrl.yaml", "rtrrl")
    index = tmp_path / "index.yaml"
    index.write_text(
        "protocol_version: '1'\nscripts:\n  - rtrrl.yaml\n  - rtrrl.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"index.yaml.*duplicate catalog entry.*rtrrl.yaml"):
        load_catalog_index(index)


def test_index_loader_rejects_duplicate_script_name(tmp_path: Path) -> None:
    write_descriptor(tmp_path, "first.yaml", "same")
    write_descriptor(tmp_path, "second.yaml", "same")
    index = tmp_path / "index.yaml"
    index.write_text(
        "protocol_version: '1'\nscripts:\n  - first.yaml\n  - second.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"second.yaml.*duplicate script name.*same"):
        load_catalog_index(index)


def test_index_loader_uses_task_1_model_validation(tmp_path: Path) -> None:
    descriptor = catalog_data()["scripts"]["rtrrl"]
    descriptor["argv"] = "python rtrrl.py"
    (tmp_path / "rtrrl.yaml").write_text(json.dumps(descriptor), encoding="utf-8")
    index = tmp_path / "index.yaml"
    index.write_text(
        "protocol_version: '1'\nscripts:\n  - rtrrl.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=r"rtrrl.yaml|argv"):
        load_catalog_index(index)


def test_repository_descriptors_form_a_real_complete_catalog() -> None:
    infra = Path(__file__).parents[2]

    catalog = load_catalog_index(infra / "scripts" / "index.yaml")

    assert set(catalog.scripts) == {"rtrrl", "ppo_baseline", "sac_baseline"}
    for name, descriptor in catalog.scripts.items():
        assert descriptor.name == name
        assert descriptor.argv[:2] == ("python", f"{name}.py")
        assert descriptor.fields
        assert descriptor.defaults.training_budget.env_steps > 0
        assert descriptor.objective.metric
        assert descriptor.sdk_protocol_version == "1"


def test_rtrrl_descriptor_uses_actual_reward_metric() -> None:
    infra = Path(__file__).parents[2]

    catalog = load_catalog_index(infra / "scripts" / "index.yaml")

    assert catalog.scripts["rtrrl"].objective.metric == "eval/rewards"


def test_catalog_cli_prints_encoded_validated_index(capsys: pytest.CaptureFixture[str]) -> None:
    index = Path(__file__).parents[2] / "scripts" / "index.yaml"

    exit_code = image_catalog.main([str(index)])

    assert exit_code == 0
    assert decode_catalog(capsys.readouterr().out.strip()) == load_catalog_index(index)


def test_catalog_cli_is_registered_as_console_script() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    contents = pyproject.read_text(encoding="utf-8")

    assert 'trainer-image-catalog = "trainer_infra.image_catalog:main"' in contents


@pytest.mark.parametrize("filename", ["Dockerfile", "Dockerfile.gpu"])
def test_dockerfile_accepts_catalog_build_arg_and_copies_descriptors(filename: str) -> None:
    dockerfile = Path(__file__).parents[2] / "docker" / filename
    contents = dockerfile.read_text(encoding="utf-8")

    assert "ARG TRAINER_SCRIPT_CATALOG" in contents
    guard = 'RUN test -n "${TRAINER_SCRIPT_CATALOG}"'
    assert guard in contents
    assert f'LABEL {LABEL}="${{TRAINER_SCRIPT_CATALOG}}"' in contents
    assert contents.index(guard) < contents.index(f"LABEL {LABEL}")
    assert "COPY infra/scripts /opt/trainer/scripts" in contents


def test_shared_build_entrypoint_passes_catalog_arg_conditionally() -> None:
    script = Path(__file__).parents[4] / "infra" / "build-and-push.sh"
    contents = script.read_text(encoding="utf-8")

    assert 'CATALOG_INDEX="${PROJECT_DIR}/infra/scripts/index.yaml"' in contents
    assert 'if [ -f "${CATALOG_INDEX}" ]; then' in contents
    assert 'uv run --project "${PROJECT_DIR}/infra/control-plane"' in contents
    assert 'trainer-image-catalog "${CATALOG_INDEX}"' in contents
    assert 'BUILD_ARGS+=(--build-arg "TRAINER_SCRIPT_CATALOG=${TRAINER_SCRIPT_CATALOG}")' in contents
    assert '"${BUILD_ARGS[@]}"' in contents


def test_github_build_generates_and_passes_catalog_arg() -> None:
    workflow = Path(__file__).parents[4] / ".github" / "workflows" / "build-rtrrl-image.yml"
    contents = workflow.read_text(encoding="utf-8")

    assert "uses: astral-sh/setup-uv@" in contents
    assert "id: catalog" in contents
    assert "trainer-image-catalog rtrrl/infra/scripts/index.yaml" in contents
    assert "build-args: |" in contents
    assert "TRAINER_SCRIPT_CATALOG=${{ steps.catalog.outputs.value }}" in contents
