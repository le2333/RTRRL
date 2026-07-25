from __future__ import annotations

import json
import hashlib
from typing import Any

import pytest

from trainer_infra.ecr import BotoEcrCatalogReader
from trainer_infra.image_catalog import LABEL, encode_catalog
from trainer_infra.models import ScriptCatalog
from tests.test_image_catalog import catalog_data

DIGEST = "sha256:" + "a" * 64
REGISTRY = "123456789012.dkr.ecr.eu-north-1.amazonaws.com"
REPOSITORY = "team/trainer"


class FakeEcr:
    def __init__(self, catalog: ScriptCatalog) -> None:
        self.calls: list[dict[str, Any]] = []
        self.catalog = catalog
        self.config_bytes = json.dumps(
            {"config": {"Labels": {LABEL: encode_catalog(catalog)}}}
        ).encode()
        self.config_digest = "sha256:" + hashlib.sha256(self.config_bytes).hexdigest()
        self.failures: list[dict[str, str]] = []
        self.ambiguous = False

    def batch_get_image(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        image_id = kwargs["imageIds"][0]
        if self.failures:
            return {"images": [], "failures": self.failures}
        manifest = json.dumps({"config": {"digest": self.config_digest}})
        image = {
            "imageId": {"imageDigest": DIGEST, **image_id},
            "imageManifest": manifest,
        }
        images = [image, image] if self.ambiguous else [image]
        return {"images": images, "failures": []}

    def get_download_url_for_layer(self, **kwargs: Any) -> dict[str, str]:
        assert kwargs == {
            "registryId": "123456789012",
            "repositoryName": REPOSITORY,
            "layerDigest": self.config_digest,
        }
        return {"downloadUrl": "https://example.invalid/config"}


def make_reader(client: FakeEcr) -> BotoEcrCatalogReader:
    return BotoEcrCatalogReader(
        client,
        account_id="123456789012",
        region="eu-north-1",
        read_url=lambda url: client.config_bytes,
    )


def test_tag_resolution_happens_once_then_all_reads_use_canonical_digest() -> None:
    catalog = ScriptCatalog.model_validate(catalog_data())
    client = FakeEcr(catalog)
    reference = f"{REGISTRY}/{REPOSITORY}:release"

    image = make_reader(client).resolve_and_fetch(reference)

    assert image.reference == f"{REGISTRY}/{REPOSITORY}@{DIGEST}"
    assert image.catalog == catalog
    assert [call["imageIds"] for call in client.calls] == [
        [{"imageTag": "release"}],
        [{"imageDigest": DIGEST}],
    ]
    assert all(call["repositoryName"] == REPOSITORY for call in client.calls)
    assert all(call["registryId"] == "123456789012" for call in client.calls)


def test_digest_reference_never_performs_a_tag_lookup() -> None:
    catalog = ScriptCatalog.model_validate(catalog_data())
    client = FakeEcr(catalog)

    image = make_reader(client).resolve_and_fetch(f"{REGISTRY}/{REPOSITORY}@{DIGEST}")

    assert image.catalog == catalog
    assert [call["imageIds"] for call in client.calls] == [[{"imageDigest": DIGEST}]]


@pytest.mark.parametrize("mode", ["missing", "ambiguous", "failure"])
def test_missing_ambiguous_and_failed_ecr_reads_fail_closed(mode: str) -> None:
    catalog = ScriptCatalog.model_validate(catalog_data())
    client = FakeEcr(catalog)
    if mode == "missing":
        client.batch_get_image = lambda **kwargs: {"images": [], "failures": []}  # type: ignore[method-assign]
    elif mode == "ambiguous":
        client.ambiguous = True
    else:
        client.failures = [{"failureCode": "ImageNotFound", "failureReason": "gone"}]

    with pytest.raises(ValueError, match="exactly one|failure"):
        make_reader(client).resolve_and_fetch(f"{REGISTRY}/{REPOSITORY}:release")


def test_malformed_manifest_or_missing_catalog_label_fails_closed() -> None:
    catalog = ScriptCatalog.model_validate(catalog_data())
    client = FakeEcr(catalog)
    missing = b'{"config":{"Labels":{}}}'
    client.config_bytes = missing
    client.config_digest = "sha256:" + hashlib.sha256(missing).hexdigest()
    reader = make_reader(client)
    with pytest.raises(ValueError, match=LABEL):
        reader.resolve_and_fetch(f"{REGISTRY}/{REPOSITORY}@{DIGEST}")

    original = client.batch_get_image

    def malformed(**kwargs: Any) -> dict[str, Any]:
        response = original(**kwargs)
        response["images"][0]["imageManifest"] = "{}"
        return response

    client.batch_get_image = malformed  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="config digest"):
        make_reader(client).resolve_and_fetch(f"{REGISTRY}/{REPOSITORY}@{DIGEST}")


def test_manifest_requires_canonical_config_digest() -> None:
    catalog = ScriptCatalog.model_validate(catalog_data())
    client = FakeEcr(catalog)
    original = client.batch_get_image

    def malformed(**kwargs: Any) -> dict[str, Any]:
        response = original(**kwargs)
        response["images"][0]["imageManifest"] = json.dumps(
            {"config": {"digest": "sha256:not-a-digest"}}
        )
        return response

    client.batch_get_image = malformed  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="invalid config digest"):
        make_reader(client).resolve_and_fetch(f"{REGISTRY}/{REPOSITORY}@{DIGEST}")


def test_digest_response_and_config_blob_hash_must_match() -> None:
    catalog = ScriptCatalog.model_validate(catalog_data())
    client = FakeEcr(catalog)
    original = client.batch_get_image

    def wrong_digest(**kwargs: Any) -> dict[str, Any]:
        response = original(**kwargs)
        if "imageDigest" in kwargs["imageIds"][0]:
            response["images"][0]["imageId"]["imageDigest"] = "sha256:" + "c" * 64
        return response

    client.batch_get_image = wrong_digest  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="requested digest"):
        make_reader(client).resolve_and_fetch(f"{REGISTRY}/{REPOSITORY}@{DIGEST}")

    client = FakeEcr(catalog)
    client.config_bytes += b"tampered"
    with pytest.raises(ValueError, match="config blob digest"):
        make_reader(client).resolve_and_fetch(f"{REGISTRY}/{REPOSITORY}@{DIGEST}")


@pytest.mark.parametrize(
    "reference",
    [
        f"999999999999.dkr.ecr.eu-north-1.amazonaws.com/{REPOSITORY}:release",
        f"123456789012.dkr.ecr.us-east-1.amazonaws.com/{REPOSITORY}:release",
        f"registry.example/{REPOSITORY}:release",
    ],
)
def test_registry_host_must_match_expected_account_and_region(reference: str) -> None:
    catalog = ScriptCatalog.model_validate(catalog_data())
    client = FakeEcr(catalog)
    with pytest.raises(ValueError, match="registry"):
        make_reader(client).resolve_and_fetch(reference)
    assert client.calls == []


def test_raw_boto_error_propagates_and_reader_has_no_mutation_api() -> None:
    catalog = ScriptCatalog.model_validate(catalog_data())
    client = FakeEcr(catalog)
    error = RuntimeError("raw ECR error")

    def fail(**_kwargs: Any) -> Any:
        raise error

    client.batch_get_image = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError) as caught:
        make_reader(client).resolve_and_fetch(f"{REGISTRY}/{REPOSITORY}:release")
    assert caught.value is error

    reader = make_reader(FakeEcr(catalog))
    for name in ("put_image", "delete_repository", "batch_delete_image"):
        assert not hasattr(reader, name)
