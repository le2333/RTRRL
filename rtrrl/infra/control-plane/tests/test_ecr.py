from __future__ import annotations

import json
from typing import Any

import pytest

from trainer_infra.ecr import BotoEcrCatalogReader
from trainer_infra.image_catalog import LABEL, encode_catalog
from trainer_infra.models import ScriptCatalog
from test_image_catalog import catalog_data

DIGEST = "sha256:" + "a" * 64
CONFIG_DIGEST = "sha256:" + "b" * 64
REGISTRY = "123456789012.dkr.ecr.eu-north-1.amazonaws.com"
REPOSITORY = "team/trainer"


class FakeEcr:
    def __init__(self, catalog: ScriptCatalog) -> None:
        self.calls: list[dict[str, Any]] = []
        self.catalog = catalog
        self.failures: list[dict[str, str]] = []
        self.ambiguous = False

    def batch_get_image(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        image_id = kwargs["imageIds"][0]
        if self.failures:
            return {"images": [], "failures": self.failures}
        manifest = json.dumps({"config": {"digest": CONFIG_DIGEST}})
        image = {
            "imageId": {"imageDigest": DIGEST, **image_id},
            "imageManifest": manifest,
        }
        images = [image, image] if self.ambiguous else [image]
        return {"images": images, "failures": []}

    def get_download_url_for_layer(self, **kwargs: Any) -> dict[str, str]:
        assert kwargs == {"repositoryName": REPOSITORY, "layerDigest": CONFIG_DIGEST}
        return {"downloadUrl": "https://example.invalid/config"}


def make_reader(client: FakeEcr) -> BotoEcrCatalogReader:
    config = {"config": {"Labels": {LABEL: encode_catalog(client.catalog)}}}
    return BotoEcrCatalogReader(
        client,
        read_url=lambda url: json.dumps(config).encode(),
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
    reader = BotoEcrCatalogReader(client, read_url=lambda _url: b'{"config":{"Labels":{}}}')
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
