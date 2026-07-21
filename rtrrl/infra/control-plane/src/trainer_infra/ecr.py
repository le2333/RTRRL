from __future__ import annotations

import json
from dataclasses import replace
import re
from typing import Any, Callable, Mapping
from urllib.request import urlopen

from trainer_infra.image_catalog import (
    LABEL,
    ResolvedImage,
    _validate_catalog_identity,
    decode_catalog,
    resolve_image,
)
from trainer_infra.models import ScriptCatalog

_MANIFEST_TYPES = [
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
]
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _read_url(url: str) -> bytes:
    with urlopen(url) as response:  # noqa: S310 - URL is issued by ECR.
        return response.read()


def _repository_and_tag(reference: str) -> tuple[str, str, str]:
    if "://" in reference or "/" not in reference:
        raise ValueError(f"image {reference!r} is not an ECR image reference")
    registry, repository_reference = reference.split("/", 1)
    last_component = repository_reference.rsplit("/", 1)[-1]
    if ":" not in last_component:
        raise ValueError(f"image {reference!r} must contain a tag or digest")
    repository, tag = repository_reference.rsplit(":", 1)
    if not registry or not repository or not tag:
        raise ValueError(f"image {reference!r} is not an ECR image reference")
    return registry, repository, tag


def _one_image(response: Mapping[str, Any], *, context: str) -> Mapping[str, Any]:
    failures = response.get("failures", [])
    if failures:
        raise ValueError(f"{context}: ECR returned failure: {failures!r}")
    images = response.get("images", [])
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], Mapping):
        raise ValueError(f"{context}: expected exactly one ECR image")
    return images[0]


class BotoEcrCatalogReader:
    def __init__(
        self,
        client: Any,
        *,
        read_url: Callable[[str], bytes] = _read_url,
    ) -> None:
        self._client = client
        self._read_url = read_url

    def _batch_get(self, repository: str, image_id: dict[str, str]) -> Mapping[str, Any]:
        response = self._client.batch_get_image(
            repositoryName=repository,
            imageIds=[image_id],
            acceptedMediaTypes=_MANIFEST_TYPES,
        )
        return _one_image(response, context=f"repository {repository!r}")

    def resolve(self, reference: str) -> ResolvedImage:
        if "@" in reference:
            return resolve_image(reference)
        registry, repository, tag = _repository_and_tag(reference)
        image = self._batch_get(repository, {"imageTag": tag})
        image_id = image.get("imageId")
        digest = image_id.get("imageDigest") if isinstance(image_id, Mapping) else None
        if not isinstance(digest, str):
            raise ValueError(f"image {reference!r}: ECR response is missing image digest")
        return resolve_image(f"{registry}/{repository}@{digest}")

    def fetch(self, image: ResolvedImage) -> ScriptCatalog:
        manifest_image = self._batch_get(
            image.repository.split("/", 1)[1],
            {"imageDigest": image.digest},
        )
        manifest_raw = manifest_image.get("imageManifest")
        if not isinstance(manifest_raw, (str, bytes)):
            raise ValueError(f"image {image.reference}: ECR response is missing manifest")
        try:
            manifest = json.loads(manifest_raw)
            config_digest = manifest["config"]["digest"]
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as error:
            raise ValueError(
                f"image {image.reference}: manifest is missing config digest"
            ) from error
        if not isinstance(config_digest, str) or _DIGEST.fullmatch(config_digest) is None:
            raise ValueError(f"image {image.reference}: manifest has invalid config digest")

        repository = image.repository.split("/", 1)[1]
        response = self._client.get_download_url_for_layer(
            repositoryName=repository,
            layerDigest=config_digest,
        )
        download_url = response.get("downloadUrl")
        if not isinstance(download_url, str) or not download_url:
            raise ValueError(f"image {image.reference}: config download URL is missing")
        try:
            config = json.loads(self._read_url(download_url))
            label = config["config"]["Labels"][LABEL]
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"image {image.reference}: config is missing label {LABEL!r}") from error
        if not isinstance(label, str) or not label:
            raise ValueError(f"image {image.reference}: config label {LABEL!r} is invalid")
        catalog = decode_catalog(label)
        _validate_catalog_identity(catalog, context=f"image {image.reference}")
        return catalog

    def resolve_and_fetch(self, reference: str) -> ResolvedImage:
        image = self.resolve(reference)
        return replace(image, catalog=self.fetch(image))
