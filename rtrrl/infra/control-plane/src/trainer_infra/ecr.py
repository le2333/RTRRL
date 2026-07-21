from __future__ import annotations

import json
from dataclasses import replace
import hashlib
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
_REGISTRY = re.compile(
    r"(?P<account>[0-9]{12})\.dkr\.ecr\."
    r"(?P<region>[a-z0-9-]+)\.amazonaws\.com\Z"
)


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
        account_id: str,
        region: str,
        read_url: Callable[[str], bytes] = _read_url,
    ) -> None:
        if re.fullmatch(r"[0-9]{12}", account_id) is None or not region:
            raise ValueError("expected ECR account and region are invalid")
        self._client = client
        self._account_id = account_id
        self._region = region
        self._read_url = read_url

    def _batch_get(self, repository: str, image_id: dict[str, str]) -> Mapping[str, Any]:
        response = self._client.batch_get_image(
            registryId=self._account_id,
            repositoryName=repository,
            imageIds=[image_id],
            acceptedMediaTypes=_MANIFEST_TYPES,
        )
        return _one_image(response, context=f"repository {repository!r}")

    def _require_registry(self, repository_reference: str) -> str:
        registry = repository_reference.split("/", 1)[0]
        match = _REGISTRY.fullmatch(registry)
        if (
            match is None
            or match.group("account") != self._account_id
            or match.group("region") != self._region
        ):
            raise ValueError(
                f"ECR registry must match account {self._account_id} "
                f"and region {self._region}: {registry!r}"
            )
        return registry

    def resolve(self, reference: str) -> ResolvedImage:
        if "@" in reference:
            self._require_registry(reference)
            return resolve_image(reference)
        registry, repository, tag = _repository_and_tag(reference)
        self._require_registry(reference)
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
        returned_id = manifest_image.get("imageId")
        returned_digest = (
            returned_id.get("imageDigest") if isinstance(returned_id, Mapping) else None
        )
        if returned_digest != image.digest:
            raise ValueError(
                f"image {image.reference}: ECR response digest does not match requested digest"
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
            registryId=self._account_id,
            repositoryName=repository,
            layerDigest=config_digest,
        )
        download_url = response.get("downloadUrl")
        if not isinstance(download_url, str) or not download_url:
            raise ValueError(f"image {image.reference}: config download URL is missing")
        config_bytes = self._read_url(download_url)
        actual_config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
        if actual_config_digest != config_digest:
            raise ValueError(
                f"image {image.reference}: config blob digest does not match manifest"
            )
        try:
            config = json.loads(config_bytes)
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
