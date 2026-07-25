from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from training_sdk.contract import Catalog

from trainer_infra.preflight import PreflightError

CATALOG_LABEL = "org.rtrrl.trainer.catalog.v2"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REGISTRY = re.compile(
    r"(?P<account>[0-9]{12})\.dkr\.ecr\."
    r"(?P<region>[a-z0-9-]+)\.amazonaws\.com\Z"
)
_MANIFEST_TYPES = [
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
]


@dataclass(frozen=True)
class ResolvedImage:
    reference: str
    repository: str
    digest: str
    catalog: Catalog


def encode_catalog(catalog: Catalog) -> str:
    raw = catalog.model_dump_json(exclude_none=True).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")


def decode_catalog(value: str) -> Catalog:
    try:
        compressed = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise PreflightError("catalog label is not valid base64") from error
    try:
        raw = gzip.decompress(compressed)
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise PreflightError("catalog label is not valid gzip data") from error
    try:
        return Catalog.model_validate_json(raw)
    except (ValidationError, ValueError) as error:
        raise PreflightError(f"catalog label does not contain a valid catalog: {error}") from error


def _parse_digest_reference(reference: str) -> tuple[str, str, str]:
    if "@" not in reference:
        raise PreflightError(f"image {reference!r} is not an immutable digest reference")
    repository, digest = reference.rsplit("@", 1)
    if not repository or "://" in repository or _DIGEST.fullmatch(digest) is None:
        raise PreflightError(f"image {reference!r} is not an immutable sha256 digest reference")
    last_slash = repository.rfind("/")
    last_colon = repository.rfind(":")
    if last_colon > last_slash:
        repository = repository[:last_colon]
    if not repository:
        raise PreflightError(f"image {reference!r} has no repository")
    return f"{repository}@{digest}", repository, digest


def _registry_account(reference: str) -> str:
    registry = reference.split("/", 1)[0]
    match = _REGISTRY.fullmatch(registry)
    if match is None:
        raise PreflightError(f"image {reference!r} is not an ECR image reference")
    return match.group("account")


def _repository_and_tag(reference: str) -> tuple[str, str, str]:
    if "://" in reference or "/" not in reference:
        raise PreflightError(f"image {reference!r} is not an ECR image reference")
    registry, repository_reference = reference.split("/", 1)
    last_component = repository_reference.rsplit("/", 1)[-1]
    if ":" not in last_component:
        raise PreflightError(f"image {reference!r} must contain a tag or digest")
    repository, tag = repository_reference.rsplit(":", 1)
    if not registry or not repository or not tag:
        raise PreflightError(f"image {reference!r} is not an ECR image reference")
    _registry_account(reference)
    return registry, repository, tag


def _one_image(response: dict[str, Any], *, context: str) -> dict[str, Any]:
    failures = response.get("failures", [])
    if failures:
        raise PreflightError(f"{context}: ECR returned failure: {failures!r}")
    images = response.get("images", [])
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise PreflightError(f"{context}: expected exactly one ECR image")
    return images[0]


def _fetch_catalog(
    ecr_client: Any,
    *,
    registry_id: str,
    repository: str,
    digest: str,
    reference: str,
    read_url: Callable[[str], bytes],
) -> Catalog:
    manifest_image = _one_image(
        ecr_client.batch_get_image(
            registryId=registry_id,
            repositoryName=repository,
            imageIds=[{"imageDigest": digest}],
            acceptedMediaTypes=_MANIFEST_TYPES,
        ),
        context=f"image {reference!r}",
    )
    returned_id = manifest_image.get("imageId")
    returned_digest = (
        returned_id.get("imageDigest") if isinstance(returned_id, dict) else None
    )
    if returned_digest != digest:
        raise PreflightError(
            f"image {reference!r}: ECR response digest does not match requested digest"
        )
    manifest_raw = manifest_image.get("imageManifest")
    if not isinstance(manifest_raw, (str, bytes)):
        raise PreflightError(f"image {reference!r}: ECR response is missing manifest")
    try:
        manifest = json.loads(manifest_raw)
        config_digest = manifest["config"]["digest"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as error:
        raise PreflightError(
            f"image {reference!r}: manifest is missing config digest"
        ) from error
    if not isinstance(config_digest, str) or _DIGEST.fullmatch(config_digest) is None:
        raise PreflightError(f"image {reference!r}: manifest has invalid config digest")

    response = ecr_client.get_download_url_for_layer(
        registryId=registry_id,
        repositoryName=repository,
        layerDigest=config_digest,
    )
    download_url = response.get("downloadUrl")
    if not isinstance(download_url, str) or not download_url:
        raise PreflightError(f"image {reference!r}: config download URL is missing")
    config_bytes = read_url(download_url)
    actual_config_digest = "sha256:" + hashlib.sha256(config_bytes).hexdigest()
    if actual_config_digest != config_digest:
        raise PreflightError(f"image {reference!r}: config blob digest does not match manifest")
    try:
        config = json.loads(config_bytes)
        label = config["config"]["Labels"][CATALOG_LABEL]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as error:
        raise PreflightError(
            f"image {reference!r}: config is missing label {CATALOG_LABEL!r}"
        ) from error
    if not isinstance(label, str) or not label:
        raise PreflightError(f"image {reference!r}: config label {CATALOG_LABEL!r} is invalid")
    return decode_catalog(label)


def resolve_image(
    reference: str,
    ecr_client: Any,
    read_url: Callable[[str], bytes],
) -> ResolvedImage:
    if "@" in reference:
        canonical_reference, full_repository, digest = _parse_digest_reference(reference)
        registry_id = _registry_account(reference)
        repository = full_repository.split("/", 1)[1]
    else:
        registry, repository, tag = _repository_and_tag(reference)
        registry_id = _registry_account(reference)
        response = ecr_client.describe_images(
            registryId=registry_id,
            repositoryName=repository,
            imageIds=[{"imageTag": tag}],
        )
        details = response.get("imageDetails", [])
        if not isinstance(details, list) or not details:
            raise PreflightError(f"image {reference!r}: no image found in ECR")
        digest_value = details[0].get("imageDigest")
        if not isinstance(digest_value, str) or _DIGEST.fullmatch(digest_value) is None:
            raise PreflightError(f"image {reference!r}: ECR response is missing image digest")
        digest = digest_value
        canonical_reference = f"{registry}/{repository}@{digest}"
        full_repository = f"{registry}/{repository}"

    catalog = _fetch_catalog(
        ecr_client,
        registry_id=registry_id,
        repository=repository,
        digest=digest,
        reference=canonical_reference,
        read_url=read_url,
    )
    return ResolvedImage(
        reference=canonical_reference,
        repository=full_repository,
        digest=digest,
        catalog=catalog,
    )
