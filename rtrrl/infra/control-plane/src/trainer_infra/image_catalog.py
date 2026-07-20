from __future__ import annotations

import base64
import binascii
import gzip
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml
from pydantic import ValidationError

from trainer_infra.models import ScriptCatalog, ScriptDescriptor

LABEL = "org.rtrrl.trainer.scripts.v1"
CATALOG_PROTOCOL_VERSION = "1"
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ResolvedImage:
    reference: str
    repository: str
    digest: str
    catalog: ScriptCatalog | None = None


class EcrClient(Protocol):
    """The complete ECR behavior needed by catalog discovery."""

    def resolve_tag(self, reference: str) -> str: ...

    def get_manifest(self, reference: str) -> Mapping[str, Any] | str | bytes: ...

    def get_config_blob(
        self, repository: str, digest: str
    ) -> Mapping[str, Any] | str | bytes: ...


def encode_catalog(catalog: ScriptCatalog) -> str:
    """Encode a catalog as a reproducible Docker-label value."""
    raw = catalog.model_dump_json(exclude_none=True).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")


def decode_catalog(value: str) -> ScriptCatalog:
    """Decode and validate a catalog Docker-label value."""
    try:
        compressed = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("catalog label is not valid base64") from error
    try:
        raw = gzip.decompress(compressed)
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise ValueError("catalog label is not valid gzip data") from error
    try:
        return ScriptCatalog.model_validate_json(raw)
    except (ValidationError, ValueError) as error:
        raise ValueError(f"catalog label does not contain a valid catalog: {error}") from error


def resolve_image(reference: str) -> ResolvedImage:
    """Parse an immutable image reference."""
    if "@" not in reference:
        raise ValueError(f"image {reference!r} is not an immutable digest reference")
    repository, digest = reference.rsplit("@", 1)
    if not repository or not _DIGEST_PATTERN.fullmatch(digest):
        raise ValueError(f"image {reference!r} is not an immutable sha256 digest reference")
    return ResolvedImage(reference=reference, repository=repository, digest=digest)


def _json_object(value: Mapping[str, Any] | str | bytes, *, context: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise ValueError(f"{context}: expected a JSON object") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{context}: expected a JSON object")
    return parsed


def _validate_catalog_identity(catalog: ScriptCatalog, *, context: str) -> None:
    if catalog.protocol_version != CATALOG_PROTOCOL_VERSION:
        raise ValueError(
            f"{context}: unsupported protocol_version {catalog.protocol_version!r}; "
            f"expected {CATALOG_PROTOCOL_VERSION!r}"
        )

    names = [descriptor.name for descriptor in catalog.scripts.values()]
    duplicate_names = {name for name in names if names.count(name) > 1}
    if duplicate_names:
        duplicate = sorted(duplicate_names)[0]
        raise ValueError(f"{context}: duplicate script name {duplicate!r}")

    for key, descriptor in catalog.scripts.items():
        if key != descriptor.name:
            raise ValueError(
                f"{context}: catalog key {key!r} does not match "
                f"descriptor name {descriptor.name!r}"
            )


class EcrCatalogReader:
    def __init__(self, client: EcrClient) -> None:
        self._client = client

    def resolve(self, reference: str) -> ResolvedImage:
        if "@" in reference:
            return resolve_image(reference)

        try:
            resolved = self._client.resolve_tag(reference)
        except Exception as error:
            raise ValueError(f"image {reference!r}: failed to resolve tag") from error
        immutable_reference = resolved if "@" in resolved else f"{reference.rsplit(':', 1)[0]}@{resolved}"
        try:
            return resolve_image(immutable_reference)
        except ValueError as error:
            raise ValueError(
                f"image {reference!r}: ECR returned invalid digest {resolved!r}"
            ) from error

    def fetch(self, image: ResolvedImage) -> ScriptCatalog:
        if not isinstance(image, ResolvedImage):
            image = resolve_image(str(image))
        context = f"image {image.reference}"
        try:
            manifest_value = self._client.get_manifest(image.reference)
        except Exception as error:
            raise ValueError(f"{context}: failed to read manifest") from error
        manifest = _json_object(manifest_value, context=f"{context} manifest")
        try:
            config_digest = manifest["config"]["digest"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{context}: manifest is missing config digest") from error
        if not isinstance(config_digest, str) or not _DIGEST_PATTERN.fullmatch(config_digest):
            raise ValueError(f"{context}: manifest has invalid config digest {config_digest!r}")

        try:
            config_value = self._client.get_config_blob(image.repository, config_digest)
        except Exception as error:
            raise ValueError(
                f"{context}: failed to read config blob {config_digest}"
            ) from error
        config = _json_object(config_value, context=f"{context} config blob")
        try:
            label = config["config"]["Labels"][LABEL]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{context}: config is missing label {LABEL!r}") from error
        if not isinstance(label, str) or not label:
            raise ValueError(f"{context}: config label {LABEL!r} must be a non-empty string")

        try:
            catalog = decode_catalog(label)
        except ValueError as error:
            raise ValueError(f"{context}: invalid catalog label {LABEL!r}: {error}") from error
        _validate_catalog_identity(catalog, context=context)
        return catalog

    def resolve_and_fetch(self, reference: str) -> ResolvedImage:
        image = self.resolve(reference)
        return replace(image, catalog=self.fetch(image))


def _load_yaml(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"{path}: failed to load YAML: {error}") from error


def load_catalog_index(path: Path) -> ScriptCatalog:
    """Load descriptor files named by an index and validate one complete catalog."""
    index = _load_yaml(path)
    if not isinstance(index, dict):
        raise ValueError(f"{path}: catalog index must be a mapping")
    unexpected = set(index) - {"protocol_version", "scripts"}
    if unexpected:
        raise ValueError(f"{path}: unexpected catalog index fields: {sorted(unexpected)!r}")
    protocol_version = index.get("protocol_version")
    entries = index.get("scripts")
    if not isinstance(protocol_version, str):
        raise ValueError(f"{path}: protocol_version must be a string")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: scripts must be a non-empty array")

    seen_entries: set[str] = set()
    descriptors: dict[str, ScriptDescriptor] = {}
    for entry in entries:
        if not isinstance(entry, str) or not entry:
            raise ValueError(f"{path}: every catalog entry must be a non-empty string")
        if entry in seen_entries:
            raise ValueError(f"{path}: duplicate catalog entry {entry!r}")
        seen_entries.add(entry)
        descriptor_path = path.parent / entry
        raw_descriptor = _load_yaml(descriptor_path)
        try:
            descriptor = ScriptDescriptor.model_validate(raw_descriptor)
        except ValidationError as error:
            raise ValidationError.from_exception_data(
                f"invalid script descriptor {descriptor_path}", error.errors()
            ) from error
        if descriptor.name in descriptors:
            raise ValueError(
                f"{descriptor_path}: duplicate script name {descriptor.name!r}"
            )
        descriptors[descriptor.name] = descriptor

    catalog = ScriptCatalog(protocol_version=protocol_version, scripts=descriptors)
    _validate_catalog_identity(catalog, context=str(path))
    return catalog


def encode_catalog_file(path: Path) -> str:
    """Load, validate, and encode an index for a Docker build argument."""
    return encode_catalog(load_catalog_index(path))
