from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlsplit

_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]\Z")
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str

    def __str__(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def parse_s3_uri(uri: str, *, allow_trailing_slash: bool = False) -> S3Uri:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or _BUCKET.fullmatch(parsed.hostname or "") is None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
        or "%" in parsed.path
        or "\\" in parsed.path
    ):
        raise ValueError(f"invalid S3 URI: {uri!r}")
    key = parsed.path[1:]
    parts = key.split("/")
    if not key or "." in parts or ".." in parts or "" in parts[:-1]:
        raise ValueError(f"invalid S3 URI key: {uri!r}")
    if parts[-1] == "" and not allow_trailing_slash:
        raise ValueError(f"S3 URI must name an object: {uri!r}")
    if parts[-1] != "" and allow_trailing_slash:
        raise ValueError(f"S3 URI must end with '/': {uri!r}")
    return S3Uri(bucket=parsed.hostname or "", key=key)


@dataclass(frozen=True)
class ExperimentS3Namespace:
    bucket: str
    experiment_id: str

    @property
    def key_prefix(self) -> str:
        return f"experiments/{self.experiment_id}/"

    @property
    def prefix_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key_prefix}"

    @classmethod
    def from_prefix(cls, uri: str) -> ExperimentS3Namespace:
        parsed = parse_s3_uri(uri, allow_trailing_slash=True)
        parts = parsed.key.split("/")
        if (
            len(parts) != 3
            or parts[0] != "experiments"
            or _SEGMENT.fullmatch(parts[1]) is None
            or parts[1] in {".", ".."}
        ):
            raise ValueError("experiment prefix must be experiments/<experiment-id>/")
        return cls(parsed.bucket, parts[1])

    @classmethod
    def from_bundle_uri(cls, uri: str) -> tuple[ExperimentS3Namespace, str]:
        parsed = parse_s3_uri(uri)
        parts = parsed.key.split("/")
        if (
            len(parts) != 5
            or parts[0] != "experiments"
            or parts[2] != "jobs"
            or parts[4] != "bundle.json"
            or _SEGMENT.fullmatch(parts[1]) is None
            or _SEGMENT.fullmatch(parts[3]) is None
        ):
            raise ValueError("bundle URI must be experiments/<id>/jobs/<job-id>/bundle.json")
        return cls(parsed.bucket, parts[1]), parts[3]

    def uri(self, relative_key: str) -> str:
        if relative_key.startswith("/") or relative_key.endswith("/"):
            raise ValueError("relative S3 key must name an object")
        return str(self.require_uri(f"{self.prefix_uri}{relative_key}"))

    def require_uri(self, uri: str) -> S3Uri:
        parsed = parse_s3_uri(uri)
        if parsed.bucket != self.bucket or not parsed.key.startswith(self.key_prefix):
            raise ValueError(f"S3 URI is outside configured experiment prefix: {uri!r}")
        return parsed

    def require_key(self, key: str) -> S3Uri:
        if not key.startswith(self.key_prefix):
            raise ValueError("S3 key is outside configured experiment prefix")
        parsed = parse_s3_uri(
            f"s3://{self.bucket}/{key}",
            allow_trailing_slash=key.endswith("/"),
        )
        if parsed.bucket != self.bucket:
            raise ValueError("S3 key uses the wrong bucket")
        return parsed
