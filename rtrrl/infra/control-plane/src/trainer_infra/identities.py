from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import yaml


def study_name(experiment_id: str, group: str) -> str:
    return f"{experiment_id}:{group}"


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _plain_json(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_yaml(value: Any) -> str:
    return yaml.safe_dump(
        _plain_json(value),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
