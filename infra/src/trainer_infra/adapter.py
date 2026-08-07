from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_parameter_ranges(
    declared: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    _collect(declared, overrides, prefix="", resolved=resolved)
    return resolved


def _collect(
    declared: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    prefix: str,
    resolved: dict[str, dict[str, Any]],
) -> None:
    for name, node in declared.items():
        path = f"{prefix}.{name}" if prefix else name
        if "search" in node:
            resolved[path] = _range(overrides.get(name, node["search"]))
        else:
            _collect(
                node,
                overrides.get(name, {}),
                prefix=path,
                resolved=resolved,
            )


def _range(specification: Any) -> dict[str, Any]:
    if isinstance(specification, list):
        return {"type": "choice", "values": list(specification)}
    return dict(specification)
