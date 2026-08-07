"""The image's declared search space, narrowed by what an experiment pins.

The catalog is a tree; the sampler wants a flat table keyed by the dotted path,
which is also how the worker spells a parameter. So the experiment file pins by
that same dotted path and nothing has to agree on a nesting convention twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SpaceError(ValueError):
    """An experiment narrows a search space the image does not declare."""


def resolve_parameter_ranges(
    declared: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Every declared range, with the experiment's narrower one where it gave one.

    A pin the image declares no parameter for is refused rather than ignored:
    it is a knob that turns nothing, and the sampler would never fill it, so
    the run would start with the value its author believed they had set.
    """

    resolved: dict[str, dict[str, Any]] = {}
    _collect(declared, overrides, prefix="", resolved=resolved)
    unknown = sorted(set(overrides) - set(resolved))
    if unknown:
        raise SpaceError(f"the image declares no parameter named {unknown}")
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
            resolved[path] = _range(overrides.get(path, node["search"]))
        else:
            _collect(node, overrides, prefix=path, resolved=resolved)


def _range(specification: Any) -> dict[str, Any]:
    if isinstance(specification, list):
        return {"type": "choice", "values": list(specification)}
    return dict(specification)
