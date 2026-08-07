"""The image's declared search space, narrowed by what an experiment pins.

Both are trees of the same shape, so this walks them together. The sampler
wants a flat table, and the dotted path down the tree is what it is keyed by --
which is also how a run configuration spells a parameter, so the flattening
happens once, here, and is a rendering rather than a namespace.

A leaf is a node carrying ``search``. Nothing else distinguishes a value from a
group: a component chosen among branches is a parameter called ``kind`` living
beside them, which is why a name used at two sites is two nodes and not a
clash.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
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
    unknown = sorted(_unpinnable(declared, overrides, prefix=""))
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
        path = f"{prefix}{name}"
        if "search" in node:
            resolved[path] = _range(overrides.get(name, node["search"]))
        else:
            _collect(node, overrides.get(name, {}), prefix=f"{path}.", resolved=resolved)


def _unpinnable(
    declared: Mapping[str, Any], overrides: Mapping[str, Any], *, prefix: str
) -> Iterator[str]:
    """Every pin with no parameter under it, named by where it was written."""

    for name, pinned in overrides.items():
        path = f"{prefix}{name}"
        node = declared.get(name)
        if node is None:
            yield path
        elif "search" not in node:
            yield from _unpinnable(node, pinned, prefix=f"{path}.")


def _range(specification: Any) -> dict[str, Any]:
    if isinstance(specification, list):
        return {"type": "choice", "values": list(specification)}
    return dict(specification)
