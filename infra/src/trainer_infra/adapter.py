"""The image's declared search space, narrowed by what an experiment pins.

Both are trees of the same shape, so this walks them together and the result is
a third of that shape: a group where the image had a group, a resolved range
where it had a parameter. It stays a tree because the tree is the conditional
structure -- a parameter under a branch exists only for the trials that chose
that branch, and a flat table cannot say so.

Two leaf tests, one per shape. In what the image declares, a leaf carries
``search``. In what comes out of here, a leaf carries ``type``. Nothing else
separates a value from a group: a component chosen among branches is a
parameter called ``kind`` living beside them, which is why a name used at two
sites is two nodes and not a clash.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

# The parameter a group selects its branch with. Written here rather than
# imported, because the worker's copy of this word lives in a package this side
# does not install; the two are kept equal by the round-trip test.
KIND = "kind"


class SpaceError(ValueError):
    """An experiment narrows a search space the image does not declare."""


def resolve_parameter_ranges(
    declared: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Every declared range, with the experiment's narrower one where it gave one.

    A pin the image declares no parameter for is refused rather than ignored:
    it is a knob that turns nothing, and the sampler would never fill it, so
    the run would start with the value its author believed they had set.
    """

    unknown = sorted(_unpinnable(declared, overrides, prefix=""))
    if unknown:
        raise SpaceError(f"the image declares no parameter named {unknown}")
    outside = sorted(_outside_valid_domains(declared, overrides, prefix=""))
    if outside:
        raise SpaceError(f"experiment range is outside the valid domain for {outside}")
    resolved = _resolve(declared, overrides)
    searched = sorted(_searched_structure(resolved, prefix=""))
    if searched:
        raise SpaceError(f"structure parameters must be fixed for one experiment: {searched}")
    return resolved


def _resolve(declared: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: (
            _range(overrides.get(name, node["search"]))
            if "search" in node
            else _resolve(node, overrides.get(name, {}))
        )
        for name, node in declared.items()
    }


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


def _outside_valid_domains(
    declared: Mapping[str, Any], overrides: Mapping[str, Any], *, prefix: str
) -> Iterator[str]:
    for name, override in overrides.items():
        node = declared[name]
        path = f"{prefix}{name}"
        if "search" in node:
            if not _within(node["valid"], _range(override)):
                yield path
        else:
            yield from _outside_valid_domains(node, override, prefix=f"{path}.")


def _within(valid: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    if valid["type"] == "choice":
        return candidate["type"] == "choice" and set(candidate["values"]) <= set(valid["values"])
    if candidate["type"] == "choice":
        return all(_number_within(valid, value) for value in candidate["values"])
    if candidate["type"] not in ("int", "float"):
        return False
    low = candidate.get("low")
    high = candidate.get("high")
    return _number_within(valid, low) and _number_within(valid, high)


def _number_within(valid: Mapping[str, Any], value: Any) -> bool:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if valid["type"] == "int" and not isinstance(value, int):
        return False
    low = valid.get("low")
    high = valid.get("high")
    return (low is None or value >= low) and (high is None or value <= high)


def _searched_structure(tree: Mapping[str, Any], *, prefix: str) -> Iterator[str]:
    kind = tree.get(KIND)
    if isinstance(kind, Mapping) and "type" in kind:
        if kind["type"] != "choice" or len(kind["values"]) != 1:
            yield f"{prefix}{KIND}"
            return
        branch = tree.get(str(kind["values"][0]))
        if isinstance(branch, Mapping):
            yield from _searched_structure(branch, prefix=f"{prefix}{kind['values'][0]}.")
        return
    for name, node in tree.items():
        if isinstance(node, Mapping) and "type" not in node:
            yield from _searched_structure(node, prefix=f"{prefix}{name}.")
