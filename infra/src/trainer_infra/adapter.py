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
def static_names(declared, prefix: str = "") -> frozenset[str]:
    """Every leaf the image says must be known while its graph is built.

    A width that sizes an array, a choice that selects a branch. The image
    decides which, because only it knows what its graph does with a value, and
    the catalog is where it says so.
    """

    found: set[str] = set()
    for name, node in declared.items():
        path = f"{prefix}{name}"
        if not isinstance(node, dict):
            continue
        if "valid" in node:
            if node.get("static"):
                found.add(path)
            continue
        found |= static_names(node, f"{path}.")
    return frozenset(found)


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
            parameter_range(overrides.get(name, node["search"]))
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


def parameter_range(specification: Any) -> dict[str, Any]:
    """A written domain as the resolved tree spells it: a list is a categorical.

    Public because a domain is written in two places now. An experiment writes
    one under ``space`` to narrow a parameter, and one under ``bindings`` to
    give a shared variable its own, and the two have to be the same notation or
    an author would have to remember which of them they were writing.
    """

    if isinstance(specification, list):
        return {"type": "choice", "values": list(specification)}
    return dict(specification)


def readable_domain(written: Any) -> dict[str, Any] | None:
    """A written domain as a range, or nothing where a domain is not written.

    Both trees that carry domains also carry groups, and a group is a mapping
    too. What separates them is the same word everywhere else in this side: a
    leaf written the long way says ``type``. Reading a range as a group is not a
    harmless confusion -- walked as one it has no leaves, so it reports nothing
    rather than reporting itself.
    """

    if not isinstance(written, (list, Mapping)):
        return None
    resolved = parameter_range(written)
    return resolved if "type" in resolved else None


def offers_one_value(domain: Mapping[str, Any]) -> bool:
    """Whether a resolved domain leaves anything for a sampler to choose.

    The notation does not decide it. ``[0.999]``, ``{type: choice, values:
    [0.999]}`` and ``{type: float, low: 0.999, high: 0.999}`` are one value
    each, and a rule about what a study would still search has to read the
    domain rather than the spelling it arrived in.
    """

    if domain["type"] == "choice":
        values = domain.get("values")
        return isinstance(values, list) and len(values) == 1
    low = domain.get("low")
    return low is not None and low == domain.get("high")


def _outside_valid_domains(
    declared: Mapping[str, Any], overrides: Mapping[str, Any], *, prefix: str
) -> Iterator[str]:
    for name, override in overrides.items():
        node = declared[name]
        path = f"{prefix}{name}"
        if "search" in node:
            if not domain_contains(node["valid"], parameter_range(override)):
                yield path
        else:
            yield from _outside_valid_domains(node, override, prefix=f"{path}.")


def domain_contains(valid: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Every value ``candidate`` can produce is one ``valid`` accepts."""

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
