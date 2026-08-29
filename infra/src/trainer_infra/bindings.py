"""One drawn value written into several of a run's parameters.

Experiment 2 compares update rules while holding a setting equal across the
actor, the critic and the shared torso. The tree cannot say that on its own:
three leaves with the same domain are three leaves, and a sampler draws them
three times. What that costs is not only the extra dimensions -- it is that the
comparison being run is no longer the one the experiment describes.

So a binding is a variable, declared once beside the space and named at the
paths it reaches::

    bindings:
      shared_beta2:
        domain: [0.9, 0.99, 0.995, 0.999, 0.9999]
        paths:
          - actor.optimizer.adam.b2
          - critic.optimizer.adam.b2
          - torso.optimizer.adam.b2

It is a statement about *configuration* and nothing else. The optimizer draws
``shared_beta2`` once and the run document carries an ordinary number at each of
the three paths, so the three blocks build three optimizers with three states
exactly as they did before. Sharing a value is not sharing a moment, a trace or
a bound statistic, and nothing here can make it one.

Two namespaces, kept apart by their shape. A destination is a dotted path into
the image's declared tree; a variable's name has no dots and may not be a name
that tree already uses. A variable therefore cannot name another variable, which
is what makes a cycle unspellable rather than something to detect.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trainer_infra.adapter import KIND, SpaceError, domain_contains, parameter_range

FIELDS = ("domain", "paths")


class BindingError(SpaceError):
    """A shared variable this side cannot write into the space it names."""


@dataclass(frozen=True)
class Binding:
    """One sampled variable, and the parameters it is written into.

    ``domain`` is a resolved range of the same shape the space's leaves carry,
    so the sampler suggests it exactly as it suggests any other parameter --
    under ``name``, once, which is what makes it one Optuna dimension.
    """

    name: str
    domain: Mapping[str, Any]
    paths: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        """What the study archives, so a resumed run can be read months later."""

        return {"name": self.name, "domain": dict(self.domain), "paths": list(self.paths)}


def resolve_bindings(
    *,
    declared: Mapping[str, Any],
    resolved: Mapping[str, Any],
    space: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> tuple[Binding, ...]:
    """Every declared variable, checked against the tree it writes into.

    All of it before a job is submitted. A binding that named a path the image
    does not declare, or one under a branch this experiment does not select,
    would otherwise be discovered by a worker -- a round of containers that each
    start, read a configuration missing the value they were promised, and die on
    the same line.
    """

    if not bindings:
        return ()
    if not isinstance(bindings, Mapping):
        raise BindingError("bindings must name each shared variable once, as a mapping")
    declarations = tuple(_declared(name, bindings[name]) for name in sorted(bindings))
    _names_stay_out_of_the_tree(declared, declarations)
    _destinations_are_declared(declared, declarations)
    _destinations_are_not_structural(declarations)
    _destinations_accept_the_domain(declared, declarations)
    _no_destination_is_written_twice(declarations)
    _no_destination_is_also_pinned(space, declarations)
    _destinations_are_selected(resolved, declarations)
    return declarations


def live_paths(resolved: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """The leaves this experiment's structure actually reaches, by path.

    The same descent the sampler and the worker make: a group is entered when
    nothing beside it selected a sibling instead. A structural choice is pinned
    to one value for a whole study, so which leaves are live is decided here,
    once, rather than differing between trials.
    """

    leaves: dict[str, Any] = {}
    groups: dict[str, Mapping[str, Any]] = {}
    for name, node in resolved.items():
        path = f"{prefix}{name}"
        if "type" in node:
            leaves[path] = node
        else:
            groups[name] = node

    selected = _selected_branch(leaves.get(f"{prefix}{KIND}"))
    for name, group in groups.items():
        if selected is not None and name != selected:
            continue
        leaves |= live_paths(group, f"{prefix}{name}.")
    return leaves


def expand(parameters: Mapping[str, Any], bindings: Sequence[Binding]) -> dict[str, Any]:
    """A study's recorded point as a run document reads it.

    Optuna stores the variable, because the variable is the dimension it
    searched. A worker knows nothing about variables, so what is handed to it is
    the value at each destination and no alias at all. Both views are needed:
    ``settle`` re-reads trials the study already drew, and the runs it scores
    have to be the runs that were submitted.
    """

    written = dict(parameters)
    for binding in bindings:
        if binding.name not in written:
            continue
        value = written.pop(binding.name)
        for path in binding.paths:
            written[path] = value
    return written


def searched(bindings: Mapping[str, Any] | None) -> Iterator[str]:
    """Every shared variable that still offers more than one value.

    Read off the file rather than off a resolved binding, because a formal
    launch is refused for what it says: the configuration it reports on is
    frozen, and a variable with a domain left in it is a choice still being
    made.
    """

    for name, declaration in (bindings or {}).items():
        if not isinstance(declaration, Mapping):
            continue  # resolution reports the shape; this only reads a domain
        domain = declaration.get("domain")
        if domain is None:
            continue
        listed = isinstance(domain, Sequence) and not isinstance(domain, (str, bytes))
        if listed and len(domain) == 1:
            continue  # a one-option categorical is a value, not a choice
        yield str(name)


def _selected_branch(kind: Any) -> str | None:
    """Which branch a group's ``kind`` opens, when it opens exactly one."""

    if not isinstance(kind, Mapping) or kind.get("type") != "choice":
        return None
    values = kind["values"]
    return str(values[0]) if len(values) == 1 else None


def _declared(name: Any, declaration: Any) -> Binding:
    if not isinstance(name, str) or not name:
        raise BindingError(f"a shared variable's name must be a non-empty string, not {name!r}")
    if not isinstance(declaration, Mapping):
        raise BindingError(f"binding {name!r} must say its domain and the paths it reaches")
    missing = sorted(field for field in FIELDS if field not in declaration)
    if missing:
        raise BindingError(f"binding {name!r} does not say {missing}")
    unknown = sorted(set(declaration) - set(FIELDS))
    if unknown:
        raise BindingError(f"binding {name!r} says {unknown}, which a binding has no field for")
    paths = declaration["paths"]
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise BindingError(f"binding {name!r} must list the paths it is written into")
    if len(paths) < 2:
        raise BindingError(
            f"binding {name!r} reaches {len(paths)} path(s); a variable written into one "
            "parameter is that parameter's own range, and belongs under space"
        )
    return Binding(
        name=name,
        domain=parameter_range(declaration["domain"]),
        paths=tuple(str(path) for path in paths),
    )


def _names_stay_out_of_the_tree(
    declared: Mapping[str, Any], bindings: Sequence[Binding]
) -> None:
    """A variable is drawn under its own name, so that name must be free.

    ``gamma`` as a variable would be suggested under the name the tree's own
    ``gamma`` is suggested under, and the study would hold one dimension where
    the file says two things. Dots are refused from the other side and for the
    same reason: a dot is what makes a name a destination.
    """

    dotted = sorted(binding.name for binding in bindings if "." in binding.name)
    if dotted:
        raise BindingError(
            f"shared variables {dotted} contain a dot, which is what names a destination"
        )
    taken = sorted(binding.name for binding in bindings if binding.name in declared)
    if taken:
        raise BindingError(
            f"shared variables {taken} are already names in the image's parameter tree"
        )


def _destinations_are_declared(
    declared: Mapping[str, Any], bindings: Sequence[Binding]
) -> None:
    unknown = sorted(
        _named(binding, path)
        for binding in bindings
        for path in binding.paths
        if _leaf(declared, path) is None
    )
    if unknown:
        raise BindingError(f"the image declares no parameter at {unknown}")


def _destinations_are_not_structural(bindings: Sequence[Binding]) -> None:
    """A ``kind`` decides which parameters exist, so it is pinned, not shared.

    One structure for a whole study is already the rule for the space, and a
    variable is drawn after the descent a ``kind`` directs. Binding one would be
    asking the walk to follow a branch it has not chosen yet.
    """

    structural = sorted(
        _named(binding, path)
        for binding in bindings
        for path in binding.paths
        if path.rsplit(".", 1)[-1] == KIND
    )
    if structural:
        raise BindingError(
            f"{structural} select which parameters exist; pin a structural choice "
            "under space rather than sharing it"
        )


def _destinations_accept_the_domain(
    declared: Mapping[str, Any], bindings: Sequence[Binding]
) -> None:
    outside = sorted(
        _named(binding, path)
        for binding in bindings
        for path in binding.paths
        if not domain_contains(_leaf(declared, path)["valid"], binding.domain)
    )
    if outside:
        raise BindingError(f"the shared domain is outside the valid domain at {outside}")


def _no_destination_is_written_twice(bindings: Sequence[Binding]) -> None:
    seen: dict[str, list[str]] = {}
    for binding in bindings:
        for path in binding.paths:
            seen.setdefault(path, []).append(binding.name)
    contested = sorted(path for path, names in seen.items() if len(names) != 1)
    if contested:
        raise BindingError(f"more than one value is written into {contested}")


def _no_destination_is_also_pinned(
    space: Mapping[str, Any], bindings: Sequence[Binding]
) -> None:
    """A path cannot both take a shared value and narrow one of its own.

    Two authors of one leaf, and no order between them more obviously right than
    the other. The pin is the one to delete, since a binding already carries a
    domain.
    """

    pinned = sorted(
        _named(binding, path)
        for binding in bindings
        for path in binding.paths
        if _pinned(space, path)
    )
    if pinned:
        raise BindingError(f"{pinned} are bound to a shared variable and also pinned under space")


def _destinations_are_selected(
    resolved: Mapping[str, Any], bindings: Sequence[Binding]
) -> None:
    """Every destination is a leaf this experiment's structure reaches.

    A binding into a branch nobody selected writes nothing, which would leave a
    file claiming a value was shared by paths one of which does not exist. It is
    also what a binding across incompatible branches looks like from here: at
    most one of the two is live.
    """

    live = live_paths(resolved)
    unreachable = sorted(
        _named(binding, path)
        for binding in bindings
        for path in binding.paths
        if path not in live
    )
    if unreachable:
        raise BindingError(f"{unreachable} are under branches this experiment does not select")


def _named(binding: Binding, path: str) -> str:
    return f"{binding.name} -> {path}"


def _leaf(declared: Mapping[str, Any], path: str) -> Mapping[str, Any] | None:
    """The declaration at a dotted path, or nothing where a parameter is not."""

    node: Any = declared
    for step in path.split("."):
        if not isinstance(node, Mapping) or step not in node:
            return None
        node = node[step]
    if isinstance(node, Mapping) and "search" in node:
        return node
    return None


def _pinned(space: Mapping[str, Any], path: str) -> bool:
    node: Any = space
    for step in path.split("."):
        if not isinstance(node, Mapping) or step not in node:
            return False
        node = node[step]
    return not isinstance(node, Mapping) or "type" in node
