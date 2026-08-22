"""Run one trial's seeds as a single graph, and report each as its own run.

The worker hands a group entry several run documents at once instead of one, and
this is what an algorithm's ensemble entry does with them. Every member still
produces its own metrics, its own artifacts and its own result: what changes is
that they were computed together, not that they became one run.

What members may differ in is the seed, and the parameters a sweep is allowed
to vary. Everything else -- the environment, the budget, the schedule, the entry
-- is shared, because it went into the one graph and the one loop they share. A
member disagreeing about any of it would be run under its neighbour's terms and
reported under its own: a wrong number rather than a failure, on a run nobody
would think to doubt.

A parameter may be swept when the graph reads it arithmetically. One that sizes
an array or picks a branch may not, since the members share both their shapes
and their graph, and those are declared ``static`` where the algorithm declares
them. A choice is static without being declared: its values are strings and
booleans, which no trace carries.

This needs nothing new in the manifest. Each member already arrives as its own
run document with its own parameters, which is how two trials have always
differed. Grouping them is the whole of what a sweep adds.
"""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from memorax.assembly import assemble
from memorax.parameters import flatten
from memorax.runtime import RuntimeConfig
from memorax.runtime.ensemble import EnsembleRuntime

from ._contract import RunSpec
from ._observability import build_reporter

GROUP_VARIABLE = "TRAINER_RUN_GROUP"

# Read by the catalog, and through it by the control plane, which is what
# decides whether a round's runs go into a manifest as `runs` or as `groups`.
# An entry importing this is an entry that takes a group.
GROUPED = True

# What a member carries of its own, outside the parameters. Everything else at
# this level has to match, because everything else went into the one graph and
# the one schedule they share. The artifact root is here because it *must*
# differ: members publish separately, and two sharing a root would each
# overwrite the other's metrics and result.
PERSONAL = (
    ("identity", "run_id"),
    ("identity", "seed"),
    ("training", "seed"),
    ("artifacts", "root"),
)

# The parameters are compared on their own terms, so they are lifted out of the
# document before the rest is compared for equality.
PARAMETERS = ("algorithm", "parameters")


class GroupError(RuntimeError):
    """The group the worker passed cannot be run as one graph."""


def load_group() -> tuple[tuple[RunSpec, Path], ...]:
    """Every member of the group, with the scratch directory it reports into."""

    index = Path(os.environ[GROUP_VARIABLE])
    members = json.loads(index.read_text(encoding="utf-8"))["members"]
    if not members:
        raise GroupError("the group names no members")
    return tuple(
        (
            RunSpec.model_validate(
                json.loads(Path(member["config"]).read_text(encoding="utf-8"))
            ),
            Path(member["scratch"]),
        )
        for member in members
    )


def _without_personal(document: dict[str, Any]) -> dict[str, Any]:
    shape = json.loads(json.dumps(document, sort_keys=True))
    for *parents, leaf in (*PERSONAL, PARAMETERS):
        scope = shape
        for name in parents:
            scope = scope[name]
        scope.pop(leaf, None)
    return shape


def swept_parameters(
    members: tuple[tuple[RunSpec, Path], ...], declared: Any
) -> dict[str, list[Any]]:
    """Which parameters the members disagree about, and their values in order.

    A disagreement about a leaf the algorithm declares ``static`` is refused
    here rather than discovered inside a trace. That covers the widths and the
    buffers, and it covers every ``kind``: a round whose members pick different
    backbones is two graphs, and belongs in two groups.
    """

    leaves = flatten(declared)
    documents = [spec.algorithm.parameters for spec, _ in members]

    names = sorted({name for document in documents for name in document})
    swept: dict[str, list[Any]] = {}
    unknown = []
    for name in names:
        values = [document.get(name) for document in documents]
        if all(value == values[0] for value in values):
            continue
        parameter = leaves.get(name)
        if parameter is None:
            unknown.append(name)
            continue
        if parameter.static:
            raise GroupError(
                f"the group varies {name}, which is static: it sizes an array "
                "or picks a branch, and the members of one round share both. "
                "Put those members in groups of their own."
            )
        swept[name] = values

    if unknown:
        raise GroupError(
            f"the group varies {', '.join(unknown)}, which the algorithm does "
            "not declare, so nothing here can say whether it may be swept"
        )
    missing = [name for name in names if any(name not in d for d in documents)]
    if missing:
        raise GroupError(
            f"only some members carry {', '.join(sorted(set(missing)))}; "
            "a group's members must name the same parameters"
        )
    return swept


def one_configuration(members: tuple[tuple[RunSpec, Path], ...]) -> RunSpec:
    """The configuration they share, or an error naming what they do not.

    The seeds are checked for repetition here as well as in the runtime. Two
    members on one seed is one run billed twice, and downstream nothing could
    tell the duplicate from a real second sample -- the artifacts would differ
    only in a run id.
    """

    first, _ = members[0]
    reference = _without_personal(first.model_dump(mode="json"))
    for spec, _ in members[1:]:
        shape = _without_personal(spec.model_dump(mode="json"))
        if shape == reference:
            continue
        differing = sorted(
            key
            for key in set(shape) | set(reference)
            if shape.get(key) != reference.get(key)
        )
        raise GroupError(
            f"run {spec.identity.run_id} differs from {first.identity.run_id} "
            f"in {', '.join(differing)}; a group may differ only in its seed"
        )

    for spec, _ in members:
        if spec.identity.seed != spec.training.seed:
            raise GroupError(
                f"run {spec.identity.run_id} is labelled seed "
                f"{spec.identity.seed} but trains on {spec.training.seed}"
            )
    seeds = [spec.training.seed for spec, _ in members]
    if len(set(seeds)) != len(seeds):
        raise GroupError(f"the group repeats a seed: {seeds}")

    # Two members publishing to one root is one member's results, silently.
    # Whichever wrote last would be the run that appears to have happened.
    roots = [spec.artifacts.root.rstrip("/") for spec, _ in members]
    if len(set(roots)) != len(roots):
        raise GroupError(f"the group repeats an artifact root: {roots}")
    return first


def run_group(
    definition: Any,
    members: tuple[tuple[RunSpec, Path], ...],
    *,
    build_request: Callable[[RunSpec], Any],
    runtime_config: Callable[[RunSpec], RuntimeConfig],
    declared: Any,
) -> None:
    """Run every member of the group through one graph, on one device pass."""

    shared = one_configuration(members)
    swept = swept_parameters(members, declared)
    request = build_request(shared)

    # Built here even when a sweep will rebuild it inside the map: the runtime
    # reads the observation schema off a built algorithm, and a schema does not
    # depend on the values a sweep varies. It is also the first place a
    # configuration that cannot build at all says so, before any member axis
    # makes the error harder to read.
    algorithm = assemble(definition, request)

    with ExitStack() as stack:
        reporters = [
            stack.enter_context(build_reporter(spec, scratch))
            for spec, scratch in members
        ]
        EnsembleRuntime(
            algorithm=algorithm,
            config=runtime_config(shared),
            seeds=tuple(spec.training.seed for spec, _ in members),
            build=(
                (lambda parameters: assemble(definition, replace(request, parameters=parameters)))
                if swept
                else None
            ),
            parameters=dict(shared.algorithm.parameters),
            swept=swept,
        ).run(reporters)


def main_for(
    definition: Any,
    *,
    build_request: Callable[[RunSpec], Any],
    runtime_config: Callable[[RunSpec], RuntimeConfig],
    declared: Any,
) -> Callable[[list[str] | None], int]:
    """The ``main`` an algorithm's ensemble entry exposes to the catalog."""

    def main(argv: list[str] | None = None) -> int:
        del argv
        run_group(
            definition,
            load_group(),
            build_request=build_request,
            runtime_config=runtime_config,
            declared=declared,
        )
        return 0

    return main
