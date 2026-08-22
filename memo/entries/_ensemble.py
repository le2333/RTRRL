"""Run one trial's seeds as a single graph, and report each as its own run.

The worker hands a group entry several run documents at once instead of one, and
this is what an algorithm's ensemble entry does with them. Every member still
produces its own metrics, its own artifacts and its own result: what changes is
that they were computed together, not that they became one run.

Only the seed may differ. The graph is built from the first member and used for
all of them, so a member that disagreed about anything else would be silently
run under its neighbour's parameters -- a wrong number rather than a failure, on
a run nobody would think to doubt. :func:`one_configuration` refuses that.
"""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

from memorax.assembly import assemble
from memorax.runtime import RuntimeConfig
from memorax.runtime.ensemble import EnsembleRuntime

from ._contract import RunSpec
from ._observability import build_reporter

GROUP_VARIABLE = "TRAINER_RUN_GROUP"

# What a member is allowed to carry of its own. Everything else has to match,
# because everything else went into the one graph they share. The artifact root
# is here because it *must* differ: members publish separately, and two sharing
# a root would each overwrite the other's metrics and result.
PERSONAL = (
    ("identity", "run_id"),
    ("identity", "seed"),
    ("training", "seed"),
    ("artifacts", "root"),
)


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
    for *parents, leaf in PERSONAL:
        scope = shape
        for name in parents:
            scope = scope[name]
        scope.pop(leaf, None)
    return shape


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
) -> None:
    """Build the shared graph once and run every member through it."""

    shared = one_configuration(members)
    algorithm = assemble(definition, build_request(shared))
    with ExitStack() as stack:
        reporters = [
            stack.enter_context(build_reporter(spec, scratch))
            for spec, scratch in members
        ]
        EnsembleRuntime(
            algorithm=algorithm,
            config=runtime_config(shared),
            seeds=tuple(spec.training.seed for spec, _ in members),
        ).run(reporters)


def main_for(
    definition: Any,
    *,
    build_request: Callable[[RunSpec], Any],
    runtime_config: Callable[[RunSpec], RuntimeConfig],
) -> Callable[[list[str] | None], int]:
    """The ``main`` an algorithm's ensemble entry exposes to the catalog."""

    def main(argv: list[str] | None = None) -> int:
        del argv
        run_group(
            definition,
            load_group(),
            build_request=build_request,
            runtime_config=runtime_config,
        )
        return 0

    return main
