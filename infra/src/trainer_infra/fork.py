"""Three branches of one run, from one checkpoint, as ordinary run documents.

R3.4 asks what the three update rules do from the moment before a collapse.
That is three runs which differ in their optimizer parameters and in nothing
else -- not in their environment, their seed, their budget's step axis, or the
state they start from -- so a branch is not a new kind of job. It is a run
document with a ``fork`` block, and the three of them go into the manifest
format the Worker already reads.

What this module owns is the part that must not be typed by hand: that the two
D-RTRRL arms are written over the *same* threshold as the original clip they
are the limits of, that all three name the same parent object at the same
boundary, and that a branch whose rule cannot hold the parent's optimizer state
says so in its own document rather than discovering it in the container.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

# The order is the issue's: the rule that produced the collapse, then the two
# controls for it. Kept as a tuple because a comparison between three arms is
# read in a fixed order and a set would not have one.
ARMS: tuple[str, ...] = ("original_clip", "fixed_step", "td_out")

# Whose state a branch cannot take from a parent that stepped a different rule.
# Adam carries moments per parameter and the D-RTRRL arms carry nothing at all,
# so this is the one path a fork legitimately declares.
RULE_STATE = "core.rule"

OPTIMIZER_GROUPS = ("torso", "heads")

# Where a run files its whole state inside its artifact root, and what one
# checkpoint is called. This is the artifact contract rather than this side's
# choice -- the image writes these names -- and it is spelled here for the same
# reason `metrics.jsonl` is spelled on both sides of the boundary: the control
# plane names objects the image produced without importing the image.
CHECKPOINTS = "checkpoints"
CHECKPOINT_NAME = "step-{steps:012d}.msgpack"


class ForkError(ValueError):
    """A branch cannot be built from this parent and this decision."""


def threshold(parameters: Mapping[str, Any]) -> float:
    """The number the original clip bounds an update by, which the arms share.

    Read off the parent rather than passed in. The fixed-step arm is the
    saturated limit of *this* clip and the TD-out arm is the same clip with the
    TD error left in; both stop being controls for it the moment their ``c`` is
    chosen independently, and the way that happens is somebody typing it.
    """

    clip = parameters.get("torso.grad_clip")
    if clip is None:
        raise ForkError(
            "the parent declares no torso.grad_clip, so there is no original "
            "clip for the two arms to be written over"
        )
    if float(clip) <= 0.0:
        raise ForkError(
            "the parent's original clip is off, so its saturated limit is not "
            "defined and neither arm is a control for anything"
        )
    return float(clip)


def arm_parameters(parent: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """What each of the three arms changes about the parent's parameters.

    ``original_clip`` changes nothing: it is the parent's own rule, and it is
    what makes the fork falsifiable, since two branches of it must agree.
    """

    c = threshold(parent)
    arms: dict[str, dict[str, Any]] = {"original_clip": {}}
    for name, magnitude in (("fixed_step", "sign"), ("td_out", "td_out")):
        overrides: dict[str, Any] = {}
        for group in OPTIMIZER_GROUPS:
            overrides |= {
                f"{group}.optimizer.kind": "d_rtrrl",
                f"{group}.optimizer.d_rtrrl.c": c,
                f"{group}.optimizer.d_rtrrl.magnitude": magnitude,
                f"{group}.optimizer.d_rtrrl.scope": "block",
                f"{group}.optimizer.d_rtrrl.denominator": "shifted",
                f"{group}.optimizer.d_rtrrl.eps": 1e-8,
            }
        arms[name] = overrides
    return arms


def replacing(parent: Mapping[str, Any], overrides: Mapping[str, Any]) -> tuple[str, ...]:
    """Which state the branch supplies itself, from what its rule changed.

    One rule state covers both groups, so a change to either group's optimizer
    means the parent's has nowhere to go. Derived rather than declared by hand
    because getting it wrong in either direction is silent: named when it need
    not be, the branch throws away the parent's moments; not named when it must
    be, the run dies in the container after the queue time.
    """

    changed = any(
        overrides.get(f"{group}.optimizer.kind", parent.get(f"{group}.optimizer.kind"))
        != parent.get(f"{group}.optimizer.kind")
        for group in OPTIMIZER_GROUPS
    )
    return (RULE_STATE,) if changed else ()


def preceding_checkpoint(parent: Mapping[str, Any], collapse_step: int) -> int:
    """The last boundary the parent filed a checkpoint at before the collapse.

    Strictly before: the checkpoint *at* the collapse is the state after the
    decline had already begun, and branching from it would compare three rules
    on recovering from a collapse rather than on causing one.
    """

    declared = parent.get("checkpoint")
    if not declared:
        raise ForkError(
            f"run {parent['identity']['run_id']} filed no checkpoints, so there "
            "is no state to branch from; a run that may need forking declares a "
            "checkpoint block before it starts"
        )
    every = int(declared["every_steps"])
    boundary = ((int(collapse_step) - 1) // every) * every
    if boundary <= 0:
        raise ForkError(
            f"the collapse at {collapse_step} steps is at or before the first "
            f"checkpoint of every {every}; nothing precedes it to branch from"
        )
    return boundary


def checkpoint_uri(parent: Mapping[str, Any], boundary: int) -> str:
    """Where the parent's checkpoint for that boundary was uploaded to."""

    root = str(parent["artifacts"]["root"]).rstrip("/")
    return f"{root}/{CHECKPOINTS}/{CHECKPOINT_NAME.format(steps=int(boundary))}"


def branch_documents(
    parent: Mapping[str, Any],
    *,
    checkpoint: str,
    from_steps: int,
    steps: int,
    arms: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """One run document per arm, all naming the same parent checkpoint.

    ``steps`` is the branch's own budget. The document's ``total_steps`` is the
    parent's boundary plus it, because a branch continues the parent's step
    axis: its evaluation at 750k is the parent's 750k, and the two curves are
    read against the same numbers.
    """

    if steps <= 0:
        raise ForkError("a branch that runs no steps measures nothing")
    parameters = dict(parent["algorithm"]["parameters"])
    selected = dict(arms) if arms is not None else arm_parameters(parameters)
    if not selected:
        raise ForkError("a fork with no arms is not a comparison")

    documents = []
    for trial, name in enumerate(sorted(selected, key=_arm_order)):
        overrides = dict(selected[name])
        run_id = f"{parent['identity']['run_id']}-{name.replace('_', '-')}"
        documents.append(
            {
                **{
                    key: value
                    for key, value in parent.items()
                    # Rebuilt below, or belonging to the parent alone.
                    if key not in {"identity", "artifacts", "algorithm", "training", "checkpoint"}
                },
                "identity": {**parent["identity"], "run_id": run_id, "trial": trial},
                "artifacts": {"root": _sibling(parent["artifacts"]["root"], run_id)},
                "algorithm": {
                    **parent["algorithm"],
                    "parameters": {**parameters, **overrides},
                },
                "training": {
                    **parent["training"],
                    "total_steps": from_steps + steps,
                },
                "fork": {
                    "parent": checkpoint,
                    "from_steps": from_steps,
                    "replacing": list(replacing(parameters, overrides)),
                },
            }
        )
    return tuple(documents)


def manifest(uris: Sequence[str]) -> str:
    """The Worker manifest naming the branch documents, in the arms' order."""

    return json.dumps({"runs": list(uris)}, sort_keys=True)


def _arm_order(name: str) -> tuple[int, str]:
    return (ARMS.index(name) if name in ARMS else len(ARMS), name)


def _sibling(root: str, run_id: str) -> str:
    """A branch's artifacts beside its parent's, under its own run id."""

    prefix = str(root).rstrip("/").rsplit("/", 1)[0]
    return f"{prefix}/{run_id}"
