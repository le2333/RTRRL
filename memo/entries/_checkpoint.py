"""Project a run document's checkpoint and fork blocks onto Runtime's.

Both directions use the artifact contract that already exists. A parent files
its checkpoints under ``scratch/artifacts/checkpoints/``, which Worker uploads
whole to ``artifacts.root`` like everything else in that directory; a branch
names one of the resulting objects by URI and reads it back through the same
object access Worker publishes with. There is no fork-specific worker, no fork
transport and no second artifact layout -- a branch is an ordinary run whose
document says where it started.
"""

from __future__ import annotations

from pathlib import Path

import jax

from memorax.runtime import Checkpoint, CheckpointDirectory
from memorax.runtime.checkpoint import loads
from worker import objects

from ._contract import RunSpec

DIRECTORY = "checkpoints"


def checkpoint_directory(config: RunSpec, scratch: Path) -> CheckpointDirectory | None:
    """Where this run files its whole state, if it files any."""

    if config.checkpoint is None:
        return None
    return CheckpointDirectory(
        Path(scratch) / "artifacts" / DIRECTORY, keep=config.checkpoint.keep
    )


def checkpoint_every_steps(config: RunSpec) -> int:
    return 0 if config.checkpoint is None else config.checkpoint.every_steps


def resume(config: RunSpec, program) -> Checkpoint | None:
    """The parent checkpoint this run continues from, if it is a branch.

    The template is a fresh state of *this* run's graph, so what is checked is
    that the parent fits the branch about to use it -- every parameter, trace,
    carry and counter, leaf by leaf. The declared boundary is checked against
    the object rather than trusted: a manifest naming one boundary and pointing
    at another would date a branch to a moment it did not come from, and the
    metrics it wrote would be wrong on an axis nothing downstream re-derives.
    """

    fork = config.fork
    if fork is None:
        return None
    key = jax.random.key(config.training.seed)
    key, init_key = jax.random.split(key)
    checkpoint = loads(
        objects.get_bytes(fork.parent),
        state=jax.jit(program.init)(init_key),
        key=key,
        replacing=fork.replacing,
    )
    if checkpoint.env_steps != fork.from_steps:
        raise ValueError(
            f"{fork.parent} is the boundary at {checkpoint.env_steps} steps, but "
            f"this run says it forked from {fork.from_steps}"
        )
    return checkpoint
