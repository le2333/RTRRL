"""What the Entry does with the two blocks a fork is made of.

The mechanism is tested next door, against states and rules; what is left here
is the projection -- that a document asking for checkpoints files them where
the artifact contract says, that a branch reads its parent through the same
object access Worker publishes with, and that a branch whose document and
whose object disagree about the boundary is refused rather than run and
mislabelled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from entries._checkpoint import (
    checkpoint_directory,
    checkpoint_every_steps,
    resume,
)
from entries._contract import ForkSpec, RunSpec
from memorax.runtime.checkpoint import CheckpointError, dumps
from tests.support.builders import D_RTRRL, assemble_rtrrl
from tests.support.numerics import flattened

SEED = 3
BOUNDARY = 20000
# The serialized version-12 documents both sides of the deployment boundary
# read, so what these project from is the shape the Entry will actually be
# handed rather than a stand-in that cannot go out of date with it.
FIXTURES = Path(__file__).resolve().parents[4] / "tests" / "contracts" / "v12"


def document(*, checkpoint=None, fork=None) -> RunSpec:
    """A validated run document with the two blocks a fork is made of."""

    payload = json.loads((FIXTURES / "run.json").read_text(encoding="utf-8"))
    payload["training"] |= {"seed": SEED, "total_steps": 70000, "chunk_steps": 10000}
    payload["evaluation"] = {
        "every_steps": 10000,
        "episodes": 2,
        "chunk_steps": 100,
        "seed": 7,
    }
    payload["algorithm"]["num_envs"] = 1
    if checkpoint is None:
        payload.pop("checkpoint", None)
    else:
        payload["checkpoint"] = checkpoint
    if fork is not None:
        payload["fork"] = fork
    return RunSpec.model_validate(payload)


def forked(uri: str, *, from_steps: int = BOUNDARY, replacing=()) -> RunSpec:
    return document(
        fork={
            "parent": uri,
            "from_steps": from_steps,
            "replacing": list(replacing),
        }
    )


def parent_object(tmp_path: Path, *, steps=BOUNDARY, optimizer=None) -> tuple[str, Any]:
    """A checkpoint of a trained run, addressed the way an artifact is."""

    built = assemble_rtrrl(optimizer=optimizer)
    key = jax.random.key(SEED)
    key, init_key = jax.random.split(key)
    state, _ = built.program.train(key, built.program.init(init_key), 4)
    path = tmp_path / "checkpoints" / "step.msgpack"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dumps(env_steps=steps, state=state, key=key))
    return path.resolve().as_uri(), state


def test_a_document_without_a_checkpoint_block_files_nothing(tmp_path):
    assert checkpoint_directory(document(), tmp_path) is None
    assert checkpoint_every_steps(document()) == 0


def test_checkpoints_land_under_the_artifact_directory_worker_uploads(tmp_path):
    """No fork-specific transport: they go where everything else goes.

    Worker uploads `scratch/artifacts/` whole and keeps the relative paths, so
    a checkpoint reaches S3 by being an artifact rather than by anything new.
    """

    config = document(checkpoint={"every_steps": 10000, "keep": 2})
    directory = checkpoint_directory(config, tmp_path)

    assert directory is not None
    assert checkpoint_every_steps(config) == 10000
    built = assemble_rtrrl()
    state = built.program.init(jax.random.key(0))
    saved = directory.save(env_steps=10000, state=state, key=jax.random.key(1))

    assert saved.parent == tmp_path / "artifacts" / "checkpoints"
    assert saved.name == "step-000000010000.msgpack"


def test_a_run_that_is_not_a_branch_restores_nothing(tmp_path):
    built = assemble_rtrrl()

    assert resume(document(), built.program) is None


def test_a_branch_reads_its_parent_by_uri_and_takes_its_state(tmp_path):
    uri, state = parent_object(tmp_path)

    restored = resume(forked(uri), assemble_rtrrl().program)

    assert restored is not None
    assert restored.env_steps == BOUNDARY
    got, wanted = flattened(restored.state), flattened(state)
    assert all(np.array_equal(leaf, got[path]) for path, leaf in wanted.items())


def test_a_branch_onto_another_rule_carries_what_it_declared(tmp_path):
    uri, state = parent_object(tmp_path)
    config = forked(uri, replacing=("core.rule",))

    restored = resume(config, assemble_rtrrl(optimizer=D_RTRRL).program)

    assert restored is not None and restored.replaced == ("core.rule",)
    parent, branch = flattened(state.core.torso), flattened(restored.state.core.torso)
    assert all(np.array_equal(leaf, branch[path]) for path, leaf in parent.items())


def test_a_branch_onto_another_rule_that_did_not_declare_it_is_refused(tmp_path):
    uri, _ = parent_object(tmp_path)

    with pytest.raises(CheckpointError):
        resume(forked(uri), assemble_rtrrl(optimizer=D_RTRRL).program)


def test_a_branch_whose_document_and_object_disagree_is_refused(tmp_path):
    """A branch dated to a boundary it did not come from is undetectable later.

    Nothing downstream re-derives which state a run started from: the step axis
    of everything it reports comes from this number.
    """

    uri, _ = parent_object(tmp_path, steps=30000)

    with pytest.raises(ValueError, match="says it forked from"):
        resume(forked(uri), assemble_rtrrl().program)


def test_a_fork_block_that_names_no_parent_is_refused():
    with pytest.raises(ValueError, match="must name the checkpoint"):
        ForkSpec(parent="", from_steps=BOUNDARY)
