"""R3.4 end to end: one run, one boundary, three branches, through Runtime.

The pieces are driven apart elsewhere -- the checkpoint against a state tree,
the schedule against an arithmetic program. What is left, and what R3.4 rests
on, is the whole path: a real RTRRL run files a whole state at a measured
boundary, three runs restore it, and the two that were given the same rule are
the same run while the one that was given another is not.

The environment's reward does not depend on the action, which is deliberate:
returns cannot tell these branches apart, so the comparison is made on what
the updates did. A test that compared returns here would pass for a fork
mechanism that restored nothing at all.
"""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from memorax.runtime import CheckpointDirectory, Runtime, RuntimeConfig
from memorax.runtime.checkpoint import CheckpointError, read
from tests.support.builders import D_RTRRL, assemble_rtrrl
from tests.support.fakes import EpisodeRecorder

BOUNDARY = 16
TOTAL = 24
INTERVAL = 8
WATCHED = "update.torso.realized_update_norm"


def schedule(**overrides) -> RuntimeConfig:
    return RuntimeConfig(
        **{
            "total_steps": TOTAL,
            "chunk_steps": INTERVAL,
            "max_episode_steps": 16,
            "evaluate_every_steps": INTERVAL,
            "rollout_steps": 6,
            "num_envs": 1,
            "seed": 5,
            **overrides,
        }
    )


def parent(tmp_path: Path) -> Path:
    """A run that files its whole state at every boundary it measures."""

    built = assemble_rtrrl(record=frozenset())
    directory = tmp_path / "checkpoints"
    Runtime(
        algorithm=built,
        config=schedule(checkpoint_every_steps=INTERVAL),
        checkpoints=CheckpointDirectory(directory),
    ).run(EpisodeRecorder())
    return directory / f"step-{BOUNDARY:012d}.msgpack"


def branch(path: Path, *, optimizer=None, replacing=()) -> list[tuple]:
    """One branch from that boundary, and what its updates did."""

    built = assemble_rtrrl(record=frozenset(), optimizer=optimizer)
    restored = read(
        path,
        state=built.program.init(jax.random.key(0)),
        key=jax.random.key(0),
        replacing=replacing,
    )
    recorder = EpisodeRecorder()
    Runtime(algorithm=built, config=schedule()).run(recorder, resume=restored)
    return [
        (episode.phase, episode.end_env_steps, tuple(episode.series.get(WATCHED, ())))
        for episode in recorder.episodes
    ]


def test_three_branches_of_one_boundary_agree_exactly_where_they_should(tmp_path):
    """Two Original-clip branches are one run; the fixed-step arm is another.

    Both halves are the claim. Without the first, a difference between arms
    could be the fork mechanism rather than the rule. Without the second, the
    branches could be agreeing because none of them is reading its checkpoint.
    """

    path = parent(tmp_path)

    original = branch(path)
    twin = branch(path)
    fixed_step = branch(path, optimizer=D_RTRRL, replacing=("core.rule",))

    assert original, "the branch reported no episode, so nothing was compared"
    assert original == twin
    assert original != fixed_step
    # And every branch picked up where the parent stopped rather than at zero.
    assert min(end for _, end, _ in original) > BOUNDARY


def test_a_branch_reports_on_the_parents_step_axis(tmp_path):
    """750k in a branch is the parent's 750k, in miniature.

    The boundaries already behind the checkpoint are not re-run: they happened
    once, and the parent reported them.
    """

    path = parent(tmp_path)

    evaluations = [end for phase, end, _ in branch(path) if phase == "eval"]

    # However many episodes one evaluation completes, they are all dated to
    # the boundary that ran them -- and the only boundary this branch ran is
    # the one after the checkpoint.
    assert evaluations and set(evaluations) == {TOTAL}


def test_a_branch_onto_another_rule_must_declare_what_it_cannot_hold(tmp_path):
    """The one honest mismatch, undeclared, is an error rather than a guess."""

    path = parent(tmp_path)

    with pytest.raises(CheckpointError):
        branch(path, optimizer=D_RTRRL)
