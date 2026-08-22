"""What a group has to agree about before it can be one graph.

The graph is built from the first member and used for all of them, so every
disagreement these tests provoke would otherwise be a member reported under its
own identity and computed under somebody else's. That is the failure worth
spending a check on: not a crash, but a plausible number on a run nobody would
think to doubt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deployment.contract import CONTRACT_VERSION
from entries._contract import RunSpec
from entries._ensemble import GroupError, one_configuration, swept_parameters
from memorax.algorithms.drqn import PARAMETERS as DECLARED


def spec(
    *,
    seed: int,
    label: int | None = None,
    root: str | None = None,
    parameters: dict | None = None,
    total_steps: int = 8,
) -> RunSpec:
    labelled = seed if label is None else label
    return RunSpec.model_validate(
        {
            "contract": CONTRACT_VERSION,
            "identity": {
                "run_id": f"group-t0-s{labelled}",
                "experiment": "group",
                "launch_id": "20260822-000000",
                "trial": 0,
                "seed": labelled,
                "role": "tuning",
                "digest": "local@sha256:" + "a" * 64,
            },
            "entry": "drqn_ensemble",
            "artifacts": {"root": root or f"s3://bucket/seed-{labelled}"},
            "algorithm": {
                "environment": {
                    "id": "gymnax::CartPole-v1",
                    "backend": None,
                    "episode_length": 8,
                    "observed": None,
                },
                "num_envs": 1,
                "parameters": parameters or {"gamma": 0.9},
            },
            "training": {"seed": seed, "total_steps": total_steps, "chunk_steps": 4},
            "evaluation": {
                "every_steps": 4,
                "episodes": 0,
                "chunk_steps": 4,
                "seed": 1000,
            },
            "logging": {"aim": {"url": "aim://localhost:1"}},
        }
    )


def group(*specs: RunSpec):
    return tuple((item, Path("/tmp/scratch")) for item in specs)


def test_seeds_and_the_names_that_follow_them_may_differ():
    shared = one_configuration(group(spec(seed=0), spec(seed=1), spec(seed=2)))
    assert shared.training.seed == 0


def test_a_value_that_differs_is_what_the_group_sweeps():
    """A discount the members disagree about is the sweep, not an error.

    This is the rule that changed once the algorithms stopped coercing their
    value-only leaves. Under seeds alone it was a refusal, because the graph was
    built from one member and a difference could only have been silently
    dropped; now the difference is carried on the member axis.
    """

    members = group(
        spec(seed=0, parameters={"gamma": 0.9}),
        spec(seed=1, parameters={"gamma": 0.5}),
    )
    one_configuration(members)
    assert swept_parameters(members, DECLARED) == {"gamma": [0.9, 0.5]}


def test_what_the_members_agree_about_is_not_swept():
    members = group(
        spec(seed=0, parameters={"gamma": 0.9, "grad_clip": 1.0}),
        spec(seed=1, parameters={"gamma": 0.5, "grad_clip": 1.0}),
    )
    assert swept_parameters(members, DECLARED) == {"gamma": [0.9, 0.5]}


def test_a_static_value_that_differs_is_refused():
    """A width sizes an array, and the members of one map share their shapes.

    Nothing about `hidden_dim`'s domain tells it apart from `gamma`'s -- both
    are numbers a search may draw -- so this is the declaration doing the work,
    and doing it before a job compiles rather than several frames into a trace.
    """

    members = group(
        spec(seed=0, parameters={"core.lru.hidden_dim": 32}),
        spec(seed=1, parameters={"core.lru.hidden_dim": 64}),
    )
    with pytest.raises(GroupError, match="core.lru.hidden_dim, which is static"):
        swept_parameters(members, DECLARED)


def test_a_branch_that_differs_is_refused():
    """Two cores is two graphs, and a graph is what a group shares.

    No declaration was needed: a choice's values are strings, which no trace
    carries, so the domain settles it.
    """

    members = group(
        spec(seed=0, parameters={"core.kind": "lru"}),
        spec(seed=1, parameters={"core.kind": "rtu"}),
    )
    with pytest.raises(GroupError, match="core.kind, which is static"):
        swept_parameters(members, DECLARED)


def test_a_name_the_algorithm_does_not_declare_is_refused():
    """Nothing can say whether an undeclared name may be swept, so nothing does."""

    members = group(
        spec(seed=0, parameters={"invented": 1.0}),
        spec(seed=1, parameters={"invented": 2.0}),
    )
    with pytest.raises(GroupError, match="does not declare"):
        swept_parameters(members, DECLARED)


def test_a_budget_that_differs_is_refused():
    """Not only the algorithm. The schedule is shared too, and silently.

    Members are advanced together, so a member asking for a longer run would
    simply be stopped where the first member stopped.
    """

    with pytest.raises(GroupError, match="differs from .*in training"):
        one_configuration(group(spec(seed=0), spec(seed=1, total_steps=16)))


def test_a_label_that_does_not_match_what_ran_is_refused():
    """The artifacts say which seed produced them, so the label cannot lie."""

    with pytest.raises(GroupError, match="labelled seed 9 but trains on 1"):
        one_configuration(group(spec(seed=0), spec(seed=1, label=9)))


def test_a_repeated_seed_is_refused():
    with pytest.raises(GroupError, match="repeats a seed"):
        one_configuration(group(spec(seed=0), spec(seed=0, root="s3://bucket/b")))


def test_a_repeated_artifact_root_is_refused():
    """Two members publishing to one prefix is one member's results.

    Both would exit zero. The run that appears to have happened is whichever
    wrote last, and nothing downstream could see that the other ever ran.
    """

    with pytest.raises(GroupError, match="repeats an artifact root"):
        one_configuration(
            group(
                spec(seed=0, root="s3://bucket/same"),
                spec(seed=1, root="s3://bucket/same/"),
            )
        )
