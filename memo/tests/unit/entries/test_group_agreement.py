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
from entries._ensemble import GroupError, one_configuration


def spec(
    *,
    seed: int,
    label: int | None = None,
    root: str | None = None,
    gamma: float = 0.9,
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
                "parameters": {"gamma": gamma},
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


def test_a_parameter_that_differs_is_refused():
    with pytest.raises(GroupError, match="differs from .*in algorithm"):
        one_configuration(group(spec(seed=0), spec(seed=1, gamma=0.5)))


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
