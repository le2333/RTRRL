"""Three branches of one run, and the ways of building them that would lie.

A fork is only a comparison of update rules if everything else is held: the
same parent state, the same boundary, the same threshold, the same axis. Each
test here is one of those, written as the mistake it prevents -- an arm whose
threshold was chosen independently is not a control for the clip it is supposed
to be the limit of, and a branch dated to its own step zero cannot be laid over
the curve that decided to take it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainer_infra.fork import (
    ARMS,
    ForkError,
    arm_parameters,
    branch_documents,
    checkpoint_uri,
    manifest,
    preceding_checkpoint,
    replacing,
    threshold,
)

CONTRACT = Path(__file__).resolve().parents[2] / "tests" / "contracts" / "v12"
CLIP = 2.5


def parent(**overrides):
    document = {
        "contract": 12,
        "identity": {
            "run_id": "rtrrl-r2-halfcheetah-t0-s11",
            "experiment": "rtrrl-issue45-r2",
            "launch_id": "20260817-000000",
            "trial": 0,
            "seed": 11,
            "role": "formal",
            "digest": "sha256:abc",
        },
        "entry": "rtrrl",
        "artifacts": {"root": "s3://bucket/rtrrl-issue45-r2/launch/run-t0-s11"},
        "algorithm": {
            "environment": {
                "id": "brax::halfcheetah",
                "backend": "spring",
                "observed": [0, 1],
                "episode_length": 1000,
            },
            "num_envs": 1,
            "parameters": {
                "gamma": 0.99,
                "torso.grad_clip": CLIP,
                "torso.optimizer.kind": "adam",
                "torso.optimizer.adam.lr": 1e-4,
                "heads.optimizer.kind": "adam",
                "heads.optimizer.adam.lr": 1e-4,
            },
        },
        "training": {"seed": 11, "total_steps": 1000000, "chunk_steps": 10000},
        "evaluation": {
            "every_steps": 10000,
            "episodes": 5,
            "chunk_steps": 5000,
            "seed": 7,
        },
        "checkpoint": {"every_steps": 10000, "keep": None},
        "logging": {"aim": {"url": "aim://aim:53800"}},
    }
    document.update(overrides)
    return document


def branches(**overrides):
    return branch_documents(
        parent(**overrides),
        checkpoint="s3://bucket/run-t0/checkpoints/step-000000700000.msgpack",
        from_steps=700000,
        steps=50000,
    )


def by_name(documents):
    return {
        document["identity"]["run_id"].rsplit("-s11-", 1)[1]: document
        for document in documents
    }


def test_the_three_arms_are_the_rule_that_collapsed_and_its_two_controls():
    found = branches()

    assert [document["identity"]["run_id"] for document in found] == [
        "rtrrl-r2-halfcheetah-t0-s11-original-clip",
        "rtrrl-r2-halfcheetah-t0-s11-fixed-step",
        "rtrrl-r2-halfcheetah-t0-s11-td-out",
    ]
    assert ARMS == ("original_clip", "fixed_step", "td_out")
    assert [document["identity"]["trial"] for document in found] == [0, 1, 2]


def test_every_branch_names_the_same_parent_state_at_the_same_boundary():
    """Anything else and the three are not a comparison of rules."""

    for document in branches():
        assert document["fork"]["parent"].endswith("step-000000700000.msgpack")
        assert document["fork"]["from_steps"] == 700000


def test_both_arms_are_written_over_the_original_clips_own_threshold():
    """The arms are limits of *this* clip, so ``c`` is read off the parent.

    Chosen independently they would be a different optimizer being compared
    with the original rather than a decomposition of it, and the way that
    happens is somebody typing the number.
    """

    assert threshold(parent()["algorithm"]["parameters"]) == CLIP

    arms = by_name(branches())
    for name in ("fixed-step", "td-out"):
        parameters = arms[name]["algorithm"]["parameters"]
        assert parameters["torso.optimizer.d_rtrrl.c"] == CLIP
        assert parameters["heads.optimizer.d_rtrrl.c"] == CLIP
        # And the clip they are the limit of is still declared, unchanged.
        assert parameters["torso.grad_clip"] == CLIP


def test_the_two_arms_differ_from_each_other_in_the_magnitude_alone():
    arms = by_name(branches())
    fixed = arms["fixed-step"]["algorithm"]["parameters"]
    td_out = arms["td-out"]["algorithm"]["parameters"]

    differ = {key for key in fixed if fixed[key] != td_out.get(key)}
    assert differ == {
        "torso.optimizer.d_rtrrl.magnitude",
        "heads.optimizer.d_rtrrl.magnitude",
    }
    assert fixed["torso.optimizer.d_rtrrl.magnitude"] == "sign"
    assert td_out["torso.optimizer.d_rtrrl.magnitude"] == "td_out"


def test_the_unchanged_arm_changes_nothing_at_all():
    """It is the parent's own rule, and it is what makes the fork falsifiable.

    Two branches of it must agree with each other and with the parent's own
    continuation; if they do not, no difference between the changed arms means
    anything.
    """

    original = by_name(branches())["original-clip"]

    assert original["algorithm"]["parameters"] == parent()["algorithm"]["parameters"]
    assert original["fork"]["replacing"] == []


def test_a_branch_that_cannot_hold_the_parents_optimizer_state_declares_it():
    """Adam carries moments; the D-RTRRL arms carry nothing.

    Derived from what the arm changed rather than declared by hand, because
    both mistakes are silent: named needlessly, the branch throws away the
    parent's moments; missing when it is needed, the run dies in the container
    after the queue time.
    """

    arms = by_name(branches())
    assert arms["fixed-step"]["fork"]["replacing"] == ["core.rule"]
    assert arms["td-out"]["fork"]["replacing"] == ["core.rule"]

    parameters = parent()["algorithm"]["parameters"]
    assert replacing(parameters, {}) == ()
    assert replacing(parameters, {"heads.optimizer.kind": "d_rtrrl"}) == ("core.rule",)


def test_a_branch_continues_its_parents_step_axis():
    """750k in a branch is the parent's 750k, or the curves cannot be overlaid."""

    for document in branches():
        assert document["training"]["total_steps"] == 750000
        assert document["training"]["seed"] == 11
        assert document["evaluation"] == parent()["evaluation"]


def test_a_branch_is_the_same_repetition_of_the_same_configuration():
    """The seed and the role are the parent's; only the trial distinguishes arms.

    A branch that renumbered the seed would claim to be another sample of the
    configuration, when it is the same run continued under another rule -- and
    a branch that called itself tuning would be a formal result nothing may
    report.
    """

    for document in branches():
        assert document["identity"]["seed"] == parent()["identity"]["seed"]
        assert document["identity"]["role"] == "formal"


def test_a_branch_writes_beside_its_parent_and_files_no_checkpoints_of_its_own():
    roots = {document["artifacts"]["root"] for document in branches()}

    assert roots == {
        "s3://bucket/rtrrl-issue45-r2/launch/rtrrl-r2-halfcheetah-t0-s11-original-clip",
        "s3://bucket/rtrrl-issue45-r2/launch/rtrrl-r2-halfcheetah-t0-s11-fixed-step",
        "s3://bucket/rtrrl-issue45-r2/launch/rtrrl-r2-halfcheetah-t0-s11-td-out",
    }
    # A 50k branch has nothing to fork from itself, and the parent's block
    # would otherwise make three more copies of the state.
    assert all("checkpoint" not in document for document in branches())


def test_the_checkpoint_taken_is_the_last_one_strictly_before_the_collapse():
    """At the collapse the decline has already begun.

    Branching from there would compare three rules on recovering from a
    collapse, which is a different experiment from the one that was asked for.
    """

    assert preceding_checkpoint(parent(), 700000) == 690000
    assert preceding_checkpoint(parent(), 705000) == 700000


def test_a_checkpoint_interval_wider_than_the_evaluation_is_respected():
    """A run that files one state in five measures its fork boundary in fives."""

    sparse = parent(checkpoint={"every_steps": 50000, "keep": None})

    assert preceding_checkpoint(sparse, 720000) == 700000
    assert checkpoint_uri(sparse, 700000).endswith(
        "/checkpoints/step-000000700000.msgpack"
    )


def test_a_parent_that_filed_nothing_cannot_be_forked():
    childless = parent()
    childless.pop("checkpoint")

    with pytest.raises(ForkError, match="filed no checkpoints"):
        preceding_checkpoint(childless, 700000)


def test_a_collapse_at_the_first_checkpoint_has_nothing_before_it():
    with pytest.raises(ForkError, match="nothing precedes it"):
        preceding_checkpoint(parent(), 10000)


def test_a_parent_with_no_clip_has_no_saturated_limit_to_compare_with():
    for clip in (0.0, None):
        parameters = dict(parent()["algorithm"]["parameters"])
        if clip is None:
            parameters.pop("torso.grad_clip")
        else:
            parameters["torso.grad_clip"] = clip
        with pytest.raises(ForkError):
            arm_parameters(parameters)


def test_a_branch_that_runs_no_steps_measures_nothing():
    with pytest.raises(ForkError, match="no steps"):
        branch_documents(
            parent(),
            checkpoint="s3://bucket/checkpoints/step-000000700000.msgpack",
            from_steps=700000,
            steps=0,
        )


def test_the_manifest_names_the_branches_in_the_arms_order():
    uris = [f"s3://bucket/configs/{document['identity']['run_id']}.json" for document in branches()]

    assert json.loads(manifest(uris))["runs"] == uris


def test_a_branch_document_carries_the_blocks_the_image_contract_declares():
    """The shape is the version-11 run document, which the image side validates.

    Both sides read `tests/contracts/v11/fork.json`: the image proves a
    document of that shape projects onto its Entry and its Worker, and this
    proves the branches this side builds are documents of that shape.
    """

    fixture = json.loads((CONTRACT / "fork.json").read_text(encoding="utf-8"))

    for document in branches():
        assert set(document) == set(fixture)
        assert set(document["fork"]) == set(fixture["fork"])
        assert set(document["identity"]) == set(fixture["identity"])
