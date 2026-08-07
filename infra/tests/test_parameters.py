"""Drawing one point out of a tree, which is what makes the space conditional.

A parameter under a branch belongs to the trials that took that branch. That is
not something a flat table of ranges can say, so the sampler walks the tree and
draws ``kind`` a line above the branch it opens -- the same walk the worker does
when it builds the graph, differing only in where the chosen name comes from.
"""

import optuna
import pytest

from trainer_infra import sample_parameters

BOUNDED = {
    "kind": {"type": "choice", "values": ["ob", "none"]},
    "ob": {"kappa": {"type": "float", "low": 0.5, "high": 10.0, "log": False}},
}


def test_sample_parameters_samples_every_declared_range() -> None:
    ranges = {
        "gamma": {"type": "choice", "values": [0.9, 0.95, 0.99]},
        "actor": {"optim": {"lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}}},
        "batch_size": {"type": "int", "low": 32, "high": 128, "step": 32},
        "meta_rl": {"type": "choice", "values": [False]},
    }
    trial = optuna.trial.FixedTrial(
        {"gamma": 0.95, "actor.optim.lr": 1e-4, "batch_size": 64, "meta_rl": False}
    )

    assert sample_parameters(trial, ranges) == {
        "gamma": 0.95,
        "actor.optim.lr": 1e-4,
        "batch_size": 64,
        "meta_rl": False,
    }


def test_single_choice_is_still_declared_as_a_trial_parameter() -> None:
    study = optuna.create_study()
    trial = study.ask()

    sampled = sample_parameters(trial, {"backbone": {"kind": {"type": "choice", "values": ["rtu"]}}})

    assert sampled == {"backbone.kind": "rtu"}
    assert trial.params == {"backbone.kind": "rtu"}


def test_a_branch_nobody_chose_contributes_no_dimension() -> None:
    """The point of the walk. Otherwise every SGD trial also carries a kappa.

    Those are dimensions nothing reads, which the sampler must nonetheless
    model, and which a run configuration would then carry as values that look
    as though someone chose them.
    """

    study = optuna.create_study()
    trial = study.ask()

    sampled = sample_parameters(
        trial, {"bound": {**BOUNDED, "kind": {"type": "choice", "values": ["none"]}}}
    )

    assert sampled == {"bound.kind": "none"}
    assert "bound.ob.kappa" not in trial.params


def test_a_choice_of_component_may_itself_be_searched() -> None:
    """Which the old flat table could not express, so it was forbidden instead.

    Two branches with different subspaces is a conditional space, and the tree
    already is one; nothing new had to be invented to say it.
    """

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    seen = [sample_parameters(study.ask(), {"bound": BOUNDED}) for _ in range(40)]
    by_branch = {point["bound.kind"]: point for point in seen}

    assert set(by_branch) == {"ob", "none"}
    assert "bound.ob.kappa" in by_branch["ob"]
    assert "bound.ob.kappa" not in by_branch["none"]


def test_a_group_that_chooses_nothing_is_only_grouping() -> None:
    trial = optuna.trial.FixedTrial({"actor.lr": 0.1, "critic.lr": 0.2})

    sampled = sample_parameters(
        trial,
        {
            "actor": {"lr": {"type": "choice", "values": [0.1]}},
            "critic": {"lr": {"type": "choice", "values": [0.2]}},
        },
    )

    assert sampled == {"actor.lr": 0.1, "critic.lr": 0.2}


@pytest.mark.parametrize("chosen", ("ob", "none"))
def test_the_same_name_under_two_groups_is_two_parameters(chosen: str) -> None:
    trial = optuna.trial.FixedTrial(
        {
            "actor.kind": chosen,
            "actor.ob.kappa": 2.0,
            "critic.kind": "ob",
            "critic.ob.kappa": 3.0,
        }
    )

    sampled = sample_parameters(trial, {"actor": BOUNDED, "critic": BOUNDED})

    assert sampled["critic.ob.kappa"] == 3.0
    assert ("actor.ob.kappa" in sampled) is (chosen == "ob")
