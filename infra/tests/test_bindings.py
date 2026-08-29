"""One drawn value at several of a run's parameters, and what that must not be.

Experiment 2's comparisons hold a setting equal across the actor, the critic and
the shared torso. Three leaves with one domain do not say that -- the sampler
draws three numbers and the study searches three dimensions -- so the experiment
being run stops being the one the file describes.

What is asserted here is the whole of the claim and no more of it:

- the study searches *one* dimension and every destination carries the same
  ordinary number, in the run document a worker will read;
- what is not bound is still drawn apart, learning rates above all;
- a binding is configuration. Four intentional rules given one ``clip`` are
  still four rules with four states, which the algorithm side proves and this
  side simply never gets in the way of;
- a study can be reopened and still knows what its one dimension stood for;
- and everything a binding can get wrong is refused while it is still a file,
  before a round of containers is paid for.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import optuna
import pytest
import yaml
from test_cli import WORKER

from trainer_infra import BindingError, ExperimentError, ExperimentRunner
from trainer_infra.cli import main
from trainer_infra.rounds import partition

LAUNCH = "20260829-120000"

BETA2 = [0.9, 0.99, 0.999, 0.9999]
ADAM_B2 = (
    "actor.optimizer.adam.b2",
    "critic.optimizer.adam.b2",
    "torso.optimizer.adam.b2",
)
# The four rules an output aggregation runs: one at each head, and the torso's
# two branches, which are whole intentional updates rather than settings on one.
IU_CLIP = (
    "actor.optimizer.iu.clip",
    "critic.optimizer.iu.clip",
    "torso.optimizer.output_iu.actor.clip",
    "torso.optimizer.output_iu.critic.clip",
)


def runner(experiment: Any, catalog: Any, tmp_path: Path, name: str = "study") -> ExperimentRunner:
    return ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path / f"{name}.db",
        launch_id=LAUNCH,
    )


def bind(experiment: Any, name: str, domain: Any, paths: Any) -> Any:
    experiment.setdefault("bindings", {})[name] = {"domain": domain, "paths": list(paths)}
    return experiment


def one_trial(experiment: Any) -> Any:
    """A round of one, for the cases whose subject is a single drawn point."""

    experiment["hpo"]["trials_per_round"] = 1
    return experiment


def intentional(experiment: Any) -> Any:
    """The same experiment with every block stepped by an intentional rule."""

    experiment["space"]["actor"]["optimizer"]["kind"] = ["iu"]
    experiment["space"]["critic"]["optimizer"]["kind"] = ["iu"]
    experiment["space"]["torso"]["optimizer"]["kind"] = ["output_iu"]
    return experiment


def parameters(runs: Any) -> dict[str, Any]:
    return runs[0]["algorithm"]["parameters"]


def study_of(built: ExperimentRunner) -> optuna.Study:
    return built.hpo._open()


# ------------------------------------------------------------------- the claim
def test_one_variable_reaches_every_path_it_names(
    blocks: Any, catalog: Any, tmp_path: Path
) -> None:
    """The acceptance case: one Adam beta, three optimizers, one dimension.

    Both halves matter. The run document has to carry an ordinary number at each
    of the three paths, because the worker knows nothing about variables; and the
    study has to hold one parameter, because a study that recorded three would go
    on modelling a space three dimensions wider than the experiment.
    """

    built = runner(one_trial(bind(blocks, "shared_beta2", BETA2, ADAM_B2)), catalog, tmp_path)
    drawn = parameters(built.next_round())

    values = {drawn[path] for path in ADAM_B2}
    assert len(values) == 1
    assert values <= set(BETA2)

    (recorded,) = study_of(built).trials
    assert recorded.params["shared_beta2"] == drawn[ADAM_B2[0]]
    assert not any(path in recorded.params for path in ADAM_B2)


def test_a_binding_costs_one_dimension_where_three_leaves_cost_three(
    blocks: Any, catalog: Any, tmp_path: Path
) -> None:
    """What the experiment is for. Counted against the same file unbound."""

    unbound = runner(one_trial(copy.deepcopy(blocks)), catalog, tmp_path, "unbound")
    bound = runner(
        one_trial(bind(blocks, "shared_beta2", BETA2, ADAM_B2)), catalog, tmp_path, "bound"
    )

    unbound.next_round()
    bound.next_round()

    (loose,) = study_of(unbound).trials
    (tied,) = study_of(bound).trials
    assert len(loose.params) - len(tied.params) == len(ADAM_B2) - 1


def test_what_is_not_bound_is_still_drawn_apart(
    blocks: Any, catalog: Any, tmp_path: Path
) -> None:
    """Sharing one setting does not tie the block down.

    The learning rates are the case the experiment cares about: the three blocks
    are compared under one beta and their own rates, and a facility that tied a
    whole optimizer would be a different comparison.
    """

    blocks["hpo"]["trials_per_round"] = 4
    built = runner(bind(blocks, "shared_beta2", BETA2, ADAM_B2), catalog, tmp_path)

    for run in built.next_round():
        drawn = run["algorithm"]["parameters"]
        rates = {
            drawn[f"{block}.optimizer.adam.lr"] for block in ("actor", "critic", "torso")
        }
        assert len(rates) == 3, drawn


def test_one_setting_reaches_four_rules_that_stay_four_rules(
    blocks: Any, catalog: Any, tmp_path: Path
) -> None:
    """IU-output: two heads and the torso's two branches, given one ``clip``.

    Each destination is a rule with its own state, and each keeps its own
    ``eta``: the published actor-critic intends a different fraction for a policy
    than for a value function, and a binding that reached the step sizes too
    would be expressing something the experiment does not claim.
    """

    experiment = one_trial(bind(intentional(blocks), "shared_clip", [10.0, 20.0], IU_CLIP))
    built = runner(experiment, catalog, tmp_path)
    drawn = parameters(built.next_round())

    assert len({drawn[path] for path in IU_CLIP}) == 1
    etas = [path.replace(".clip", ".eta") for path in IU_CLIP]
    (recorded,) = study_of(built).trials
    assert all(eta in recorded.params for eta in etas)
    assert recorded.params["shared_clip"] == drawn[IU_CLIP[0]]


def test_a_reopened_study_still_knows_what_its_dimension_stood_for(
    blocks: Any, catalog: Any, tmp_path: Path
) -> None:
    """``settle`` reads trials a dead controller drew, so it must read them back.

    Optuna stored the variable, which is the dimension it searched and the only
    thing a resumed study can be told. The runs being settled are the runs that
    were submitted, and those carried the destinations -- so what comes back out
    of storage has to be written out to them again, and no per-destination
    parameter may appear in the study as a result.
    """

    experiment = one_trial(bind(blocks, "shared_beta2", BETA2, ADAM_B2))
    first = runner(experiment, catalog, tmp_path)
    submitted = parameters(first.next_round())

    resumed = runner(copy.deepcopy(experiment), catalog, tmp_path)
    (open_trial,) = resumed.hpo.running()

    assert "shared_beta2" not in open_trial.parameters
    for path in ADAM_B2:
        assert open_trial.parameters[path] == submitted[path]
    assert set(study_of(resumed).trials[0].params) == set(
        study_of(first).trials[0].params
    )


def test_the_binding_is_archived_beside_the_study_it_shaped(
    blocks: Any, catalog: Any, tmp_path: Path
) -> None:
    """What the variable meant is not recoverable from what Optuna stores."""

    built = runner(bind(blocks, "shared_beta2", BETA2, ADAM_B2), catalog, tmp_path)
    built.next_round()

    assert study_of(built).user_attrs["bindings"] == [
        {
            "name": "shared_beta2",
            "domain": {"type": "choice", "values": BETA2},
            "paths": list(ADAM_B2),
        }
    ]


def test_a_file_without_bindings_is_the_file_it_always_was(
    blocks: Any, catalog: Any, tmp_path: Path
) -> None:
    plain = runner(copy.deepcopy(blocks), catalog, tmp_path, "plain")
    empty = runner({**copy.deepcopy(blocks), "bindings": {}}, catalog, tmp_path, "empty")

    assert plain.bindings == empty.bindings == ()
    assert parameters(plain.next_round()) == parameters(empty.next_round())


# ------------------------------------------------------------ the two channels
def test_the_grouped_channel_resolves_the_same_binding(
    blocks: Any, catalog: Any, tmp_path: Path
) -> None:
    """One vmapped job and a round of separate ones draw the same point.

    Grouping happens over finished configurations, so a binding has to be
    resolved before it -- which is what makes the two channels' parameters the
    same object to compare rather than two spellings to reconcile.
    """

    blocks["hpo"]["trials_per_round"] = 2
    single = bind(copy.deepcopy(blocks), "shared_beta2", BETA2, ADAM_B2)
    many = copy.deepcopy(single)
    many["entry"] = "blocks_ensemble"

    ungrouped = runner(single, catalog, tmp_path, "ungrouped")
    grouped = runner(many, catalog, tmp_path, "grouped")
    assert grouped.grouped and not ungrouped.grouped

    apart = ungrouped.next_round()
    together = grouped.next_round()
    assert [run["algorithm"]["parameters"] for run in apart] == [
        run["algorithm"]["parameters"] for run in together
    ]

    # And the round still packs into one graph: a shared beta is not a width or
    # a branch, so it never separates two members of a group.
    uris = tuple(f"s3://exchange/{index}" for index in range(len(together)))
    assert len(partition(together, uris, grouped.static_parameters)) == 1



def test_a_launch_reports_what_its_one_dimension_stood_for(
    blocks: Any, catalog: Any, tmp_path: Path, capsys: Any
) -> None:
    """The whole way through, from a file on disk to the report and the runs.

    The report is what an operator freezes a configuration out of, and a trial's
    parameters name the variable rather than the paths -- so a report that did
    not also say which paths those were could not be used for that. The run
    documents the worker actually read are checked here too, since they are the
    only place the destinations were ever going to appear as plain numbers.
    """

    blocks["storage"] = (tmp_path / "artifacts").resolve().as_uri()
    blocks["score"] = {
        "metric": "objective",
        "window_steps": [0, 10],
        "reduce": "last",
        "non_finite": "worst",
        "direction": "maximize",
    }
    catalog["entries"]["blocks"]["metrics"] = ["objective"]
    one_trial(bind(blocks, "shared_beta2", BETA2, ADAM_B2))

    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER, encoding="utf-8")

    assert (
        main(
            [
                "run",
                str(experiment_path),
                "--backend",
                "local",
                "--catalog",
                str(catalog_path),
                "--database",
                str(tmp_path / "study.db"),
                "--launch-id",
                LAUNCH,
                "--exchange",
                str(tmp_path / "exchange"),
                "--workspace",
                str(tmp_path / "scratch"),
                "--worker-command",
                sys.executable,
                str(worker),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["bindings"] == [
        {
            "name": "shared_beta2",
            "domain": {"type": "choice", "values": BETA2},
            "paths": list(ADAM_B2),
        }
    ]
    (trial,) = payload["trials"]
    assert "shared_beta2" in trial["parameters"]
    assert not any(path in trial["parameters"] for path in ADAM_B2)

    written = [
        json.loads(document.read_text(encoding="utf-8"))
        for document in sorted((tmp_path / "exchange").rglob("*.json"))
    ]
    published = [document for document in written if "algorithm" in document]
    assert published
    for run in published:
        drawn = run["algorithm"]["parameters"]
        assert "shared_beta2" not in drawn
        assert {drawn[path] for path in ADAM_B2} == {trial["parameters"]["shared_beta2"]}


# ---------------------------------------------------------------- the refusals
@pytest.mark.parametrize(
    ("name", "domain", "paths", "message"),
    [
        pytest.param(
            "shared_beta2",
            BETA2,
            ["actor.optimizer.adam.b2", "actor.optimizer.adam.b3"],
            "declares no parameter",
            id="a path the image does not declare",
        ),
        pytest.param(
            "shared_beta2",
            [0.9, 4.0],
            ADAM_B2,
            "outside the valid domain",
            id="a value no destination accepts",
        ),
        pytest.param(
            "shared_kappa",
            [0.5, 0.9],
            ["actor.optimizer.iu.eta", "actor.optimizer.adam.b2"],
            "does not select",
            id="a branch this experiment does not select",
        ),
        pytest.param(
            "shared_kind",
            ["adam"],
            ["actor.optimizer.kind", "critic.optimizer.kind"],
            "select which parameters exist",
            id="a structural choice",
        ),
        pytest.param(
            "gamma",
            [0.9, 0.99],
            ADAM_B2,
            "already names in the image",
            id="a name the tree already uses",
        ),
        pytest.param(
            "actor.shared",
            BETA2,
            ADAM_B2,
            "contain a dot",
            id="a name shaped like a destination",
        ),
        pytest.param(
            "shared_beta2",
            BETA2,
            ["actor.optimizer.adam.b2"],
            "belongs under space",
            id="one destination, which is not a shared variable",
        ),
    ],
)
def test_a_binding_that_cannot_be_written_is_refused(
    blocks: Any,
    catalog: Any,
    tmp_path: Path,
    name: str,
    domain: Any,
    paths: Any,
    message: str,
) -> None:
    with pytest.raises(BindingError, match=message):
        runner(bind(blocks, name, domain, paths), catalog, tmp_path)


def test_two_variables_cannot_both_write_one_parameter(
    blocks: Any, catalog: Any, tmp_path: Path
) -> None:
    bind(blocks, "shared_beta2", BETA2, ADAM_B2)
    bind(blocks, "also_beta2", BETA2, ADAM_B2[:2])

    with pytest.raises(BindingError, match="more than one value is written into"):
        runner(blocks, catalog, tmp_path)


def test_a_bound_parameter_cannot_also_be_pinned_under_space(
    blocks: Any, catalog: Any, tmp_path: Path
) -> None:
    """Two authors of one leaf, and no order between them worth inventing."""

    blocks["space"]["actor"]["optimizer"]["adam"] = {"b2": [0.999]}

    with pytest.raises(BindingError, match="also pinned under space"):
        runner(bind(blocks, "shared_beta2", BETA2, ADAM_B2), catalog, tmp_path)


def test_a_formal_launch_cannot_leave_a_variable_still_offering_a_choice(
    blocks: Any, catalog: Any, tmp_path: Path
) -> None:
    """A formal launch runs what it froze, and a domain is a choice not made."""

    blocks["environment"]["seeds"] = [10, 11]
    blocks["selection"] = {"study": "s", "trial": 3, "tuning_seeds": [0]}
    bind(blocks, "shared_beta2", BETA2, ADAM_B2)

    with pytest.raises(ExperimentError, match="shared_beta2"):
        runner(blocks, catalog, tmp_path)

    blocks["bindings"]["shared_beta2"]["domain"] = [0.999]
    assert runner(blocks, catalog, tmp_path).role == "formal"


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        pytest.param({"paths": list(ADAM_B2)}, "does not say", id="no domain"),
        pytest.param({"domain": BETA2}, "does not say", id="no paths"),
        pytest.param(
            {"domain": BETA2, "paths": list(ADAM_B2), "note": "why"},
            "has no field for",
            id="a field a binding does not have",
        ),
        pytest.param(
            {"domain": BETA2, "paths": "actor.optimizer.adam.b2"},
            "must list the paths",
            id="one path written as a string",
        ),
    ],
)
def test_a_binding_written_wrong_is_refused_by_its_shape(
    blocks: Any, catalog: Any, tmp_path: Path, declaration: Any, message: str
) -> None:
    blocks["bindings"] = {"shared_beta2": declaration}

    with pytest.raises(BindingError, match=message):
        runner(blocks, catalog, tmp_path)


def test_a_binding_is_refused_before_anything_is_submitted(
    blocks: Any, catalog: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    """Preflight, which is the only place this can be caught for free.

    A binding into a path the image does not declare would otherwise be found by
    a worker: a round of containers that each start, read a configuration missing
    the value they were promised, and die on the same line. So the file is
    refused before the session that would submit them is even opened.
    """

    bind(blocks, "shared_beta2", BETA2, ["actor.optimizer.adam.b2", "nowhere.at.all"])
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    def refuse() -> Any:
        raise AssertionError("a rejected binding reached AWS")

    monkeypatch.setattr("trainer_infra.cli._batch_session", refuse)

    with pytest.raises(BindingError, match="declares no parameter"):
        main(
            [
                "run",
                str(experiment_path),
                "--backend",
                "batch",
                "--catalog",
                str(catalog_path),
                "--database",
                str(tmp_path / "study.db"),
            ]
        )
