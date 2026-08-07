"""The whole chain, from what the image declares to an agent taking a step.

    catalog  ->  the control plane's resolver and sampler  ->  RunConfig  ->  build

Neither side imports the other in production. They answer to a shape written
down in docs/contract.md, and two copies of one shape stay equal only if
something runs them against each other. The other half of that lives in
``infra/tests/test_round_trip.py``, which validates what the control plane
produces without needing an array library. This half is the one that does: it
ends where the names are actually read, in the entry's ``build``.

A name renamed on either side fails here, which is the only place it can fail
before a job has been dispatched.
"""

from __future__ import annotations

from pathlib import Path

import jax
import pytest
import yaml
from trainer_infra.adapter import resolve_parameter_ranges
from trainer_infra.experiment import ExperimentRunner
from trainer_infra.hpo import sample_parameters

from entries import stream_ac
from memorax.parameters import KIND, expand, flatten
from runner.catalog import build_catalog
from worker.contract import CONTRACT_VERSION, RunConfig

optuna = pytest.importorskip("optuna")

TEMPLATE = (
    Path(__file__).resolve().parents[2] / "experiments" / "streamac template.yaml"
)
LAUNCH = "20260807-120000"

# What the entry reads out of the training block. The rest of a budget is the
# runtime's, and this test never runs one.
STREAMS = 2


@pytest.fixture(scope="module")
def experiment() -> dict:
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalog() -> dict:
    """The real one, built the way the image build builds it."""

    return build_catalog().model_dump(mode="json")


@pytest.fixture(scope="module")
def configurations(experiment, catalog, tmp_path_factory) -> tuple[dict, ...]:
    return ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path_factory.mktemp("study") / "study.db",
        launch_id=LAUNCH,
    ).next_round()


def test_the_control_plane_accepts_the_catalog_this_image_would_ship(catalog):
    assert catalog["contract"] == CONTRACT_VERSION
    assert "stream_ac" in catalog["entries"]


def test_every_configuration_a_round_produces_is_one_this_side_accepts(configurations):
    assert configurations
    for configuration in configurations:
        assert RunConfig.model_validate(configuration).contract == CONTRACT_VERSION


def test_the_sampler_draws_only_names_the_entry_declares(configurations, catalog):
    declared = set(flatten(stream_ac.PARAMETERS))

    for configuration in configurations:
        unknown = sorted(set(configuration["params"]) - declared)

        assert not unknown, f"the sampler produced {unknown}"


def test_both_sides_walk_the_tree_to_the_same_names(experiment, catalog):
    """The two copies of the walk agree on which parameters a run has.

    One is in ``memorax.parameters``, reading a chosen branch out of a
    mapping; the other is in ``trainer_infra.hpo``, drawing that branch a line
    above it. They are written separately on purpose -- the two sides share no
    package -- so nothing but this keeps them the same walk.
    """

    ranges = resolve_parameter_ranges(
        catalog["entries"]["stream_ac"]["parameters"], experiment["space"]
    )
    drawn = sample_parameters(optuna.create_study().ask(), ranges)

    assert set(expand(stream_ac.PARAMETERS, drawn)) == set(drawn)


def test_each_configuration_is_the_whole_of_one_walk_and_no_more(configurations):
    """Per configuration, because two trials need not carry the same names.

    A searched choice of component means the parameter set is the trial's own:
    the one that drew ``sparse`` carries a sparsity and the one that drew
    ``lecun`` does not. That is the conditional space working, so the invariant
    is that each configuration reproduces itself, not that they match.
    """

    for configuration in configurations:
        params = configuration["params"]

        assert set(expand(stream_ac.PARAMETERS, params)) == set(params)


def test_a_branch_the_experiment_did_not_choose_is_absent(configurations):
    """Not filled in with something that would read as chosen."""

    params = configurations[0]["params"]

    assert params[f"backbone.{KIND}"] == "rtu"
    assert not [name for name in params if name.startswith("backbone.mlp.")]
    assert not [name for name in params if ".adam." in name]


def test_the_entry_builds_and_steps_on_what_the_sampler_drew(configurations):
    """The end of the chain: the names line up where they are finally read."""

    config = RunConfig.model_validate(configurations[0])
    agent = stream_ac.build(
        config.params,
        config.environment,
        # The only field build reads here is the width, and two streams answer
        # whether the names line up as well as sixteen do, faster.
        config.training.model_copy(update={"num_envs": STREAMS}),
    )
    state = agent.init(jax.random.key(config.environment.seed))
    _, metrics = agent.train(jax.random.key(1), state, 2 * STREAMS)

    assert metrics.interaction.reward.shape == (2, STREAMS)


def test_the_score_names_a_metric_this_entry_reports(configurations):
    """Otherwise the run finishes and the control plane has nothing to read."""

    assert configurations[0]["score"]["metric"] in stream_ac.METRICS
