"""The whole chain for R2D2, from what the image declares to an agent stepping.

    catalog -> control-plane resolver and sampler -> RunSpec -> assembly

The same chain ``tests/test_round_trip.py`` walks for StreamAC. R2D2 is worth
walking separately because it is the first entry whose graph shape depends on
a deployment field rather than only on its own parameters: a full-episode
replay window is the run document's ``episode_length``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import pytest
import yaml
from trainer_infra.adapter import resolve_parameter_ranges
from trainer_infra.experiment import ExperimentRunner
from trainer_infra.hpo import sample_parameters

from deployment.catalog import build_catalog
from deployment.contract import CONTRACT_VERSION
from entries import r2d2
from entries._contract import RunSpec
from memorax.parameters import KIND, expand, flatten
from tests.support.builders import assemble_r2d2

optuna = pytest.importorskip("optuna")

pytestmark = pytest.mark.integration

TEMPLATE = Path(__file__).resolve().parents[3] / "experiments" / "r2d2 template.yaml"
LAUNCH = "20260815-120000"

# What this test steps. The rest of a budget is the runtime's.
STREAMS = 1


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


def test_the_image_catalog_discovers_r2d2_beside_the_other_entries(catalog):
    assert catalog["contract"] == CONTRACT_VERSION
    assert {"r2d2", "rtrrl", "stream_ac"} <= set(catalog["entries"])
    assert catalog["entries"]["r2d2"]["command"] == ["python", "-m", "entries.r2d2"]


def test_the_catalog_ships_exactly_what_the_entry_declares(catalog):
    from memorax.parameters import describe

    assert catalog["entries"]["r2d2"]["parameters"] == describe(r2d2.PARAMETERS)
    assert tuple(catalog["entries"]["r2d2"]["metrics"]) == tuple(r2d2.METRICS)


def test_every_pin_the_experiment_writes_is_a_name_the_entry_declares(experiment):
    declared = set(flatten(r2d2.PARAMETERS))

    pinned = set(flatten_space(experiment["space"]))

    assert pinned <= declared, f"the experiment pins {sorted(pinned - declared)}"


def flatten_space(space, prefix="") -> dict:
    """The experiment's overrides by the dotted path the catalog knows them by."""

    found = {}
    for name, node in space.items():
        key = f"{prefix}{name}"
        if isinstance(node, dict):
            found |= flatten_space(node, f"{key}.")
        else:
            found[key] = node
    return found


def test_every_configuration_a_round_produces_is_one_this_side_accepts(configurations):
    assert configurations
    for configuration in configurations:
        assert RunSpec.model_validate(configuration).contract == CONTRACT_VERSION


def test_the_sampler_draws_only_names_the_entry_declares(configurations):
    declared = set(flatten(r2d2.PARAMETERS))

    for configuration in configurations:
        unknown = sorted(set(configuration["algorithm"]["parameters"]) - declared)

        assert not unknown, f"the sampler produced {unknown}"


def test_both_sides_walk_the_tree_to_the_same_names(experiment, catalog):
    ranges = resolve_parameter_ranges(
        catalog["entries"]["r2d2"]["parameters"], experiment["space"]
    )
    drawn = sample_parameters(optuna.create_study().ask(), ranges)

    assert set(expand(r2d2.PARAMETERS, drawn)) == set(drawn)


def test_each_configuration_is_the_whole_of_one_walk_and_no_more(configurations):
    for configuration in configurations:
        params = configuration["algorithm"]["parameters"]

        assert set(expand(r2d2.PARAMETERS, params)) == set(params)


def test_a_branch_the_experiment_did_not_choose_is_absent(configurations):
    """Not filled in with something that would read as chosen."""

    params = configurations[0]["algorithm"]["parameters"]

    assert params[f"backbone.{KIND}"] == "lru"
    assert params[f"learning.{KIND}"] == "tbptt"
    assert not [name for name in params if name.startswith("backbone.rtu.")]
    assert not [name for name in params if name.startswith("learning.full_bptt.")]
    assert not [name for name in params if ".signed_hyperbolic." in name]


def test_the_resolved_manifest_assembles_and_steps(configurations):
    """The end of the chain: the names line up where they are finally read."""

    config = RunSpec.model_validate(configurations[0])
    program = assemble_r2d2(
        config.algorithm.parameters,
        config.algorithm.environment,
        num_envs=STREAMS,
    )
    state = program.init(jax.random.key(config.training.seed))
    _, metrics = program.train(jax.random.key(1), state, 2 * STREAMS)

    assert metrics.interaction.reward.shape == (2, STREAMS)


def test_the_score_names_a_metric_this_entry_reports(experiment):
    """Otherwise the run finishes and the control plane has nothing to read."""

    assert experiment["score"]["metric"] in r2d2.METRICS
