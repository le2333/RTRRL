"""What this side produces is what the other side accepts.

Neither side imports the other in production: they answer to a shape written
down in docs/contract.md, and to CONTRACT_VERSION. Two copies of one shape stay
equal only if something compares them, and this is that something -- it runs
the real sampler and then the real validator, so a field renamed on one side
fails here rather than on a job that has already started.

The worker's contract module is imported from source rather than installed. It
is pydantic and standard library, nothing else: memorax.parameters draws no
array library and memorax's own __init__ imports lazily, so this costs the
control plane nothing it would have to carry.

This is the upper half. It ends at validation, with the parameters supplied by
hand, because the catalog's parameter tree is mid-refactor and the adapter is
already written for its finished shape. The lower half -- the real catalog
sampled and the entry built from it -- is docs/roadmap.md R1d.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import EXPERIMENT
from pydantic import BaseModel
from worker.contract import CONTRACT_VERSION, RunConfig

from trainer_infra.experiment import (
    PASSED_THROUGH,
    REQUIRED,
    ExperimentError,
    ExperimentRunner,
    _absent,
)

TEMPLATE = Path(__file__).resolve().parents[2] / "experiments" / "streamac template.yaml"


def test_the_fixture_catalog_still_claims_the_contract_the_worker_implements(
    catalog: Any,
) -> None:
    """Otherwise the round trip below would be run against a shape nobody ships."""

    assert catalog["contract"] == CONTRACT_VERSION


def test_every_configuration_a_round_produces_is_one_the_worker_accepts(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    configurations = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path / "study.db",
        launch_id="20260807-120000",
    ).next_round()

    assert configurations
    for configuration in configurations:
        validated = RunConfig.model_validate(configuration)

        # extra="forbid" already refuses a field the worker does not know; this
        # is the other direction, that none of them arrived as a default.
        assert validated.model_dump(exclude_unset=True).keys() == configuration.keys()
        assert validated.contract == CONTRACT_VERSION


def test_a_configuration_carries_what_the_assembler_reads(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    """The three the entry's build touches: params, the environment, the width.

    Everything else in a run configuration is budget or bookkeeping. These are
    what decide the shape of the graph, so they are the ones whose absence is
    not a smaller run but no run at all.
    """

    configuration = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path / "study.db",
    ).next_round()[0]
    validated = RunConfig.model_validate(configuration)

    assert validated.params
    assert validated.environment.id and validated.environment.backend
    assert validated.environment.episode_length > 0
    assert validated.training.num_envs > 0


def test_a_configuration_says_where_its_own_result_goes(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    """The worker writes a score somewhere; only this side knows where."""

    validated = RunConfig.model_validate(
        ExperimentRunner(
            experiment=experiment,
            catalog=catalog,
            database=tmp_path / "study.db",
        ).next_round()[0]
    )

    assert validated.score.s3.startswith(experiment["storage"])
    assert validated.score.s3.endswith(f"{validated.run_id}/score.json")


def test_the_shipped_template_is_a_file_this_side_can_run(catalog: Any) -> None:
    """The artifact a person edits, checked as the artifact rather than a copy."""

    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))

    assert sorted(_absent(template)) == []
    assert "@sha256:" in template["image"]
    assert template["entry"] in catalog["entries"]


def test_a_template_whose_budget_is_ragged_is_refused_on_arrival(
    experiment: Any, catalog: Any, tmp_path: Path
) -> None:
    """This side counts nothing; the worker refuses a budget it cannot divide.

    Which is the division of labour working: infra checks that the file said
    something, the worker checks that what it said means something.
    """

    experiment["training"]["epoch_steps"] = 99

    configuration = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=tmp_path / "study.db",
    ).next_round()[0]

    with pytest.raises(ValueError, match="whole epochs"):
        RunConfig.model_validate(configuration)


def test_the_two_sides_pass_through_the_same_blocks() -> None:
    """Read off both sides rather than listed here, so a new block cannot hide.

    A test that names the blocks itself is blind to exactly the drift it is
    for: a block added on one side is absent from the list and so filtered out
    of both halves of the comparison.
    """

    nested = {
        name
        for name, field in RunConfig.model_fields.items()
        if isinstance(field.annotation, type)
        and issubclass(field.annotation, BaseModel)
    }

    assert set(PASSED_THROUGH) == nested


@pytest.mark.parametrize("block", PASSED_THROUGH)
def test_a_field_the_worker_requires_is_one_the_experiment_file_must_say(
    block: str,
) -> None:
    """Except the ones this side fills in, which an experiment cannot know.

    A field the worker requires and the file need not mention is one a run
    discovers is missing after it has been dispatched, which is a round of jobs
    that each start, read the same absence, and die.
    """

    derived = {"score": {"s3"}}.get(block, set())
    model = RunConfig.model_fields[block].annotation
    required = {
        name for name, field in model.model_fields.items() if field.is_required()
    }

    assert required - derived <= set(REQUIRED[block])


def test_an_empty_file_is_refused_before_anything_is_asked_of_the_catalog() -> None:
    with pytest.raises(ExperimentError, match="does not say"):
        ExperimentRunner(experiment={}, catalog={}, database=Path("unused.db"))


def test_the_experiment_fixture_and_the_shipped_template_agree_on_their_blocks() -> None:
    """A fixture that drifts from the template stops testing the template."""

    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))

    assert set(EXPERIMENT) <= set(template)
