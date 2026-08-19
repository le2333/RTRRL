"""Every image-side consumer projects the same serialized version-11 run.

Version 11 adds the two blocks a checkpoint fork needs: ``checkpoint``, which
says how often a run files its whole state, and ``fork``, which says which
filed state a run continues from. Both are optional, so a version-10 document
means the same thing here; what makes it a new version is that a version-10
image would reject a document carrying either, and a branch launched at one
would silently be a fresh run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from deployment.catalog import build_catalog, write_catalog
from deployment.contract import CONTRACT_VERSION, Catalog
from entries._contract import RunSpec
from worker.envelope import WorkerEnvelope

FIXTURES = Path(__file__).resolve().parents[4] / "tests" / "contracts" / "v11"


def read_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_one_serialized_run_has_worker_and_entry_projections() -> None:
    payload = read_json("run.json")

    worker = WorkerEnvelope.model_validate(payload)
    entry = RunSpec.model_validate(payload)

    assert worker.contract == entry.contract == CONTRACT_VERSION == 11
    assert worker.identity.model_dump() == entry.identity.model_dump()
    assert worker.artifacts.model_dump() == entry.artifacts.model_dump()
    assert worker.algorithm == payload["algorithm"]
    assert entry.algorithm.parameters == payload["algorithm"]["parameters"]


def test_each_training_scope_states_its_interval_in_its_own_unit() -> None:
    """A scope's schedule cannot be written in another scope's unit."""

    training = RunSpec.model_validate(read_json("run.json")).logging.aim.training

    assert training is not None
    assert training.step is not None and training.step.every_steps == 50
    assert training.episode is not None and training.episode.every_episodes == 20
    assert training.window is not None and training.window.length_steps == 25


def test_worker_does_not_interpret_entry_owned_payload() -> None:
    payload = read_json("run.json")
    payload["algorithm"]["future_algorithm_field"] = {"anything": True}

    worker = WorkerEnvelope.model_validate(payload)

    assert worker.algorithm["future_algorithm_field"] == {"anything": True}
    with pytest.raises(ValidationError):
        RunSpec.model_validate(payload)


def test_run_contains_one_artifact_root_and_no_score_policy() -> None:
    payload = read_json("run.json")

    assert payload["artifacts"] == {
        "root": "s3://artifacts/trainer/stream-ac/launch/run-t0"
    }
    assert "score" not in payload
    assert "s3" not in payload["logging"]["rerun"]


def test_an_environment_with_one_implementation_names_no_backend() -> None:
    """Brax chooses a physics backend; Gymnax has none to choose.

    The field stays required reading for the namespaces that mean it, so this
    is null rather than absent from the document.
    """

    payload = read_json("run.json")
    payload["algorithm"]["environment"] |= {"id": "gymnax::CartPole-v1"}
    payload["algorithm"]["environment"]["backend"] = None

    assert RunSpec.model_validate(payload).algorithm.environment.backend is None


def test_entry_owns_schedule_and_graph_width_validation() -> None:
    payload = read_json("run.json")
    payload["algorithm"]["num_envs"] = 3

    WorkerEnvelope.model_validate(payload)
    with pytest.raises(ValidationError, match="whole environment steps"):
        RunSpec.model_validate(payload)


def test_a_branch_names_its_parent_and_worker_hands_it_through() -> None:
    """Worker carries the fork block without acquiring an opinion about it.

    Which object a branch restores, and what it does not take from it, is the
    Entry's to interpret; Worker's part is the artifact root it already fills.
    """

    payload = read_json("fork.json")

    worker = WorkerEnvelope.model_validate(payload)
    entry = RunSpec.model_validate(payload)

    assert worker.fork == payload["fork"]
    assert entry.fork is not None
    assert entry.fork.from_steps == 700000
    assert entry.fork.replacing == ("core.rule",)
    # The budget is the parent's boundary plus this branch's own 50k, on the
    # parent's step axis, so the two curves are read against the same numbers.
    assert entry.training.total_steps - entry.fork.from_steps == 50000


def test_a_checkpoint_that_is_not_a_measurement_is_refused() -> None:
    """Both schedules answer to the evaluation's; neither rounds to it."""

    payload = read_json("run.json")
    payload["checkpoint"]["every_steps"] = 150

    with pytest.raises(ValidationError, match="whole evaluation intervals"):
        RunSpec.model_validate(payload)


def test_a_branch_from_between_two_measurements_is_refused() -> None:
    payload = read_json("fork.json")
    payload["fork"]["from_steps"] = 705000

    with pytest.raises(ValidationError, match="evaluation boundary"):
        RunSpec.model_validate(payload)


def test_a_branch_with_no_budget_beyond_its_parent_is_refused() -> None:
    payload = read_json("fork.json")
    payload["training"]["total_steps"] = payload["fork"]["from_steps"]

    with pytest.raises(ValidationError, match="must exceed the boundary"):
        RunSpec.model_validate(payload)


def test_manifest_names_serialized_run_documents() -> None:
    assert read_json("manifest.json") == {
        "runs": ["s3://artifacts/trainer/configs/stream-ac-t0.json"]
    }


def test_image_catalog_uses_the_deployment_contract(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    written = write_catalog(path)
    parsed = Catalog.model_validate_json(path.read_text(encoding="utf-8"))

    assert parsed == written == build_catalog()
    assert parsed.contract == CONTRACT_VERSION
    assert set(parsed.entries) == {"r2d2", "rtrrl", "stream_ac"}
    assert parsed.entries["rtrrl"].command == ("python", "-m", "entries.rtrrl")
