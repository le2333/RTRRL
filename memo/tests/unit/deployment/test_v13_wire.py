"""Every image-side consumer projects the same serialized version-13 run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from deployment.catalog import build_catalog, write_catalog
from deployment.contract import CONTRACT_VERSION, Catalog
from entries._contract import RunSpec
from worker.envelope import WorkerEnvelope

FIXTURES = Path(__file__).resolve().parents[4] / "tests" / "contracts" / "v13"


def read_json(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_one_serialized_run_has_worker_and_entry_projections() -> None:
    payload = read_json("run.json")

    worker = WorkerEnvelope.model_validate(payload)
    entry = RunSpec.model_validate(payload)

    assert worker.contract == entry.contract == CONTRACT_VERSION == 13
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


def test_environment_constructor_kwargs_survive_the_entry_contract() -> None:
    payload = read_json("run.json")
    payload["algorithm"]["environment"]["kwargs"] = {"mapping_seed": 3}

    environment = RunSpec.model_validate(payload).algorithm.environment

    assert environment.kwargs == {"mapping_seed": 3}


def test_entry_owns_schedule_and_graph_width_validation() -> None:
    payload = read_json("run.json")
    payload["algorithm"]["num_envs"] = 3

    WorkerEnvelope.model_validate(payload)
    with pytest.raises(ValidationError, match="whole environment steps"):
        RunSpec.model_validate(payload)


def test_manifest_names_serialized_run_documents() -> None:
    assert read_json("manifest.json") == {
        "runs": ["s3://artifacts/trainer/configs/stream-ac-t0.json"],
        # A group is several configurations handed to one entry at once, so a
        # trial's seeds are computed as one graph. It is a list of lists rather
        # than a flat list because a manifest may carry more than one group, and
        # which members belong together is the whole of what it says.
        "groups": [
            [
                "s3://artifacts/trainer/configs/drqn-t0-s0.json",
                "s3://artifacts/trainer/configs/drqn-t0-s1.json",
            ]
        ],
    }


def test_image_catalog_uses_the_deployment_contract(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    written = write_catalog(path)
    parsed = Catalog.model_validate_json(path.read_text(encoding="utf-8"))

    assert parsed == written == build_catalog()
    assert parsed.contract == CONTRACT_VERSION
    assert set(parsed.entries) == {
        "drqn",
        "drqn_ensemble",
        "r2d2",
        "rtrrl",
        "rtrrl_ensemble",
        "stream_ac",
    }
    assert parsed.entries["rtrrl"].command == ("python", "-m", "entries.rtrrl")
    # An ensemble entry is its algorithm's, so it declares the same metrics and
    # the same parameters. What the catalog says about it is only that this
    # image can be asked to run a group -- which is why it is an entry rather
    # than a flag on the other one, and why the control plane can read the
    # capability instead of inferring it from a version.
    assert parsed.entries["drqn_ensemble"].command == (
        "python",
        "-m",
        "entries.drqn_ensemble",
    )
    # Which entries take a group is what the control plane reads to decide
    # whether a round's runs go into a manifest as `runs` or as `groups`. An
    # entry that predates groups says so rather than being absent, so nothing
    # downstream has to treat a missing key as an answer.
    assert {name: entry.grouped for name, entry in sorted(parsed.entries.items())} == {
        "drqn": False,
        "drqn_ensemble": True,
        "r2d2": False,
        "rtrrl": False,
        "rtrrl_ensemble": True,
        "stream_ac": False,
    }
    for algorithm in ("drqn", "rtrrl"):
        alone = parsed.entries[algorithm]
        grouped = parsed.entries[f"{algorithm}_ensemble"]
        assert grouped.metrics == alone.metrics
        assert grouped.parameters == alone.parameters
