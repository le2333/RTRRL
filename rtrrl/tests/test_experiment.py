"""The experiment file for this arm, read against the entry it names.

`memo/tests/test_experiments.py` does this for the four arms whose entries live
in the memo image. This one cannot: its entry lives here, and importing it there
would mean installing two incompatible jaxes side by side. So the same
guarantees are asserted from this side, over the one file that names this entry,
plus the ones that only matter for this arm -- that the decisions the comparison
was designed around actually arrive at their dataclass.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from training_sdk.contract import BudgetConfig, EnvironmentConfig

from entries.rtrrl_aaai import METRICS, SPACE, settings

EXPERIMENT = (
    Path(__file__).resolve().parents[2] / "experiments" / "rtrrl-hopper-aaai.yaml"
)
# The registry, repository and account the control plane is allowed to run from.
IMAGE = re.compile(
    r"^007122174918\.dkr\.ecr\.eu-north-1\.amazonaws\.com/rtrrl@sha256:[0-9a-f]{64}$"
)


@pytest.fixture
def experiment() -> dict[str, Any]:
    return yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))


def chosen(space: dict[str, Any]) -> dict[str, Any]:
    """One point of the experiment's space: pinned values, and a searched low.

    Every parameter this arm holds fixed is a list of one, so reading it is
    unambiguous. The six that are searched are ranges, and any point in a range
    will do for asking what the translation does with the rest.
    """

    return {
        name: spec[0] if isinstance(spec, list) else spec["low"]
        for name, spec in space.items()
    }


def test_it_names_every_parameter_the_entry_declares_and_no_other(experiment) -> None:
    """An omitted parameter is not an error anywhere, which is the problem.

    The control plane resolves a run's space as the entry's space updated with
    the experiment's, so a name left out keeps the whole domain the entry
    declared and the sampler draws from it. Nothing fails and the run is not the
    run the file describes.
    """

    assert set(experiment["space"]) == set(SPACE)


def test_it_is_scored_on_something_the_entry_reports(experiment) -> None:
    assert experiment["score"]["metric"] in METRICS


def test_the_score_window_is_the_budget(experiment) -> None:
    """Preflight refuses a window wider than the smallest budget in the space."""

    assert experiment["score"]["window_steps"] == [
        0,
        experiment["space"]["total_steps"][0],
    ]


def test_the_image_is_pinned_by_digest(experiment) -> None:
    """A tag can be moved; a run has to be attributable to what ran it."""

    assert IMAGE.match(experiment["image"])
    assert experiment["entry"] == "rtrrl_aaai"


def test_the_file_configures_the_run_the_comparison_was_designed_around(
    experiment,
) -> None:
    """Every decision that makes this arm the fifth arm, in one place.

    Each of these has a way of being quietly wrong. Their `rflo` default does
    not reach the first step under an LRU; their `patience` default gives this
    arm a shorter budget than the other four; a Brax backend left unset is
    whichever one Brax defaults to that month rather than the one the other four
    ran on; and the budget is only two million steps if the outer count and the
    scan length multiply to it.
    """

    resolved = settings(
        chosen(experiment["space"]),
        EnvironmentConfig(
            id="brax::hopper",
            backend="spring",
            num_envs=1,
            observed=(0, 1, 2, 3, 4),
        ),
        BudgetConfig(total_steps=2_000_000, epoch_steps=100_000, eval_steps=1000),
    )

    assert resolved["rnn_model"] == "lru"
    assert resolved["gradient_mode"] == "rtrl"
    assert resolved["patience"] == 0
    assert resolved["hidden_size"] == 32
    assert resolved["meta_rl"] is True
    assert not resolved["normalize_obs"]
    assert not resolved["normalize_reward"]
    assert not resolved["f_align"]
    assert not resolved["mlp_actor"]
    assert not resolved["layer_norm"]
    assert resolved["gamma"] == 0.99
    assert resolved["eta_pi"] == 1.0
    assert resolved["eta_f"] == 1.0
    assert resolved["trace_mode"] == "accumulate"
    assert resolved["seed"] == 1

    assert resolved["episodes"] == 2000
    assert resolved["steps"] == 1000
    assert resolved["eval_every"] == 100
    assert resolved["eval_steps"] == 1000
    assert resolved["eval_batch_size"] == 10

    assert resolved["environment"]["env_name"] == "brax-hopper"
    assert resolved["environment"]["batch_size"] == 1
    assert resolved["environment"]["obs_mask"] == (0, 1, 2, 3, 4)
    assert resolved["environment"]["env_kwargs"] == {"backend": "spring"}
    assert resolved["environment"]["render"] is False

    assert resolved["rnn"]["gradient_clip"] == 1.0
    assert "gradient_clip" not in resolved["td"]


def test_the_six_searched_dimensions_are_the_ones_the_other_arms_search(
    experiment,
) -> None:
    """Identical in all five files, or the arms are compared on their searches.

    Held against the memo files by value rather than by reference: those live in
    an image this project cannot import, and a range that drifts in one file is
    exactly the drift this asserts against.
    """

    searched = {
        name: spec
        for name, spec in experiment["space"].items()
        if not isinstance(spec, list)
    }

    assert searched == {
        "td_lr": {"type": "float", "low": 1.0e-5, "high": 1.0e-3, "log": True},
        "rnn_lr": {"type": "float", "low": 1.0e-5, "high": 1.0e-3, "log": True},
        "lambda_pi": {"type": "float", "low": 0.5, "high": 0.99},
        "lambda_v": {"type": "float", "low": 0.5, "high": 0.99},
        "lambda_rnn": {"type": "float", "low": 0.5, "high": 0.99},
        "entropy_rate": {"type": "float", "low": 1.0e-6, "high": 1.0e-3, "log": True},
    }
