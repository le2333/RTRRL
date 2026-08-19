"""The R2 experiment files say what a formal run of them is.

These are not a check that the files run -- that needs the image's catalog and
a queue. They are a check that what the files claim about themselves is true
and stays true: the budget is the formal one, the seeds are declared fresh and
disjoint from the tuning seed, and the run keeps every checkpoint, because
which one a fork wants is decided from a collapse the run has not had yet.

A formal config that quietly drifted to a shorter budget, or to the tuning
seed, would produce results that look like the protocol's and are not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trainer_infra.experiment import ExperimentError, _absent, _formal

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
FORMAL = ("rtrrl issue45 r2 halfcheetah.yaml", "rtrrl issue45 r2 hopper.yaml")


def read(name: str) -> dict:
    return yaml.safe_load((EXPERIMENTS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", FORMAL)
def test_a_formal_config_says_everything_the_control_plane_needs(name: str) -> None:
    assert sorted(_absent(read(name))) == []


@pytest.mark.parametrize("name", FORMAL)
def test_a_formal_config_runs_one_of_five_fresh_seeds_for_a_million_steps(
    name: str,
) -> None:
    experiment = read(name)

    assert experiment["training"]["total_steps"] == 1_000_000
    assert experiment["evaluation"]["every_steps"] == 10_000
    assert len(experiment["formal"]["seeds"]) == 5
    assert experiment["formal"]["tuning_seed"] not in experiment["formal"]["seeds"]
    # And the file passes the check the control plane makes of it.
    _formal(experiment)


@pytest.mark.parametrize("name", FORMAL)
def test_a_formal_config_keeps_every_checkpoint_it_files(name: str) -> None:
    """All of them, at the boundary the evaluation measures.

    A run that kept the last few would have thrown away the state R3.4 needs by
    the time the collapse it needs it for has been found.
    """

    experiment = read(name)
    checkpoint = experiment["checkpoint"]

    assert checkpoint["keep"] is None
    assert checkpoint["every_steps"] % experiment["evaluation"]["every_steps"] == 0


@pytest.mark.parametrize("name", FORMAL)
def test_a_formal_config_declares_the_clip_the_fork_arms_are_written_over(
    name: str,
) -> None:
    """`trainerctl fork` reads `C` off the parent, so the parent must pin it."""

    clip = read(name)["space"]["torso"]["grad_clip"]

    assert clip == [1.0]


@pytest.mark.parametrize("name", FORMAL)
def test_a_formal_config_sends_the_update_scale_readings_to_the_dashboard(
    name: str,
) -> None:
    """The window the analysis reads comes from `metrics.jsonl` regardless.

    What this block buys is the moment-by-moment view: the collapse looked at
    transition by transition rather than through each episode's mean.
    """

    assert read(name)["logging"]["aim"]["training"]["step"]["every_steps"] > 0


def test_the_two_environments_do_not_share_a_seed_set() -> None:
    """Fresh per environment: reusing a seed reuses an initialization."""

    seeds = [set(read(name)["formal"]["seeds"]) for name in FORMAL]

    assert not seeds[0] & seeds[1]


def test_the_formal_configs_run_the_masked_observation_the_issue_names() -> None:
    """Partial observation is what the recurrent torso is under test for."""

    halfcheetah, hopper = (read(name) for name in FORMAL)

    assert halfcheetah["environment"]["id"] == "brax::halfcheetah"
    assert hopper["environment"]["id"] == "brax::hopper"
    for experiment in (halfcheetah, hopper):
        assert experiment["environment"]["observed"]


def test_a_config_that_ran_the_tuning_seed_would_be_caught() -> None:
    """The check is live on these files, not merely available to them."""

    experiment = read(FORMAL[0])
    experiment["environment"]["seed"] = experiment["formal"]["tuning_seed"]

    with pytest.raises(ExperimentError, match="formal set"):
        _formal(experiment)
