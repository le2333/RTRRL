"""The pinned StreamAC Hopper settings are ones this build can actually run.

The point of reproducing a recorded score is that every setting behind it
still means what it meant. These check the settings survive the trip from the
experiment file into the kernel config, so a silently dropped or renamed
parameter shows up here rather than as a number that fails to reproduce.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from runner.registry import topology

EXPERIMENT = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "streamac-hopper-reproduce.yaml"
)

# Aim run d9fe0986, the best recorded StreamAC-RTRL result on masked Hopper.
RECORDED = {
    "gamma": 0.99,
    "trace_lambda": 0.9060749966425912,
    "actor_lr": 0.532246925974616,
    "critic_lr": 0.11990103985952048,
    "actor_kappa": 2.722352816476039,
    "critic_kappa": 1.4401857247174306,
    "entropy_coefficient": 0.000271109456772485,
    "adaptive": False,
}


@pytest.fixture(scope="module")
def experiment() -> dict:
    return yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pinned(experiment) -> dict:
    values = {}
    for name, choices in experiment["space"].items():
        assert len(choices) == 1, f"{name} is being searched, not reproduced"
        values[name] = choices[0]
    return values


def test_the_experiment_names_the_entry_and_the_metric_it_scores(experiment):
    entry = topology(experiment["entry"])
    assert experiment["score"]["metric"] in entry.metrics


def test_every_pinned_setting_is_one_the_topology_declares(experiment, pinned):
    space = topology(experiment["entry"]).space
    assert not set(pinned) - set(space)


def test_nothing_is_left_for_the_sampler_to_choose(experiment, pinned):
    """A parameter the experiment forgets keeps the whole domain the entry
    declared, and the sampler is then free to pick from it. For a run that
    exists to reproduce one recorded number, every parameter must be pinned or
    the number is not the one being reproduced.
    """

    space = topology(experiment["entry"]).space
    assert not set(space) - set(pinned)


def test_the_recorded_hyperparameters_survive_into_the_kernel(pinned):
    from memorax.algorithms.stream_ac_rtrl import StreamACRTRLConfig
    from runner.registry import _select

    config = StreamACRTRLConfig(**_select(pinned, StreamACRTRLConfig))
    for name, value in RECORDED.items():
        assert getattr(config, name) == value, name
    assert config.num_envs == 16


def test_the_masked_partially_observed_hopper_is_what_gets_built(pinned):
    from runner.registry import _environment

    env, env_params = _environment(pinned)
    # Hopper reports 11 numbers; mode P zeroes the six velocities, so a
    # recurrent policy is the only thing that can recover them.
    assert env.observation_space(env_params).shape == (11,)
    assert env.action_space(env_params).shape == (3,)


def test_both_normalizers_are_on_as_they_were_when_the_score_was_set(pinned):
    from runner.registry import _normalization

    normalization = _normalization(pinned)
    assert normalization.normalize_observation
    assert normalization.normalize_reward
    assert normalization.reward_gamma == 0.99


def test_the_budget_divides_into_whole_epochs(pinned):
    assert pinned["total_steps"] % pinned["epoch_steps"] == 0
    assert pinned["epoch_steps"] % pinned["num_envs"] == 0
