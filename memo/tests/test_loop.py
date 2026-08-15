"""The driver, and the catalog that says what there is to drive.

Neither knows an algorithm, so neither is tested with one: the driver runs a
few lines of arithmetic dressed as an algorithm, which is enough to check that
the arrows go where they should and much faster than the real thing.

The arithmetic is arranged so that the two streams end their episodes at
different steps. Anything that averages the stream axis away turns those two
answers into one, which is the arrangement this file is here to refuse.
"""

from __future__ import annotations

from typing import Any

import pytest

import entries
from deployment.catalog import build_catalog, discover
from deployment.contract import Catalog
from entries import stream_ac
from memorax.observability import check_names, statistics
from memorax.runtime import (
    BuiltAlgorithm,
    ObservationSchema,
    Program,
    Runtime,
    RuntimeConfig,
    evaluation_boundaries,
)
from tests.support.fakes import EpisodeRecorder as Recorder
from tests.support.programs import (
    EPOCH_STEPS,
    EVAL_STEPS,
    NUM_ENVS,
    SERIES,
    TOTAL_STEPS,
    arithmetic_program,
)


def run_arithmetic(recorder, **overrides):
    init_fn, train_fn, evaluate_fn, interact_fn = arithmetic_program()
    settings: dict[str, Any] = {
        "total_steps": TOTAL_STEPS,
        "chunk_steps": EPOCH_STEPS,
        "evaluate_every_steps": EPOCH_STEPS,
        "rollout_steps": EVAL_STEPS,
        "num_envs": NUM_ENVS,
        "seed": 0,
        # Four transitions is what the arithmetic's longest episode runs to, so
        # one train call is one whole interval and the chunking is not the subject.
        "max_episode_steps": 4,
    }
    reward = overrides.pop("reward", "interaction.reward")
    series = overrides.pop("series", SERIES)
    Runtime(
        algorithm=BuiltAlgorithm(
            program=Program(
                init=init_fn,
                train=train_fn,
                evaluate=evaluate_fn,
                interact=interact_fn,
            ),
            observations=ObservationSchema(
                reward=reward,
                done="interaction.done",
                terminal="interaction.terminal",
                observation="interaction.observation",
                next_observation="interaction.next_observation",
                action="interaction.action",
                series=series,
            ),
        ),
        config=RuntimeConfig(**(settings | overrides)),
    ).run(recorder)


def test_a_completed_episode_is_the_only_reporting_occasion():
    recorder = Recorder()
    run_arithmetic(recorder)

    # Two endings per evaluation interval and one per evaluation, twice. A
    # boundary is not an ending, so the second pair began in the first.
    assert len(recorder.of("train")) == 4
    assert len(recorder.of("eval")) == 2
    assert [episode.number for episode in recorder.of("train")] == [1, 2, 3, 4]
    assert [episode.number for episode in recorder.of("eval")] == [1, 2]


def test_an_episode_carries_what_it_was_paid_and_where_it_ended():
    recorder = Recorder()
    run_arithmetic(recorder)

    first, second = recorder.of("train")[:2]
    assert first.rewards == (1.0, 3.0)
    assert second.rewards == (5.0, 5.0, 5.0)
    assert statistics(first) == pytest.approx(
        {
            "train/episode/length": 2.0,
            "train/episode/return": 4.0,
            "train/episode/return_per_step": 2.0,
            "train/episode/return_per_step_variance": 1.0,
            "train/episode/loss": 1.0,
            "train/episode/loss_variance": 1.0,
            "train/episode/by_part.torso": 1.0,
            "train/episode/by_part.torso_variance": 1.0,
        }
    )


def test_two_streams_are_two_series_and_not_one_average():
    """Each episode belongs to one stream, so per-env comes for free.

    A single number per interval is what averaging the stream axis produces, and
    the two streams here disagree about both when their episode ended and what
    it was worth. Asserting one scalar per interval is asserting the arrangement
    this replaces.
    """

    recorder = Recorder()
    run_arithmetic(recorder)

    ends = [episode.end_env_steps for episode in recorder.of("train")]
    assert ends == [4, 7, 12, 15]
    returns = [sum(episode.rewards) for episode in recorder.of("train")]
    assert returns == [4.0, 15.0, 20.0, 20.0]


def test_an_episode_the_budget_cut_off_emits_nothing():
    """Both streams run past their ending without reaching another one."""

    recorder = Recorder()
    run_arithmetic(recorder)

    assert all(episode.terminals[-1] for episode in recorder.episodes)
    assert [len(episode.rewards) for episode in recorder.of("train")] == [2, 3, 4, 4]


def test_an_episode_that_outlives_a_train_call_is_reported_once_and_whole():
    """A call boundary is a transport size, so it ends nothing.

    The third and fourth episodes each begin in the first interval and end in
    the second; cutting per call would report each of them as two shorter ones.
    """

    recorder = Recorder()
    run_arithmetic(recorder)

    third, fourth = recorder.of("train")[2:]
    assert third.rewards == (7.0, 9.0, 1.0, 3.0)
    assert (third.start_env_steps, third.end_env_steps) == (4, 12)
    assert fourth.rewards == (5.0, 5.0, 5.0, 5.0)
    assert (fourth.start_env_steps, fourth.end_env_steps) == (7, 15)


def test_a_diagnostic_the_configuration_names_but_never_produces_is_refused():
    """A reading the schema names and no chunk carries is a build fault.

    Reporting the episode without it would hide a graph that was asked for a
    diagnostic and never wired one up.
    """

    with pytest.raises(ValueError, match="only_sometimes"):
        run_arithmetic(Recorder(), series=(*SERIES, "only_sometimes"))


def test_a_diagnostic_the_configuration_declared_reaches_every_episode():
    recorder = Recorder()
    run_arithmetic(recorder)

    for episode in recorder.of("train"):
        assert set(episode.series) == {"loss", "by_part.torso"}


def test_an_evaluation_episode_is_dated_by_the_training_it_measures():
    """Evaluation steps are not training steps, so they do not move the axis.

    A rollout run at the boundary measures the policy as it stood there.
    Spreading its episodes along the axis as though they were training would
    put them in the interval that has not happened yet.
    """

    recorder = Recorder()
    run_arithmetic(recorder)

    for episode in recorder.of("eval"):
        assert episode.start_env_steps == episode.end_env_steps
    assert [episode.end_env_steps for episode in recorder.of("eval")] == [8, 16]


def test_an_evaluation_episode_carries_the_trajectory_it_walked():
    recorder = Recorder()
    run_arithmetic(recorder)

    for episode in recorder.of("eval"):
        assert len(episode.observations) == len(episode.actions) + 1
        assert len(episode.rewards) == len(episode.actions)


def test_evaluation_can_be_switched_off_entirely():
    recorder = Recorder()
    run_arithmetic(recorder, rollout_steps=0)

    assert not recorder.of("eval")
    assert len(recorder.of("train")) == 4


def test_an_episode_is_worth_what_was_paid_not_what_the_algorithm_was_shown():
    """The scale an algorithm learns on is not the scale a score is read on.

    An entry that normalises in an environment wrapper has had the reward
    overwritten before the algorithm ever saw it, and the untouched one is
    somewhere else in the same summary. The entry names where.
    """

    shown, paid = Recorder(), Recorder()
    run_arithmetic(shown)
    run_arithmetic(paid, reward="interaction.paid")

    assert [episode.rewards for episode in paid.episodes] != [
        episode.rewards for episode in shown.episodes
    ]
    for one, other in zip(paid.episodes, shown.episodes, strict=True):
        assert one.rewards == pytest.approx([reward / 10 for reward in other.rewards])


@pytest.mark.parametrize(
    "budget, complaint",
    [
        ({"every_steps": 3}, "every_steps"),
        ({"total_steps": 9}, "total_steps"),
    ],
)
def test_a_ragged_budget_is_refused_rather_than_rounded(budget, complaint):
    with pytest.raises(ValueError, match=complaint):
        evaluation_boundaries(
            **({"total_steps": 8, "every_steps": 4, "num_envs": 2} | budget)
        )


def test_a_budget_that_divides_lands_on_the_last_step():
    boundaries = evaluation_boundaries(total_steps=8, every_steps=4, num_envs=2)
    assert list(boundaries) == [4, 8]


def test_the_names_an_entry_declares_are_names_a_sink_will_accept():
    """Three parts, and the middle one a window something is reduced over.

    The rule lives in the SDK because the sinks are where a name is finally
    read; an entry that spells one itself would be a second vocabulary.
    """

    check_names(stream_ac.METRICS)


def test_the_catalog_is_the_entries_directory_and_nothing_written_down():
    catalog = Catalog.model_validate(build_catalog().model_dump(mode="json"))
    modules = discover()

    # Naming the entries here would be the written-down copy this test exists to
    # refuse; that the directory is not empty is discover()'s own guarantee.
    assert modules
    assert set(catalog.entries) == set(modules)
    for name, entry in catalog.entries.items():
        assert entry.command == ("python", "-m", f"{entries.__name__}.{name}")
        assert set(entry.parameters) == set(modules[name].PARAMETERS)
        assert entry.metrics == tuple(modules[name].METRICS)
