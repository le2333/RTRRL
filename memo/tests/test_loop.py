"""The driver, and the catalog that says what there is to drive.

Neither knows an algorithm, so neither is tested with one: the driver runs a
few lines of arithmetic dressed as an algorithm, which is enough to check that
the arrows go where they should and much faster than the real thing.

The arithmetic is arranged so that the two streams end their episodes at
different steps. Anything that averages the stream axis away turns those two
answers into one, which is the arrangement this file is here to refuse.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import pytest
from training_sdk.contract import Catalog

import entries
from entries import stream_ac
from memorax.runtime import drive, whole_epochs
from memorax.runtime.episode import check_names, statistics
from runner.catalog import build_catalog, discover


class Recorder:
    """A reporter that keeps what it was told instead of shipping it."""

    def __init__(self) -> None:
        self.episodes: list = []

    def log_episode(self, episode):
        self.episodes.append(episode)

    def of(self, phase: str) -> list:
        return [episode for episode in self.episodes if episode.phase == phase]


NUM_ENVS = 2
EPOCH_STEPS = 8
TOTAL_STEPS = 16

# Stream 0 ends its episode two transitions in, stream 1 three; both run past
# their ending without reaching another, so each chunk holds two whole episodes
# and two partial ones.
TRAIN_REWARD = jnp.array([[1.0, 5.0], [3.0, 5.0], [7.0, 5.0], [9.0, 5.0]])
TRAIN_DONE = jnp.array([[False, False], [True, False], [False, True], [False, False]])
TRAIN_LOSS = jnp.array([[0.0, 0.0], [2.0, 1.0], [4.0, 1.0], [6.0, 1.0]])

EVAL_REWARD = jnp.array([[2.0, 0.0], [4.0, 0.0], [0.0, 0.0]])
EVAL_DONE = jnp.array([[False, False], [True, False], [False, False]])
EVAL_STEPS = EVAL_DONE.shape[0]

SERIES = ("loss", "by_part.torso", "only_sometimes")


class InteractionMetrics(NamedTuple):
    """Where the kernels group the transition, which is all the loop reads."""

    reward: Any
    done: Any
    terminal: Any = None
    observation: Any = None
    next_observation: Any = None
    action: Any = None
    paid: Any = None


class Metrics(NamedTuple):
    interaction: InteractionMetrics
    loss: Any = None
    by_part: Any = None
    only_sometimes: Any = None


def arithmetic():
    """Something with the shape of an algorithm and none of the substance."""

    def init_fn(key):
        del key
        return jnp.asarray(0.0)

    def train_fn(key, state, num_steps):
        del key, num_steps
        return state + EPOCH_STEPS, Metrics(
            interaction=InteractionMetrics(
                reward=TRAIN_REWARD,
                done=TRAIN_DONE,
                terminal=TRAIN_DONE,
                paid=TRAIN_REWARD / 10,
            ),
            loss=TRAIN_LOSS,
            by_part={"torso": TRAIN_LOSS},
        )

    def evaluate_fn(key, state, num_steps):
        del key
        column = jnp.arange(num_steps * NUM_ENVS, dtype=jnp.float32)
        grid = column.reshape(num_steps, NUM_ENVS)
        return state, Metrics(
            interaction=InteractionMetrics(
                observation=grid[..., None],
                next_observation=grid[..., None] + 1,
                action=grid[..., None],
                reward=EVAL_REWARD,
                done=EVAL_DONE,
                terminal=EVAL_DONE,
                paid=EVAL_REWARD / 10,
            )
        )

    return init_fn, train_fn, evaluate_fn


def run_arithmetic(recorder, **overrides):
    init_fn, train_fn, evaluate_fn = arithmetic()
    settings: dict[str, Any] = {
        "total_steps": TOTAL_STEPS,
        "epoch_steps": EPOCH_STEPS,
        "eval_steps": EVAL_STEPS,
        "num_envs": NUM_ENVS,
        "seed": 0,
        "series": SERIES,
    }
    drive(
        recorder,
        init_fn=init_fn,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
        **(settings | overrides),
    )


def test_a_completed_episode_is_the_only_reporting_occasion():
    recorder = Recorder()
    run_arithmetic(recorder)

    # Two whole episodes per training chunk and one per evaluation, twice.
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

    A single number per epoch is what averaging the stream axis produces, and
    the two streams here disagree about both when their episode ended and what
    it was worth. Asserting one scalar per epoch is asserting the arrangement
    this replaces.
    """

    recorder = Recorder()
    run_arithmetic(recorder)

    ends = [episode.end_env_steps for episode in recorder.of("train")]
    assert ends == [4, 6, 12, 14]
    returns = [sum(episode.rewards) for episode in recorder.of("train")]
    assert returns == [4.0, 15.0, 4.0, 15.0]


def test_an_episode_the_budget_cut_off_emits_nothing():
    """Both streams run past their ending without reaching another one."""

    recorder = Recorder()
    run_arithmetic(recorder)

    assert all(episode.terminals[-1] for episode in recorder.episodes)
    assert [len(episode.rewards) for episode in recorder.of("train")] == [2, 3, 2, 3]


def test_a_diagnostic_the_configuration_never_produced_is_left_out():
    recorder = Recorder()
    run_arithmetic(recorder)

    for episode in recorder.of("train"):
        assert set(episode.series) == {"loss", "by_part.torso"}


def test_an_evaluation_episode_is_dated_by_the_training_it_measures():
    """Evaluation steps are not training steps, so they do not move the axis.

    A rollout run at the epoch boundary measures the policy as it stood there.
    Spreading its episodes along the axis as though they were training would
    put them in the epoch that has not happened yet.
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
    run_arithmetic(recorder, eval_steps=0)

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
        ({"epoch_steps": 3}, "epoch_steps"),
        ({"total_steps": 9}, "total_steps"),
    ],
)
def test_a_ragged_budget_is_refused_rather_than_rounded(budget, complaint):
    with pytest.raises(ValueError, match=complaint):
        whole_epochs(**({"total_steps": 8, "epoch_steps": 4, "num_envs": 2} | budget))


def test_a_budget_that_divides_lands_on_the_last_step():
    epochs = whole_epochs(total_steps=8, epoch_steps=4, num_envs=2)
    assert list(epochs) == [4, 8]


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
