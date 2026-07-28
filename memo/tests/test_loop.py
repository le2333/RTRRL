"""The driver, and the catalog that says what there is to drive.

Neither knows an algorithm, so neither is tested with one: the driver runs a
few lines of arithmetic dressed as an algorithm, which is enough to check that
the arrows go where they should and much faster than the real thing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest
from training_sdk.contract import Catalog

import entries
from runner.catalog import build_catalog, discover, source_hash
from runner.episodes import complete_episodes
from runner.loop import drive, named_scalars, whole_epochs


class Recorder:
    """A reporter that keeps what it was told instead of shipping it."""

    def __init__(self) -> None:
        self.reports: list[tuple[int, dict[str, float]]] = []
        self.episodes: list = []

    def report(self, step, metrics):
        self.reports.append((step, dict(metrics)))

    def log_episode(self, episode):
        self.episodes.append(episode)


class Metrics(NamedTuple):
    loss: Any
    only_sometimes: Any = None
    by_part: Any = None


class Rollout(NamedTuple):
    observation: Any
    next_observation: Any
    action: Any
    reward: Any
    done: Any


NUM_ENVS = 2
# One episode ends in the first stream; the second stream never terminates.
DONE = jnp.array([[False, False], [True, False], [False, False]])


def arithmetic():
    """Something with the shape of an algorithm and none of the substance."""

    def init_fn(key):
        del key
        return jnp.asarray(0.0)

    def train_fn(key, state, num_steps):
        del key
        updates = num_steps // NUM_ENVS
        return state + num_steps, Metrics(loss=jnp.arange(updates, dtype=jnp.float32))

    def evaluate_fn(key, state, num_steps):
        del key
        column = jnp.arange(num_steps * NUM_ENVS, dtype=jnp.float32)
        grid = column.reshape(num_steps, NUM_ENVS)
        return state, Rollout(
            observation=grid[..., None],
            next_observation=grid[..., None] + 1,
            action=grid[..., None],
            reward=grid,
            done=DONE,
        )

    return init_fn, train_fn, evaluate_fn


def run_arithmetic(recorder, **overrides):
    init_fn, train_fn, evaluate_fn = arithmetic()
    settings: dict[str, Any] = {
        "total_steps": 8,
        "epoch_steps": 4,
        "eval_steps": DONE.shape[0],
        "num_envs": NUM_ENVS,
        "seed": 0,
        "training_report": lambda metrics: named_scalars(
            metrics, ("loss", "only_sometimes"), prefix="train/"
        ),
    }
    drive(
        recorder,
        init_fn=init_fn,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
        **(settings | overrides),
    )


def test_each_epoch_is_reported_at_the_step_it_ends_on():
    recorder = Recorder()
    run_arithmetic(recorder)

    assert [step for step, _ in recorder.reports] == [4, 4, 8, 8]


def test_a_metric_the_configuration_never_produced_is_left_out():
    recorder = Recorder()
    run_arithmetic(recorder)

    training = [report for _, report in recorder.reports if "train/loss" in report]
    assert len(training) == 2
    assert all("train/only_sometimes" not in report for report in training)


def test_the_whole_episode_is_scored_and_the_partial_ones_are_not():
    recorder = Recorder()
    run_arithmetic(recorder)

    assert [episode.number for episode in recorder.episodes] == [1, 2]
    scored = [report for _, report in recorder.reports if "eval/reward" in report]
    assert all("eval/episode_return" in report for report in scored)


def test_an_episode_is_worth_what_was_paid_not_what_the_algorithm_was_shown():
    """The scale an algorithm learns on is not the scale a score is read on.

    A kernel that normalises inside itself keeps the environment's own reward for
    the summary, and this argument is for the other arrangement: normalisation in
    an environment wrapper, which overwrites the reward before the algorithm ever
    sees it. The entry that wrapped the environment is what knows where the
    untouched one went, so it hands it in, and both the episode returns and the
    reported mean have to come from it rather than from the summary.
    """

    shown, paid = Recorder(), Recorder()
    run_arithmetic(shown)
    run_arithmetic(paid, eval_reward=lambda summary: summary.reward / 10)

    assert [episode.rewards for episode in paid.episodes] != [
        episode.rewards for episode in shown.episodes
    ]
    for one, other in zip(paid.episodes, shown.episodes, strict=True):
        assert one.rewards == pytest.approx([reward / 10 for reward in other.rewards])

    def scores(recorder, key):
        return [report[key] for _, report in recorder.reports if key in report]

    for key in ("eval/reward", "eval/episode_return"):
        assert scores(paid, key) == pytest.approx(
            [value / 10 for value in scores(shown, key)]
        )


def test_evaluation_can_be_switched_off_entirely():
    recorder = Recorder()
    run_arithmetic(recorder, eval_steps=0)

    assert [step for step, _ in recorder.reports] == [4, 8]
    assert not recorder.episodes


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


def test_the_catalog_is_the_entries_directory_and_nothing_written_down():
    catalog = Catalog.model_validate(build_catalog().model_dump(mode="json"))
    modules = discover()

    assert set(catalog.entries) == set(modules)
    assert set(catalog.entries) == {"rtrrl", "stream_ac", "upstream_stream_ac"}
    for name, entry in catalog.entries.items():
        assert entry.command == ("python", "-m", f"{entries.__name__}.{name}")
        assert set(entry.space) == set(modules[name].SPACE)
        assert entry.metrics == tuple(modules[name].METRICS)
        # The control plane refuses an entry without it, since a run with no
        # budget has nothing to stop it.
        assert "total_steps" in entry.space


def test_the_source_hash_ignores_bytecode(tmp_path: Path):
    root = tmp_path / "pkg"
    (root / "__pycache__").mkdir(parents=True)
    (root / "kernel.py").write_text("x = 1\n")
    before = source_hash((root,))
    (root / "__pycache__" / "kernel.pyc").write_bytes(b"\x00compiled")
    assert source_hash((root,)) == before
    (root / "kernel.py").write_text("x = 2\n")
    assert source_hash((root,)) != before


class Summary:
    def __init__(self, before, after, action, reward, done):
        self.observation = before
        self.next_observation = after
        self.action = action
        self.reward = reward
        self.done = done


def test_a_partial_episode_at_either_end_is_left_out():
    #                   env 0: done at 1 and 4     env 1: never done
    done = np.array([[0, 0], [1, 0], [0, 0], [0, 0], [1, 0]], dtype=bool)
    steps, envs = done.shape
    summary = Summary(
        before=np.arange(steps * envs, dtype=float).reshape(steps, envs, 1),
        after=np.arange(steps * envs, dtype=float).reshape(steps, envs, 1) + 100,
        action=np.zeros((steps, envs, 1)),
        reward=np.ones((steps, envs)),
        done=done,
    )
    episodes = list(
        complete_episodes(summary, phase="eval", start_env_steps=0, num_envs=envs)
    )

    # Steps 2 and 3 of env 0 run past the end without terminating, and env 1
    # never terminates at all; neither is a return anyone can report.
    assert [len(episode.actions) for episode in episodes] == [2, 3]
    assert [episode.number for episode in episodes] == [1, 2]
    assert episodes[0].observations[-1] == [102.0]


def test_every_episode_carries_one_more_observation_than_action():
    recorder = Recorder()
    run_arithmetic(recorder)

    for episode in recorder.episodes:
        assert len(episode.observations) == len(episode.actions) + 1
        assert len(episode.rewards) == len(episode.actions)
        assert episode.terminals[-1] is True
        assert not any(episode.terminals[:-1])


def test_a_named_family_is_read_one_part_at_a_time():
    """A kernel that measures parts of a network cannot name them.

    It reports a mapping and the entry, which built the network, names the parts
    it wants watched. A part that is absent is skipped like any other absent
    field rather than failing the epoch.
    """

    metrics = Metrics(
        loss=jnp.asarray([1.0]),
        by_part={"torso": jnp.asarray([1.0, 3.0])},
    )
    assert named_scalars(
        metrics, ("by_part/torso", "by_part/absent", "absent/torso"), prefix="train/"
    ) == {"train/by_part/torso": 2.0}
