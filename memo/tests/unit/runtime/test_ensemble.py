"""What a member of an ensemble is owed.

The ensemble exists to fill a device with a sweep rather than with any one run,
and it is only worth having if a member gets what it would have got on its own.
"On its own" needs care, though, and the shape of that care is what these tests
fix.

A member is *not* bit-identical to the same seed under the single-member driver,
and cannot be. ``jax.vmap`` rewrites the computation into batched operations,
XLA compiles those to different kernels, and different kernels reduce in
different orders. Measured on DRQN, a one-member ensemble already diverges from
the driver -- at the same episode and in the same direction as a three-member
one -- so the divergence is vmap's arithmetic and not the batching of several
members together. An ensemble run is therefore not comparable digit-for-digit
with an acceptance number taken on the driver.

What must hold instead, and what is tested here, is that a member is a function
of its seed and of nothing else about the round: not its size, not the member's
position in it, not which other seeds it travelled with. Without that, a score
would depend on how a sweep happened to be packed, and two runs of the same
experiment could not be compared.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import pytest

from memorax.runtime import (
    BuiltAlgorithm,
    ObservationSchema,
    Program,
    RuntimeConfig,
)
from memorax.runtime.ensemble import EnsembleRuntime
from tests.support.fakes import EpisodeRecorder

NUM_ENVS = 2

OBSERVATIONS = ObservationSchema(
    reward="interaction.reward",
    done="interaction.done",
    terminal="interaction.terminal",
)


class Interaction(NamedTuple):
    reward: Any
    done: Any
    terminal: Any


class Metrics(NamedTuple):
    interaction: Interaction


def train(key, state, num_steps):
    """Training whose episodes end where the key says, so seeds differ."""

    rows = num_steps // NUM_ENVS
    draws = jax.random.uniform(key, (rows, NUM_ENVS))
    done = draws < 0.7
    return state + num_steps, Metrics(
        interaction=Interaction(reward=draws, done=done, terminal=done)
    )


def evaluate(key, state, num_steps):
    """Evaluation that ends every second row, so a quota always fills.

    Deterministic on purpose: an evaluation whose *length* varied with the key
    would make the loop's stopping condition the thing under test rather than
    the member's independence.
    """

    rows = num_steps // NUM_ENVS
    rank = state + jnp.arange(1, rows + 1, dtype=jnp.int32)[:, None]
    done = jnp.broadcast_to(rank % 2 == 0, (rows, NUM_ENVS))
    return state + rows, Metrics(
        interaction=Interaction(
            reward=jax.random.uniform(key, (rows, NUM_ENVS)),
            done=done,
            terminal=done,
        )
    )


def algorithm() -> BuiltAlgorithm:
    return BuiltAlgorithm(
        program=Program(
            init=lambda key: jax.random.randint(key, (), 0, 1000, dtype=jnp.int32),
            train=train,
            open_evaluation=lambda key, state: jnp.asarray(0, jnp.int32),
            evaluate=evaluate,
            interact=lambda key, state: None,
        ),
        observations=OBSERVATIONS,
    )


def config() -> RuntimeConfig:
    return RuntimeConfig(
        total_steps=8,
        chunk_steps=4,
        max_episode_steps=16,
        evaluate_every_steps=4,
        evaluation_episodes=2,
        evaluation_chunk_steps=4,
        evaluation_seed=1000,
        num_envs=NUM_ENVS,
        # A round has no single training seed. The field is ignored, and a
        # value that could be mistaken for one in a log is worse than a nonsense
        # one that cannot.
        seed=-1,
    )


def run(seeds: tuple[int, ...]) -> list[EpisodeRecorder]:
    recorders = [EpisodeRecorder() for _ in seeds]
    EnsembleRuntime(algorithm=algorithm(), config=config(), seeds=seeds).run(recorders)
    return recorders


def summarize(recorder: EpisodeRecorder) -> list[tuple]:
    """An episode reduced to what a score would ever read from it."""

    return [
        (episode.phase, episode.number, episode.stream, tuple(episode.rewards))
        for episode in recorder.episodes
    ]


@pytest.mark.parametrize("round_", [(3,), (3, 4), (9, 3, 4), (4, 1, 3, 8, 6)])
def test_a_member_does_not_depend_on_the_round_it_travelled_in(round_):
    """Seed 3 gets the same episodes whoever else is in the job.

    This is the property the ensemble has to have. A score that moved with the
    size of the round, or with a member's index in it, would mean the number an
    experiment reports depends on how its sweep was packed -- and two runs of
    one experiment could then disagree without either being wrong.
    """

    alone = summarize(run((3,))[0])
    among = summarize(run(round_)[round_.index(3)])
    assert among == alone


def test_members_are_not_copies_of_each_other():
    """The seeds must actually reach the members.

    A vmap that broadcast one member's key would satisfy every other test here
    -- the round would be internally consistent and reproducible -- while
    computing one run N times.
    """

    recorders = run((3, 4, 5))
    trained = [summarize(recorder) for recorder in recorders]
    assert trained[0] != trained[1]
    assert trained[1] != trained[2]


def test_every_member_gets_its_own_destination():
    recorders = run((3, 4, 5))
    assert len(recorders) == 3
    for recorder in recorders:
        assert recorder.episodes
        assert {episode.phase for episode in recorder.episodes} == {"train", "eval"}


def test_evaluation_is_paired_across_members():
    """Every member is measured on one evaluation stream.

    What separates two members' scores has to be the policy each learned. If
    each drew its own evaluation episodes, a difference between them would be
    partly the difference between two rollouts, and the members would be
    comparable only in the loose sense that both were measured.
    """

    recorders = run((3, 4, 5))
    numbers = [
        tuple(
            (episode.number, episode.stream)
            for episode in recorder.episodes
            if episode.phase == "eval"
        )
        for recorder in recorders
    ]
    assert numbers[0] == numbers[1] == numbers[2]
    assert numbers[0]


def test_an_ensemble_needs_a_member():
    with pytest.raises(ValueError, match="at least one member"):
        EnsembleRuntime(algorithm=algorithm(), config=config(), seeds=())


def test_two_members_that_are_the_same_run_are_refused():
    """One run billed twice, and indistinguishable in the results.

    Which is the worse half: nothing downstream could tell the duplicate from a
    real second sample.
    """

    with pytest.raises(ValueError, match="the same run"):
        EnsembleRuntime(algorithm=algorithm(), config=config(), seeds=(3, 4, 3))


def test_one_seed_under_two_swept_values_is_two_members():
    """A round runs every seed under every trial, so seeds repeat across it.

    Refusing that was right while only seeds could differ and became wrong the
    moment values could: seed 3 at one gamma and seed 3 at another are the same
    start under different parameters, which is the shape of a sweep rather than
    a duplicate. This is the second invariant that had quietly outlived its
    reason, and it blocked the first swept launch that reached a device.
    """

    recorders = [EpisodeRecorder(), EpisodeRecorder()]
    EnsembleRuntime(
        algorithm=scaled({"scale": 1.0}),
        config=config(),
        seeds=(3, 3),
        build=scaled,
        parameters={"scale": 1.0},
        swept={"scale": [1.0, 4.0]},
    ).run(recorders)
    assert summarize(recorders[0]) != summarize(recorders[1])


def test_a_destination_is_required_for_every_member():
    with pytest.raises(ValueError, match="2 destinations for 3 members"):
        EnsembleRuntime(algorithm=algorithm(), config=config(), seeds=(3, 4, 5)).run(
            [EpisodeRecorder(), EpisodeRecorder()]
        )


# --------------------------------------------------------------- swept values


def scaled(parameters) -> BuiltAlgorithm:
    """A graph whose episodes end at a rate its parameters decide.

    The value is read where a real algorithm reads a discount: while the graph
    is being built, closed into it, and never seen again. That is what makes it
    the thing a member axis has to carry rather than pass.
    """

    scale = parameters["scale"]

    def train(key, state, num_steps):
        rows = num_steps // NUM_ENVS
        draws = jax.random.uniform(key, (rows, NUM_ENVS)) * scale
        done = draws < 0.5
        return state + num_steps, Metrics(
            interaction=Interaction(reward=draws, done=done, terminal=done)
        )

    return BuiltAlgorithm(
        program=Program(
            init=lambda key: jax.random.randint(key, (), 0, 1000, dtype=jnp.int32),
            train=train,
            open_evaluation=lambda key, state: jnp.asarray(0, jnp.int32),
            evaluate=evaluate,
            interact=lambda key, state: None,
        ),
        observations=OBSERVATIONS,
    )


def run_swept(seeds, scales) -> list[EpisodeRecorder]:
    recorders = [EpisodeRecorder() for _ in seeds]
    EnsembleRuntime(
        algorithm=scaled({"scale": scales[0]}),
        config=config(),
        seeds=seeds,
        build=scaled,
        parameters={"scale": scales[0]},
        swept={"scale": list(scales)},
    ).run(recorders)
    return recorders


def test_a_swept_value_reaches_each_member():
    """Hold the seeds and move only the value.

    Two members with different seeds always differ, so a sweep that did nothing
    at all would still pass that comparison. What isolates the value is running
    the same two seeds twice: once where the members share a value and once
    where the second member's differs. The first member is the control -- its
    value never moved, so it must not move -- and the second is the reading.
    """

    same = run_swept((3, 4), (1.0, 1.0))
    moved = run_swept((3, 4), (1.0, 4.0))

    assert summarize(moved[0]) == summarize(same[0])
    assert summarize(moved[1]) != summarize(same[1])


def test_a_swept_member_does_not_depend_on_the_round_it_travelled_in():
    """Same seed, same value, same result, whatever else was in the job."""

    alone = summarize(run_swept((3,), (2.0,))[0])
    among = summarize(run_swept((9, 3, 4), (5.0, 2.0, 0.5))[1])
    assert among == alone


def test_a_sweep_needs_both_a_build_and_its_values():
    with pytest.raises(ValueError, match="needs both a build"):
        EnsembleRuntime(
            algorithm=algorithm(),
            config=config(),
            seeds=(3, 4),
            swept={"scale": [1.0, 2.0]},
        )
    with pytest.raises(ValueError, match="needs both a build"):
        EnsembleRuntime(
            algorithm=algorithm(), config=config(), seeds=(3, 4), build=scaled
        )


def test_every_member_needs_a_value_of_its_own():
    """A sweep short of a value would silently broadcast somebody else's."""

    with pytest.raises(ValueError, match="2 values for 3 members"):
        EnsembleRuntime(
            algorithm=algorithm(),
            config=config(),
            seeds=(3, 4, 5),
            build=scaled,
            swept={"scale": [1.0, 2.0]},
        )
