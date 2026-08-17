"""What the platform reads out of R2D2: its parameters, its entry, its graph."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from entries import r2d2 as entry
from memorax.algorithms import r2d2
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import Choice
from memorax.parameters import expand as expand_parameters
from memorax.parameters import flatten
from tests.support.builders import graph_of
from tests.support.environments import TinyDiscreteEnv

EPISODE_LENGTH = 8


def parameters(**overrides):
    """One resolved manifest, structurally pinned the way an experiment pins it."""

    pinned = {
        "backbone.kind": "lru",
        "backbone.lru.feature_dim": 4,
        "backbone.lru.hidden_dim": 3,
        "head.kind": "dueling",
        "learning.kind": "tbptt",
        "learning.tbptt.burn_in_length": 1,
        "learning.tbptt.unroll_length": 2,
        "optimizer.kind": "adam",
        "optimizer.adam.lr": 1e-3,
        "replay.capacity": 64,
        "replay.minimum_size": 8,
        "replay.batch_size": 2,
        "replay.priority_exponent": 0.9,
        "replay.importance_sampling_exponent": 0.6,
        "replay.max_priority_weight": 0.9,
        "target.update_period": 2,
        "returns.n_step": 2,
        "returns.value_transform.kind": "identity",
        "gamma": 0.99,
        "exploration.epsilon_start": 0.2,
        "exploration.epsilon_end": 0.05,
        "exploration.epsilon_decay_steps": 1000,
        "exploration.evaluation_epsilon": 0.0,
    }
    pinned.update(overrides)
    return expand_parameters(r2d2.PARAMETERS, pinned)


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyDiscreteEnv()
    return environment, environment.default_params


def assembled(record=None, **overrides):
    return assemble(
        r2d2.R2D2,
        BuildRequest(
            parameters=parameters(**overrides),
            environment=EnvironmentSpec(
                id="tiny",
                backend=None,
                observed=None,
                episode_length=EPISODE_LENGTH,
            ),
            num_envs=1,
            record=(r2d2.OBSERVATIONS.trajectory_fields if record is None else record),
        ),
        environment_factory=tiny_environment,
    )


def run_document(*, every_steps=None, total_steps=50, episode_length=7):
    rerun = (
        None if every_steps is None else SimpleNamespace(log_every_steps=every_steps)
    )
    return SimpleNamespace(
        algorithm=SimpleNamespace(
            parameters={"gamma": 0.9},
            environment=SimpleNamespace(
                id="tiny",
                backend="test",
                observed=[0, 1],
                episode_length=episode_length,
            ),
            num_envs=2,
        ),
        training=SimpleNamespace(seed=7, total_steps=total_steps, chunk_steps=10),
        evaluation=SimpleNamespace(every_steps=10, episodes=3, chunk_steps=4, seed=11),
        logging=SimpleNamespace(aim=SimpleNamespace(training=None), rerun=rerun),
    )


def test_r2d2_declarations_live_with_its_graph_and_entry_reexports_them():
    assert entry.PARAMETERS is r2d2.PARAMETERS
    assert entry.METRICS is r2d2.METRICS
    assert r2d2.OBSERVATIONS.series == r2d2.TRAINING_METRICS


def test_the_tree_declares_both_of_every_choice_and_no_online_differentiation():
    declared = flatten(r2d2.PARAMETERS)

    assert not [path for path in declared if "differentiation" in path]
    assert {"backbone.lru.feature_dim", "backbone.rtu.feature_dim"} <= set(declared)
    for chooser, choices in (
        ("backbone.kind", ("lru", "rtu")),
        ("head.kind", ("dueling", "linear")),
        ("learning.kind", ("tbptt", "full_bptt")),
        ("returns.value_transform.kind", ("signed_hyperbolic", "identity")),
    ):
        valid = declared[chooser].valid
        assert isinstance(valid, Choice), f"{chooser} is not a choice"
        assert set(valid.values) == set(choices)


def test_the_entry_composes_and_calculates_nothing():
    """An entry projects a run document. Anything it computed would be a copy."""

    defined = sorted(
        name
        for name, value in vars(entry).items()
        if inspect.isfunction(value) and value.__module__ == entry.__name__
    )
    assert defined == ["build_request", "main", "run", "runtime_config"]

    body = inspect.getsource(entry)
    for owned_by_the_graph in (
        "q_value",
        "burn_in",
        "unroll",
        "priorit",
        "target_params",
        "loss",
        "grad",
        "epsilon",
        "n_step",
    ):
        assert owned_by_the_graph not in body


def test_entry_only_projects_the_run_config_for_assembly():
    config = run_document(every_steps=10)

    request = entry.build_request(config)

    assert request.parameters is config.algorithm.parameters
    assert request.environment.id == "tiny"
    assert request.environment.episode_length == 7
    assert request.num_envs == 2
    assert request.record == r2d2.OBSERVATIONS.trajectory_fields


def test_entry_projects_only_the_runtime_schedule():
    config = run_document(every_steps=10)

    schedule = entry.runtime_config(config)

    assert schedule.total_steps == 50
    assert schedule.chunk_steps == 10
    assert schedule.evaluation_episodes == 3
    assert schedule.evaluation_chunk_steps == 4
    assert schedule.evaluation_seed == 11
    assert schedule.num_envs == 2
    assert schedule.seed == 7
    assert schedule.trajectory_at_steps == (10, 20, 30, 40, 50)
    assert schedule.max_episode_steps == 7


def test_a_run_without_rerun_asks_for_no_sample_and_keeps_no_walk():
    config = run_document(every_steps=None)

    assert entry.build_request(config).record == frozenset()
    assert entry.runtime_config(config).trajectory_at_steps == ()


@pytest.mark.parametrize("backbone", ["lru", "rtu"])
@pytest.mark.parametrize("head", ["dueling", "linear"])
def test_the_graph_builds_the_branches_the_manifest_selected(backbone, head):
    built = assembled(
        **{
            "backbone.kind": backbone,
            f"backbone.{backbone}.feature_dim": 4,
            f"backbone.{backbone}.hidden_dim": 3,
            "head.kind": head,
        }
    )

    graph = graph_of(built)
    q_function = graph.core.q_function

    assert q_function.backbone_kind == backbone
    assert q_function.head_kind == head
    assert q_function.action_dim == 2
    assert q_function.feature_dim == 4
    assert q_function.hidden_dim == 3


def sampled_window(graph) -> int:
    """The transition count the assembled replay actually draws.

    Read off the sampler the graph bound, because the buffer is a record of
    functions and does not carry the lengths it was built with.
    """

    return graph.buffer.sample.keywords["sample_sequence_length"]


def test_tbptt_reads_the_window_its_horizons_name():
    """Burn-in plus unroll transitions, which is one fewer than its inputs."""

    graph = graph_of(assembled())

    assert graph.core.learning_kind == "tbptt"
    assert graph.core.burn_in_length == 1
    assert graph.core.unroll_length == 2
    assert r2d2.SelectedLearning("tbptt", 1, 2).window(EPISODE_LENGTH) == 3
    assert sampled_window(graph) == 3


def test_full_bptt_reads_the_configured_episode_and_declares_no_horizon():
    graph = graph_of(
        assembled(
            **{"learning.kind": "full_bptt", "replay.minimum_size": EPISODE_LENGTH}
        )
    )

    assert graph.core.learning_kind == "full_bptt"
    assert graph.core.burn_in_length == 0
    assert graph.core.unroll_length == 0
    assert r2d2.SelectedLearning("full_bptt", 0, 0).window(EPISODE_LENGTH) == (
        EPISODE_LENGTH
    )
    assert sampled_window(graph) == EPISODE_LENGTH


def test_the_value_transform_the_manifest_names_is_the_one_the_core_holds():
    identity = graph_of(assembled())
    transformed = graph_of(
        assembled(**{"returns.value_transform.kind": "signed_hyperbolic"})
    )

    assert identity.core.transform(3.0) == 3.0
    assert transformed.core.transform is r2d2.signed_hyperbolic
    assert transformed.core.inverse_transform is r2d2.signed_parabolic


def test_a_graph_keeps_the_walk_only_when_the_build_asked_for_it():
    kept = assembled()
    dropped = assembled(record=frozenset())

    state = kept.program.init(jax.random.key(0))
    _, walked = kept.program.train(jax.random.key(1), state, 2)
    state = dropped.program.init(jax.random.key(0))
    _, plain = dropped.program.train(jax.random.key(1), state, 2)

    for reading in (walked, plain):
        assert reading.interaction.reward.shape == (2, 1)
        assert reading.interaction.done.shape == (2, 1)
    assert walked.interaction.observation is not None
    assert walked.interaction.action is not None
    assert plain.interaction.observation is None
    assert plain.interaction.action is None

    assert kept.observations is r2d2.OBSERVATIONS
    assert dropped.observations.trajectory_fields == frozenset()
    assert dropped.observations.episode_fields == r2d2.OBSERVATIONS.episode_fields


def test_generic_assembly_closes_r2d2_over_the_runtime_program():
    built = assembled()

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, 2)
    opened = built.program.open_evaluation(jax.random.key(2), trained)
    _, evaluated = built.program.evaluate(jax.random.key(3), opened, 2)

    assert int(trained.step) == 2
    assert metrics.interaction.reward.shape == (2, 1)
    assert evaluated.interaction.reward.shape == (2, 1)


def test_program_exposes_no_learning_interaction():
    built = assembled()
    state = built.program.init(jax.random.key(0))

    advanced, metrics = built.program.interact(jax.random.key(1), state)

    assert metrics.interaction.reward.shape == (1,)
    assert metrics.update is None
    assert int(advanced.step) == int(state.step)
    assert int(advanced.core.update_step) == int(state.core.update_step)
    assert int(advanced.env_state.step_count[0]) == (
        int(state.env_state.step_count[0]) + 1
    )


def test_interaction_moves_no_learned_quantity():
    built = assembled()
    state = built.program.init(jax.random.key(0))

    advanced, _ = built.program.interact(jax.random.key(2), state)

    for what, left, right in (
        ("params", advanced.core.params, state.core.params),
        ("target params", advanced.core.target_params, state.core.target_params),
        (
            "optimizer state",
            advanced.core.optimizer_state,
            state.core.optimizer_state,
        ),
        ("replay", advanced.buffer_state, state.buffer_state),
    ):
        moved = [
            index
            for index, (got, wanted) in enumerate(
                zip(jax.tree.leaves(left), jax.tree.leaves(right))
            )
            if not np.array_equal(np.asarray(got), np.asarray(wanted))
        ]
        assert not moved, f"{what}: leaves {moved} moved"
