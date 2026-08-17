from importlib import import_module
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import memorax
from entries import rtrrl as entry
from memorax import algorithms
from memorax.algorithms import RTRRL
from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.networks.sequence import PLACES
from memorax.observability.metrics import metric_names
from memorax.parameters import expand
from memorax.readings import taken
from tests.support.builders import graph_of
from tests.support.environments import TinyContinuousEnv
from tests.support.numerics import flattened
from tests.support.parameters import kinds


def assert_tree_equal(actual, expected, what):
    """Assert two trees hold the same leaves bit for bit."""

    got, wanted = flattened(actual), flattened(expected)
    assert set(got) == set(wanted), f"{what}: the trees have different leaves"
    moved = [
        path for path, leaf in wanted.items() if not np.array_equal(got[path], leaf)
    ]
    assert not moved, f"{what}: {moved} moved"


def parameters(backbone="lru", differentiation="exact_rtrl"):
    branch = f"torso.backbone.{backbone}"
    return expand(
        rtrrl.PARAMETERS,
        {
            "torso.backbone.kind": backbone,
            **({f"{branch}.feature_dim": 4} if backbone == "lru" else {}),
            f"{branch}.hidden_dim": 2,
            f"{branch}.differentiation.kind": differentiation,
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 1e-3,
            "torso.grad_clip": 1.0,
            "torso.follow": 0.25,
            "heads.optimizer.kind": "adam",
            "heads.optimizer.adam.lr": 5e-4,
            "actor.head.kind": "state_std",
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": False,
        },
    )


def assembled(backbone="lru", differentiation="exact_rtrl", record=None):
    return assemble(
        rtrrl.RTRRL,
        BuildRequest(
            parameters=parameters(backbone, differentiation),
            environment=EnvironmentSpec(
                id="tiny",
                backend=None,
                observed=None,
                episode_length=8,
            ),
            num_envs=1,
            record=(rtrrl.OBSERVATIONS.trajectory_fields if record is None else record),
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
        training=SimpleNamespace(
            seed=7,
            total_steps=total_steps,
            chunk_steps=10,
        ),
        evaluation=SimpleNamespace(every_steps=10, episodes=3, chunk_steps=4, seed=11),
        logging=SimpleNamespace(aim=SimpleNamespace(training=None), rerun=rerun),
    )


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


@pytest.mark.parametrize("backbone", ("lru", "rtu"))
def test_each_recurrent_backbone_registers_its_differentiation_scope(backbone):
    assert set(kinds(rtrrl.PARAMETERS, "torso.backbone")) == {"lru", "rtu"}
    assert set(
        kinds(rtrrl.PARAMETERS, f"torso.backbone.{backbone}.differentiation")
    ) == {"exact_rtrl", "tbptt"}


@pytest.mark.parametrize("backbone", ("lru", "rtu"))
@pytest.mark.parametrize("kind", ("exact_rtrl", "tbptt"))
def test_rtrrl_builds_the_selected_kernel_scoped_differentiation(backbone, kind):
    differentiation = import_module("memorax.networks.differentiation")
    built = assembled(backbone, kind)
    torso = graph_of(built).core.torso
    selected = torso._differentiation

    assert isinstance(selected, differentiation.RecurrentDifferentiation)
    if kind == "tbptt":
        assert type(selected) is differentiation.TruncatedBPTT
    else:
        assert type(selected).__module__ == (
            f"memorax.networks.sequence_models.{backbone}"
        )

    state = built.program.init(jax.random.key(0))
    differentiation_state = state.core.torso.recurrence.differentiation_state
    assert (differentiation_state is None) is (kind == "tbptt")


def test_rtrrl_declares_parameters_and_observations_beside_its_graph():
    assert RTRRL is rtrrl.RTRRL
    assert entry.PARAMETERS is rtrrl.PARAMETERS
    assert entry.METRICS is rtrrl.METRICS
    assert rtrrl.PARAMETERS
    assert rtrrl.TRAINING_METRICS == taken(rtrrl.REPORTS, parts=PLACES)
    assert rtrrl.METRICS == metric_names(
        "train", rtrrl.TRAINING_METRICS
    ) + metric_names("eval")


def test_observation_schema_separates_episode_and_trajectory_fields():
    schema = rtrrl.OBSERVATIONS

    assert schema.episode_fields == frozenset(
        (schema.reward, schema.done, schema.terminal, *schema.series)
    )
    assert schema.trajectory_fields == frozenset(
        (schema.observation, schema.next_observation, schema.action)
    )
    assert set(rtrrl.TRAINING_METRICS) <= schema.episode_fields


def test_only_the_current_rtrrl_contract_is_public():
    assert memorax.RTRRL is algorithms.RTRRL is rtrrl.RTRRL
    for obsolete in ("EvalSummary", "IndependentRTRRL"):
        assert obsolete not in algorithms.__all__
        assert obsolete not in memorax.__all__


def test_entry_only_projects_the_run_config_for_assembly_and_runtime():
    config = run_document(every_steps=10)

    request = entry.build_request(config)
    schedule = entry.runtime_config(config)

    assert request.parameters is config.algorithm.parameters
    assert request.environment.id == "tiny"
    assert request.num_envs == 2
    assert schedule.total_steps == 50
    assert schedule.chunk_steps == 10
    assert schedule.evaluation_episodes == 3
    assert schedule.evaluation_chunk_steps == 4
    assert schedule.evaluation_seed == 11
    assert schedule.num_envs == 2
    assert schedule.seed == 7


def test_entry_expands_the_rerun_interval_into_the_steps_it_names():
    config = run_document(every_steps=10)

    request = entry.build_request(config)
    schedule = entry.runtime_config(config)

    # The schedule begins at the first interval, and the environment's own
    # episode limit is what bounds a train call.
    assert schedule.trajectory_at_steps == (10, 20, 30, 40, 50)
    assert schedule.max_episode_steps == 7
    assert request.record == rtrrl.OBSERVATIONS.trajectory_fields


def test_a_run_without_rerun_asks_for_no_sample_and_keeps_no_walk():
    config = run_document(every_steps=None)

    assert entry.build_request(config).record == frozenset()
    assert entry.runtime_config(config).trajectory_at_steps == ()


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
    assert walked.interaction.next_observation is not None
    assert walked.interaction.action is not None
    assert plain.interaction.observation is None
    assert plain.interaction.next_observation is None
    assert plain.interaction.action is None

    # Runtime is handed the schema the graph will actually answer.
    assert kept.observations is rtrrl.OBSERVATIONS
    assert dropped.observations.trajectory_fields == frozenset()
    assert dropped.observations.episode_fields == rtrrl.OBSERVATIONS.episode_fields


def test_generic_assembly_closes_one_shared_torso_rtrrl_graph():
    built = assembled()

    graph = graph_of(built)
    assert isinstance(graph.core.torso, rtrrl.Torso)
    assert graph.core.actor is not graph.core.critic
    assert graph.cfg.torso_optimizer.lr == 1e-3
    assert graph.cfg.heads_optimizer.lr == 5e-4
    assert built.observations is rtrrl.OBSERVATIONS

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
    assert int(advanced.update_step) == int(state.update_step)
    assert int(advanced.step) == int(state.step)
    assert not np.array_equal(
        np.asarray(advanced.timestep.obs), np.asarray(state.timestep.obs)
    )
    assert (
        int(advanced.env_state.step_count[0]) == int(state.env_state.step_count[0]) + 1
    )


def test_interaction_moves_no_learned_quantity():
    built = assembled()
    state = built.program.init(jax.random.key(0))
    before = state.core

    advanced, _ = built.program.interact(jax.random.key(2), state)

    assert_tree_equal(advanced.core.torso.params, before.torso.params, "torso params")
    assert_tree_equal(advanced.core.torso.traces, before.torso.traces, "torso traces")
    assert_tree_equal(
        advanced.core.torso.slow_params, before.torso.slow_params, "torso slow params"
    )
    assert_tree_equal(advanced.core.actor.params, before.actor.params, "actor params")
    assert_tree_equal(advanced.core.actor.traces, before.actor.traces, "actor traces")
    assert_tree_equal(
        advanced.core.critic.params, before.critic.params, "critic params"
    )
    assert_tree_equal(
        advanced.core.critic.traces, before.critic.traces, "critic traces"
    )
    assert_tree_equal(advanced.core.rule, before.rule, "rule state")
    assert_tree_equal(advanced.scales, state.scales, "normalization scales")


def test_semantic_subgraphs_do_not_expose_the_old_generic_network_layer():
    assert not hasattr(rtrrl, "Network")

    graph = graph_of(assembled())

    for subgraph in (graph.core.torso, graph.core.actor, graph.core.critic):
        assert not hasattr(subgraph, "block")
        assert not hasattr(subgraph, "module")
        assert not hasattr(subgraph, "credit")
