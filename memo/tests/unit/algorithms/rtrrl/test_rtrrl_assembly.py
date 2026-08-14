from importlib import import_module
from types import SimpleNamespace

import jax
import memorax
import numpy as np
import pytest
from entries import rtrrl as entry
from memorax import algorithms
from memorax.algorithms import RTRRL
from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.networks.sequence import PLACES
from memorax.observability.metrics import metric_names
from memorax.parameters import expand
from memorax.readings import taken
from tests.support.environments import TinyContinuousEnv
from tests.support.numerics import flattened


def assert_tree_equal(actual, expected, what):
    """Assert two trees hold the same leaves bit for bit."""

    got, wanted = flattened(actual), flattened(expected)
    assert set(got) == set(wanted), f"{what}: the trees have different leaves"
    moved = [path for path, leaf in wanted.items() if not np.array_equal(got[path], leaf)]
    assert not moved, f"{what}: {moved} moved"


def parameters(backbone="lru", differentiation="exact_rtrl"):
    branch = f"torso.backbone.{backbone}"
    return expand(
        rtrrl.PARAMETERS,
        {
            "torso.backbone.kind": backbone,
            f"{branch}.feature_dim": 4,
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


def assembled(backbone="lru", differentiation="exact_rtrl"):
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
        ),
        environment_factory=tiny_environment,
    )


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


@pytest.mark.parametrize("backbone", ("lru", "rtu"))
def test_each_recurrent_backbone_registers_its_differentiation_scope(backbone):
    backbone_node = rtrrl.PARAMETERS["torso"]["backbone"]
    differentiation = backbone_node[backbone]["differentiation"]

    assert set(backbone_node["kind"].valid.values) == {"lru", "rtu"}
    assert set(differentiation["kind"].valid.values) == {
        "exact_rtrl",
        "tbptt",
    }


@pytest.mark.parametrize("backbone", ("lru", "rtu"))
@pytest.mark.parametrize("kind", ("exact_rtrl", "tbptt"))
def test_rtrrl_builds_the_selected_kernel_scoped_differentiation(backbone, kind):
    differentiation = import_module("memorax.networks.differentiation")
    built = assembled(backbone, kind)
    torso = built.program.init.__self__.core.torso
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
    assert set(rtrrl.TRAINING_METRICS) <= rtrrl.RECORD


def test_only_the_current_rtrrl_contract_is_public():
    assert memorax.RTRRL is algorithms.RTRRL is rtrrl.RTRRL
    for obsolete in ("EvalSummary", "IndependentRTRRL"):
        assert obsolete not in algorithms.__all__
        assert obsolete not in memorax.__all__


def test_entry_only_projects_the_run_config_for_assembly_and_runtime():
    config = SimpleNamespace(
        algorithm=SimpleNamespace(
            parameters={"gamma": 0.9},
            environment=SimpleNamespace(
                id="tiny",
                backend="test",
                observed=[0, 1],
                episode_length=8,
            ),
            num_envs=2,
        ),
        runtime=SimpleNamespace(
            seed=7,
            total_steps=32,
            epoch_steps=8,
            evaluation_steps=4,
        ),
    )

    request = entry.build_request(config)
    schedule = entry.runtime_config(config)

    assert request.parameters is config.algorithm.parameters
    assert request.environment.id == "tiny"
    assert request.num_envs == 2
    assert schedule.total_steps == 32
    assert schedule.epoch_steps == 8
    assert schedule.eval_steps == 4
    assert schedule.num_envs == 2
    assert schedule.seed == 7


def test_generic_assembly_closes_one_shared_torso_rtrrl_graph():
    built = assemble(
        rtrrl.RTRRL,
        BuildRequest(
            parameters=parameters(),
            environment=EnvironmentSpec(
                id="tiny",
                backend=None,
                observed=None,
                episode_length=8,
            ),
            num_envs=1,
        ),
        environment_factory=tiny_environment,
    )

    graph = built.program.init.__self__
    assert isinstance(graph.core.torso, rtrrl.Torso)
    assert graph.core.actor is not graph.core.critic
    assert graph.cfg.torso_optimizer.lr == 1e-3
    assert graph.cfg.heads_optimizer.lr == 5e-4
    assert built.observations is rtrrl.OBSERVATIONS

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, 2)
    evaluated = built.program.evaluate(jax.random.key(2), trained, 2)

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
    assert int(advanced.env_state.step_count[0]) == int(state.env_state.step_count[0]) + 1


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
    assert_tree_equal(advanced.core.critic.params, before.critic.params, "critic params")
    assert_tree_equal(advanced.core.critic.traces, before.critic.traces, "critic traces")
    assert_tree_equal(advanced.core.rule, before.rule, "rule state")
    assert_tree_equal(advanced.scales, state.scales, "normalization scales")


def test_semantic_subgraphs_do_not_expose_the_old_generic_network_layer():
    assert not hasattr(rtrrl, "Network")

    graph = assemble(
        rtrrl.RTRRL,
        BuildRequest(
            parameters=parameters(),
            environment=EnvironmentSpec(
                id="tiny",
                backend=None,
                observed=None,
                episode_length=8,
            ),
            num_envs=1,
        ),
        environment_factory=tiny_environment,
    ).program.init.__self__

    for subgraph in (graph.core.torso, graph.core.actor, graph.core.critic):
        assert not hasattr(subgraph, "block")
        assert not hasattr(subgraph, "module")
        assert not hasattr(subgraph, "credit")
