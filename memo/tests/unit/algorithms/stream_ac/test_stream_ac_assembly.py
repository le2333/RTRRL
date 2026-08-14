from types import SimpleNamespace

import jax
import numpy as np

from entries import stream_ac as entry
from memorax.algorithms import stream_ac
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import expand
from tests.support.environments import TinyContinuousEnv
from tests.support.numerics import flattened


def assert_tree_equal(actual, expected, what):
    """Assert two trees hold the same leaves bit for bit."""

    got, wanted = flattened(actual), flattened(expected)
    assert set(got) == set(wanted), f"{what}: the trees have different leaves"
    moved = [path for path, leaf in wanted.items() if not np.array_equal(got[path], leaf)]
    assert not moved, f"{what}: {moved} moved"


def parameters():
    return expand(
        stream_ac.PARAMETERS,
        {
            "actor.head.kind": "global_std",
            "actor.optimizer.bound.kind": "none",
            "actor.optimizer.base.kind": "sgd",
            "critic.head.kind": "value",
            "critic.optimizer.bound.kind": "none",
            "critic.optimizer.base.kind": "sgd",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "backbone.kind": "mlp",
            "meta_rl": False,
        },
    )


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


def assembled():
    return assemble(
        stream_ac.StreamAC,
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


def test_stream_ac_declarations_live_with_its_graph_and_entry_reexports_them():
    assert entry.PARAMETERS is stream_ac.PARAMETERS
    assert entry.METRICS is stream_ac.METRICS
    assert stream_ac.OBSERVATIONS.series == stream_ac.TRAINING_METRICS


def test_only_the_recurrent_backbone_declares_differentiation():
    backbone = stream_ac.PARAMETERS["backbone"]

    assert "differentiation" in backbone["rtu"]
    assert "differentiation" not in backbone["mlp"]
    assert "credit" not in stream_ac.PARAMETERS


def test_entry_only_projects_the_run_config_for_assembly():
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
    )

    request = entry.build_request(config)

    assert request.parameters is config.algorithm.parameters
    assert request.environment.id == "tiny"
    assert request.num_envs == 2


def test_entry_projects_only_the_runtime_schedule():
    config = SimpleNamespace(
        algorithm=SimpleNamespace(num_envs=2),
        runtime=SimpleNamespace(
            seed=7, total_steps=32, epoch_steps=8, evaluation_steps=4
        ),
    )

    schedule = entry.runtime_config(config)

    assert schedule.total_steps == 32
    assert schedule.epoch_steps == 8
    assert schedule.eval_steps == 4
    assert schedule.num_envs == 2
    assert schedule.seed == 7


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

    advanced, _ = built.program.interact(jax.random.key(2), state)

    assert_tree_equal(advanced.actor.params, state.actor.params, "actor params")
    assert_tree_equal(advanced.actor.rule, state.actor.rule, "actor rule state")
    assert_tree_equal(advanced.critic.params, state.critic.params, "critic params")
    assert_tree_equal(advanced.critic.rule, state.critic.rule, "critic rule state")
    assert_tree_equal(
        advanced.critic.recurrence, state.critic.recurrence, "critic recurrence"
    )
    assert_tree_equal(advanced.scales, state.scales, "normalization scales")


def test_generic_assembly_closes_stream_ac_over_the_runtime_program():
    built = assembled()

    graph = built.program.init.__self__
    assert graph.core.actor.block.network is not graph.core.critic.block.network
    assert built.observations is stream_ac.OBSERVATIONS

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, 2)
    evaluated = built.program.evaluate(jax.random.key(2), trained, 2)

    assert int(trained.step) == 2
    assert metrics.interaction.reward.shape == (2, 1)
    assert evaluated.interaction.reward.shape == (2, 1)
