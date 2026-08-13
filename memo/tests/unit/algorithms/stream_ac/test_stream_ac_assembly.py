from types import SimpleNamespace

import jax

from entries import stream_ac as entry
from memorax.algorithms import stream_ac
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import expand
from tests.support.environments import TinyContinuousEnv


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
            "credit.kind": "tbptt",
            "meta_rl": False,
        },
    )


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


def test_stream_ac_declarations_live_with_its_graph_and_entry_reexports_them():
    assert entry.PARAMETERS is stream_ac.PARAMETERS
    assert entry.METRICS is stream_ac.METRICS
    assert stream_ac.OBSERVATIONS.series == stream_ac.TRAINING_METRICS


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


def test_generic_assembly_closes_stream_ac_over_the_runtime_program():
    built = assemble(
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

    graph = built.program.init.__self__
    assert graph.core.actor.block.network is not graph.core.critic.block.network
    assert built.observations is stream_ac.OBSERVATIONS

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, 2)
    evaluated = built.program.evaluate(jax.random.key(2), trained, 2)

    assert int(trained.step) == 2
    assert metrics.interaction.reward.shape == (2, 1)
    assert evaluated.interaction.reward.shape == (2, 1)
