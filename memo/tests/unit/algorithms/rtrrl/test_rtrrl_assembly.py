import jax

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.networks.sequence import PLACES
from memorax.observability.metrics import metric_names
from memorax.parameters import expand
from memorax.readings import taken
from tests.support.environments import TinyContinuousEnv


def parameters():
    return expand(
        rtrrl.PARAMETERS,
        {
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 4,
            "torso.backbone.lru.hidden_dim": 2,
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


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


def test_rtrrl_declares_parameters_and_observations_beside_its_graph():
    assert rtrrl.PARAMETERS
    assert rtrrl.TRAINING_METRICS == taken(rtrrl.REPORTS, parts=PLACES)
    assert rtrrl.METRICS == metric_names(
        "train", rtrrl.TRAINING_METRICS
    ) + metric_names("eval")
    assert set(rtrrl.TRAINING_METRICS) <= rtrrl.RECORD


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
