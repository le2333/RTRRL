"""What the platform reads out of DRQN: its parameters, its entry, its graph."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from entries import drqn as entry
from memorax.algorithms import drqn
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import Choice
from memorax.parameters import expand as expand_parameters
from memorax.parameters import flatten
from tests.support.builders import graph_of
from tests.support.environments import TinyDiscreteEnv

EPISODE_LENGTH = 8


def _only(updates):
    """The single array in a one-parameter update, as numpy.

    Indexing the tree by key would ask the type checker to believe an
    ``ArrayTree`` is a dict, which it is not obliged to be.
    """

    return np.asarray(jax.tree.leaves(updates)[0])


def parameters(**overrides):
    """One resolved manifest, structurally pinned the way an experiment pins it."""

    pinned = {
        "core.kind": "lru",
        "core.lru.hidden_dim": 3,
        "core.lru.feature_dim": 4,
        "learning.kind": "truncated",
        "learning.truncated.length": 3,
        "optimizer.kind": "adadelta",
        "optimizer.adadelta.lr": 0.1,
        "optimizer.adadelta.rho": 0.95,
        "optimizer.adadelta.eps": 1e-8,
        "grad_clip": 10.0,
        "replay.capacity": 64,
        "replay.minimum_size": 8,
        "replay.batch_size": 2,
        "target.update_period": 2,
        "gamma": 0.99,
        "exploration.epsilon_start": 0.2,
        "exploration.epsilon_end": 0.05,
        "exploration.epsilon_decay_steps": 1000,
        "exploration.evaluation_epsilon": 0.0,
    }
    pinned.update(overrides)
    return expand_parameters(drqn.PARAMETERS, pinned)


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyDiscreteEnv()
    return environment, environment.default_params


def assembled(record=None, **overrides):
    return assemble(
        drqn.DRQN,
        BuildRequest(
            parameters=parameters(**overrides),
            environment=EnvironmentSpec(
                id="tiny",
                backend=None,
                observed=None,
                episode_length=EPISODE_LENGTH,
            ),
            num_envs=1,
            record=(drqn.OBSERVATIONS.trajectory_fields if record is None else record),
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


def test_drqn_declarations_live_with_its_graph_and_entry_reexports_them():
    assert entry.PARAMETERS is drqn.PARAMETERS
    assert entry.METRICS is drqn.METRICS
    assert drqn.OBSERVATIONS.series == drqn.TRAINING_METRICS


def test_the_tree_declares_both_cores_both_windows_and_one_head():
    declared = flatten(drqn.PARAMETERS)

    for chooser, choices in (
        ("core.kind", ("lru", "rtu")),
        ("learning.kind", ("truncated", "full_bptt")),
        ("optimizer.kind", ("adadelta", "adam")),
    ):
        valid = declared[chooser].valid
        assert isinstance(valid, Choice), f"{chooser} is not a choice"
        assert set(valid.values) == set(choices)
    # The head is not a choice at all: the published one is linear.
    assert not [path for path in declared if path.startswith("head")]
    # The published solver's own settings, and the clip that goes with it.
    assert {"optimizer.adadelta.lr", "optimizer.adadelta.rho", "grad_clip"} <= set(
        declared
    )


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
        "unroll",
        "truncat",
        "target_params",
        "loss",
        "grad",
        "epsilon",
    ):
        assert owned_by_the_graph not in body


def test_entry_only_projects_the_run_config_for_assembly():
    config = run_document(every_steps=10)

    request = entry.build_request(config)

    assert request.parameters is config.algorithm.parameters
    assert request.environment.id == "tiny"
    assert request.environment.episode_length == 7
    assert request.num_envs == 2
    assert request.record == drqn.OBSERVATIONS.trajectory_fields


def test_entry_projects_only_the_runtime_schedule():
    config = run_document(every_steps=10)

    schedule = entry.runtime_config(config)

    assert schedule.total_steps == 50
    assert schedule.chunk_steps == 10
    assert schedule.num_envs == 2
    assert schedule.seed == 7
    # Episodes, not steps: a checkpoint is scored on exactly this many, and the
    # evaluation opens a key stream of its own so a measurement cannot move the
    # training one.
    assert schedule.evaluation_episodes == 3
    assert schedule.evaluation_chunk_steps == 4
    assert schedule.evaluation_seed == 11
    assert schedule.trajectory_at_steps == (10, 20, 30, 40, 50)
    assert schedule.max_episode_steps == 7


@pytest.mark.parametrize("core_kind", ["lru", "rtu"])
def test_the_graph_builds_the_core_the_manifest_selected(core_kind):
    built = assembled(**{"core.kind": core_kind, f"core.{core_kind}.hidden_dim": 3})

    graph = graph_of(built)
    q_function = graph.core.q_function

    assert q_function.core_kind == core_kind
    assert q_function.action_dim == 2
    assert q_function.hidden_dim == 3
    assert q_function.observation_dim == 2
    # Only the LRU has a readout to size; see the parameter declarations.
    assert q_function.feature_dim == (4 if core_kind == "lru" else None)


def test_the_core_reads_the_observation_directly_and_normalises_after_it():
    """The matched representation, which is the online arm's own.

    Anything in front of the cell would be a parameter the online arm carries no
    exact sensitivity for, and the comparison would stop being a comparison of
    learners.
    """

    graph = graph_of(assembled())
    components = graph.core.q_function.network.sequence().components

    assert len(components) == 2
    assert getattr(components[0], "recurrent", False)
    normalization = components[1]
    assert type(normalization).__name__ == "LayerNorm"
    assert not normalization.use_scale
    assert not normalization.use_bias


def test_the_truncation_is_the_window_and_full_bptt_is_the_episode():
    truncated = graph_of(assembled())
    full = graph_of(
        assembled(
            **{"learning.kind": "full_bptt", "replay.minimum_size": EPISODE_LENGTH}
        )
    )

    assert drqn.SelectedLearning("truncated", 3).window(EPISODE_LENGTH) == 3
    assert drqn.SelectedLearning("full_bptt", 0).window(EPISODE_LENGTH) == (
        EPISODE_LENGTH
    )
    # The graph holds no truncation of its own: the window is the buffer's, and
    # the loss differentiates whatever it is handed.
    assert not hasattr(truncated.core, "truncation")
    assert not hasattr(full.core, "truncation")


def test_the_graph_holds_no_r2d2_machinery():
    graph = graph_of(assembled())

    for absent in (
        "n_step",
        "transform",
        "inverse_transform",
        "burn_in_length",
        "importance_sampling_exponent",
        "max_priority_weight",
    ):
        assert not hasattr(graph.core, absent), absent
    assert not hasattr(graph.buffer, "set_priorities")


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
    assert plain.interaction.observation is None

    assert kept.observations is drqn.OBSERVATIONS
    assert dropped.observations.trajectory_fields == frozenset()


def test_generic_assembly_closes_drqn_over_the_runtime_program():
    built = assembled()

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, 20)
    opened = built.program.open_evaluation(jax.random.key(2), trained)
    _, evaluated = built.program.evaluate(jax.random.key(3), opened, 2)

    assert int(trained.step) == 20
    assert int(trained.core.update_step) > 0
    assert metrics.interaction.reward.shape == (20, 1)
    assert evaluated.interaction.reward.shape == (2, 1)
    assert np.isfinite(np.asarray(metrics.update.loss)).all()


def test_measuring_the_policy_moves_nothing_it_measures():
    """A scored checkpoint is a read of the training state, never a write.

    The rollout the two evaluation arrows pass between them is the
    evaluation's own, so replay, the optimizer, the target copy and the
    learner's own step counter all have to come back where they were.
    """

    built = assembled()
    state = built.program.init(jax.random.key(0))
    trained, _ = built.program.train(jax.random.key(1), state, 20)

    opened = built.program.open_evaluation(jax.random.key(2), trained)
    advanced, metrics = built.program.evaluate(jax.random.key(3), opened, 4)

    assert metrics.update is None
    for what, left, right in (
        ("params", advanced.core.params, trained.core.params),
        ("target params", advanced.core.target_params, trained.core.target_params),
        (
            "optimizer state",
            advanced.core.optimizer_state,
            trained.core.optimizer_state,
        ),
        ("replay", advanced.buffer_state, trained.buffer_state),
    ):
        moved = [
            index
            for index, (got, wanted) in enumerate(
                zip(jax.tree.leaves(left), jax.tree.leaves(right))
            )
            if not np.array_equal(np.asarray(got), np.asarray(wanted))
        ]
        assert not moved, f"{what}: leaves {moved} moved"
    assert int(advanced.core.update_step) == int(trained.core.update_step)


def test_program_exposes_no_learning_interaction():
    built = assembled()
    state = built.program.init(jax.random.key(0))

    advanced, metrics = built.program.interact(jax.random.key(1), state)

    assert metrics.interaction.reward.shape == (1,)
    assert metrics.update is None
    assert int(advanced.step) == int(state.step)
    assert int(advanced.core.update_step) == int(state.core.update_step)


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


def test_a_truncation_longer_than_an_episode_is_refused_at_build_time():
    """A window that cannot fit is a configuration error, not an empty draw."""

    with pytest.raises(ValueError, match="exceeds the declared episode length"):
        assembled(**{"learning.truncated.length": EPISODE_LENGTH + 1})


def test_the_assembled_graph_steps_the_solver_the_manifest_named():
    """The graph's own optimizer, not a helper that resembles it.

    A test that exercised ``step_transform`` alone would pass while the graph
    built something else entirely -- which is exactly what happened: the chain
    was written, the graph was still calling the shared base transform, and a
    manifest naming adadelta got plain SGD with no clip and no error. So the
    thing under test here is the transform the assembled graph is holding.
    """

    import optax

    from memorax.algorithms.drqn import Adadelta, step_transform

    optimizer = graph_of(assembled()).core.optimizer
    grads = {"w": jnp.asarray([300.0, -400.0])}  # global norm 500, over the clip

    stepped, _ = optimizer.update(grads, optimizer.init(grads), grads)

    published = step_transform(Adadelta(lr=0.1, rho=0.95, eps=1e-8), grad_clip=10.0)
    wanted, _ = published.update(grads, published.init(grads), grads)
    np.testing.assert_allclose(_only(stepped), _only(wanted), rtol=1e-6)

    # And is not what falling through to the shared base transform would give.
    sgd = optax.sgd(0.1)
    fallen_through, _ = sgd.update(grads, sgd.init(grads), grads)
    assert not np.allclose(_only(stepped), _only(fallen_through))


def test_the_manifest_grad_clip_reaches_the_assembled_optimizer():
    """Not merely declared: the manifest's clip has to change what is stepped.

    AdaDelta divides a gradient by a running average of itself, so a *constant*
    gradient produces the same trajectory clipped or not -- the clip cancels out
    of both sides. It becomes visible when the magnitude changes between steps:
    a clipped first step leaves a much smaller accumulator behind, and the
    second step is taken against that.
    """

    steep = {"w": jnp.asarray([300.0, -400.0])}  # global norm 500, over the clip
    gentle = {"w": jnp.asarray([0.6, -0.8])}  # global norm 1, under it

    clipped = graph_of(assembled()).core.optimizer
    loose = graph_of(assembled(**{"grad_clip": 0.0})).core.optimizer

    def after_both(optimizer):
        state = optimizer.init(steep)
        _, state = optimizer.update(steep, state, steep)
        update, _ = optimizer.update(gentle, state, gentle)
        return _only(update)

    assert not np.allclose(after_both(clipped), after_both(loose))
