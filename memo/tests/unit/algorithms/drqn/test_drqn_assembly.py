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
from memorax.buffers import EpisodeWindowBuffer
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


def run_document(
    *, every_steps=None, total_steps=50, episode_length=7, snapshot_every_steps=0
):
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
            snapshot_every_steps=snapshot_every_steps,
        ),
        evaluation=SimpleNamespace(every_steps=10, episodes=3, chunk_steps=4, seed=11),
        logging=SimpleNamespace(aim=SimpleNamespace(training=None), rerun=rerun),
    )


def test_drqn_declarations_live_with_its_graph_and_entry_reexports_them():
    assert entry.PARAMETERS is drqn.PARAMETERS
    assert entry.METRICS is drqn.METRICS
    assert drqn.OBSERVATIONS.series == drqn.TRAINING_METRICS


def test_the_tree_declares_every_core_both_windows_and_one_head():
    declared = flatten(drqn.PARAMETERS)

    for chooser, choices in (
        ("core.kind", ("lru", "rtu", "lstm")),
        ("learning.kind", ("truncated", "full_bptt")),
        ("optimizer.kind", ("adadelta", "adam")),
    ):
        valid = declared[chooser].valid
        assert isinstance(valid, Choice), f"{chooser} is not a choice"
        assert set(valid.values) == set(choices)
    # Only the LRU has a readout width beside its hidden size; the RTU's output
    # is its carries and the LSTM's is its hidden state.
    assert "core.lru.feature_dim" in declared
    for widthless in ("rtu", "lstm"):
        assert [path for path in declared if path.startswith(f"core.{widthless}.")] == [
            f"core.{widthless}.hidden_dim"
        ]
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
    # Off unless the document asks, and carried straight through when it does.
    assert schedule.snapshot_every_steps == 0
    assert (
        entry.runtime_config(
            run_document(every_steps=10, snapshot_every_steps=10)
        ).snapshot_every_steps
        == 10
    )


@pytest.mark.parametrize("core_kind", ["lru", "rtu", "lstm"])
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


def test_a_manifest_selecting_the_published_cell_gets_one_lstm_and_the_head():
    """``core.kind: lstm`` at the paper's width, assembled from a manifest.

    The paper's network is an LSTM read directly by a linear Q head, so what a
    manifest selecting it must get is one recurrent component and nothing else:
    no readout width, and none of the normalisation the matched cores carry.
    """

    built = assembled(**{"core.kind": "lstm", "core.lstm.hidden_dim": 32})

    graph = graph_of(built)
    q_function = graph.core.q_function

    assert q_function.core_kind == "lstm"
    assert q_function.hidden_dim == 32
    assert q_function.feature_dim is None
    components = q_function.network.sequence().components
    assert len(components) == 1
    assert getattr(components[0], "recurrent", False)
    # An LSTM carries a cell state and a hidden state, both at the declared
    # width, and both zero where a window opens.
    carry = jax.tree.leaves(q_function.reset(drqn.ZERO_MEMORY, 1))
    assert [np.asarray(leaf).shape for leaf in carry] == [(1, 32), (1, 32)]


def test_the_published_cell_trains_through_the_same_replay_and_update():
    """The acceptance smoke: a discrete run reaches an update on ``lstm`` at 32.

    Nothing below the core changes when the cell does, so what this asks is only
    that the whole path -- replay, window, unroll, loss, step -- closes over a
    carry that is a pair of states rather than a single array.
    """

    built = assembled(**{"core.kind": "lstm", "core.lstm.hidden_dim": 32})

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, 40)

    assert int(trained.core.update_step) > 0
    applied = np.asarray(metrics.update.applied)
    assert np.all(np.isfinite(np.asarray(metrics.update.loss)[applied]))


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


def test_the_assembled_graph_draws_episodes_and_not_stored_positions():
    """Read off the built graph, because a helper tested alone proves nothing.

    The one thing that has already gone wrong here is a correct helper the
    assembled graph never called. What the sampling claims are about is the
    buffer this object holds, so that is what is asked.
    """

    built = assembled()
    graph = graph_of(built)
    state = built.program.init(jax.random.key(0)).buffer_state
    blank = jax.tree.map(lambda value: value[:, 0], state.trajectory.experience)

    def stored(state, ending):
        return graph.buffer.add(
            state,
            blank.replace(
                done=jnp.full_like(blank.done, ending),
                terminal=jnp.full_like(blank.terminal, ending),
            ),
        )

    assert isinstance(graph.buffer, EpisodeWindowBuffer)
    # Ten transitions is past the declared minimum size of eight, so a buffer
    # reporting on length alone would call this ready. It is five episodes of
    # two, and the truncation is three: there is nothing here to draw.
    for _ in range(5):
        state = stored(stored(state, False), True)
    assert int(state.written) == 10
    assert int(graph.buffer.retained(state)) == 10
    assert bool(graph.buffer.can_sample(state)) is False

    # One episode long enough to hold a window, and now there is.
    state = stored(stored(stored(state, False), False), True)
    assert bool(graph.buffer.can_sample(state)) is True
    assert int(jnp.sum(graph.buffer.sample(state, jax.random.key(0)).batch_valid)) == 1


@pytest.mark.parametrize(
    "branch",
    [
        {"learning.kind": "truncated", "learning.truncated.length": 3},
        {"learning.kind": "full_bptt", "replay.minimum_size": EPISODE_LENGTH},
    ],
)
def test_both_branches_train_through_the_episode_sampler(branch):
    """Full BPTT is not a second replay path, so it has to reach an update too.

    A full-episode window is drawn from any completed episode and padded out to
    the declared limit, where a truncated one is drawn only from episodes long
    enough to hold it. Those are different eligibility rules over the same
    buffer, and only running both says both reach a learner update.
    """

    built = assembled(**branch)

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, 40)

    assert int(trained.core.update_step) > 0
    assert np.isfinite(np.asarray(metrics.update.loss)).all()
    # A masked-out row must not leave the loss undefined for the rows that are
    # there, which is the failure a padded window would produce.
    applied = np.asarray(metrics.update.applied)
    assert np.all(np.isfinite(np.asarray(metrics.update.loss)[applied]))


def test_more_than_one_environment_is_refused_rather_than_given_a_cadence():
    """One update per environment transition is a claim, so it is enforced.

    This loop adds every stream's transition and then updates once, so several
    streams would silently make it one update per ``num_envs`` transitions.
    That is a change to the published learner's cadence, and an arm calling
    itself a reproduction may not make one quietly. A vectorised DRQN is not
    what this arm is for, so the answer is a refusal rather than an invented
    schedule.
    """

    with pytest.raises(ValueError, match="drqn runs one environment"):
        assemble(
            drqn.DRQN,
            BuildRequest(
                parameters=parameters(),
                environment=EnvironmentSpec(
                    id="tiny",
                    backend=None,
                    observed=None,
                    episode_length=EPISODE_LENGTH,
                ),
                num_envs=4,
                record=frozenset(),
            ),
            environment_factory=tiny_environment,
        )


def test_the_stored_reward_is_clipped_and_the_reported_one_is_not():
    """Read off a real run, because the helper alone proves only arithmetic.

    This environment pays ``0.4 + 0.35 * step``, so it never pays in units and
    the clip is not the identity: replay must hold ones where the run reports
    0.4, 0.75, 1.1 and so on. That separation is the claim -- a run is scored
    on what the environment paid, and the learner reads what DQN would have
    stored.
    """

    built = assembled()

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, 6)

    paid = np.asarray(metrics.interaction.reward)
    written = int(trained.buffer_state.written)
    stored = np.asarray(trained.buffer_state.trajectory.experience.reward)[:, :written]

    assert np.any(paid != np.sign(paid)), paid
    np.testing.assert_allclose(stored, np.sign(paid).T, atol=0)


def test_exploration_holds_still_inside_an_episode():
    """The published schedule is read once per episode, not once per step.

    A rate that moved mid-episode would make the episode a mixture of two
    policies, which is not what the agent whose schedule this is plays.
    """

    built = assembled()
    graph = graph_of(built)
    state = built.program.init(jax.random.key(0))

    # Mid-episode, whatever the update counter says, the rate is the one the
    # episode began under.
    inside = state.replace(
        episode_start=jnp.zeros_like(state.episode_start),
        epsilon=jnp.full_like(state.epsilon, 0.17),
        core=state.core.replace(update_step=jnp.asarray(500, dtype=jnp.int32)),
    )
    np.testing.assert_allclose(np.asarray(graph._episode_epsilon(inside)), 0.17)

    # At a boundary it is re-read, and 500 updates of a 1000-update anneal from
    # 0.2 to 0.05 is halfway.
    opening = inside.replace(episode_start=jnp.ones_like(state.episode_start))
    np.testing.assert_allclose(
        np.asarray(graph._episode_epsilon(opening)), 0.125, rtol=1e-6
    )


def test_epsilon_anneals_on_learner_updates_and_not_on_environment_steps():
    """No update has happened, so no progress has been made.

    The published schedule counts solver iterations, so it sits at
    ``epsilon_start`` for the whole replay warmup: an agent that has learned
    nothing yet acts uniformly at random rather than at a rate that has already
    begun to decay because the clock ran. Annealing on environment steps is a
    different exploration profile, not a rescaling of the same one.
    """

    warming = assembled(**{"replay.capacity": 4096, "replay.minimum_size": 1000})
    learning = assembled()

    held, _ = warming.program.train(
        jax.random.key(1), warming.program.init(jax.random.key(0)), 40
    )
    moved, _ = learning.program.train(
        jax.random.key(1), learning.program.init(jax.random.key(0)), 40
    )

    assert int(held.core.update_step) == 0
    np.testing.assert_allclose(np.asarray(held.epsilon), 0.2)

    assert int(moved.core.update_step) > 0
    assert float(np.asarray(moved.epsilon).item()) < 0.2


def test_an_update_cannot_draw_the_episode_it_is_still_finishing():
    """Replay is read as it stood before this transition, which is the loop.

    The published agent accumulates an episode locally, updates once per frame
    against the episodes it has already remembered, and remembers the current
    one at its ending. Adding first would let the update on an episode's last
    frame draw the episode that frame just completed.

    The arithmetic here is exact rather than approximate. This environment ends
    every three steps and the declared minimum size is eight, so eight
    transitions of *finished* episodes first exist after step nine. The update
    on step nine reads replay as of step eight -- six transitions, two
    episodes -- and does not fire; the update on step ten reads nine and does.
    """

    built = assembled()
    graph = graph_of(built)
    state = built.program.init(jax.random.key(0))

    before, _ = built.program.train(jax.random.key(1), state, 9)
    after, _ = built.program.train(jax.random.key(1), state, 10)

    assert int(before.buffer_state.written) == 9
    assert int(graph.buffer.retained(before.buffer_state)) == 9
    assert int(before.core.update_step) == 0
    assert int(after.core.update_step) == 1


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
