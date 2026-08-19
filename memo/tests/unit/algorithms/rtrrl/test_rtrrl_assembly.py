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
from memorax.networks.sequence import PLACES
from memorax.observability.metrics import metric_names
from memorax.readings import taken
from tests.support.builders import (
    D_RTRRL,
    C,
    assemble_rtrrl,
    graph_of,
    rtrrl_parameters,
)
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


parameters = rtrrl_parameters
assembled = assemble_rtrrl


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
        checkpoint=None,
        fork=None,
    )


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


def test_the_d_rtrrl_optimizer_is_reachable_from_a_run_configuration():
    """The rule is only real if a run document can name it and a step can take it.

    The rule's own arithmetic is driven in ``tests/unit/components``. What is
    asserted here is everything between a configuration and that arithmetic:
    that ``d_rtrrl`` survives the parameter surface, that the two groups
    build a rule each, and that a scan of real transitions through the whole
    graph -- torso trace, two readouts, emphasis, resets -- stays finite.
    Finiteness is the claim the version this replaces could not make.
    """

    built = assembled(optimizer=D_RTRRL)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 32)

    for path, leaf in flattened(stepped.core).items():
        assert np.all(np.isfinite(leaf)), f"{path} went non-finite"
    assert np.all(np.isfinite(metrics.update.td_error))
    # The rate every rule reports under one name, and here a constant one.
    np.testing.assert_allclose(metrics.update.torso_step.step_size, C)
    np.testing.assert_allclose(metrics.update.heads_step.step_size, C)

    # Every block took a step. Without this the assertions above are satisfied
    # by a run that never moved: a TD error of zero reaching `sign` leaves the
    # traced update at exactly zero, which is finite and constant and means
    # nothing was learned.
    for block in ("torso", "actor", "critic"):
        before = flattened(getattr(state.core, block).params)
        after = flattened(getattr(stepped.core, block).params)
        assert any(
            not np.array_equal(leaf, after[path]) for path, leaf in before.items()
        ), f"{block} never moved"


def test_the_two_arms_differ_in_exactly_the_magnitude_they_keep():
    """The R3 comparison in miniature: same C, same everything, one difference.

    This is the property the formal arms are read for, asserted here on a tiny
    environment so it is checkable without a cluster. Over a run of real
    transitions:

    ``sign``
        every step moves the torso the same distance, whatever that step's TD
        error was. The realized update norm is flat.
    ``td_out``
        every step moves it a distance proportional to ``|delta|``. The
        realized update norm divided by ``|delta|`` is what is flat instead.

    The outer clip is off here. At ``grad_clip == C`` it binds on nearly every
    step and would flatten ``td_out`` too, which is the interaction worth
    knowing about and not the thing being measured.
    """

    def realized(magnitude):
        built = assembled(
            optimizer={
                **D_RTRRL,
                "torso.optimizer.d_rtrrl.magnitude": magnitude,
                "heads.optimizer.d_rtrrl.magnitude": magnitude,
                "torso.grad_clip": 0.0,
            }
        )
        state = built.program.init(jax.random.key(0))
        moves, surprises = [], []
        for step in range(8):
            before = flattened(state.core.critic.params)
            state, metrics = built.program.train(jax.random.key(step + 1), state, 1)
            after = flattened(state.core.critic.params)
            moves.append(
                float(
                    np.sqrt(
                        sum(
                            np.sum(np.square(np.asarray(after[k]) - np.asarray(leaf)))
                            for k, leaf in before.items()
                        )
                    )
                )
            )
            surprises.append(abs(float(np.asarray(metrics.update.td_error).ravel()[0])))
        # The first step steps with the initial trace, which is zero.
        return np.array(moves[1:]), np.array(surprises[1:])

    flat, flat_delta = realized("sign")
    scaled, scaled_delta = realized("td_out")

    # The TD errors are not all equal, or neither claim below means anything.
    assert flat_delta.std() > 0.05 * flat_delta.mean(), "no spread in |delta| to see"

    # `sign`: the distance is C and says nothing about the surprise.
    np.testing.assert_allclose(flat, C, rtol=1e-4)

    # `td_out`: the distance is proportional to the surprise instead. The trace
    # is long enough to be clipped throughout, which is what makes the ratio
    # exactly C rather than merely increasing.
    np.testing.assert_allclose(scaled / scaled_delta, C, rtol=1e-4)
    assert scaled.std() > 0.05 * scaled.mean(), "td_out did not vary with |delta|"


def test_the_shared_recurrent_direction_is_formed_before_it_is_normalized():
    """The torso's two sources are summed first and the resultant normalized.

    RTRRL's torso is fed by both readouts: ``upstream(actor_upward +
    critic_upward)``. The sum happens before the pullback and there is one
    torso trace, so the norm that is removed is the resultant's.

    The failure this is written against is normalizing the two sources
    separately and adding them afterwards. That version is *invariant* to the
    actor's source magnitude -- scaling the actor's sensitivity would rescale
    a vector that is about to be divided by its own norm, and the torso would
    not feel it. So asymmetric sensitivities are exactly what tells the two
    orderings apart, and ``eta_pi`` is the knob that makes them asymmetric:
    the actor's traced objective is ``eta_pi * log_prob``, so its upward
    gradient is proportional to it while the critic's is not.

    Two update steps, because the first one steps with the initial trace,
    which is zero.
    """

    def torso_after(sensitivity):
        built = assembled(optimizer={**D_RTRRL, "eta_pi": sensitivity})
        state = built.program.init(jax.random.key(0))
        return built.program.train(jax.random.key(1), state, 2)[0].core.torso.params

    balanced, actor_loud = torso_after(1.0), torso_after(50.0)

    moved = [
        path
        for path, leaf in flattened(balanced).items()
        if not np.array_equal(leaf, flattened(actor_loud)[path])
    ]
    assert moved, (
        "the torso step did not feel the actor's source magnitude, which is "
        "what separately normalizing the two sources would look like"
    )


def test_a_signed_step_does_not_read_the_torso_td_scaling():
    """``eta_f`` is inert end to end, not only where the sign is taken.

    The rule-level assertion pins the arithmetic; this one pins the wiring, so
    that a future change routing ``eta_f`` somewhere other than the TD error
    handed to the torso rule is caught here rather than in a sweep over a knob
    that turns out to move nothing.
    """

    def trained(magnitude, scaling):
        built = assembled(
            optimizer={
                **D_RTRRL,
                "torso.optimizer.d_rtrrl.magnitude": magnitude,
                "eta_f": scaling,
            }
        )
        state = built.program.init(jax.random.key(0))
        return built.program.train(jax.random.key(1), state, 16)[0]

    signed = (trained("sign", 1.0), trained("sign", 100.0))
    assert_tree_equal(signed[0].core.torso.params, signed[1].core.torso.params, "eta_f")

    # The control, without which the assertion above would also pass if `eta_f`
    # never reached the torso rule at all. Under the ablation the same knob has
    # to move the same parameters, or this test is measuring the wiring being
    # absent rather than the sign discarding it. `td_out` keeps |delta|, so it
    # is the arm where the same knob must still be live.
    ablated = (trained("td_out", 1.0), trained("td_out", 100.0))
    moved = [
        path
        for path, leaf in flattened(ablated[0].core.torso.params).items()
        if not np.array_equal(leaf, flattened(ablated[1].core.torso.params)[path])
    ]
    assert moved, "eta_f reaches the torso rule under neither setting: it is unwired"


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
