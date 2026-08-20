"""The published learner, checked against an independent reading of it.

Each test here names one sentence of Hausknecht and Stone and computes what that
sentence says out of the same networks, rather than asking whether the code
agrees with itself. The point of the arm is that it is DRQN; a reproduction that
had quietly become something else would still train.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict

from memorax.algorithms import drqn
from memorax.algorithms.drqn import Core, LearnerSequence, QFunction, RecurrentInputs
from memorax.parameters import flatten


def q_function(**overrides):
    settings = {
        "action_dim": 2,
        "observation_dim": 2,
        "hidden_dim": 3,
        "feature_dim": 4,
        "core_kind": "lru",
    }
    settings.update(overrides)
    return QFunction(**settings)


def core(**overrides):
    settings = {
        "q_function": q_function(),
        "optimizer": optax.adam(0.05),
        "gamma": 0.5,
        "target_update_period": 4,
    }
    settings.update(overrides)
    return Core(**settings)


def observations(key, time, width=2):
    return jax.random.normal(key, (1, time, width))


def sample(key, *, dones, terminals, actions, rewards):
    """A window of ``len(actions)`` transitions with nontrivial observations."""

    transitions = len(actions)
    walk = observations(key, transitions + 1)
    return LearnerSequence(
        inputs=RecurrentInputs(
            observation=walk[:, :-1],
            episode_start=jnp.asarray([[True] + [False] * (transitions - 1)]),
        ),
        # The states the transitions arrived at. Within one episode these are
        # the walk shifted by one, which is what replay stores.
        bootstrap_inputs=RecurrentInputs(
            observation=walk[:, 1:],
            episode_start=jnp.zeros((1, transitions), dtype=jnp.bool_),
        ),
        actions=jnp.asarray([actions]),
        rewards=jnp.asarray([rewards]),
        dones=jnp.asarray([dones]),
        terminals=jnp.asarray([terminals]),
        valid=jnp.asarray([[True] * transitions]),
    )


def diverged(learner, drawn):
    """Online and target parameters that are not each other.

    A test that cannot tell two networks apart cannot say which one chose an
    action, so the arrangement every target test needs is one update away from
    initialisation.
    """

    state = learner.core.init(jax.random.key(11), drawn.inputs)
    moved, _ = learner.core.update_parameters(jax.random.key(12), state, drawn)
    return moved.params, state.target_params


class _Learner:
    """The core under test, with the sample it was arranged around."""

    def __init__(self, **overrides):
        self.core = core(**overrides)


def test_the_declared_tree_offers_no_r2d2_enhancement():
    """None of R2D2's additions can be selected, because none is declared."""

    declared = set(flatten(drqn.PARAMETERS))

    for absent in (
        "priority",
        "importance_sampling",
        "max_priority_weight",
        "n_step",
        "value_transform",
        "burn_in",
        "dueling",
        "differentiation",
    ):
        assert not [path for path in declared if absent in path], absent
    assert {"core.lru.hidden_dim", "core.rtu.hidden_dim"} <= declared
    assert "learning.truncated.length" in declared


def test_the_head_is_one_linear_map_from_the_recurrent_output():
    """A linear Q head, so no value and advantage streams to combine."""

    function = q_function()
    params, _ = function.init(
        jax.random.key(0),
        RecurrentInputs(
            observation=observations(jax.random.key(1), 1),
            episode_start=jnp.asarray([[True]]),
        ),
    )

    head = params["params"]["DiscreteQNetwork_0"]
    assert set(head) == {"Dense_0"}
    assert head["Dense_0"]["kernel"].shape[-1] == function.action_dim


def test_nothing_the_learner_reads_carries_the_actors_recurrence():
    """Zero hidden state at a window start makes the stored one dead weight."""

    stored = set(drqn.ReplayTransition.__dataclass_fields__)

    assert stored == {
        "observation",
        "episode_start",
        "action",
        "reward",
        "next_observation",
        "done",
        "terminal",
    }
    assert not [
        name for name in drqn.LearnerSequence.__dataclass_fields__ if "recur" in name
    ]


def test_a_window_starts_from_a_zero_hidden_state():
    """The state a window opens on is the reset one, and the reset one is zero."""

    function = q_function()
    opened = function.reset(drqn.ZERO_MEMORY, 4)

    for leaf in jax.tree.leaves(opened):
        assert np.allclose(np.asarray(leaf), 0.0) or np.allclose(np.asarray(leaf), 1.0)
    # The carry the LRU opens on is a zero state and a unit decay; what must be
    # zero is the state it accumulates into.
    assert np.allclose(np.asarray(jax.tree.leaves(opened)[0]), 0.0)
    assert np.array_equal(
        np.asarray(jax.tree.leaves(function.reset(jax.random.key(3), 4))[0]),
        np.asarray(jax.tree.leaves(opened)[0]),
    )


@pytest.mark.parametrize(
    "dones,terminals",
    [
        ([False, False], [False, False]),
        ([False, True], [False, True]),
    ],
)
def test_the_target_is_one_step_and_greedy_under_the_target_network(dones, terminals):
    """r + gamma (1 - terminal) max_a Q_target(s', a), and no other term.

    Computed here from a separate unroll of the same two networks, so the only
    thing shared with the code under test is the network.
    """

    learner = _Learner()
    drawn = sample(
        jax.random.key(4),
        dones=dones,
        terminals=terminals,
        actions=[0, 1],
        rewards=[1.0, -2.0],
    )
    params, target_params = diverged(learner, drawn)

    _, readings = learner.core._loss(params, target_params, drawn)

    start = learner.core.q_function.reset(drqn.ZERO_MEMORY, 1)
    _, online_q, _ = learner.core.q_function.unroll(params, drawn.inputs, start)
    _, successor_q, _ = learner.core.q_function.unroll(
        target_params, drawn.bootstrap_inputs, start
    )
    q_value = jnp.take_along_axis(online_q, drawn.actions[..., None], axis=-1).squeeze(
        axis=-1
    )
    successor = jnp.max(successor_q, axis=-1)
    expected_target = (
        drawn.rewards + 0.5 * (1.0 - drawn.terminals.astype(jnp.float32)) * successor
    )

    np.testing.assert_allclose(
        np.asarray(readings.q_value), np.asarray(q_value), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(readings.td_error),
        np.asarray(expected_target - q_value),
        rtol=1e-5,
        atol=1e-6,
    )


def ranked_differently(params):
    """The same network with its two Q outputs exchanged.

    Double Q-learning and plain Q-learning are only two numbers where the two
    networks disagree about which action is best, so the arrangement is built
    rather than hoped for: initialisation and a few small steps leave both
    networks preferring the same action, and then no assertion could tell the
    two rules apart.
    """

    flat = dict(flatten_dict(params))
    head = ("params", "DiscreteQNetwork_0", "Dense_0")
    flat[(*head, "kernel")] = flat[(*head, "kernel")][:, ::-1]
    flat[(*head, "bias")] = flat[(*head, "bias")][::-1]
    return unflatten_dict(flat)


def test_the_greedy_action_is_the_target_networks_and_not_the_online_ones():
    """Not double Q-learning: no online argmax is handed to the target network."""

    learner = _Learner()
    drawn = sample(
        jax.random.key(5),
        dones=[False, False],
        terminals=[False, False],
        actions=[1, 0],
        rewards=[0.5, 0.25],
    )
    params, target_params = diverged(learner, drawn)
    target_params = ranked_differently(target_params)

    _, readings = learner.core._loss(params, target_params, drawn)

    start = learner.core.q_function.reset(drqn.ZERO_MEMORY, 1)
    _, successor_q, _ = learner.core.q_function.unroll(
        target_params, drawn.bootstrap_inputs, start
    )
    _, online_successor_q, _ = learner.core.q_function.unroll(
        params, drawn.bootstrap_inputs, start
    )
    online_choice = jnp.argmax(online_successor_q, axis=-1)
    double_q = jnp.take_along_axis(
        successor_q, online_choice[..., None], axis=-1
    ).squeeze(axis=-1)
    greedy = jnp.max(successor_q, axis=-1)

    target = readings.q_value + readings.td_error
    plain = drawn.rewards + 0.5 * greedy
    doubled = drawn.rewards + 0.5 * double_q

    np.testing.assert_allclose(np.asarray(target), np.asarray(plain), rtol=1e-5)
    # The test can only mean something where the two readings differ, so make
    # the arrangement prove that it does before believing the agreement above.
    assert not np.allclose(np.asarray(plain), np.asarray(doubled))


def test_every_target_is_read_from_the_stored_successor():
    """Including at an ending cut off by a step limit, which is the hard case.

    The row after an ending holds the next episode's first observation, so a
    target read from the following row would value a state the transition never
    entered. Reading the whole target pass off what replay stored makes that
    true everywhere rather than in a special case.
    """

    learner = _Learner()
    drawn = sample(
        jax.random.key(6),
        dones=[False, True],
        terminals=[False, False],
        actions=[0, 1],
        rewards=[1.0, 1.0],
    )
    # Move only the last stored successor -- the state the cut-off transition
    # actually reached. Recurrence runs forward, so nothing earlier may move.
    elsewhere = drawn.replace(
        bootstrap_inputs=drawn.bootstrap_inputs.replace(
            observation=drawn.bootstrap_inputs.observation
            + jnp.asarray([[[0.0, 0.0], [3.0, -3.0]]])
        )
    )
    params, target_params = diverged(learner, drawn)

    _, kept = learner.core._loss(params, target_params, drawn)
    _, moved = learner.core._loss(params, target_params, elsewhere)

    kept_targets = kept.q_value + kept.td_error
    moved_targets = moved.q_value + moved.td_error

    assert not np.allclose(
        np.asarray(kept_targets)[:, 1], np.asarray(moved_targets)[:, 1]
    )
    np.testing.assert_allclose(
        np.asarray(kept_targets)[:, 0], np.asarray(moved_targets)[:, 0], rtol=1e-6
    )


def test_a_terminal_ending_has_no_successor_at_all():
    learner = _Learner()
    drawn = sample(
        jax.random.key(7),
        dones=[False, True],
        terminals=[False, True],
        actions=[1, 1],
        rewards=[0.0, 3.0],
    )
    params, target_params = diverged(learner, drawn)

    _, readings = learner.core._loss(params, target_params, drawn)

    target = readings.q_value + readings.td_error
    assert float(target[0, 1]) == pytest.approx(3.0, abs=1e-6)


def test_positions_past_an_ending_do_not_enter_the_loss():
    learner = _Learner()
    drawn = sample(
        jax.random.key(8),
        dones=[True, False],
        terminals=[True, False],
        actions=[0, 0],
        rewards=[1.0, 5.0],
    )
    cut = drawn.replace(valid=jnp.asarray([[True, False]]))
    kept = drawn.replace(valid=jnp.asarray([[True, True]]))
    params, target_params = diverged(learner, drawn)

    masked, _ = learner.core._loss(params, target_params, cut)
    scored, _ = learner.core._loss(params, target_params, kept)

    _, readings = learner.core._loss(params, target_params, cut)
    np.testing.assert_allclose(
        float(masked), 0.5 * float(readings.td_error[0, 0]) ** 2, rtol=1e-6
    )
    assert not np.allclose(float(masked), float(scored))


def test_the_target_reads_the_successor_sequence_from_its_own_zero_state():
    """The target pass is over successors, not over the online pass shifted.

    The published update begins its target pass on the first successor with the
    continuation flag cleared, so the target network's hidden state at ``s_1``
    has seen ``s_1`` and nothing else. Reading the target off the online
    sequence one step along would give it a state that had also consumed
    ``s_0``. On a feed-forward Q network the two are the same number; here they
    are not, which is what this perturbs to show.
    """

    learner = _Learner()
    drawn = sample(
        jax.random.key(21),
        dones=[False, False],
        terminals=[False, False],
        actions=[0, 1],
        rewards=[1.0, 1.0],
    )
    # Move only the first current state. Every successor, and therefore every
    # target, is untouched.
    moved = drawn.replace(
        inputs=drawn.inputs.replace(
            observation=drawn.inputs.observation
            + jnp.asarray([[[5.0, -5.0], [0.0, 0.0]]])
        )
    )
    params, target_params = diverged(learner, drawn)

    _, kept = learner.core._loss(params, target_params, drawn)
    _, shifted = learner.core._loss(params, target_params, moved)

    kept_targets = kept.q_value + kept.td_error
    shifted_targets = shifted.q_value + shifted.td_error

    np.testing.assert_allclose(
        np.asarray(kept_targets), np.asarray(shifted_targets), rtol=1e-6, atol=1e-6
    )
    # The arrangement has to be one where s_0 mattered to something, or the
    # agreement above would be vacuous.
    assert not np.allclose(np.asarray(kept.q_value), np.asarray(shifted.q_value))


def test_the_online_and_target_passes_are_over_different_sequences():
    """Same length, different states: current versus arrived-at."""

    drawn = sample(
        jax.random.key(22),
        dones=[False, False],
        terminals=[False, False],
        actions=[0, 0],
        rewards=[0.0, 0.0],
    )

    assert drawn.inputs.observation.shape == drawn.bootstrap_inputs.observation.shape
    assert not np.allclose(
        np.asarray(drawn.inputs.observation),
        np.asarray(drawn.bootstrap_inputs.observation),
    )
