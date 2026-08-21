"""``core.kind: lstm``: the paper's own cell, checked where selecting it differs.

The rest of the DRQN suite runs against the matched cores -- the structured
diagonal cells the online arm carries exact recurrent sensitivity through --
because that is what most of this learner is answerable to. Selecting ``lstm``
changes exactly one thing, the network between the observation and the linear Q
head, and this file checks that the learner reads that network at each of the
four places it reads a core: the state a window opens on, one step of acting,
the loss over a drawn window, and the target pass.

Nothing here re-checks replay, the schedule or the solver. Those do not know
which cell they are running over, and a copy of their tests under a second core
would say only that a parameter had been threaded.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.traverse_util import flatten_dict

from memorax.algorithms import drqn
from memorax.algorithms.drqn import Core, LearnerSequence, QFunction, RecurrentInputs

# The width the acceptance manifest pins, and the paper's own: DRQN replaces
# DQN's first fully-connected layer with an LSTM of the same size.
HIDDEN = 32
OBSERVATION = 3
ACTIONS = 2
TRANSITIONS = 4


def q_function(**overrides):
    settings = {
        "action_dim": ACTIONS,
        "observation_dim": OBSERVATION,
        "hidden_dim": HIDDEN,
        # An LSTM's output is its hidden state, so there is no readout to size.
        "feature_dim": None,
        "core_kind": "lstm",
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


def step(key):
    """One timestep for one environment, which is what this learner acts on."""

    return RecurrentInputs(
        observation=jax.random.normal(key, (1, 1, OBSERVATION)),
        episode_start=jnp.asarray([[True]]),
    )


def window(key, *, terminals=None):
    """A drawn window: ``TRANSITIONS`` transitions and the states they reached."""

    terminals = [False] * TRANSITIONS if terminals is None else terminals
    walk = jax.random.normal(key, (1, TRANSITIONS + 1, OBSERVATION))
    return LearnerSequence(
        inputs=RecurrentInputs(
            observation=walk[:, :-1],
            episode_start=jnp.asarray([[True] + [False] * (TRANSITIONS - 1)]),
        ),
        bootstrap_inputs=RecurrentInputs(
            observation=walk[:, 1:],
            episode_start=jnp.zeros((1, TRANSITIONS), dtype=jnp.bool_),
        ),
        actions=jnp.asarray([[0, 1, 1, 0]]),
        rewards=jnp.asarray([[1.0, -1.0, 0.0, 1.0]]),
        dones=jnp.asarray([terminals]),
        terminals=jnp.asarray([terminals]),
        valid=jnp.asarray([[True] * TRANSITIONS]),
        batch_valid=jnp.asarray([True]),
    )


def weight(tree, *path):
    """One weight out of a flattened parameter tree, as numpy.

    ``flatten_dict`` types its values as whatever the tree it walked held, which
    is a union numpy will not take directly, so the read walks to the leaf the
    way the rest of the DRQN tests do.
    """

    return np.asarray(jax.tree.leaves(tree[path])[0])


def diverged(learner, drawn):
    """Online and target parameters that are not each other."""

    state = learner.init(jax.random.key(11), drawn.inputs)
    moved, _ = learner.update_parameters(jax.random.key(12), state, drawn)
    return moved.params, state.target_params


def test_the_cell_is_read_by_the_head_and_by_nothing_in_between():
    """One component, no projection in front of it and no normalisation behind.

    The matched cores are normalised after the cell because that is the online
    arm's own topology. The published network has nothing there: the LSTM output
    is the Q head's input, so the head's kernel is as wide as the hidden state.
    """

    function = q_function()
    components = function.network.sequence().components

    assert len(components) == 1
    assert getattr(components[0], "recurrent", False)
    assert not [
        component for component in components if type(component).__name__ == "LayerNorm"
    ]

    params, _ = function.init(jax.random.key(0), step(jax.random.key(1)))
    tree = flatten_dict(params)
    gates = ("i", "f", "g", "o")

    # Four gates, each reading the observation at its own width and the hidden
    # state at the cell's. The input kernels being ``OBSERVATION`` wide is what
    # says there is no projection in front of the cell.
    for gate in gates:
        cell = ("params", "OptimizedLSTMCell_0")
        assert weight(tree, *cell, f"i{gate}", "kernel").shape == (OBSERVATION, HIDDEN)
        assert weight(tree, *cell, f"h{gate}", "kernel").shape == (HIDDEN, HIDDEN)
    # And the head reads the hidden state, at the hidden state's width.
    head = ("params", "DiscreteQNetwork_0", "Dense_0", "kernel")
    assert weight(tree, *head).shape == (HIDDEN, ACTIONS)
    # Nothing else has weights, which is the whole of "one cell and one head".
    assert {path[1] for path in tree} == {"OptimizedLSTMCell_0", "DiscreteQNetwork_0"}


def test_a_window_opens_on_a_zero_cell_state_and_a_zero_hidden_state():
    """Both halves of the carry, because only one of them reaches the head.

    A hidden state zeroed over a cell state that was not would leave the first
    step of a window reading memory the sampler never gave it, and the head
    would not show it -- the head sees the hidden state alone.
    """

    function = q_function()
    opened = function.reset(drqn.ZERO_MEMORY, 2)

    leaves = jax.tree.leaves(opened)
    assert len(leaves) == 2
    for leaf in leaves:
        assert leaf.shape == (2, HIDDEN)
        assert np.allclose(np.asarray(leaf), 0.0)
    # The key carries no information, which is what lets every window in the
    # learner open on the same ``ZERO_MEMORY`` without that being a seed.
    for keyed, zeroed in zip(
        jax.tree.leaves(function.reset(jax.random.key(3), 2)), leaves
    ):
        assert np.array_equal(np.asarray(keyed), np.asarray(zeroed))


def test_acting_one_step_moves_both_states_and_takes_the_head_argmax():
    learner = core()
    timestep = step(jax.random.key(1))
    state = learner.init(jax.random.key(0), timestep)

    recurrence, action, metrics = learner.act(
        jax.random.key(2), state, timestep, epsilon=jnp.asarray(0.0)
    )

    _, q_values = learner.q_function.apply(state.params, timestep, state.recurrence)
    assert action.shape == (1,)
    assert int(action[0]) == int(jnp.argmax(q_values[:, 0], axis=-1)[0])
    assert float(metrics.epsilon) == 0.0
    # Both carries advanced. A cell state left where it was would be an LSTM
    # with no memory beyond the step it is on, which the head cannot report.
    for before, after in zip(
        jax.tree.leaves(state.recurrence), jax.tree.leaves(recurrence)
    ):
        assert not np.allclose(np.asarray(before), np.asarray(after))


def test_the_window_loss_scores_the_lstm_unrolled_from_the_zero_state():
    """The same expression as on a matched core, over this cell's outputs."""

    learner = core()
    drawn = window(jax.random.key(4))
    params, target_params = diverged(learner, drawn)

    loss, readings = learner._loss(params, target_params, drawn)

    opening = learner.q_function.reset(drqn.ZERO_MEMORY, 1)
    _, online_q, _ = learner.q_function.unroll(params, drawn.inputs, opening)
    _, successor_q, _ = learner.q_function.unroll(
        target_params, drawn.bootstrap_inputs, opening
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
    np.testing.assert_allclose(
        float(loss),
        float(drqn.published_loss(readings.td_error, drawn.valid)),
        rtol=1e-6,
    )


def test_the_gradient_of_a_window_reaches_back_to_its_first_observation():
    """TBPTT(t) over a dense-gated cell is still TBPTT(t).

    Only the last transition is scored, so whatever the gradient reaches at the
    window's first observation it reached through the carry. A cell wired up
    without one -- an LSTM the wrapper never scanned, say -- would leave that
    gradient at exactly zero while every other test here still passed.
    """

    learner = core()
    drawn = window(jax.random.key(5))
    scored_at_the_end = drawn.replace(
        valid=jnp.asarray([[False] * (TRANSITIONS - 1) + [True]])
    )
    state = learner.init(jax.random.key(6), drawn.inputs)

    def loss_of(observation):
        moved = scored_at_the_end.replace(
            inputs=scored_at_the_end.inputs.replace(observation=observation)
        )
        loss, _ = learner._loss(state.params, state.target_params, moved)
        return loss

    gradient = np.asarray(jax.grad(loss_of)(drawn.inputs.observation))

    assert np.any(np.abs(gradient[0, 0]) > 1e-8)
    # And the first step is not the only one it touched, which is what separates
    # a window from a sequence of independent one-step updates.
    assert np.any(np.abs(gradient[0, 1:-1]) > 1e-8)


def test_the_target_pass_reads_the_successors_from_its_own_zero_state():
    """Moving the first current state moves no target, on this cell too.

    The target network unrolls the successor sequence from zero rather than
    reading the online pass shifted by one. On an LSTM the difference is a
    hidden state that has or has not already consumed ``s_0``, so a target that
    moved here would be the shifted reading.
    """

    learner = core()
    drawn = window(jax.random.key(21))
    displacement = jnp.zeros((1, TRANSITIONS, OBSERVATION)).at[0, 0].set(5.0)
    moved = drawn.replace(
        inputs=drawn.inputs.replace(observation=drawn.inputs.observation + displacement)
    )
    params, target_params = diverged(learner, drawn)

    _, kept = learner._loss(params, target_params, drawn)
    _, shifted = learner._loss(params, target_params, moved)

    np.testing.assert_allclose(
        np.asarray(kept.q_value + kept.td_error),
        np.asarray(shifted.q_value + shifted.td_error),
        rtol=1e-6,
        atol=1e-6,
    )
    # Vacuous unless s_0 mattered to the online pass, so make it say that it did.
    assert not np.allclose(np.asarray(kept.q_value), np.asarray(shifted.q_value))
