"""``backbone.kind: lstm``: R2D2 on the cell it was published on.

The rest of the R2D2 suite runs against the matched cores -- the structured
diagonal cells the online arm carries exact recurrent sensitivity through --
because a matched pair of runs is what most of this repository reads R2D2 for.
Selecting ``lstm`` changes exactly one component, the recurrent one, and this
file checks the places where a dense-gated cell could differ from the cores the
suite already covers: the two states a reset has to clear rather than one, the
burn-in boundary the carry crosses, the unroll boundary the gradient stops at,
and the target pass reading its own parameters through its own carry.

Two things it deliberately does not check. Replay, the schedule, the priority
arithmetic and the solver do not know which cell they run over, so a copy of
their tests under a third core would say only that a parameter had been
threaded. And nothing here is DRQN's: the learner around this cell is R2D2's,
which is the whole point of being able to name the same cell in both.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.traverse_util import flatten_dict

from memorax.algorithms.r2d2 import (
    Core,
    LearnerSequence,
    QFunction,
    RecurrentInputs,
    _burn_in,
)

# The width R1.1.2 pins on both sides of the comparison: DRQN-LSTM runs the
# published cell at 32, and an R2D2-LSTM read against it runs at 32 too.
HIDDEN = 32
# R2D2's encoder width, which is R2D2's and not the cell's. DRQN's LSTM reads
# the observation directly and so declares no such width; here the cell reads
# whatever the encoder emits, and this is how wide that is.
FEATURE = 4
OBSERVATION = 2
ACTIONS = 2
TRANSITIONS = 4
BURN_IN = 2
UNROLL = 2

CORES = ("lru", "rtu", "lstm")


def q_function(backbone_kind="lstm", head_kind="dueling"):
    return QFunction(
        action_dim=ACTIONS,
        feature_dim=FEATURE,
        hidden_dim=HIDDEN,
        backbone_kind=backbone_kind,
        head_kind=head_kind,
    )


def learner(function, *, learning_kind="tbptt", burn_in_length=BURN_IN):
    return Core(
        q_function=function,
        optimizer=optax.sgd(0.01),
        gamma=0.5,
        n_step=1,
        burn_in_length=burn_in_length,
        unroll_length=UNROLL,
        importance_sampling_exponent=0.4,
        max_priority_weight=0.75,
        target_update_period=2,
        transform=lambda value: value,
        inverse_transform=lambda value: value,
        learning_kind=learning_kind,
    )


def one_step(key):
    return RecurrentInputs(
        observation=jax.random.normal(key, (1, 1, OBSERVATION)),
        previous_action=jnp.zeros((1, 1), dtype=jnp.int32),
        previous_reward=jnp.zeros((1, 1)),
        episode_start=jnp.asarray([[True]]),
    )


def drawn_sequence(initial_recurrence, observation):
    """One replay item: ``TRANSITIONS`` transitions and the states around them.

    The shape a burn-in reads: ``TRANSITIONS + 1`` inputs, of which the first
    ``BURN_IN`` are there only to rebuild a carry.
    """

    inputs = RecurrentInputs(
        observation=observation,
        previous_action=jnp.zeros((1, TRANSITIONS + 1), dtype=jnp.int32),
        previous_reward=jnp.zeros((1, TRANSITIONS + 1)),
        episode_start=jnp.zeros((1, TRANSITIONS + 1), dtype=jnp.bool_),
    )
    return LearnerSequence(
        inputs=inputs,
        bootstrap_inputs=RecurrentInputs(
            observation=observation[:, 1:],
            previous_action=jnp.zeros((1, TRANSITIONS), dtype=jnp.int32),
            previous_reward=jnp.zeros((1, TRANSITIONS)),
            episode_start=jnp.zeros((1, TRANSITIONS), dtype=jnp.bool_),
        ),
        actions=jnp.asarray([[0, 1, 1, 0]]),
        rewards=jnp.asarray([[1.0, -1.0, 0.0, 1.0]]),
        dones=jnp.zeros((1, TRANSITIONS), dtype=jnp.bool_),
        terminals=jnp.zeros((1, TRANSITIONS), dtype=jnp.bool_),
        valid=jnp.ones((1, TRANSITIONS), dtype=jnp.bool_),
        initial_recurrence=initial_recurrence,
        probabilities=jnp.asarray([1.0]),
        indices=jnp.asarray([0]),
        buffer_size=jnp.asarray(8),
    )


def walk(key):
    return jax.random.normal(key, (1, TRANSITIONS + 1, OBSERVATION))


def poisoned(recurrence, index, value=3.0):
    """The same carry with one of its states set to something a step notices."""

    leaves, treedef = jax.tree_util.tree_flatten(recurrence)
    replaced = [
        jnp.full_like(leaf, value) if position == index else leaf
        for position, leaf in enumerate(leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, replaced)


def test_the_encoder_feeds_the_cell_and_the_head_reads_it_directly():
    """R2D2's own graph, with the cell in the one place a core goes.

    The encoder in front is what makes this R2D2's network rather than DRQN's:
    the published R2D2 runs its recurrent layer on encoded features, where the
    published DRQN replaces a fully-connected layer and reads the observation.
    Selecting ``lstm`` moves the recurrent component and nothing else, which is
    what the gate kernels being ``FEATURE`` wide says.
    """

    function = q_function()
    components = function.network.sequence().components
    recurrent = [
        component for component in components if getattr(component, "recurrent", False)
    ]

    assert [type(component).__name__ for component in components] == [
        "FFN",
        "LayerNorm",
        "Tanh",
        "RNN",
    ]
    assert len(recurrent) == 1

    params, _ = function.init(jax.random.key(0), one_step(jax.random.key(1)))
    tree = flatten_dict(params)
    cell = ("params", "OptimizedLSTMCell_0")
    for gate in ("i", "f", "g", "o"):
        assert np.asarray(tree[(*cell, f"i{gate}", "kernel")]).shape == (
            FEATURE,
            HIDDEN,
        )
        assert np.asarray(tree[(*cell, f"h{gate}", "kernel")]).shape == (HIDDEN, HIDDEN)
    assert (
        np.asarray(tree[("params", "FFN_0", "Dense_0", "kernel")]).shape[1] == FEATURE
    )
    # The dueling head reads the cell's output, so both its streams are as wide
    # as the hidden state. Nothing stands between them.
    for stream in ("value", "advantage"):
        kernel = np.asarray(tree[("params", "DuelingQHead_0", stream, "kernel")])
        assert kernel.shape[0] == HIDDEN


def test_a_reset_clears_the_cell_state_as_well_as_the_hidden_state():
    """Both halves of the carry, because only one of them reaches the head.

    A hidden state zeroed over a cell state that was not would leave the next
    step reading memory from before the episode, and no reading of the network
    would show it: the head sees the hidden state alone.
    """

    function = q_function()
    opened = function.reset(jax.random.key(0), 2)

    leaves = jax.tree.leaves(opened)
    assert len(leaves) == 2
    for leaf in leaves:
        assert leaf.shape == (2, HIDDEN)
        np.testing.assert_array_equal(np.asarray(leaf), 0.0)
    # The key carries no information: an LSTM opens on zeros, not on a draw.
    for keyed, zeroed in zip(
        jax.tree.leaves(function.reset(jax.random.key(3), 2)), leaves
    ):
        np.testing.assert_array_equal(np.asarray(keyed), np.asarray(zeroed))


@pytest.mark.parametrize("index", (0, 1))
def test_an_episode_start_clears_every_carry_before_the_input_is_read(index):
    """Neither state survives an episode boundary inside a sequence.

    Checked one state at a time, so that a reset which cleared only the hidden
    state -- the one the head reads, and so the one an end-to-end comparison
    would still look right on -- fails on the other.
    """

    function = q_function()
    inputs = one_step(jax.random.key(4))
    params, opened = function.init(jax.random.key(5), inputs)
    carried = poisoned(opened, index)

    started = inputs.replace(episode_start=jnp.asarray([[True]]))
    from_poison, poisoned_q = function.apply(params, started, carried)
    from_zero, zeroed_q = function.apply(params, started, opened)

    np.testing.assert_allclose(np.asarray(poisoned_q), np.asarray(zeroed_q))
    for left, right in zip(jax.tree.leaves(from_poison), jax.tree.leaves(from_zero)):
        np.testing.assert_allclose(np.asarray(left), np.asarray(right))

    # Vacuous unless that state is one a step reads, so make it say that it is.
    continued = inputs.replace(episode_start=jnp.asarray([[False]]))
    _, carried_q = function.apply(params, continued, carried)
    _, fresh_q = function.apply(params, continued, opened)
    assert not np.allclose(np.asarray(carried_q), np.asarray(fresh_q))


@pytest.mark.parametrize("backbone_kind", CORES)
def test_the_burn_in_rebuilds_the_carry_without_a_gradient_crossing_it(backbone_kind):
    """Every core's burn-in, checked as one property over all three.

    The stored actor recurrence and the burn-in inputs reach the loss only
    through a carry ``_burn_in`` has stopped the gradient at, so their gradients
    are exactly zero while the cell's own parameters still take one. Parametrised
    rather than written for the LSTM alone because "the same boundary as the
    matched cores" is the claim, and a per-core copy would not state it.
    """

    function = q_function(backbone_kind)
    params, opened = function.init(jax.random.key(7), one_step(jax.random.key(6)))
    core = learner(function)
    observations = walk(jax.random.key(8))

    def loss_of(weights, observation, recurrence):
        sample = drawn_sequence(recurrence, observation)
        loss, _ = core._tbptt_loss(weights, params, sample, jnp.ones((1,)))
        return loss

    weight_gradient, input_gradient, carry_gradient = jax.grad(
        loss_of, argnums=(0, 1, 2)
    )(params, observations, opened)

    np.testing.assert_array_equal(np.asarray(input_gradient)[:, :BURN_IN], 0.0)
    for leaf in jax.tree.leaves(carry_gradient):
        np.testing.assert_array_equal(np.asarray(leaf), 0.0)
    recurrent = [
        np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(weight_gradient)
        if "Cell" in "/".join(str(part) for part in path)
    ]
    assert recurrent
    assert any(np.any(np.abs(gradient) > 0.0) for gradient in recurrent)


@pytest.mark.parametrize("backbone_kind", CORES)
def test_the_unroll_gradient_reaches_the_whole_window_and_no_further(backbone_kind):
    """The TBPTT boundary, asserted as the same mask for all three cores.

    Three regions, and the same three whichever core is selected. Zero over the
    burn-in, because the carry it built was stopped. Non-zero over every input a
    scored transition reads -- including the first, which the gradient can only
    have reached through the carry, and which is what separates a window from a
    sequence of independent one-step updates. Zero again at the input past the
    last scored transition: it enters only the bootstrap value, whose gradient
    the target stops, and the argmax that selects against it.
    """

    function = q_function(backbone_kind)
    params, opened = function.init(jax.random.key(11), one_step(jax.random.key(10)))
    core = learner(function)
    observations = walk(jax.random.key(12))

    def loss_of(observation):
        sample = drawn_sequence(opened, observation)
        loss, _ = core._tbptt_loss(params, params, sample, jnp.ones((1,)))
        return loss

    gradient = np.asarray(jax.grad(loss_of)(observations))
    touched = np.any(np.abs(gradient[0]) > 1e-9, axis=-1)

    scored = [False] * BURN_IN + [True] * UNROLL
    np.testing.assert_array_equal(
        touched, scored + [False] * (len(touched) - len(scored))
    )


def test_the_target_pass_reads_its_own_parameters_through_its_own_carry():
    """One backbone kind, two networks, and no state shared between them.

    Both carries open on the stored actor recurrence -- that is R2D2's rule, and
    it does not change with the cell -- but each is warmed through its own
    parameters from there, so a target that had been handed the online carry
    would agree with the online pass here, and does not.
    """

    function = q_function()
    params, opened = function.init(jax.random.key(15), one_step(jax.random.key(14)))
    target_params = jax.tree.map(lambda leaf: leaf * 0.5 + 0.1, params)
    core = learner(function)
    sample = drawn_sequence(opened, walk(jax.random.key(16)))

    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
        target_params
    )

    warmed, target_warmed, _ = _burn_in(
        function,
        params,
        target_params,
        sample.inputs,
        sample.initial_recurrence,
        sample.initial_recurrence,
        burn_in_length=BURN_IN,
    )
    for online_state, target_state in zip(
        jax.tree.leaves(warmed), jax.tree.leaves(target_warmed)
    ):
        assert not np.allclose(np.asarray(online_state), np.asarray(target_state))

    # And the loss is routed to them: moving the target alone moves it.
    kept, _ = core._tbptt_loss(params, target_params, sample, jnp.ones((1,)))
    moved, _ = core._tbptt_loss(
        params,
        jax.tree.map(lambda leaf: leaf + 0.05, target_params),
        sample,
        jnp.ones((1,)),
    )
    assert not np.allclose(float(kept), float(moved))


def test_the_target_network_takes_the_online_parameters_at_the_period():
    """The copy R2D2 already performs, over an LSTM's parameter tree.

    Worth one assertion rather than none: the tree has four gates' worth of
    leaves where the matched cores have their own, and a copy that reached only
    part of it would leave a run looking healthy.
    """

    inputs = one_step(jax.random.key(17))
    core = learner(q_function())
    state = core.init(jax.random.key(18), inputs)
    sample = drawn_sequence(state.recurrence, walk(jax.random.key(19)))

    first, _, _ = core.update_parameters(
        jax.random.key(20), state, sample, step=jnp.asarray(1)
    )
    second, _, _ = core.update_parameters(
        jax.random.key(21), first, sample, step=jnp.asarray(2)
    )

    # ``target_update_period`` is 2, so the first update leaves the target where
    # it was and the second hands it the online parameters entire.
    for before, after in zip(
        jax.tree.leaves(state.target_params), jax.tree.leaves(first.target_params)
    ):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    for online, target in zip(
        jax.tree.leaves(second.params), jax.tree.leaves(second.target_params)
    ):
        np.testing.assert_allclose(np.asarray(online), np.asarray(target))


def test_full_bptt_opens_the_cell_on_zeros_and_ignores_the_stored_carry():
    """The full-episode branch works on this core, and on the same terms.

    Its window is the episode, so there is no state to carry in; a stored actor
    recurrence handed to it is padding it must not read. On a cell with two
    states that is two things to ignore rather than one.
    """

    function = q_function()
    params, opened = function.init(jax.random.key(23), one_step(jax.random.key(22)))
    core = learner(function, learning_kind="full_bptt", burn_in_length=0)
    observations = walk(jax.random.key(24))

    def loss_from(recurrence):
        loss, _ = core._full_bptt_loss(
            params,
            params,
            drawn_sequence(recurrence, observations),
            jnp.ones((1,)),
        )
        return float(loss)

    both = jax.tree.map(lambda leaf: jnp.full_like(leaf, 3.0), opened)

    assert np.isfinite(loss_from(opened))
    np.testing.assert_allclose(loss_from(opened), loss_from(poisoned(opened, 0)))
    np.testing.assert_allclose(loss_from(opened), loss_from(poisoned(opened, 1)))
    np.testing.assert_allclose(loss_from(opened), loss_from(both))
