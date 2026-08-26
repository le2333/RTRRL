"""Why exact RTRL needs no RFLO approximation on a structured diagonal core.

RFLO and RTRL differ in one term. Writing one step of a recurrence as
``h_t = g(h_{t-1}, theta)``, the forward sensitivity obeys

    S_t = (dg/dh) S_{t-1} + dg/dtheta ,

and RFLO is what is left when ``dg/dh`` is replaced by the part of it that stays
inside a hidden unit: the cross-unit block is dropped, and the approximation bias
RFLO carries is exactly that dropped contribution, accumulated.

For the declared structured diagonal cores the cross-unit block is not small --
it is identically zero, because unit ``h``'s next state is a function of unit
``h``'s own previous state and of the input. So there is nothing for RFLO to
drop, the two recurrences coincide, and exact RTRL is available at RFLO's cost
rather than as an upgrade over it. That is the whole argument for running exact
RTRL directly, and it is the reason neither of these two cores offers RFLO:
these are tests *about* an approximation, not an implementation of one, and
`LRU_DIFFERENTIATION_FAMILY` and `RTU_DIFFERENTIATION_FAMILY` offer
`exact_rtrl` and `tbptt` and nothing else.

The CTRNN is where that stops holding, which is why it is a separate core with a
family of its own. Its unit reads every other unit's previous state, the block
below is genuinely there, and `CTRNN_DIFFERENTIATION_FAMILY` offers `rflo`
because the published `RTRRL-CTRNN-RFLO` runs it. `tests/test_ctrnn_rflo.py`
holds that gap from the other side: there RFLO and exact credit must disagree,
and by the same term.

A unit here is one complex mode: two real coordinates for both cores, since each
keeps a real and an imaginary part per mode. Within a unit the recurrent
Jacobian is a full 2x2 rotation and is carried in full; it is across units that
there is nothing.

The last section runs the same two recurrences over a deliberately non-diagonal
core, where the dropped block is not zero, and holds the difference to exactly
the term the algebra says was dropped. Without it, "the difference vanishes"
would be a statement about the harness rather than about the core.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from memorax.networks.backbones import backbone
from memorax.networks.differentiation import TruncatedBPTT
from memorax.networks.sequence import Sequence
from memorax.networks.sequence_models.lru import (
    LRU_DIFFERENTIATION_FAMILY,
    LRUCarry,
    LRUStructuredRTRL,
)
from memorax.networks.sequence_models.rtu import (
    RTU_DIFFERENTIATION_FAMILY,
    RTUCarry,
    RTUStructuredRTRL,
)
from tests.support.numerics import assert_within, deviations, flattened

CORES = ("lru", "rtu")
LENGTHS = (2, 5)

FEATURES = 3
HIDDEN = 2
UNIT = 2  # real and imaginary coordinate of one complex mode

JACOBIAN_BITS = 128.0

EXACT = {"lru": LRUStructuredRTRL, "rtu": RTUStructuredRTRL}


# ------------------------------------------------------- the two recurrences
def inside_one_unit(jacobian, coordinates: int):
    """``dg/dh`` with everything that crosses a hidden unit removed.

    This is the only difference between the two recurrences below, so it is
    written once and both read it.

    The state is laid out coordinate-major: both cores write every real part and
    then every imaginary part, so one unit's ``coordinates`` entries sit ``units``
    apart rather than adjacent, and the mask that keeps them is a tiled identity
    rather than a block diagonal.
    """

    units = jacobian.shape[-1] // coordinates
    return jacobian * jnp.tile(jnp.eye(units), (coordinates, coordinates))


def sensitivities(step, params, states, sequence, *, coordinates: int, whole: bool):
    """Run the forward sensitivity recurrence over a frozen sequence.

    ``whole`` selects RTRL over RFLO: the recurrent Jacobian is used entire, or
    only inside a unit. Everything else -- the local Jacobian, the order, the
    frozen states it is measured at -- is shared, so a difference between two
    calls is the dropped block and nothing else.
    """

    carried = jax.tree.map(
        lambda leaf: jnp.zeros((states[0].size, *leaf.shape)), params
    )
    for index, inputs in enumerate(sequence):
        previous = states[index]
        recurrent = jax.jacobian(lambda state: step(params, state, inputs))(previous)
        if not whole:
            recurrent = inside_one_unit(recurrent, coordinates)
        local = jax.jacobian(lambda tree: step(tree, previous, inputs))(params)
        carried = jax.tree.map(
            lambda held, immediate: jnp.einsum("ij,j...->i...", recurrent, held)
            + immediate,
            carried,
            local,
        )
    return carried


def visited(step, params, sequence):
    """The states the recurrence passes through, which both recurrences read."""

    state = jnp.zeros((HIDDEN * UNIT,), dtype=jnp.float32)
    states = [state]
    for inputs in sequence[:-1]:
        state = step(params, state, inputs)
        states.append(state)
    return states


# ------------------------------------------------------- the structured cores
def structured(kind: str, length: int, seed: int = 0):
    """One recurrent component, its parameters, and a frozen input sequence."""

    keys = jax.random.split(jax.random.key(seed), 3)
    network = Sequence(components=backbone(kind, features=FEATURES, hidden_dim=HIDDEN))
    live = jnp.zeros((1, 1), dtype=jnp.bool_)
    sequence = [
        jax.random.normal(jax.random.fold_in(keys[0], index), (FEATURES,))
        for index in range(length)
    ]

    carry = network.initialize_carry(keys[1], (1, FEATURES))
    params = network.init(
        keys[2], sequence[0][None, None], done=live, initial_carry=carry
    )["params"]
    leaves, structure = jax.tree.flatten(params)
    params = jax.tree.unflatten(
        structure,
        [
            leaf
            + 0.3 * jax.random.normal(jax.random.fold_in(keys[1], index), leaf.shape)
            for index, leaf in enumerate(leaves)
        ],
    )

    def step(tree, state, inputs):
        (carries, _), _ = network.walk(
            tree,
            inputs[None, None],
            done=live,
            carries=[_carry(kind, state)],
            differentiation_state=None,
            differentiation=TruncatedBPTT(network.core),
        )
        return _vector(kind, carries[0])

    return network, params, step, sequence


def _carry(kind: str, state):
    """A hidden state written as one real vector, back in the core's own shape."""

    real, imaginary = state[:HIDDEN][None], state[HIDDEN:][None]
    if kind == "rtu":
        return RTUCarry(real=real, imaginary=imaginary)
    return LRUCarry(
        state=(real + 1j * imaginary)[:, None],
        decay=jnp.ones((1, 1, HIDDEN), dtype=jnp.complex64),
    )


def _vector(kind: str, carry):
    if kind == "rtu":
        return jnp.concatenate([carry.real[0], carry.imaginary[0]])
    state = carry.state[0, 0]
    return jnp.concatenate([state.real, state.imag])


@pytest.mark.parametrize("kind", CORES)
def test_the_recurrent_jacobian_of_a_structured_core_crosses_no_unit(kind):
    """Exactly zero, not small: the coupling is never formed.

    Measured on the forward walk rather than read off the equations, because
    what the RFLO argument needs is a property of the code that runs.
    """

    _, params, step, sequence = structured(kind, 3)

    for index, state in enumerate(visited(step, params, sequence)):
        recurrent = jax.jacobian(lambda value: step(params, value, sequence[index]))(
            state
        )
        across = recurrent - inside_one_unit(recurrent, UNIT)
        assert jnp.count_nonzero(across) == 0, (
            f"{kind}: step {index} moves one hidden unit's state with another's, "
            "so RFLO would be dropping something after all"
        )
        assert jnp.count_nonzero(inside_one_unit(recurrent, UNIT)) > 0, (
            f"{kind}: step {index} has no recurrent Jacobian at all, so this "
            "sequence cannot say anything about dropping part of one"
        )


@pytest.mark.parametrize("kind", CORES)
@pytest.mark.parametrize("length", LENGTHS)
def test_rflo_and_rtrl_coincide_on_a_structured_core(kind, length):
    """The consequence, run rather than argued: the same numbers, bit for bit.

    Both recurrences are driven over the same frozen states, so the only thing
    that can separate them is the block one of them dropped.
    """

    _, params, step, sequence = structured(kind, length)
    states = visited(step, params, sequence)

    exact = sensitivities(step, params, states, sequence, coordinates=UNIT, whole=True)
    approximate = sensitivities(
        step, params, states, sequence, coordinates=UNIT, whole=False
    )

    assert_within(
        flattened(approximate),
        flattened(exact),
        f"{kind}: RFLO against RTRL over {length} steps",
    )


@pytest.mark.parametrize("kind", CORES)
@pytest.mark.parametrize("length", LENGTHS)
def test_that_recurrence_is_the_sensitivity_the_core_carries(kind, length):
    """And the algebra above is about this implementation, not about a cousin.

    The recurrence in this module is written from ``dg/dh`` and ``dg/dtheta``
    measured on the forward walk; the core carries its own analytic sensitivity.
    Holding one to the other is what stops the RFLO argument from being true of
    a recurrence nothing runs.
    """

    network, params, step, sequence = structured(kind, length)
    states = visited(step, params, sequence)
    exact = sensitivities(step, params, states, sequence, coordinates=UNIT, whole=True)

    # The differentiation addresses the recurrent component alone, so it is
    # handed that component's slice of the tree, exactly as ``walk`` hands it.
    place = network.recurrent
    assert place is not None, f"{kind} built no recurrent component"
    held = network.names[place]
    differentiation = EXACT[kind](network.core)
    carried = differentiation.initialize(jax.random.key(0), (1, FEATURES))
    carry = network.initialize_carry(jax.random.key(0), (1, FEATURES))[place]
    live = jnp.zeros((1, 1), dtype=jnp.bool_)
    for inputs in sequence:
        carry, _, carried = differentiation(
            params[held], inputs[None, None], live, carry, carried
        )

    wanted, got = {}, {}
    for name, leaf in exact[held]["cell"].items():
        if name not in carried:
            continue
        # (2 * hidden, hidden, *parameter) -> the unit-matched part, which is
        # all the structured core keeps and all the test above says there is.
        block = leaf.reshape(2, HIDDEN, *leaf.shape[1:])
        wanted[name] = jnp.moveaxis(jnp.diagonal(block, axis1=1, axis2=2), -1, 1)
        got[name] = _carried_pair(kind, carried[name])

    assert wanted, f"{kind}: no recurrent parameter block was compared"
    assert_within(
        flattened(got),
        flattened(wanted),
        f"{kind}: carried sensitivity against the measured recurrence",
        allowed=JACOBIAN_BITS,
    )


def _carried_pair(kind: str, sensitivity):
    """One stream's sensitivity with its two real components on one axis."""

    if kind == "rtu":
        return sensitivity[0]
    value = sensitivity[0, 0]
    return jnp.stack([value.real, value.imag])


# ------------------------------------------------------------ the counterexample
def dense(length: int, units: int = 3, seed: int = 1):
    """``h_t = tanh(W h_{t-1} + U x_t)``: a core that does cross its units.

    Written here rather than registered anywhere. It exists to show that the
    two recurrences above are capable of disagreeing, which is what makes their
    agreement on the structured cores worth reporting.
    """

    keys = jax.random.split(jax.random.key(seed), 3)
    params = {
        "W": jax.random.normal(keys[0], (units, units)) / jnp.sqrt(units),
        "U": jax.random.normal(keys[1], (units, FEATURES)) / jnp.sqrt(FEATURES),
    }
    sequence = [
        jax.random.normal(jax.random.fold_in(keys[2], index), (FEATURES,))
        for index in range(length)
    ]

    def step(tree, state, inputs):
        return jnp.tanh(tree["W"] @ state + tree["U"] @ inputs)

    return params, step, sequence


@pytest.mark.parametrize("length", LENGTHS)
def test_rflo_and_rtrl_part_company_on_a_core_that_crosses_its_units(length):
    """Nonzero, and nonzero by exactly the term the algebra says was dropped.

    The second assertion is the one with content. That two numbers differ says
    little; that they differ by ``(dg/dh - inside one unit) S`` accumulated says
    the difference between the methods has been located rather than observed.
    """

    params, step, sequence = dense(length)
    units = params["W"].shape[0]
    states = [jnp.zeros((units,), dtype=jnp.float32)]
    for inputs in sequence[:-1]:
        states.append(step(params, states[-1], inputs))

    exact = sensitivities(step, params, states, sequence, coordinates=1, whole=True)
    approximate = sensitivities(
        step, params, states, sequence, coordinates=1, whole=False
    )

    assert deviations(flattened(approximate), flattened(exact)), (
        "dropping the cross-unit recurrent Jacobian of a dense tanh recurrence "
        "changed nothing, so the comparison is not measuring what it says"
    )

    # The same recursion again, carrying only the difference: it is driven by
    # the dropped block acting on the exact sensitivity, and propagated by the
    # part RFLO kept.
    carried = jax.tree.map(lambda leaf: jnp.zeros((units, *leaf.shape)), params)
    running = jax.tree.map(lambda leaf: jnp.zeros((units, *leaf.shape)), params)
    for index, inputs in enumerate(sequence):
        previous = states[index]
        recurrent = jax.jacobian(lambda state: step(params, state, inputs))(previous)
        kept = inside_one_unit(recurrent, 1)
        local = jax.jacobian(lambda tree: step(tree, previous, inputs))(params)
        dropped = recurrent - kept
        running = jax.tree.map(
            lambda gap, held: jnp.einsum("ij,j...->i...", kept, gap)
            + jnp.einsum("ij,j...->i...", dropped, held),
            running,
            carried,
        )
        carried = jax.tree.map(
            lambda held, immediate: jnp.einsum("ij,j...->i...", recurrent, held)
            + immediate,
            carried,
            local,
        )

    assert_within(
        flattened(jax.tree.map(jnp.subtract, exact, approximate)),
        flattened(running),
        "the RFLO-RTRL gap against the dropped cross-unit term",
        allowed=JACOBIAN_BITS,
    )


# ------------------------------------------------------------------ the naming
def test_no_rflo_implementation_is_offered_under_any_name():
    """The argument above is a reason not to have one, so there must not be one.

    Naming exact RTRL ``rflo``, or offering an ``rflo`` branch that resolves to
    it, would make every R1 result answer to a method description it does not
    implement -- which is the confusion this module exists to settle.
    """

    for family in (LRU_DIFFERENTIATION_FAMILY, RTU_DIFFERENTIATION_FAMILY):
        assert set(family.branches) == {"exact_rtrl", "tbptt"}

    for kind in CORES:
        network = Sequence(
            components=backbone(kind, features=FEATURES, hidden_dim=HIDDEN)
        )
        chosen = EXACT[kind](network.core)
        assert "rflo" not in type(chosen).__name__.lower()
