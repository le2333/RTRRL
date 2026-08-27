"""The LSTM cell, and RFLO as the thing that credits it.

Five questions, and they need different judges:

``the transition``
    One LSTM step against the equations, computed here in numpy from the
    parameters alone. Nothing about differentiation is involved, so a failure
    is the forward and nothing else.

``the local derivatives``
    Each factor the trace's immediate term is built from, against autodiff of
    the same step with the two carried states held. This is the piece a reader
    has to take on trust everywhere else: the recurrence below is written from
    the algebra, and this is the only thing saying the algebra was done right.

``the sensitivity``
    The three trace recurrences against a numpy reference, over a sequence that
    contains an ending. The forget gate is the leak, so this is where a trace
    that forgot at the wrong rate would show.

``the phantom``
    That the gradient autodiff produces through the differentiated scan is
    exactly the cotangent contracted with the carried trace. The trace and the
    gradient are written in two places, and this is the only thing holding them
    to being one statement. Read with a cotangent on the cell state it is an
    equality; read with one on the output it acquires the ``o * tanh'(c)``
    factor, and both are asserted, because that factor is the whole of what
    separates the state the trace is a derivative of from the value the network
    hands on.

``what RFLO drops``
    RFLO against exact credit through a truncation-free unroll. Over one
    transition the two must agree -- there is no history to approximate -- and
    over three they must not, with the gap held to exactly the term the algebra
    says was dropped. Without both halves, "RFLO is implemented" would be a
    claim no test could distinguish from either "RTRL is implemented" or "the
    gradient is wrong".

The derivation these hold is in ``docs/rtrrl-lstm-rflo.md``.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.networks.differentiation import TruncatedBPTT
from memorax.networks.sequence_models import RNN
from memorax.networks.sequence_models.lstm import (
    TRACED,
    LSTMCarry,
    LSTMCell,
    LSTMConfig,
    LSTMRflo,
    packed_kernel,
)
from memorax.utils.typing import Array

FEATURES = 3
HIDDEN = 4
STREAMS = 2
STEPS = 6
WIDTH = FEATURES + HIDDEN + 1

# Not one, and not zero. The forget gate is the factor every trace is
# multiplied by, so a bias the cell quietly substituted its own value for would
# be invisible at either of the two numbers someone would think to hard-code.
FORGET_BIAS = 0.6

GATES = ("W_i", "W_f", "W_g", "W_o")


def build(*, forget_bias=FORGET_BIAS):
    """A cell and the scan around it."""

    return RNN(
        cell=LSTMCell(
            config=LSTMConfig(
                features=FEATURES, hidden_dim=HIDDEN, forget_bias=forget_bias
            )
        )
    )


def inputs(steps=STEPS, *, ending=True):
    """A sequence, and where one stream's episode ends inside it."""

    x = jax.random.normal(jax.random.key(11), (STREAMS, steps, FEATURES))
    done = jnp.zeros((STREAMS, steps), dtype=bool)
    if ending and steps > 3:
        done = done.at[0, 3].set(True)
    return x, done


def started(core, x, done):
    """Parameters, an empty carry and an empty trace for one sequence."""

    carry = core.cell.initialize_carry(jax.random.key(0), (STREAMS, FEATURES))
    rflo = LSTMRflo(core)
    state = rflo.initialize(jax.random.key(0), (STREAMS, FEATURES))
    params = core.init(jax.random.key(2), x, done=done, initial_carry=carry)["params"]
    return params, carry, state, rflo


def leaf(params, name) -> Array:
    """One of the cell's four matrices, as an array rather than a tree."""

    return jnp.asarray(params["cell"][name])


def matrices(params) -> dict[str, np.ndarray]:
    return {name: np.asarray(leaf(params, name)) for name in GATES}


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def walked(core, params, x, done, carry) -> Array:
    """``RNN.apply`` with the output it returns beside the carry named."""

    _, output = cast(
        "tuple[LSTMCarry, Array]",
        core.apply({"params": params}, x, done=done, initial_carry=carry),
    )
    return output


def close(got, wanted, *, tolerance=1e-5):
    return float(jnp.max(jnp.abs(jnp.asarray(got) - jnp.asarray(wanted)))) < tolerance


def reference(params, x, done):
    """The equations, stepped one stream and one transition at a time.

    Written from ``docs/rtrrl-lstm-rflo.md`` rather than from the cell, so that
    the two are two statements and the comparison has content.
    """

    weights = matrices(params)
    xs, ds = np.asarray(x), np.asarray(done)
    streams, steps, _ = xs.shape
    cell = np.zeros((streams, HIDDEN))
    hidden = np.zeros((streams, HIDDEN))
    trace = {name: np.zeros((streams, HIDDEN, WIDTH)) for name in TRACED}
    outputs = np.zeros((streams, steps, HIDDEN))

    for step in range(steps):
        for stream in range(streams):
            if ds[stream, step]:
                cell[stream] = 0
                hidden[stream] = 0
                for name in trace:
                    trace[name][stream] = 0
            row = np.concatenate([xs[stream, step], hidden[stream], [1.0]])
            opened = sigmoid(weights["W_i"] @ row)
            forget = sigmoid(weights["W_f"] @ row)
            candidate = np.tanh(weights["W_g"] @ row)
            output = sigmoid(weights["W_o"] @ row)
            immediate = {
                "W_f": (cell[stream] * forget * (1 - forget))[:, None] * row[None, :],
                "W_i": (candidate * opened * (1 - opened))[:, None] * row[None, :],
                "W_g": (opened * (1 - candidate**2))[:, None] * row[None, :],
            }
            for name in trace:
                trace[name][stream] = (
                    forget[:, None] * trace[name][stream] + immediate[name]
                )
            cell[stream] = forget * cell[stream] + opened * candidate
            hidden[stream] = output * np.tanh(cell[stream])
            outputs[stream, step] = hidden[stream]
    return outputs, trace


# ------------------------------------------------------------- the transition
def test_one_step_is_the_lstm_recurrence():
    core = build()
    x, done = inputs(steps=1, ending=False)
    params, carry, _, _ = started(core, x, done)

    output = walked(core, params, x, done, carry)

    weights = matrices(params)
    row = np.concatenate(
        [np.asarray(x)[:, 0], np.zeros((STREAMS, HIDDEN)), np.ones((STREAMS, 1))],
        axis=-1,
    )
    # From an empty carry the forget gate multiplies nothing, so the whole of
    # the first cell state is what the input gate wrote.
    cell = sigmoid(row @ weights["W_i"].T) * np.tanh(row @ weights["W_g"].T)
    wanted = sigmoid(row @ weights["W_o"].T) * np.tanh(cell)
    assert close(output[:, 0], wanted)


def test_the_bias_column_is_read_and_the_recurrent_block_is_where_it_says():
    """Which column of each matrix means what, asserted rather than assumed.

    The layout ``[input, hidden, bias]`` is ``ctrnn.py``'s, and it is what
    makes one trace per gate have the shape of that gate's parameters. A
    transposition here would credit an input column for a recurrent connection
    and the trace would follow it there.
    """

    core = build()
    x, done = inputs(steps=1, ending=False)
    params, carry, _, _ = started(core, x, done)
    weights = matrices(params)

    zero = jnp.zeros((STREAMS, 1, FEATURES))
    only_bias = walked(core, params, zero, done, carry)
    bias = {name: weights[name][:, -1] for name in GATES}
    cell = sigmoid(bias["W_i"]) * np.tanh(bias["W_g"])
    assert close(only_bias[:, 0], sigmoid(bias["W_o"]) * np.tanh(cell))

    hidden = jax.random.normal(jax.random.key(5), (STREAMS, HIDDEN))
    state = jax.random.normal(jax.random.key(6), (STREAMS, HIDDEN))
    with_state = walked(core, params, zero, done, LSTMCarry(cell=state, hidden=hidden))
    recurrent = {name: weights[name][:, FEATURES : FEATURES + HIDDEN] for name in GATES}
    pre = {name: np.asarray(hidden) @ recurrent[name].T + bias[name] for name in GATES}
    advanced = sigmoid(pre["W_f"]) * np.asarray(state) + sigmoid(pre["W_i"]) * np.tanh(
        pre["W_g"]
    )
    assert close(with_state[:, 0], sigmoid(pre["W_o"]) * np.tanh(advanced))


def test_the_forget_gate_is_the_one_matrix_drawn_with_a_bias():
    """``forget_bias`` reaches the column it names, and only that column.

    The gate is the factor every trace is multiplied by, so where it starts is
    a statement about the horizon an online method can still credit over. The
    three other matrices are drawn with a zero column, and the connections of
    all four are drawn from a fan-in that does not count the bias as one.
    """

    core = build(forget_bias=FORGET_BIAS)
    x, done = inputs(steps=1, ending=False)
    params, _, _, _ = started(core, x, done)
    weights = matrices(params)

    assert np.allclose(weights["W_f"][:, -1], FORGET_BIAS)
    for name in ("W_i", "W_g", "W_o"):
        assert np.allclose(weights[name][:, -1], 0.0)

    drawn = np.asarray(packed_kernel(0.0)(jax.random.key(3), (256, WIDTH), jnp.float32))
    assert np.allclose(drawn[:, -1], 0.0)
    # `lecun_normal` over `WIDTH - 1` inputs, which is the fan-in the bias
    # column is deliberately not counted in.
    assert abs(float(drawn[:, :-1].std()) - (1.0 / (WIDTH - 1)) ** 0.5) < 0.02


# ------------------------------------------------------ the local derivatives
def test_each_immediate_factor_is_the_autodiff_derivative_of_the_held_step():
    """The trace's immediate term against ``jax.jacrev`` of the same step.

    ``dc_t/dtheta`` with ``c_{t-1}`` and ``h_{t-1}`` held is what the trace
    adds each transition, and everywhere else in this file it is written by
    hand -- once in the cell, once in this file's numpy reference. Neither is
    an independent judge of the algebra. This is: the same step, differentiated
    by the same machinery the rest of the repository uses, with the two carried
    states frozen so that only the explicit derivative survives.

    It also states the fourth factor, which is that ``dc_t/dW_o`` is exactly
    zero. That is why there are three traces and not four.
    """

    core = build()
    x, done = inputs(steps=1, ending=False)
    params, _, _, _ = started(core, x, done)
    cell = core.cell

    previous = jax.random.normal(jax.random.key(21), (STREAMS, HIDDEN))
    hidden = jax.random.normal(jax.random.key(22), (STREAMS, HIDDEN))
    step = x[:, 0]

    def advanced(tree):
        """``c_t`` alone, from held states, as a function of the parameters."""

        carry = LSTMCarry(
            cell=jax.lax.stop_gradient(previous),
            hidden=jax.lax.stop_gradient(hidden),
        )
        state, _ = cast(
            "tuple[LSTMCarry, Array]",
            core.cell.apply({"params": tree}, carry, step),
        )
        return state.cell

    # `jacrev` gives d c_{s,k} / d W[a, b]; the trace is the k == a diagonal of
    # it, which is what "unit k owns row k" means and is asserted below rather
    # than assumed by only ever reading that diagonal.
    jacobian = cast("dict[str, Array]", jax.jacrev(advanced)(params["cell"]))

    _, _, immediate = cast(
        "tuple[LSTMCarry, Array, dict[str, Array]]",
        cell.apply(
            {"params": params["cell"]},
            LSTMCarry(cell=previous, hidden=hidden),
            step,
            jnp.zeros((STREAMS, HIDDEN)),
            {name: jnp.zeros((STREAMS, HIDDEN, WIDTH)) for name in TRACED},
            method=LSTMCell.local_jacobian,
        ),
    )

    index = jnp.arange(HIDDEN)
    for name in TRACED:
        block = jacobian[name]
        assert block.shape == (STREAMS, HIDDEN, HIDDEN, WIDTH)
        assert close(immediate[name], block[:, index, index, :])
        off = block.at[:, index, index, :].set(0.0)
        assert float(jnp.abs(off).max()) == 0.0, f"{name} credited another unit"
        assert float(jnp.abs(immediate[name]).max()) > 1e-3, "the comparison is empty"

    assert float(jnp.abs(jacobian["W_o"]).max()) == 0.0
    assert "W_o" not in immediate


# ------------------------------------------------------------- the sensitivity
def test_the_three_trace_recurrences_follow_the_equations():
    core = build()
    x, done = inputs()
    params, carry, state, rflo = started(core, x, done)

    _, output, sensitivity = rflo(params, x, done, carry, state)
    wanted_output, wanted = reference(params, x, done)

    assert set(sensitivity) == set(TRACED)
    assert close(output, wanted_output)
    for name in TRACED:
        assert close(sensitivity[name], wanted[name])
        assert float(jnp.abs(sensitivity[name]).max()) > 1e-3


def test_the_forget_gate_is_the_rate_the_trace_forgets_at():
    """The leak is the gate rather than a constant, which is what differs here.

    On the CTRNN the leak is ``1 - dt/tau`` and moving it means moving a
    declared parameter. Here it is ``sigma(W_f v_t)``: a learned function of
    the state, different for every unit and every transition. Drawing the gate
    with a large negative bias closes it, and a closed gate is a trace with no
    memory -- it holds this transition's immediate term and nothing earlier.
    """

    x, done = inputs(steps=4, ending=False)

    def last_step_alone(core):
        """The final transition's trace with, and without, the history before it.

        Both walks reach the last transition from the same carry, so their
        immediate terms are identical and the only thing that can separate them
        is what the leak let through from the three steps before.
        """

        params, carry, state, rflo = started(core, x, done)
        _, _, whole = rflo(params, x, done, carry, state)
        carried, _, held = rflo(params, x[:, :-1], done[:, :-1], carry, state)
        forgotten = jax.tree.map(jnp.zeros_like, held)
        _, _, alone = rflo(params, x[:, -1:], done[:, -1:], carried, forgotten)
        return whole, alone

    whole, alone = last_step_alone(build(forget_bias=-12.0))
    for name in TRACED:
        assert close(whole[name], alone[name], tolerance=1e-4)

    # And an open gate does keep its history, so the above is a statement about
    # the gate rather than about the trace never carrying anything.
    whole, alone = last_step_alone(build(forget_bias=4.0))
    for name in TRACED:
        assert not close(whole[name], alone[name], tolerance=1e-3)


def test_an_ending_restarts_the_state_and_the_trace_together():
    """A stream that ended carries the trace of a sequence that has not run.

    Both halves of the carry and all three traces, because the cell state and
    the hidden state are cleared by two different mechanisms -- ``reset_carry``
    over the tree, and the ``where`` on the sensitivity -- and a reset that
    reached one and not the other would leave a restarted episode reading the
    last one's memory through the gates.
    """

    core = build()
    x, done = inputs()
    params, carry, state, rflo = started(core, x, done)

    ended = done.at[0, :].set(False).at[0, STEPS - 1].set(True)
    advanced, output, sensitivity = rflo(params, x, ended, carry, state)

    fresh_x = x[:1, STEPS - 1 :]
    fresh_done = jnp.zeros((1, 1), dtype=bool)
    first_carry, first_output, first = rflo(
        params,
        fresh_x,
        fresh_done,
        jax.tree.map(lambda value: value[:1], carry),
        jax.tree.map(lambda value: value[:1], state),
    )
    assert close(output[:1, STEPS - 1 :], first_output)
    assert close(advanced.cell[:1], first_carry.cell)
    assert close(advanced.hidden[:1], first_carry.hidden)
    for name in TRACED:
        assert close(sensitivity[name][:1], first[name])

    # And a live step does not restart anything, so this is not a test of
    # everything being reset: the stream that never ended kept its whole
    # history, and the one that did no longer has it.
    _, _, uninterrupted = rflo(params, x, jnp.zeros_like(ended), carry, state)
    for name in TRACED:
        assert close(sensitivity[name][1:], uninterrupted[name][1:])
        assert not close(sensitivity[name][:1], uninterrupted[name][:1], tolerance=1e-3)


# ------------------------------------------------------------------ the phantom
def test_a_cotangent_on_the_cell_state_is_the_trace_exactly():
    """RFLO's claim, in the coordinates the trace is a derivative in.

    ``dc_t/dW_ij = p_ij``, one number per weight and no backward pass through
    time. Read here on ``c`` rather than on the output, because that is the
    state the recurrence is written for and the equality is then exact.

    The output gate is the other half of the same statement: with the cotangent
    on the cell state it receives *nothing*, because it does not enter ``c``.
    That is not an approximation and not a defect -- it is the reason ``W_o``
    has no trace, checked where it is a zero rather than where it is small.
    """

    core = build()
    x, done = inputs()
    params, carry, state, rflo = started(core, x, done)

    carried, _, held = rflo(params, x[:, :-1], done[:, :-1], carry, state)
    cotangent = jax.random.normal(jax.random.key(7), (STREAMS, HIDDEN))

    def read(tree):
        advanced, _, _ = rflo(tree, x[:, -1:], done[:, -1:], carried, held)
        return jnp.sum(advanced.cell * cotangent)

    gradient = jax.grad(read)(params)["cell"]
    _, _, advanced = rflo(params, x[:, -1:], done[:, -1:], carried, held)

    for name in TRACED:
        assert close(gradient[name], (cotangent[..., None] * advanced[name]).sum(0))
        assert float(jnp.abs(gradient[name]).max()) > 1e-3, "the comparison is vacuous"
    assert float(jnp.abs(gradient["W_o"]).max()) == 0.0


def test_a_cotangent_on_the_output_picks_up_the_gate_and_the_squashing():
    """What the network hands on is ``o * tanh(c)``, and the trace is not.

    So the gradient the algorithm receives is the trace scaled per unit by
    ``o_t (1 - tanh^2(c_t))``, and the output gate -- silent in the test above
    -- takes its whole gradient here, through the step it does enter.
    """

    core = build()
    x, done = inputs()
    params, carry, state, rflo = started(core, x, done)

    carried, _, held = rflo(params, x[:, :-1], done[:, :-1], carry, state)
    cotangent = jax.random.normal(jax.random.key(7), (STREAMS, HIDDEN))

    def read(tree):
        _, output, _ = rflo(tree, x[:, -1:], done[:, -1:], carried, held)
        return jnp.sum(output[:, 0] * cotangent)

    gradient = jax.grad(read)(params)["cell"]
    step, _, advanced = rflo(params, x[:, -1:], done[:, -1:], carried, held)

    weights = matrices(params)
    row = np.concatenate(
        [np.asarray(x)[:, -1], np.asarray(carried.hidden), np.ones((STREAMS, 1))], -1
    )
    squashed = np.tanh(np.asarray(step.cell))
    opened = sigmoid(row @ weights["W_o"].T)
    through = np.asarray(cotangent) * opened * (1 - squashed**2)

    for name in TRACED:
        assert close(gradient[name], (through[..., None] * advanced[name]).sum(0))
    wanted = (np.asarray(cotangent) * squashed * opened * (1 - opened))[
        ..., None
    ] * row[:, None, :]
    assert close(gradient["W_o"], wanted.sum(0))
    assert float(jnp.abs(gradient["W_o"]).max()) > 1e-3


def test_the_input_keeps_its_own_immediate_jacobian():
    """The gradient reaching the input is the step's, not zero.

    RTRRL's own torso reads the observation directly and has nothing in front
    of it, so nothing in this repository would notice a cell that returned zero
    here -- and the first graph that put a projection in front of one would
    train it on nothing at all.
    """

    core = build()
    x, done = inputs(steps=1, ending=False)
    params, carry, state, rflo = started(core, x, done)
    cotangent = jax.random.normal(jax.random.key(7), (STREAMS, HIDDEN))

    def read(sequence):
        _, output, _ = rflo(params, sequence, done, carry, state)
        return jnp.sum(output[:, 0] * cotangent)

    gradient = jax.grad(read)(x)[:, 0]

    # From an empty carry and an empty trace one transition is exact, so the
    # judge is autodiff through the plain forward rather than an expression
    # copied out of the cell.
    def plain(sequence):
        return jnp.sum(walked(core, params, sequence, done, carry)[:, 0] * cotangent)

    assert close(gradient, jax.grad(plain)(x)[:, 0])
    assert float(jnp.abs(gradient).max()) > 1e-3


# ------------------------------------------------------------ what RFLO drops
def unrolled_gradient(core, params, x, done, carry, cotangent):
    """Exact credit for the whole prefix, by backpropagation through it."""

    truncated = TruncatedBPTT(core)

    def read(tree):
        _, output, _ = truncated(tree, x, done, carry, None)
        return jnp.sum(output[:, -1] * cotangent)

    return jax.grad(read)(params)["cell"]


def rflo_gradient(rflo, params, x, done, carry, state, cotangent):
    """What the online method produces, through the differentiated scan."""

    def read(tree):
        _, output, _ = rflo(tree, x, done, carry, state)
        return jnp.sum(output[:, -1] * cotangent)

    return jax.grad(read)(params)["cell"]


def test_one_transition_from_an_empty_trace_is_exact():
    """The approximation has nothing to drop on the first step of an episode.

    ``S_{t-1}`` is zero, so the term RFLO leaves out is zero and the online
    gradient is the unrolled one -- for all four matrices, the output gate
    included. This is the case in which the two must agree, and it is what
    makes the disagreement below a statement about the dropped term rather than
    about the implementation.

    The trace is empty and the *carry* is not, which matters for exactly one
    gate: ``dc_t/dW_f`` is ``c_{t-1} sigma'(a^f) v_t``, so from a cell state of
    zero the forget gate's whole gradient is zero and an agreement on it would
    be an agreement about nothing -- for the one gate this method's leak is.
    """

    core = build()
    x, done = inputs(steps=1, ending=False)
    params, empty, state, rflo = started(core, x, done)
    carry = LSTMCarry(
        cell=jax.random.normal(jax.random.key(31), empty.cell.shape),
        hidden=jax.random.normal(jax.random.key(32), empty.hidden.shape),
    )
    cotangent = jax.random.normal(jax.random.key(7), (STREAMS, HIDDEN))

    exact = unrolled_gradient(core, params, x, done, carry, cotangent)
    approximate = rflo_gradient(rflo, params, x, done, carry, state, cotangent)
    for name in GATES:
        assert close(exact[name], approximate[name], tolerance=1e-5)
        assert float(jnp.abs(exact[name]).max()) > 1e-3


def test_rflo_is_not_exact_on_this_core_and_the_gap_is_the_dropped_term():
    """The path through ``h_{t-1}`` is real here, so the approximation shows.

    Three transitions, and the exact recurrence run beside RFLO with the
    dropped block put back. The restored one reproduces backpropagation through
    the unroll; RFLO does not. The first half says the harness can see an exact
    answer, the second says what is being run is the approximation and not that
    answer.
    """

    core = build()
    x, done = inputs(steps=3, ending=False)
    params, carry, state, rflo = started(core, x, done)
    cotangent = jax.random.normal(jax.random.key(7), (STREAMS, HIDDEN))

    exact = unrolled_gradient(core, params, x, done, carry, cotangent)
    approximate = rflo_gradient(rflo, params, x, done, carry, state, cotangent)
    restored = _exact_sensitivity(params, x, cotangent)

    for name in GATES:
        assert close(exact[name], restored[name], tolerance=1e-4)
        assert not close(exact[name], approximate[name], tolerance=1e-3)


def _exact_sensitivity(params, x, cotangent):
    """RFLO's recurrence with the ``h_{t-1}`` block of ``dc/dc`` restored.

    Written here and registered nowhere. It exists to name the term RFLO drops,
    so that "they differ" can be replaced by "they differ by this". The state
    is the pair ``(c, h)``, because the dropped path leaves ``c`` through ``h``
    and comes back through the gates -- which is exactly why carrying ``c``
    alone is an approximation and not a change of coordinates.
    """

    weights = matrices(params)
    recurrent = {name: weights[name][:, FEATURES : FEATURES + HIDDEN] for name in GATES}
    xs = np.asarray(x)
    streams, steps, _ = xs.shape

    cell = np.zeros((streams, HIDDEN))
    hidden = np.zeros((streams, HIDDEN))
    sc = {name: np.zeros((streams, HIDDEN, HIDDEN, WIDTH)) for name in GATES}
    sh = {name: np.zeros((streams, HIDDEN, HIDDEN, WIDTH)) for name in GATES}
    index = np.arange(HIDDEN)

    for step in range(steps):
        for stream in range(streams):
            row = np.concatenate([xs[stream, step], hidden[stream], [1.0]])
            opened = sigmoid(weights["W_i"] @ row)
            forget = sigmoid(weights["W_f"] @ row)
            candidate = np.tanh(weights["W_g"] @ row)
            output = sigmoid(weights["W_o"] @ row)
            slope = {
                "W_f": cell[stream] * forget * (1 - forget),
                "W_i": candidate * opened * (1 - opened),
                "W_g": opened * (1 - candidate**2),
            }
            # d c_t / d h_{t-1}: the block RFLO drops, summed over the three
            # gates that write the cell state.
            crossing = sum(
                slope[name][:, None] * recurrent[name] for name in ("W_f", "W_i", "W_g")
            )
            advanced = forget * cell[stream] + opened * candidate
            squashed = np.tanh(advanced)
            through = output * (1 - squashed**2)
            # d h_t / d h_{t-1}, the part that does not pass through c_t.
            gating = (squashed * output * (1 - output))[:, None] * recurrent["W_o"]

            for name in GATES:
                immediate = np.zeros((HIDDEN, HIDDEN, WIDTH))
                if name in slope:
                    immediate[index, index] = slope[name][:, None] * row[None, :]
                advanced_sc = (
                    forget[:, None, None] * sc[name][stream]
                    + np.einsum("km,mab->kab", crossing, sh[name][stream])
                    + immediate
                )
                explicit = np.zeros((HIDDEN, HIDDEN, WIDTH))
                if name == "W_o":
                    explicit[index, index] = (squashed * output * (1 - output))[
                        :, None
                    ] * row[None, :]
                sh[name][stream] = (
                    np.einsum("km,mab->kab", gating, sh[name][stream])
                    + explicit
                    + through[:, None, None] * advanced_sc
                )
                sc[name][stream] = advanced_sc
            cell[stream] = advanced
            hidden[stream] = output * squashed

    return {
        name: np.einsum("sk,skab->ab", np.asarray(cotangent), sh[name])
        for name in GATES
    }


@pytest.mark.parametrize("name", TRACED)
def test_the_trace_is_one_row_per_unit_and_not_a_full_jacobian(name):
    """What the approximation buys, stated as the shape it is carried in.

    Exact forward sensitivity of this cell would carry ``dc_k/dW[a, b]`` for
    every pair, which is a factor of the hidden width more memory and the same
    factor of arithmetic per transition -- ``_exact_sensitivity`` above carries
    exactly that and is why it is a test fixture rather than a mode. RFLO
    carries one row per unit because unit ``k``'s explicit derivative lives on
    row ``k`` alone and the dropped block is the only thing that would have
    moved credit off it.
    """

    core = build()
    x, done = inputs(steps=2, ending=False)
    _, _, state, _ = started(core, x, done)

    assert state[name].shape == (STREAMS, HIDDEN, WIDTH)
