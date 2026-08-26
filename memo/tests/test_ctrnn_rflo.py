"""The CTRNN cell, and RFLO as the thing that credits it.

Four questions, and they need four different judges:

``the transition``
    One Euler step against the equation, computed here in numpy from the
    parameters alone. Nothing about differentiation is involved, so a failure is
    the forward and nothing else.

``the sensitivity``
    Both RFLO recurrences against the same numpy reference, over a sequence that
    contains an ending. This is where ``dt`` and the wiring mask have to appear,
    and the reference is the paper's equations rather than the published code --
    the two disagree, and where they do the equations decide.

``the phantom``
    That the gradient autodiff produces through the differentiated scan is
    exactly the cotangent contracted with the carried trace. The trace and the
    gradient are written in two places, and this is the only thing holding them
    to being one statement.

``what RFLO drops``
    RFLO against exact credit through a truncation-free unroll. On this core the
    two must *not* agree -- the cross-unit block of the recurrent Jacobian is
    real here -- and the gap is held to exactly the term the algebra says was
    dropped. Without that, "RFLO is implemented" would be a claim no test could
    distinguish from "RTRL is implemented".

The corrections these hold, and what each one was in the published
implementation, are in ``docs/rtrrl-ctrnn-rflo-corrections.md``.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.networks.differentiation import TruncatedBPTT
from memorax.networks.sequence_models import RNN
from memorax.networks.sequence_models.ctrnn import (
    CTRNNCell,
    CTRNNConfig,
    CTRNNRflo,
    wiring_mask,
)
from memorax.utils.typing import Array

FEATURES = 3
HIDDEN = 4
STREAMS = 2
STEPS = 6

# Not one. Every trace recurrence carries `dt`, the published implementation
# carries it in the forward only, and at `dt = 1` the difference is invisible.
DT = 0.5

WIRINGS = ("fully_connected", "no_self")


def build(*, wiring="fully_connected", dt=DT, tau_floor=1.0):
    """A cell, the scan around it, and both ways of differentiating it."""

    cell = CTRNNCell(
        config=CTRNNConfig(
            features=FEATURES,
            hidden_dim=HIDDEN,
            dt=dt,
            wiring=wiring,
            tau_floor=tau_floor,
        )
    )
    return RNN(cell=cell)


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
    rflo = CTRNNRflo(core)
    state = rflo.initialize(jax.random.key(0), (STREAMS, FEATURES))
    params = core.init(jax.random.key(2), x, done=done, initial_carry=carry)["params"]
    return params, carry, state, rflo


def reference(params, x, done, *, wiring, dt=DT, trace_dt=None, trace_mask=True):
    """The paper's equations, stepped one stream and one transition at a time.

    Written from the equations rather than from ``models/ctrnn.py``, and the two
    disagree in two places -- so both are reachable from here. ``trace_dt``
    scales the trace's immediate Jacobian, which the published implementation
    fixes at one while integrating with ``dt``; ``trace_mask`` says whether the
    trace saw the wiring, which the published implementation does not. Left
    unset, both follow the forward, which is the equations.
    """

    trace_dt = dt if trace_dt is None else trace_dt

    weights = np.asarray(leaf(params, "W"))
    tau = np.asarray(leaf(params, "tau"))
    mask = wiring_mask(wiring, features=FEATURES, hidden_dim=HIDDEN)
    mask = None if mask is None else np.asarray(mask)
    masked = weights if mask is None else mask * weights

    traced = masked if trace_mask else weights
    xs, ds = np.asarray(x), np.asarray(done)
    streams, steps, _ = xs.shape
    width = FEATURES + HIDDEN + 1
    hidden = np.zeros((streams, HIDDEN))
    trace_w = np.zeros((streams, HIDDEN, width))
    trace_tau = np.zeros((streams, HIDDEN))
    outputs = np.zeros((streams, steps, HIDDEN))
    rate = dt / tau
    traced_rate = trace_dt / tau

    for step in range(steps):
        for stream in range(streams):
            if ds[stream, step]:
                hidden[stream] = 0
                trace_w[stream] = 0
                trace_tau[stream] = 0
            row = np.concatenate([xs[stream, step], hidden[stream], [1.0]])
            activated = np.tanh(masked @ row)
            slope = 1 - np.tanh(traced @ row) ** 2
            immediate = traced_rate[:, None] * slope[:, None] * row[None, :]
            if mask is not None and trace_mask:
                immediate = immediate * mask
            trace_w[stream] = (1 - traced_rate)[:, None] * trace_w[stream] + immediate
            trace_tau[stream] = (1 - traced_rate) * trace_tau[stream] + (
                traced_rate / tau
            ) * (hidden[stream] - np.tanh(traced @ row))
            hidden[stream] = hidden[stream] + rate * (activated - hidden[stream])
            outputs[stream, step] = hidden[stream]
    return outputs, {"W": trace_w, "tau": trace_tau}


def leaf(params, name) -> Array:
    """One of the cell's two parameters, as an array rather than a tree."""

    return jnp.asarray(params["cell"][name])


def walked(core, params, x, done, carry) -> Array:
    """``RNN.apply`` with the output it returns beside the carry named."""

    _, output = cast(
        "tuple[Array, Array]",
        core.apply({"params": params}, x, done=done, initial_carry=carry),
    )
    return output


def close(got, wanted, *, tolerance=1e-5):
    return float(jnp.max(jnp.abs(jnp.asarray(got) - jnp.asarray(wanted)))) < tolerance


# ------------------------------------------------------------- the transition
@pytest.mark.parametrize("wiring", WIRINGS)
def test_one_step_is_the_euler_integration_of_the_ctrnn_ode(wiring):
    core = build(wiring=wiring)
    x, done = inputs(steps=1, ending=False)
    params, carry, _, _ = started(core, x, done)

    output = walked(core, params, x, done, carry)

    weights = np.asarray(leaf(params, "W"))
    mask = wiring_mask(wiring, features=FEATURES, hidden_dim=HIDDEN)
    if mask is not None:
        weights = np.asarray(mask) * weights
    row = np.concatenate(
        [np.asarray(x)[:, 0], np.zeros((STREAMS, HIDDEN)), np.ones((STREAMS, 1))],
        axis=-1,
    )
    tau = np.asarray(leaf(params, "tau"))
    wanted = (DT / tau) * np.tanh(row @ weights.T)
    assert close(output[:, 0], wanted)


def test_the_bias_column_is_read_and_the_recurrent_block_is_where_it_says():
    """Which column of ``W`` means what, asserted rather than assumed.

    The layout ``[input, hidden, bias]`` is the published one and the wiring
    mask is cut against it, so a transposition here would move a mask's zeros
    onto the wrong connections -- which is precisely the defect ``no_self``
    carries in the published implementation.
    """

    core = build()
    x, done = inputs(steps=1, ending=False)
    params, carry, _, _ = started(core, x, done)
    weights, tau = leaf(params, "W"), leaf(params, "tau")

    zero = jnp.zeros((STREAMS, 1, FEATURES))
    hidden = jax.random.normal(jax.random.key(5), (STREAMS, HIDDEN))
    only_bias = walked(core, params, zero, done, jnp.zeros((STREAMS, HIDDEN)))
    assert close(only_bias[:, 0], (DT / tau) * jnp.tanh(weights[:, -1]))

    with_state = walked(core, params, zero, done, hidden)
    recurrent = weights[:, FEATURES : FEATURES + HIDDEN]
    wanted = hidden + (DT / tau) * (
        jnp.tanh(hidden @ recurrent.T + weights[:, -1]) - hidden
    )
    assert close(with_state[:, 0], wanted)


def test_the_no_self_wiring_removes_each_unit_s_own_previous_state():
    """The diagonal sits on the recurrent block, not one column to its right.

    ``fully_connected_no_self`` in the published wirings subtracts an identity
    from the *last* ``hidden_dim`` columns. Under the ``[input, hidden, bias]``
    layout those are ``h_1 .. h_{n-1}`` and the bias, so it removes unit ``i``'s
    reading of unit ``i+1`` and the last unit's bias -- which the mask builder
    then puts back, leaving that unit self-connected. It is off by one column.
    """

    mask = np.asarray(wiring_mask("no_self", features=FEATURES, hidden_dim=HIDDEN))
    recurrent = mask[:, FEATURES : FEATURES + HIDDEN]

    assert np.array_equal(recurrent, 1 - np.eye(HIDDEN))
    assert mask[:, :FEATURES].all(), "an input connection was masked out"
    assert mask[:, -1].all(), "the bias was masked out"


def test_an_unknown_wiring_is_refused_by_name():
    with pytest.raises(ValueError, match="ncp"):
        wiring_mask("ncp", features=FEATURES, hidden_dim=HIDDEN)


# ------------------------------------------------------------- the sensitivity
@pytest.mark.parametrize("wiring", WIRINGS)
def test_both_rflo_recurrences_follow_the_paper_s_equations(wiring):
    core = build(wiring=wiring)
    x, done = inputs()
    params, carry, state, rflo = started(core, x, done)

    _, output, sensitivity = rflo(params, x, done, carry, state)
    wanted_output, wanted = reference(params, x, done, wiring=wiring)

    assert close(output, wanted_output)
    assert close(sensitivity["W"], wanted["W"])
    assert close(sensitivity["tau"], wanted["tau"])


def test_the_trace_carries_the_integration_step_it_is_the_derivative_of():
    """``dt`` is in the recurrence, which at the published ``dt = 1`` is unseen.

    The published trace scales the immediate Jacobian by ``1/tau`` and forgets
    at ``1 - 1/tau``, while the integration it claims to differentiate scales by
    ``dt/tau``. So it is the derivative of a step the network did not take
    whenever ``dt`` is not one -- and at ``dt = 1`` the two coincide, which is
    why every reported run hid it.
    """

    x, done = inputs(steps=4, ending=False)
    core = build(dt=DT)
    params, carry, state, rflo = started(core, x, done)
    _, _, sensitivity = rflo(params, x, done, carry, state)

    _, wanted = reference(params, x, done, wiring="fully_connected", dt=DT)
    assert close(sensitivity["W"], wanted["W"])
    assert close(sensitivity["tau"], wanted["tau"])

    _, published = reference(
        params, x, done, wiring="fully_connected", dt=DT, trace_dt=1.0
    )
    assert not close(sensitivity["W"], published["W"], tolerance=1e-3)
    assert not close(sensitivity["tau"], published["tau"], tolerance=1e-3)

    at_one = build(dt=1.0)
    same, carry, state, rflo = started(at_one, x, done)
    _, _, unit = rflo(same, x, done, carry, state)
    _, published_at_one = reference(
        same, x, done, wiring="fully_connected", dt=1.0, trace_dt=1.0
    )
    assert close(unit["W"], published_at_one["W"])
    assert close(unit["tau"], published_at_one["tau"])


def test_a_masked_connection_is_masked_in_the_trace_too():
    """The trace forms its pre-activation from the weights the forward used.

    The published cell masks ``W`` inside ``CTRNNCell.__call__`` and hands
    ``rflo_murray`` the raw parameter tree, so the trace's ``tanh'(u)`` is taken
    at a pre-activation the network never computed and the masked entries
    accumulate credit they cannot spend. Both are checked here: the trace is
    zero where the mask is, and the entries that survive are not the ones the
    published pre-activation would have produced -- so this is not only about
    the masked columns. That the survivors are *right* is the equation
    comparison above, run under the same wiring.
    """

    core = build(wiring="no_self")
    x, done = inputs()
    params, carry, state, rflo = started(core, x, done)

    _, _, sensitivity = rflo(params, x, done, carry, state)
    mask = np.asarray(wiring_mask("no_self", features=FEATURES, hidden_dim=HIDDEN))
    assert np.abs(np.asarray(sensitivity["W"])[:, mask == 0]).max() == 0.0

    _, published = reference(params, x, done, wiring="no_self", trace_mask=False)
    surviving = np.asarray(published["W"]) * mask
    assert not close(np.asarray(sensitivity["W"]) * mask, surviving, tolerance=1e-3)


def test_an_ending_restarts_the_state_and_the_trace_together():
    """A stream that ended carries the trace of a sequence that has not run."""

    core = build()
    x, done = inputs()
    params, carry, state, rflo = started(core, x, done)

    ended = done.at[0, :].set(False).at[0, STEPS - 1].set(True)
    _, output, sensitivity = rflo(params, x, ended, carry, state)

    fresh_x = x[:1, STEPS - 1 :]
    fresh_done = jnp.zeros((1, 1), dtype=bool)
    _, first_output, first = rflo(
        params,
        fresh_x,
        fresh_done,
        carry[:1],
        jax.tree.map(lambda leaf: leaf[:1], state),
    )
    assert close(output[:1, STEPS - 1 :], first_output)
    assert close(sensitivity["W"][:1], first["W"])
    assert close(sensitivity["tau"][:1], first["tau"])

    # And a live step does not restart anything, so this is not a test of
    # everything being reset: the stream that never ended kept its whole
    # history, and the one that did no longer has it.
    _, _, uninterrupted = rflo(params, x, jnp.zeros_like(ended), carry, state)
    assert close(sensitivity["W"][1:], uninterrupted["W"][1:])
    assert not close(sensitivity["W"][:1], uninterrupted["W"][:1], tolerance=1e-3)


# ------------------------------------------------------------------ the phantom
@pytest.mark.parametrize("wiring", WIRINGS)
def test_the_gradient_is_the_cotangent_contracted_with_the_carried_trace(wiring):
    """What autodiff produces and what the trace holds are one statement.

    The recurrence is written in ``local_jacobian`` and the gradient is produced
    by differentiating the step; nothing but this holds the two to agreeing. It
    is also the sense in which the trace *is* the gradient: RFLO's whole claim
    is that ``dL/dW_ij = ybar_i p_ij``, one number per weight, with no
    backward pass through time.
    """

    core = build(wiring=wiring)
    x, done = inputs()
    params, carry, state, rflo = started(core, x, done)

    carried, _, held = rflo(params, x[:, :-1], done[:, :-1], carry, state)
    cotangent = jax.random.normal(jax.random.key(7), (STREAMS, HIDDEN))

    def read(tree):
        _, output, _ = rflo(tree, x[:, -1:], done[:, -1:], carried, held)
        return jnp.sum(output[:, 0] * cotangent)

    gradient = jax.grad(read)(params)["cell"]
    _, _, advanced = rflo(params, x[:, -1:], done[:, -1:], carried, held)

    assert close(gradient["W"], (cotangent[..., None] * advanced["W"]).sum(0))
    assert close(gradient["tau"], (cotangent * advanced["tau"]).sum(0))
    assert float(jnp.abs(gradient["W"]).max()) > 1e-3, "the comparison is vacuous"


def test_the_input_keeps_its_own_immediate_jacobian():
    """The gradient reaching the input is the step's, not zero.

    The published cell returns ``jnp.einsum(ybar, jx)`` with ``jx`` initialised
    to zeros and never written under RFLO, so anything in front of the cell
    receives exactly nothing. RTRRL's own torso reads the observation directly
    and has nothing in front of it, so the defect is invisible there -- and
    would not be for the first graph that put a projection in front.
    """

    core = build()
    x, done = inputs(steps=1, ending=False)
    params, carry, state, rflo = started(core, x, done)
    cotangent = jax.random.normal(jax.random.key(7), (STREAMS, HIDDEN))

    def read(sequence):
        _, output, _ = rflo(params, sequence, done, carry, state)
        return jnp.sum(output[:, 0] * cotangent)

    gradient = jax.grad(read)(x)[:, 0]

    weights, tau = leaf(params, "W"), leaf(params, "tau")
    row = jnp.concatenate(
        [x[:, 0], jnp.zeros((STREAMS, HIDDEN)), jnp.ones((STREAMS, 1))], axis=-1
    )
    slope = 1 - jnp.tanh(row @ weights.T) ** 2
    wanted = (cotangent * (DT / tau) * slope) @ weights[:, :FEATURES]
    assert close(gradient, wanted)
    assert float(jnp.abs(gradient).max()) > 1e-3


# ------------------------------------------------------------ what RFLO drops
def unrolled_gradient(core, params, x, done, carry, cotangent):
    """Exact credit for the whole prefix, by backpropagation through it."""

    truncated = TruncatedBPTT(core)

    def read(tree):
        _, output, _ = truncated(tree, x, done, carry, None)
        return jnp.sum(output[:, -1] * cotangent)

    return jax.grad(read)(params)["cell"]


def test_rflo_is_not_exact_on_this_core_and_the_gap_is_the_dropped_term():
    """The cross-unit block is real here, so the approximation has to show.

    ``S_t = (dh/dh) S_{t-1} + dh/dtheta``, and RFLO replaces ``dh/dh`` by its
    leak part alone. The exact recurrence run beside it, with the same forward,
    reproduces the unrolled gradient; RFLO does not, and the difference is the
    ``tanh`` path of ``dh/dh`` applied to the previous sensitivity. Both halves
    matter: the first says the harness can see an exact answer, the second says
    what is being run is the approximation and not that answer.
    """

    core = build(wiring="fully_connected")
    x, done = inputs(steps=3, ending=False)
    params, carry, state, rflo = started(core, x, done)
    cotangent = jax.random.normal(jax.random.key(7), (STREAMS, HIDDEN))

    exact = unrolled_gradient(core, params, x, done, carry, cotangent)
    _, _, sensitivity = rflo(params, x, done, carry, state)
    approximate = {
        "W": (cotangent[..., None] * sensitivity["W"]).sum(0),
        "tau": (cotangent * sensitivity["tau"]).sum(0),
    }
    assert not close(exact["W"], approximate["W"], tolerance=1e-3)
    assert not close(exact["tau"], approximate["tau"], tolerance=1e-3)

    # The same recurrence with the dropped block put back, which is exact RTRL
    # and must reproduce backpropagation through the unroll. That is what makes
    # the disagreement above a statement about RFLO rather than about either
    # implementation being wrong.
    restored = _exact_sensitivity(params, x, cotangent)
    assert close(exact["W"], restored["W"], tolerance=1e-4)
    assert close(exact["tau"], restored["tau"], tolerance=1e-4)


def _exact_sensitivity(params, x, cotangent):
    """RFLO's recurrence with the ``tanh`` block of ``dh/dh`` restored.

    Written here and registered nowhere. It exists to name the term RFLO drops,
    so that "they differ" can be replaced by "they differ by this".
    """

    weights = np.asarray(leaf(params, "W"))
    tau = np.asarray(leaf(params, "tau"))
    rate = DT / tau
    recurrent = weights[:, FEATURES : FEATURES + HIDDEN]
    xs = np.asarray(x)
    streams, steps, _ = xs.shape
    width = FEATURES + HIDDEN + 1

    hidden = np.zeros((streams, HIDDEN))
    trace_w = np.zeros((streams, HIDDEN, HIDDEN, width))
    trace_tau = np.zeros((streams, HIDDEN, HIDDEN))
    for step in range(steps):
        for stream in range(streams):
            row = np.concatenate([xs[stream, step], hidden[stream], [1.0]])
            activated = np.tanh(weights @ row)
            slope = 1 - activated**2
            # d h_i(t) / d h_k(t-1), in full.
            jacobian = np.diag(1 - rate) + (rate * slope)[:, None] * recurrent
            immediate_w = np.zeros((HIDDEN, HIDDEN, width))
            index = np.arange(HIDDEN)
            immediate_w[index, index] = (rate * slope)[:, None] * row[None, :]
            immediate_tau = np.diag((rate / tau) * (hidden[stream] - activated))
            trace_w[stream] = (
                np.einsum("ik,kjw->ijw", jacobian, trace_w[stream]) + immediate_w
            )
            trace_tau[stream] = jacobian @ trace_tau[stream] + immediate_tau
            hidden[stream] = hidden[stream] + rate * (activated - hidden[stream])
    return {
        "W": np.einsum("si,sijw->jw", np.asarray(cotangent), trace_w),
        "tau": np.einsum("si,sij->j", np.asarray(cotangent), trace_tau),
    }
