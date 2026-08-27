"""The dense state-space cell, and RFLO as what its off-diagonal costs.

This core is the control the RFLO argument in
``docs/exact-recurrent-sensitivity.md`` was missing. That document says RFLO
and exact RTRL coincide on the LRU and the RTU because their cross-unit block
is identically zero, and it says so structurally, from the parameterisation.
Here the same linear step is written with a full ``A``, so the block is a thing
that can be set to zero and switched back on -- and both directions are
measured:

``the transition`` and ``the sensitivity``
    The forward and the two trace recurrences against a numpy reference
    written from the equations, over a sequence containing an ending.

``the diagonal limit``
    With the off-diagonal of ``A`` zeroed, RFLO and backpropagation through the
    unroll agree to the last bits at every length. That is the LRU's case,
    reached by removing a term rather than by a different implementation, and
    it is the half of the argument no test in this repository had run.

``the dense case``
    With the off-diagonal restored they disagree, and an exact RTRL recurrence
    written here reproduces the unroll. The gap is one matrix product, which is
    the cleanest statement of RFLO's approximation available in this repository
    -- there is no activation Jacobian in the way to share the blame with.

``C``
    Carries no trace, and its gradient is exact at *every* length rather than
    only at one. It does not enter the state, so no approximation about the
    state's history can reach it.

``the domain``
    ``A`` is projected back onto a norm ball after every step, and the ball is
    what stops a free matrix from reaching a spectral radius the recurrence
    diverges at.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.networks.differentiation import TruncatedBPTT
from memorax.networks.sequence_models import RNN
from memorax.networks.sequence_models.dense_ssm import (
    TRACED,
    DenseSSMCell,
    DenseSSMConfig,
    DenseSSMRflo,
    contract,
)
from memorax.utils.typing import Array

FEATURES = 3
HIDDEN = 4
STREAMS = 2
STEPS = 6
WIDTH = FEATURES + 1
BOUND = 0.9

MATRICES = ("A", "B", "C")


def build(*, spectral_bound=BOUND):
    return RNN(
        cell=DenseSSMCell(
            config=DenseSSMConfig(
                features=FEATURES, hidden_dim=HIDDEN, spectral_bound=spectral_bound
            )
        )
    )


def inputs(steps=STEPS, *, ending=True):
    x = jax.random.normal(jax.random.key(11), (STREAMS, steps, FEATURES))
    done = jnp.zeros((STREAMS, steps), dtype=bool)
    if ending and steps > 3:
        done = done.at[0, 3].set(True)
    return x, done


def started(core, x, done):
    """Parameters, an empty carry and an empty trace for one sequence."""

    carry = core.cell.initialize_carry(jax.random.key(0), (STREAMS, FEATURES))
    rflo = DenseSSMRflo(core)
    state = rflo.initialize(jax.random.key(0), (STREAMS, FEATURES))
    params = core.init(jax.random.key(2), x, done=done, initial_carry=carry)["params"]
    return params, carry, state, rflo


def matrices(params) -> dict[str, np.ndarray]:
    return {name: np.asarray(params["cell"][name]) for name in MATRICES}


def close(got, wanted, *, tolerance=1e-5):
    return float(jnp.max(jnp.abs(jnp.asarray(got) - jnp.asarray(wanted)))) < tolerance


def row_norm(matrix) -> float:
    """``max_i sum_j |A_ij|``, the norm the ball is stated in."""

    return float(jnp.abs(jnp.asarray(matrix)).sum(-1).max())


def walked(core, params, x, done, carry) -> Array:
    _, output = cast(
        "tuple[Array, Array]",
        core.apply({"params": params}, x, done=done, initial_carry=carry),
    )
    return output


def reference(params, x, done, carry=None):
    """The equations, stepped one stream and one transition at a time."""

    weights = matrices(params)
    decay = np.diag(weights["A"])
    xs, ds = np.asarray(x), np.asarray(done)
    streams, steps, _ = xs.shape
    state = np.zeros((streams, HIDDEN)) if carry is None else np.asarray(carry).copy()
    trace = {
        "A": np.zeros((streams, HIDDEN, HIDDEN)),
        "B": np.zeros((streams, HIDDEN, WIDTH)),
    }
    outputs = np.zeros((streams, steps, HIDDEN))

    for step in range(steps):
        for stream in range(streams):
            if ds[stream, step]:
                state[stream] = 0
                for name in trace:
                    trace[name][stream] = 0
            row = np.concatenate([xs[stream, step], [1.0]])
            # The trace is advanced from the state as it stood, which is the
            # state this transition's explicit derivative is taken at.
            trace["A"][stream] = (
                decay[:, None] * trace["A"][stream] + state[stream][None, :]
            )
            trace["B"][stream] = decay[:, None] * trace["B"][stream] + row[None, :]
            state[stream] = weights["A"] @ state[stream] + weights["B"] @ row
            outputs[stream, step] = np.tanh(weights["C"] @ state[stream])
    return outputs, trace


# ------------------------------------------------------------- the transition
def test_one_step_is_the_state_space_recurrence():
    core = build()
    x, done = inputs()
    params, carry, _, _ = started(core, x, done)

    output = walked(core, params, x, done, carry)
    wanted, _ = reference(params, x, done)
    assert close(output, wanted)


def test_the_bias_column_is_read_and_the_state_has_its_own_matrix():
    """``B`` carries ``[input, bias]`` and the state enters through ``A`` only.

    The layout differs from ``ctrnn.py``'s and ``lstm.py``'s deliberately: a
    state-space step keeps ``A`` and ``B`` apart, and folding them together
    would make the off-diagonal this core exists to study a sub-block of
    something else.
    """

    core = build()
    x, done = inputs(steps=1, ending=False)
    params, carry, _, _ = started(core, x, done)
    weights = matrices(params)

    assert weights["A"].shape == (HIDDEN, HIDDEN)
    assert weights["B"].shape == (HIDDEN, WIDTH)

    zero = jnp.zeros((STREAMS, 1, FEATURES))
    only_bias = walked(core, params, zero, done, carry)
    assert close(only_bias[:, 0], np.tanh(weights["C"] @ weights["B"][:, -1]))

    state = jax.random.normal(jax.random.key(5), (STREAMS, HIDDEN))
    with_state = walked(core, params, zero, done, state)
    advanced = np.asarray(state) @ weights["A"].T + weights["B"][:, -1]
    assert close(with_state[:, 0], np.tanh(advanced @ weights["C"].T))


# ------------------------------------------------------------- the sensitivity
def test_both_trace_recurrences_follow_the_equations():
    core = build()
    x, done = inputs()
    params, carry, state, rflo = started(core, x, done)

    _, output, sensitivity = rflo(params, x, done, carry, state)
    wanted_output, wanted = reference(params, x, done)

    assert set(sensitivity) == set(TRACED)
    assert "C" not in sensitivity
    assert close(output, wanted_output)
    for name in TRACED:
        assert close(sensitivity[name], wanted[name])
        assert float(jnp.abs(sensitivity[name]).max()) > 1e-3


def test_an_ending_restarts_the_state_and_the_trace_together():
    """A stream that ended carries the trace of a sequence that has not run."""

    core = build()
    x, done = inputs()
    params, carry, state, rflo = started(core, x, done)

    ended = done.at[0, :].set(False).at[0, STEPS - 1].set(True)
    advanced, output, sensitivity = rflo(params, x, ended, carry, state)

    first_carry, first_output, first = rflo(
        params,
        x[:1, STEPS - 1 :],
        jnp.zeros((1, 1), dtype=bool),
        carry[:1],
        jax.tree.map(lambda value: value[:1], state),
    )
    assert close(output[:1, STEPS - 1 :], first_output)
    assert close(advanced[:1], first_carry)
    for name in TRACED:
        assert close(sensitivity[name][:1], first[name])

    _, _, uninterrupted = rflo(params, x, jnp.zeros_like(ended), carry, state)
    for name in TRACED:
        assert close(sensitivity[name][1:], uninterrupted[name][1:])
        assert not close(sensitivity[name][:1], uninterrupted[name][:1], tolerance=1e-3)


@pytest.mark.parametrize("name,width", (("A", HIDDEN), ("B", WIDTH)))
def test_the_trace_is_one_row_per_unit_and_not_a_full_jacobian(name, width):
    """What the approximation buys, stated as the shape it is carried in.

    Exact forward sensitivity carries ``dh_k/dtheta_{ab}`` for every triple,
    which is a factor of the hidden width more memory and the same factor of
    arithmetic per transition -- ``_exact_sensitivity`` below carries exactly
    that. RFLO carries one row per unit because the dropped block is the only
    thing that would have moved credit off it.
    """

    core = build()
    x, done = inputs(steps=2, ending=False)
    _, _, state, _ = started(core, x, done)

    assert state[name].shape == (STREAMS, HIDDEN, width)


# ------------------------------------------------------------------ the phantom
def test_a_cotangent_on_the_state_is_the_trace_exactly():
    """RFLO's claim, in the coordinates the trace is a derivative in.

    ``C`` is the other half of the same statement: with the cotangent on the
    state it receives exactly nothing, because it does not enter the state.
    """

    core = build()
    x, done = inputs()
    params, carry, state, rflo = started(core, x, done)

    carried, _, held = rflo(params, x[:, :-1], done[:, :-1], carry, state)
    cotangent = jax.random.normal(jax.random.key(7), (STREAMS, HIDDEN))

    def read(tree):
        advanced, _, _ = rflo(tree, x[:, -1:], done[:, -1:], carried, held)
        return jnp.sum(advanced * cotangent)

    gradient = jax.grad(read)(params)["cell"]
    _, _, advanced = rflo(params, x[:, -1:], done[:, -1:], carried, held)

    for name in TRACED:
        assert close(gradient[name], (cotangent[..., None] * advanced[name]).sum(0))
        assert float(jnp.abs(gradient[name]).max()) > 1e-3, "the comparison is vacuous"
    assert float(jnp.abs(gradient["C"]).max()) == 0.0


def test_the_input_keeps_its_own_immediate_jacobian():
    """The gradient reaching the input is the step's, not zero."""

    core = build()
    x, done = inputs(steps=1, ending=False)
    params, carry, state, rflo = started(core, x, done)
    cotangent = jax.random.normal(jax.random.key(7), (STREAMS, HIDDEN))

    def read(sequence):
        _, output, _ = rflo(params, sequence, done, carry, state)
        return jnp.sum(output[:, 0] * cotangent)

    def plain(sequence):
        return jnp.sum(walked(core, params, sequence, done, carry)[:, 0] * cotangent)

    gradient = jax.grad(read)(x)[:, 0]
    assert close(gradient, jax.grad(plain)(x)[:, 0])
    assert float(jnp.abs(gradient).max()) > 1e-3


# ------------------------------------------------------------ what RFLO drops
def cotangent_of(key=7):
    return jax.random.normal(jax.random.key(key), (STREAMS, HIDDEN))


def unrolled_gradient(core, params, x, done, carry, cotangent):
    truncated = TruncatedBPTT(core)

    def read(tree):
        _, output, _ = truncated(tree, x, done, carry, None)
        return jnp.sum(output[:, -1] * cotangent)

    return jax.grad(read)(params)["cell"]


def rflo_gradient(rflo, params, x, done, carry, state, cotangent):
    def read(tree):
        _, output, _ = rflo(tree, x, done, carry, state)
        return jnp.sum(output[:, -1] * cotangent)

    return jax.grad(read)(params)["cell"]


def diagonalized(params):
    """The same parameters with the block RFLO drops removed from ``A``."""

    matrix = params["cell"]["A"]
    return {"cell": {**params["cell"], "A": jnp.diag(jnp.diag(matrix))}}


@pytest.mark.parametrize("steps", (1, 2, 5))
def test_with_a_diagonal_a_rflo_is_exact_at_every_length(steps):
    """The LRU's case, recovered as a limit and measured rather than argued.

    ``docs/exact-recurrent-sensitivity.md`` says RFLO is not an approximation
    where the cross-unit block is identically zero. On the LRU and the RTU that
    cannot be tested by removing the block, because their parameterisation has
    no block to remove. Here it does, so this is the same claim run as an
    experiment: zero the off-diagonal and the approximate gradient becomes the
    unrolled one, for every length rather than for the first transition.

    The carry is not empty, for the same reason it is not in the test below:
    ``dh_t/dA`` is proportional to ``h_{t-1}``, so from a state of zero the
    agreement on ``A`` at one transition would be an agreement about nothing.
    """

    core = build()
    x, done = inputs(steps=steps, ending=False)
    params, empty, state, rflo = started(core, x, done)
    carry = jax.random.normal(jax.random.key(31), empty.shape)
    params = diagonalized(params)
    cotangent = cotangent_of()

    exact = unrolled_gradient(core, params, x, done, carry, cotangent)
    approximate = rflo_gradient(rflo, params, x, done, carry, state, cotangent)
    for name in MATRICES:
        assert close(exact[name], approximate[name], tolerance=1e-5)
        assert float(jnp.abs(exact[name]).max()) > 1e-3


def test_one_transition_from_an_empty_trace_is_exact():
    """With nothing carried there is nothing to drop, dense ``A`` or not."""

    core = build()
    x, done = inputs(steps=1, ending=False)
    params, empty, state, rflo = started(core, x, done)
    carry = jax.random.normal(jax.random.key(31), empty.shape)
    cotangent = cotangent_of()

    exact = unrolled_gradient(core, params, x, done, carry, cotangent)
    approximate = rflo_gradient(rflo, params, x, done, carry, state, cotangent)
    for name in MATRICES:
        assert close(exact[name], approximate[name], tolerance=1e-5)
        assert float(jnp.abs(exact[name]).max()) > 1e-3


def test_with_a_dense_a_rflo_is_not_exact_and_the_gap_is_the_dropped_term():
    """Three transitions, and an exact recurrence beside them to name the gap.

    The exact recurrence differs from the implemented one in one place: it
    applies the whole of ``A`` to the previous sensitivity where RFLO applies
    only its diagonal. That it reproduces backpropagation through the unroll,
    and that RFLO does not, is what makes the disagreement a statement about
    the off-diagonal rather than about either implementation.
    """

    core = build()
    x, done = inputs(steps=3, ending=False)
    params, carry, state, rflo = started(core, x, done)
    cotangent = cotangent_of()

    exact = unrolled_gradient(core, params, x, done, carry, cotangent)
    approximate = rflo_gradient(rflo, params, x, done, carry, state, cotangent)
    restored = _exact_sensitivity(params, x, cotangent)

    for name in TRACED:
        assert close(exact[name], restored[name], tolerance=1e-4)
        assert not close(exact[name], approximate[name], tolerance=1e-3)


@pytest.mark.parametrize("steps", (1, 3, 6))
def test_the_readout_matrix_is_exact_at_every_length(steps):
    """``C`` is not approximated, because no history reaches it.

    ``dh_t/dC`` is zero, so ``C``'s gradient is a function of the state the
    forward arrived at, and both walks arrive at the same state. This is worth
    asserting separately from the trace: a graph that quietly routed ``C``
    through the approximation would still train, and the length at which it
    started to differ would be the length at which someone noticed.
    """

    core = build()
    x, done = inputs(steps=steps, ending=False)
    params, carry, state, rflo = started(core, x, done)
    cotangent = cotangent_of()

    exact = unrolled_gradient(core, params, x, done, carry, cotangent)
    approximate = rflo_gradient(rflo, params, x, done, carry, state, cotangent)
    assert close(exact["C"], approximate["C"], tolerance=1e-5)
    assert float(jnp.abs(exact["C"]).max()) > 1e-3


def _exact_sensitivity(params, x, cotangent):
    """RFLO's recurrence with the off-diagonal of ``A`` put back.

    Written here and registered nowhere. One line differs from the reference
    above -- ``A @ S`` where RFLO has ``diag(A) * S`` -- and carrying it costs
    a factor of the hidden width in both memory and arithmetic, which is what
    the approximation is buying.
    """

    weights = matrices(params)
    xs = np.asarray(x)
    streams, steps, _ = xs.shape
    index = np.arange(HIDDEN)

    state = np.zeros((streams, HIDDEN))
    sensitivity = {
        "A": np.zeros((streams, HIDDEN, HIDDEN, HIDDEN)),
        "B": np.zeros((streams, HIDDEN, HIDDEN, WIDTH)),
    }
    for step in range(steps):
        for stream in range(streams):
            row = np.concatenate([xs[stream, step], [1.0]])
            immediate = {
                "A": np.zeros((HIDDEN, HIDDEN, HIDDEN)),
                "B": np.zeros((HIDDEN, HIDDEN, WIDTH)),
            }
            immediate["A"][index, index] = state[stream][None, :]
            immediate["B"][index, index] = row[None, :]
            for name in sensitivity:
                sensitivity[name][stream] = (
                    np.einsum("km,mab->kab", weights["A"], sensitivity[name][stream])
                    + immediate[name]
                )
            state[stream] = weights["A"] @ state[stream] + weights["B"] @ row

    # The cotangent is taken on the output, so it reaches the state through the
    # readout: `dL/dh = (ybar * (1 - tanh^2(C h))) @ C`.
    squashed = np.tanh(state @ weights["C"].T)
    through = (np.asarray(cotangent) * (1 - squashed**2)) @ weights["C"]
    return {
        name: np.einsum("sk,skab->ab", through, sensitivity[name])
        for name in sensitivity
    }


# ------------------------------------------------------------------ the domain
def test_the_initial_draw_is_inside_the_norm_ball():
    """``lecun_normal`` alone is not, and at the published width it is far out.

    Row sums grow like the square root of the width, so an unprojected ``A``
    would be a diverging recurrence at initialisation rather than after some
    number of updates.
    """

    core = build(spectral_bound=0.5)
    x, done = inputs(steps=1, ending=False)
    params, _, _, _ = started(core, x, done)

    matrix = matrices(params)["A"]
    assert row_norm(matrix) <= 0.5 + 1e-6
    assert float(np.abs(np.linalg.eigvals(matrix)).max()) < 0.5 + 1e-6
    assert float(np.abs(matrix).max()) > 0.0, "the draw was projected to nothing"


def test_the_projection_scales_the_rows_that_are_outside_and_no_others():
    """It is the projection onto the set, not a normalisation of every row."""

    inside = jnp.asarray([[0.1, 0.2], [0.0, 0.0]])
    assert close(contract(inside, 0.9), inside)

    outside = jnp.asarray([[1.0, 1.0], [0.1, 0.1]])
    projected = np.asarray(contract(outside, 0.9))
    assert close(np.abs(projected).sum(-1)[0], 0.9)
    assert close(projected[1], np.asarray(outside)[1])


def test_the_cell_names_the_set_and_it_is_the_one_a_step_can_leave():
    """The bound is stated by the component whose parameter it is.

    And it is not decorative: a matrix outside the ball diverges over an
    episode, which the walk below reaches in eight transitions while the same
    walk inside the ball stays bounded.
    """

    core = build(spectral_bound=0.5)
    x, done = inputs(steps=8, ending=False)
    params, carry, _, _ = started(core, x, done)

    weights = {name: jnp.asarray(matrices(params)[name]) for name in MATRICES}
    escaped = {"cell": {**weights, "A": weights["A"] * 6.0}}
    assert float(jnp.abs(walked(core, params, x, done, carry)).max()) <= 1.0
    grown = walked(core, escaped, x, done, carry)
    assert float(jnp.abs(grown).max()) > 0.0

    returned = cast(
        "dict[str, Array]",
        core.cell.apply(
            {"params": escaped["cell"]}, escaped["cell"], method=DenseSSMCell.constrain
        ),
    )
    assert row_norm(returned["A"]) <= 0.5 + 1e-6
    # The two matrices that are read once per transition are left alone.
    for name in ("B", "C"):
        assert close(returned[name], escaped["cell"][name])
