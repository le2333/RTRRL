"""``ctrnn.py`` against ``RTRRL-AAAI25/models/ctrnn.py``, in both directions.

Where the published cell is right this side has to match it to the last bits,
and where it is wrong this side has to *not* match it -- with the paper's
equations, computed in ``tests/test_ctrnn_rflo.py``, saying which is which. A
parity suite that only ever asserts agreement cannot carry a correction; one
that only asserts disagreement cannot tell a correction from a mistake. So both
are here, over the same two implementations and the same numbers.

The configuration where they must agree is the published one and the only one
every reported CTRNN result ran under: ``dt = 1``, ``wiring: fully_connected``.
Not one of the corrections is observable there, which is the reason all of them
survived; see ``docs/rtrrl-ctrnn-rflo-corrections.md``.

Two of the tests here characterise the *reference* rather than this side: the
published ``rtrl`` mode against backpropagation through the same three steps,
and this side's RFLO against the same judge on the one transition where the
approximation has nothing to drop. The first is what makes section 5 of the
corrections document a measurement instead of an assertion about somebody
else's code; the second is what makes the disagreements above approximation
rather than error.

The trace is compared through the gradient rather than read off the carry. The
published cell only advances its trace inside ``custom_vjp``'s forward rule, so
an undifferentiated call hands the carry's trace straight back untouched; and
the gradient is what the trace is *for*, so a comparison that ends there is the
comparison worth making.

The published code is not vendored. CI points ``RTRRL_AAAI25`` at a clone; the
local case is a skip.

    pytest tests/test_ctrnn_rflo_parity.py       # a verdict
    python tests/test_ctrnn_rflo_parity.py       # the numbers
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import pytest

from memorax.networks.sequence_models import RNN
from memorax.networks.sequence_models.ctrnn import (
    CTRNNCell,
    CTRNNConfig,
    CTRNNRflo,
    wiring_mask,
)
from tests.support.numerics import assert_within, deviations, flattened

pytestmark = [pytest.mark.parity, pytest.mark.external]

FEATURES = 3
HIDDEN = 4
STEPS = 5

# The float32 last bits two implementations of one expression may sit apart.
# They order the same sums differently -- a scan against a Python loop, one
# masked product against none -- so a few bits at the scale of the result is
# agreement rather than drift.
ALLOWED = 16.0


def published():
    """``models.ctrnn`` from an RTRRL-AAAI25 checkout, or why there is none."""

    root = os.environ.get("RTRRL_AAAI25")
    if not root:
        pytest.skip("set RTRRL_AAAI25 to an RTRRL-AAAI25 checkout to compare")
    if root not in sys.path:
        sys.path.insert(0, root)
    # Imported by name rather than by statement: there is no checkout to
    # resolve it against at type-check time, and a missing import is a skip
    # here rather than an error.
    return __import__("models.ctrnn", fromlist=["_"]).OnlineCTRNNCell


def sequence(seed=11, steps=STEPS):
    return jax.random.normal(jax.random.key(seed), (steps, FEATURES))


def cotangent(seed=7):
    return jax.random.normal(jax.random.key(seed), (HIDDEN,))


class Mine:
    """This side's cell over one stream, with the sequence walked in one call."""

    def __init__(self, *, dt, wiring):
        self.core = RNN(
            cell=CTRNNCell(
                config=CTRNNConfig(
                    features=FEATURES, hidden_dim=HIDDEN, dt=dt, wiring=wiring
                )
            )
        )
        self.rflo = CTRNNRflo(self.core)
        self.params = self.core.init(
            jax.random.key(2),
            jnp.zeros((1, 1, FEATURES)),
            done=jnp.zeros((1, 1), dtype=bool),
            initial_carry=jnp.zeros((1, HIDDEN)),
        )["params"]

    def _walk(self, params, x):
        done = jnp.zeros((1, x.shape[0]), dtype=bool)
        _, output, _ = self.rflo(
            params,
            x[None],
            done,
            jnp.zeros((1, HIDDEN)),
            self.rflo.initialize(jax.random.key(0), (1, FEATURES)),
        )
        return output[0]

    def outputs(self, x):
        return self._walk(self.params, x)

    def gradient(self, x, weight):
        """What the last step credits the parameters with, along ``weight``."""

        def read(params):
            return jnp.sum(self._walk(params, x)[-1] * weight)

        return jax.grad(read)(self.params)["cell"]

    def input_gradient(self, x, weight):
        def read(inputs):
            return jnp.sum(self._walk(self.params, inputs)[-1] * weight)

        return jax.grad(read)(x)


class Theirs:
    """The published cell over this side's parameters, stepped one at a time."""

    def __init__(self, *, dt, wiring, params):
        self.cell = published()(
            HIDDEN, dt=dt, wiring=wiring, wiring_kwargs={}, plasticity="rflo"
        )
        variables = self.cell.init(
            jax.random.key(0),
            (jnp.zeros((HIDDEN,)), None, None),
            jnp.zeros((FEATURES,)),
        )
        # Their mask, their tree, this side's weights: the two cells have to be
        # the same network before anything about credit can be compared.
        self.variables = dict(variables) | {
            "params": {
                "W": jnp.asarray(params["cell"]["W"]),
                "tau": jnp.asarray(params["cell"]["tau"]),
            }
        }

    def _walk(self, variables, x):
        carry = (
            jnp.zeros((HIDDEN,)),
            {
                "W": jnp.zeros((HIDDEN, FEATURES + HIDDEN + 1)),
                "tau": jnp.zeros((HIDDEN,)),
            },
            jnp.zeros((HIDDEN, FEATURES)),
        )
        outputs = []
        for step in range(x.shape[0]):
            carry, output = self.cell.apply(variables, carry, x[step])
            outputs.append(output)
        return jnp.stack(outputs)

    def outputs(self, x):
        return self._walk(self.variables, x)

    def gradient(self, x, weight):
        def read(params):
            walked = self._walk(self.variables | {"params": params}, x)
            return jnp.sum(walked[-1] * weight)

        return jax.grad(read)(self.variables["params"])

    def input_gradient(self, x, weight):
        def read(inputs):
            return jnp.sum(self._walk(self.variables, inputs)[-1] * weight)

        return jax.grad(read)(x)


def both(*, dt, wiring, their_wiring=None):
    mine = Mine(dt=dt, wiring=wiring)
    return mine, Theirs(dt=dt, wiring=their_wiring or wiring, params=mine.params)


# ------------------------------------------------------- where they must agree
def test_the_published_configuration_agrees_on_the_forward():
    mine, theirs = both(dt=1.0, wiring="fully_connected")
    x = sequence()

    assert_within(
        flattened(mine.outputs(x)),
        flattened(theirs.outputs(x)),
        "the forward",
        allowed=ALLOWED,
    )


def test_the_published_configuration_agrees_on_the_parameter_gradient():
    """The whole point of the trace: the gradient it stands in for.

    Both sides are asked the same question -- what the last step's output owes
    the parameters, weighted the same way -- so a difference is what they
    credit and not what either was asked.
    """

    mine, theirs = both(dt=1.0, wiring="fully_connected")
    x, weight = sequence(), cotangent()

    gradient = mine.gradient(x, weight)
    assert float(jnp.abs(gradient["W"]).max()) > 1e-3, "the comparison is vacuous"
    assert_within(
        flattened(gradient),
        flattened(theirs.gradient(x, weight)),
        "the gradient",
        allowed=ALLOWED,
    )


# --------------------------------------------------- where they must not agree
def test_the_integration_step_reaches_this_trace_and_not_the_published_one():
    """``dt`` in the trace. Absent from the published one, and this is where.

    Both cells integrate with ``dt``; only this one differentiates the step it
    took. The forward therefore still agrees and the credit no longer does,
    which is exactly the shape of the defect.
    """

    mine, theirs = both(dt=0.5, wiring="fully_connected")
    x, weight = sequence(), cotangent()

    assert_within(
        flattened(mine.outputs(x)),
        flattened(theirs.outputs(x)),
        "the forward at dt != 1",
        allowed=ALLOWED,
    )
    assert deviations(
        flattened(mine.gradient(x, weight)),
        flattened(theirs.gradient(x, weight)),
        ALLOWED,
    ), "the gradients agree at dt != 1, so nothing here carries the correction"


def test_a_wiring_reaches_this_trace_and_the_published_mask_is_a_column_out():
    """Two corrections meet in one configuration, so each is asserted alone.

    This side's credit is zero exactly where this side's mask is, and the
    published mask is not this side's mask: its identity lands one column to the
    right of the recurrent block, so it removes unit ``i``'s reading of unit
    ``i + 1`` instead of its own.
    """

    mask = wiring_mask("no_self", features=FEATURES, hidden_dim=HIDDEN)
    mine, theirs = both(
        dt=1.0, wiring="no_self", their_wiring="fully_connected_no_self"
    )
    x, weight = sequence(), cotangent()

    gradient = mine.gradient(x, weight)
    assert float(jnp.abs(gradient["W"][mask == 0]).max()) == 0.0
    assert float(jnp.abs(gradient["W"]).max()) > 1e-3

    their_gradient = theirs.gradient(x, weight)
    assert float(jnp.abs(their_gradient["W"][mask == 0]).max()) > 1e-6, (
        "the published run credits nothing this side masks out, so neither "
        "the mask correction nor the column correction is being carried here"
    )


def test_the_input_gradient_is_this_step_s_and_the_published_one_is_zero():
    """The published RFLO cell returns nothing to whatever fed it.

    ``jx`` is allocated, never written under ``rflo``, and contracted with the
    cotangent to produce the input's -- so it is structurally zero. Nothing sits
    in front of the cell in RTRRL's own torso, which is why this has never cost
    a reported result and why it would cost the first graph that put a
    projection there.
    """

    mine, theirs = both(dt=1.0, wiring="fully_connected")
    x, weight = sequence(steps=1), cotangent()

    assert float(jnp.abs(theirs.input_gradient(x, weight)).max()) == 0.0
    assert float(jnp.abs(mine.input_gradient(x, weight)).max()) > 1e-3


# -------------------------------------------- what the reference is, measured
def _unrolled(params, x, weight):
    """Backpropagation through the published equations, written out here.

    The judge for the section below. It is the CTRNN step and nothing else --
    no cell, no carry, no online rule -- so a disagreement with it is a
    statement about credit rather than about two forwards.
    """

    hidden = jnp.zeros((HIDDEN,))
    for step in range(x.shape[0]):
        row = jnp.concatenate([x[step], hidden, jnp.ones((1,))])
        hidden = hidden + (1.0 / params["tau"]) * (
            jnp.tanh(row @ params["W"].T) - hidden
        )
    return jnp.sum(hidden * weight)


def _published_credit(mode, params, x, weight):
    """One published mode's parameter gradient over the same sequence."""

    cell = published()(HIDDEN, dt=1.0, wiring=None, plasticity=mode)
    leading = () if mode == "rflo" else (HIDDEN,)

    def read(tree):
        carry = (
            jnp.zeros((HIDDEN,)),
            {name: jnp.zeros(leading + leaf.shape) for name, leaf in params.items()},
            jnp.zeros((HIDDEN, FEATURES)),
        )
        output = jnp.zeros((HIDDEN,))
        for step in range(x.shape[0]):
            carry, output = cell.apply(tree, carry, x[step])
        return jnp.sum(output * weight)

    return jax.grad(read)({"params": params})["params"]


def test_the_published_exact_mode_is_exact_on_one_leaf_of_two():
    """A characterisation of the reference, not of this side.

    ``plasticity="rtrl"`` exists to be exact, and its backward rule is
    ``df_dy @ t`` for every leaf. ``@`` against a 1-D left operand contracts
    axis 0 of a matrix and axis 1 of a stack of them, so ``tau`` -- whose
    sensitivity is ``(H, H)`` -- is contracted correctly and ``W`` -- whose
    sensitivity is ``(H, H, F + H + 1)`` -- is summed over the unit the weight
    belongs to instead of the unit the cotangent is on.

    Held here so that
    ``docs/rtrrl-ctrnn-rflo-corrections.md`` section 5 is a measurement. No
    ``rtrl`` mode exists on this side to correct, and none is proposed; what
    would go wrong if one were transcribed from that line is the point.
    """

    reference = Mine(dt=1.0, wiring="fully_connected").params["cell"]
    params = {name: jnp.asarray(reference[name]) for name in ("W", "tau")}
    x, weight = sequence(), cotangent()

    exact = jax.grad(_unrolled)(params, x, weight)
    credited = _published_credit("rtrl", params, x, weight)

    def gap(name):
        return float(jnp.abs(exact[name] - credited[name]).max()) / float(
            jnp.abs(exact[name]).max()
        )

    assert gap("tau") < 1e-5, "the one-axis leaf disagrees, so this reads wrong"
    assert gap("W") > 0.1, (
        "the published exact mode agrees with backpropagation on W, so the "
        "correction note in the corrections document is stale"
    )


def test_this_side_agrees_with_the_unroll_where_rflo_and_rtrl_must_agree():
    """One step from an empty state, where the approximation has nothing to drop.

    ``S_1 = (dh/dh) S_0 + dh/dtheta`` and ``S_0`` is zero, so RFLO and exact
    credit coincide on the first transition of a sequence. That is the horizon
    at which this side's RFLO has to equal backpropagation exactly, and it is
    what makes the disagreements elsewhere approximation rather than error.
    """

    mine = Mine(dt=1.0, wiring="fully_connected")
    params = {name: jnp.asarray(mine.params["cell"][name]) for name in ("W", "tau")}
    x, weight = sequence(steps=1), cotangent()

    exact = jax.grad(_unrolled)(params, x, weight)
    credited = mine.gradient(x, weight)

    assert_within(
        flattened({name: credited[name] for name in exact}),
        flattened(exact),
        "one step from empty",
        allowed=ALLOWED,
    )


if __name__ == "__main__":
    import pathlib as _pathlib

    sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

    if not os.environ.get("RTRRL_AAAI25"):
        print("set RTRRL_AAAI25 to an RTRRL-AAAI25 checkout to compare")
        raise SystemExit(1)

    for name, dt, wiring, their_wiring in (
        ("the published configuration", 1.0, "fully_connected", "fully_connected"),
        ("dt = 0.5", 0.5, "fully_connected", "fully_connected"),
        ("a wiring", 1.0, "no_self", "fully_connected_no_self"),
    ):
        mine, theirs = both(dt=dt, wiring=wiring, their_wiring=their_wiring)
        x, weight = sequence(), cotangent()
        forward = deviations(
            flattened(mine.outputs(x)), flattened(theirs.outputs(x)), 0.0
        )
        credit = deviations(
            flattened(mine.gradient(x, weight)),
            flattened(theirs.gradient(x, weight)),
            0.0,
        )
        print(f"\n=== {name} ===")
        print(f"  forward   {forward[0][0] if forward else 0.0:>12.1f} last bits")
        for bits, leaf in credit:
            print(f"  grad {leaf:5s} {bits:>12.1f} last bits")
