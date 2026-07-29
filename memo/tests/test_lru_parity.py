"""Our LRU and its exact credit, against the LRU the RTRRL paper published.

The reference is ``memorax.networks.sequence_models.upstream_lru``, which is
``RTRRL-AAAI25`` at ``b71fd6e`` with its arithmetic untouched; why that revision
rather than the repository's HEAD is written at the head of that file. Both sides
are driven one transition at a time, because that is the granularity the
reference computes at: its ``custom_vjp`` is a single-step rule, and comparing a
scan against it would fold our reassociation into every leaf at once instead of
naming the step where a leaf first moves.

Every comparison runs the whole stream before it judges, keeping the worst gap
each leaf reached and the step it reached it at. Stopping at the first step that
disagrees would have measured almost nothing here: the influence matrices start
at zero and ``h_0`` is zero with them, so at the first step the two sensitivities
that are built from the previous carry are zero on both sides and agree for the
one reason that proves nothing. They have to be watched for as long as they
accumulate.

Nothing here initialises either implementation from a seed. Two runtimes and two
parameter trees spend a key in different orders, so every parameter is drawn once
and injected into both, and ``_inject`` fails if either side has a parameter the
draw does not cover. That is the guard against a comparison that quietly comes
out exact because it compared nothing.

Three structural differences between the two are real and are asserted rather
than tolerated:

Our readout adds a skip term ``D x``; the published layer has no ``D`` at all and
no activation, computing ``Re(h C^T)`` alone. So ``D`` is injected as zeros here,
and a reverse test asserts that a non-zero ``D`` would have been seen -- without
it, the zero would make the readout comparison vacuous.

The published influence matrix for ``B`` adds ``outer(gamma_log, x)`` where the
derivative calls for ``outer(exp(gamma_log), x)``. Both accumulate under the same
decay, so ours is theirs scaled per hidden unit by ``exp(gamma_log) / gamma_log``
for every step, and ``input_gain`` applies exactly that factor. A reverse test
asserts the factor is not one, since a draw where it were would make the
correction untestable.

They accumulate ``dh/dLambda`` and chain it to ``nu_log`` and ``theta_log``
afterwards, through a ``vjp`` of ``get_lambda``; we fold ``dLambda/dnu`` into the
Jacobian before the scan. ``dLambda/dnu`` does not depend on the step, so
factoring it out is the same quantity, but it is not the same operation, and a
``vjp`` of a real-to-complex function is where a conjugation convention would
differ if the two had one. They do not. What those two leaves are apart by is
single last bits, growing with the step, which is a distributive law rounding
rather than a sign: conjugating one of them would put the gradient of every
recurrent parameter into a different direction, not a different final bit.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from conftest import deviations, flattened
from flax import traverse_util

from memorax.networks.sequence_models.lru import LRUCarry, LRUCell, LRUConfig
from memorax.networks.sequence_models.lru_upstream import (
    PublishedLRUCell,
    RewrittenLRUCell,
)
from memorax.networks.sequence_models.memoroid import Memoroid
from memorax.networks.sequence_models.upstream_lru import OnlineLRULayer

# The published layer reads out with one matrix whose row count is the field it
# was constructed with, so its output width and its hidden width are the same
# number. Ours can separate them; here it must not, or the two Cs differ in shape
# before any arithmetic is compared.
HIDDEN = 3
FEATURES = 4
STEPS = 5

# The reference spells two parameters differently. Everything else is spelled the
# same on both sides, which is why the injection can key on names at all.
PAPER = {"B_imaginary": "B_img", "C_imaginary": "C_img"}
OURS = {"B_imaginary": "B_imag", "C_imaginary": "C_imag"}

# Where the two agree on the arithmetic and disagree on the order of it, per leaf
# and in last bits. Every entry below has one cause, and it is the same cause for
# all four recurrent parameters: a chain factor that does not depend on the step,
# multiplied on the inside of the accumulation by us and on the outside by them.
# We build each Jacobian already chained -- the input gain into the projection for
# ``gamma_log`` and ``B``, ``dLambda/dnu`` and ``dLambda/dtheta`` into the carry
# for ``nu_log`` and ``theta_log`` -- and then accumulate. They accumulate the
# unchained influence matrices and chain afterwards, at the gradient. Distributing
# a constant over a sum is exact in arithmetic and is not exact in float32, so the
# gap is a fraction of a last bit per term and compounds along the stream, which
# is why every worst case below lands at the third step or later rather than the
# first. ``theta_log`` is the widest of them because its factor is a rotation:
# ``1j * exp(theta_log) * Lambda`` sends real parts into imaginary ones, so the
# cancellation it reassociates is the one with the least significance to spare.
# The readout carries all of this through ``h`` and adds a contraction of its own,
# an ``einsum`` over a batched time axis where they write a matrix product over a
# single step. Every allowance is twice the worst of the four seeds, which is
# headroom for the accumulation rather than a tolerance chosen to pass.
READOUT = {"y": 8.0, "h": 2.0}
INFLUENCE = {
    "nu_log": 2.0,
    "theta_log": 4.0,
    "gamma_log": 8.0,
    "B_real": 4.0,
    "B_imag": 4.0,
}
CREDITED = {
    "nu_log": 8.0,
    "theta_log": 8.0,
    "gamma_log": 8.0,
    "B_real": 8.0,
    "B_imag": 8.0,
    "C_real": 4.0,
    "C_imag": 4.0,
}
# Ours unrolled at once against ours stepped: the associative scan reassociates
# the recurrence itself, which is the one gap here that has nothing to do with
# the reference.
UNROLLED = 4.0


def drawn(seed: int) -> dict:
    """One draw of every parameter both implementations have.

    ``gamma_log`` is drawn away from zero deliberately. It is divided by, to undo
    the published influence matrix's missing exponential, and the real
    initialiser puts it at ``log(sqrt(1 - |lambda|^2))`` -- negative and not
    small -- so a draw around minus one is where the comparison would sit anyway.
    """

    keys = jax.random.split(jax.random.key(seed), 7)
    return {
        "nu_log": 0.4 * jax.random.normal(keys[0], (HIDDEN,), jnp.float32),
        "theta_log": 0.4 * jax.random.normal(keys[1], (HIDDEN,), jnp.float32),
        "gamma_log": -1.0 + 0.3 * jax.random.normal(keys[2], (HIDDEN,), jnp.float32),
        "B_real": jax.random.normal(keys[3], (HIDDEN, FEATURES), jnp.float32),
        "B_imaginary": jax.random.normal(keys[4], (HIDDEN, FEATURES), jnp.float32),
        "C_real": jax.random.normal(keys[5], (HIDDEN, HIDDEN), jnp.float32),
        "C_imaginary": jax.random.normal(keys[6], (HIDDEN, HIDDEN), jnp.float32),
    }


def inputs(seed: int):
    """One stream of observations, and the weights a scalar reads them out by."""

    keys = jax.random.split(jax.random.key(seed + 1000), 2)
    xs = jax.random.normal(keys[0], (STEPS, FEATURES), jnp.float32)
    weights = jax.random.normal(keys[1], (HIDDEN,), jnp.float32)
    return xs, weights


def _inject(tree, values: dict, spelling: dict) -> dict:
    """Put the drawn parameters into a tree, and account for every leaf.

    Keyed on the last element of each path rather than the path itself, so that
    neither side's module nesting is written down here. Every leaf of the tree
    must be covered and every drawn value must be used, which is what makes a
    parameter appearing or disappearing on either side a failure here rather than
    a comparison that silently skips it.
    """

    named = {spelling.get(name, name): value for name, value in values.items()}
    flat = cast(dict[tuple[str, ...], Any], traverse_util.flatten_dict(tree))
    used = set()
    out = {}
    for path, leaf in flat.items():
        name = path[-1]
        assert name in named, f"nothing was drawn for {'/'.join(path)}"
        value = named[name]
        assert (
            value.shape == leaf.shape
        ), f"{'/'.join(path)}: drew {value.shape}, wanted {leaf.shape}"
        out[path] = value
        used.add(name)
    unused = sorted(set(named) - used)
    assert not unused, f"drew {unused}, which this tree has no parameter for"
    return cast(dict, traverse_util.unflatten_dict(out))


def paper_side(seed: int):
    """The published layer, its parameters injected, ready to be stepped."""

    layer = OnlineLRULayer(d_hidden=HIDDEN)
    carry = (
        jnp.zeros((HIDDEN,), jnp.complex64),
        (
            jnp.zeros((HIDDEN,), jnp.complex64),
            jnp.zeros((HIDDEN,), jnp.complex64),
            jnp.zeros((HIDDEN, FEATURES), jnp.complex64),
        ),
    )
    shaped = layer.init(jax.random.key(0), carry, jnp.zeros((FEATURES,), jnp.float32))
    return layer, {"params": _inject(shaped["params"], drawn(seed), PAPER)}, carry


def paper_step(layer, params, carry, x) -> tuple[Any, Any]:
    """One transition of theirs, which is the only granularity theirs has."""

    return cast(tuple[Any, Any], layer.apply(params, carry, x))


def our_side(seed: int, *, skip: float = 0.0, cell=LRUCell):
    """Ours, its parameters injected, with the skip term the reference lacks.

    ``skip`` is the constant every element of ``D`` is set to. It is zero
    everywhere the readout is compared, since the reference has no such term at
    all, and non-zero only in the test that asserts the zero was doing work.

    ``cell`` selects which of the three this drives. All three take the same
    config and build the same tree, so the same injection reaches all of them --
    which is the property the arms depend on, exercised here rather than assumed.
    """

    core = Memoroid(
        cell=cell(
            config=LRUConfig(features=FEATURES, hidden_dim=HIDDEN, output_dim=HIDDEN)
        )
    )
    carry = LRUCarry(
        state=jnp.zeros((1, 1, HIDDEN), jnp.complex64),
        decay=jnp.ones((1, 1, HIDDEN), jnp.complex64),
    )
    sensitivity = {
        "nu_log": jnp.zeros((1, 1, HIDDEN), jnp.complex64),
        "theta_log": jnp.zeros((1, 1, HIDDEN), jnp.complex64),
        "gamma_log": jnp.zeros((1, 1, HIDDEN), jnp.complex64),
        "B_real": jnp.zeros((1, 1, HIDDEN, FEATURES), jnp.complex64),
        "B_imag": jnp.zeros((1, 1, HIDDEN, FEATURES), jnp.complex64),
    }
    values = dict(drawn(seed))
    values["D"] = jnp.full((HIDDEN, FEATURES), skip, jnp.float32)
    # Traced through the method that will be applied, not through the plain
    # forward: the two are separate compact methods, and initialising against one
    # while running the other is how a parameter comes out shaped for a call the
    # run never makes.
    shaped = core.init(
        jax.random.key(0),
        jnp.zeros((1, 1, FEATURES), jnp.float32),
        jnp.zeros((1, 1), bool),
        carry,
        sensitivity=sensitivity,
        method="local_jacobian",
    )
    return core, _inject(shaped["params"], values, OURS), carry, sensitivity


def our_step(core, params, carry, sensitivity, x) -> tuple[Any, Any, Any]:
    """One transition of ours, at the granularity the reference computes."""

    return cast(
        tuple[Any, Any, Any],
        core.apply(
            {"params": params},
            x[None, None, :],
            jnp.zeros((1, 1), bool),
            carry,
            sensitivity=sensitivity,
            method="local_jacobian",
        ),
    )


def input_gain(values: dict):
    """The factor the published influence matrix for ``B`` is missing.

    It adds the log of the input gain where the derivative calls for the gain, so
    theirs is ours divided by this, per hidden unit, at every step.
    """

    return jnp.exp(values["gamma_log"]) / values["gamma_log"]


def expected_sensitivity(values: dict, traces, *, correcting: bool = True) -> dict:
    """Our sensitivities as their influence matrices determine them.

    Theirs carry ``dh/dLambda``, ``dh/dgamma`` and ``dh/dB`` and chain the
    parameter derivatives on afterwards. Ours carry the chained quantity already,
    and since none of the three chain factors depends on the step, each is the
    other scaled by a constant.

    ``correcting`` is what the two callers differ by, and the difference is the
    point of having both. Our LRU exponentiates the input gain where theirs takes
    its logarithm, so comparing it against theirs needs the factor; the arm that
    reproduces their line needs the factor gone, and gets ``to_b`` as they compute
    it. One expectation with a flag rather than two, so that no second
    transcription of the same recursion can drift from this one.
    """

    lam = jnp.exp(-jnp.exp(values["nu_log"]) + 1j * jnp.exp(values["theta_log"]))
    gain = jnp.exp(values["gamma_log"])
    to_lambda, to_gamma, to_b = traces
    factor = input_gain(values)[:, None] if correcting else 1.0
    return {
        "nu_log": -jnp.exp(values["nu_log"]) * lam * to_lambda,
        "theta_log": 1j * jnp.exp(values["theta_log"]) * lam * to_lambda,
        "gamma_log": gain * to_gamma,
        "B_real": to_b * factor,
        "B_imag": 1j * to_b * factor,
    }


def watch(worst: dict, actual: dict, expected: dict, where: str) -> None:
    """Keep the widest gap each leaf has reached, and the step it reached it at.

    ``deviations`` still does the judging, and still fails outright if a leaf the
    comparison expects is absent or the wrong shape. This only stops the first
    disagreement from ending the stream before the leaves that need a stream to
    disagree have had one.
    """

    for bits, path in deviations(actual, expected):
        if bits > worst.get(path, (0.0, ""))[0]:
            worst[path] = (bits, where)


def assert_explained(
    worst: dict, explained: dict, what: str, *, default: float = 0.0
) -> None:
    """Nothing is further apart than this file has written down a reason for."""

    unexplained = sorted(
        (
            (bits, f"{path} ({where})")
            for path, (bits, where) in worst.items()
            if bits > explained.get(path, default)
        ),
        reverse=True,
    )
    assert not unexplained, (
        f"{what}: leaves apart from the published LRU by more than this file "
        "explains, worst first:\n"
        + "\n".join(f"  {bits:.1f} last bits  {where}" for bits, where in unexplained)
    )


@pytest.mark.parametrize("seed", range(4))
def test_the_hidden_state_and_readout_are_the_papers(seed):
    """Block: the recurrence, and the readout over it.

    Ours forms the whole input projection and then unrolls it by an associative
    scan; theirs multiplies the carry by the decay and adds the projection, one
    step at a time. Same recurrence, and at one step per call the scan has
    nothing to reassociate, so what is left is the input gain's placement and the
    readout's contraction, both of which ``READOUT`` accounts for.
    """

    layer, paper_params, paper_carry = paper_side(seed)
    core, our_params, our_carry, sensitivity = our_side(seed)
    xs, _ = inputs(seed)

    worst: dict = {}
    for step, x in enumerate(xs):
        paper_carry, paper_y = paper_step(layer, paper_params, paper_carry, x)
        our_carry, our_y, sensitivity = our_step(
            core, our_params, our_carry, sensitivity, x
        )
        watch(
            worst,
            flattened({"h": our_carry.state[0, 0], "y": our_y[0, 0]}),
            flattened({"h": paper_carry[0], "y": paper_y}),
            f"step={step}",
        )
    assert_explained(worst, READOUT, f"hidden state and readout seed={seed}")


@pytest.mark.parametrize("seed", range(4))
def test_the_influence_matrices_are_the_papers(seed):
    """Block: what the real-time credit accumulates.

    This is the quantity RTRL exists for, and the one place the two
    implementations were always going to be hardest to compare: theirs is three
    matrices carried in the cell's own carry, ours is five sensitivities keyed by
    the parameter each credits. ``expected_sensitivity`` is the map between them,
    and it is arithmetic on their matrices rather than a restatement of ours.

    The whole stream is watched because the first step cannot judge two of the
    five: ``nu_log`` and ``theta_log`` are built from the previous carry, which is
    zero there, so they agree at step zero however either side computes them.
    """

    layer, paper_params, paper_carry = paper_side(seed)
    core, our_params, our_carry, sensitivity = our_side(seed)
    values = drawn(seed)
    xs, _ = inputs(seed)

    worst: dict = {}
    for step, x in enumerate(xs):
        paper_carry, _ = paper_step(layer, paper_params, paper_carry, x)
        our_carry, _, sensitivity = our_step(
            core, our_params, our_carry, sensitivity, x
        )
        watch(
            worst,
            flattened(jax.tree.map(lambda leaf: leaf[0, 0], sensitivity)),
            flattened(expected_sensitivity(values, paper_carry[1])),
            f"step={step}",
        )
    assert_explained(worst, INFLUENCE, f"influence matrices seed={seed}")


@pytest.mark.parametrize("seed", range(4))
def test_the_credited_gradient_is_the_papers(seed):
    """Block: the gradient the accumulated credit produces.

    The seam every other block here leaves open. Theirs comes out of a
    hand-written ``custom_vjp`` that overwrites five leaves of an ordinary
    backward pass; ours comes out of an ordinary backward pass over a phantom
    term whose value is zero and whose derivative is the sensitivity. Both are
    differentiated at a fixed carry, which is what makes them the same
    single-step quantity, and the two readout matrices are in the comparison
    because they are the half of the path that is plain backpropagation on both
    sides and so has nothing to excuse.
    """

    layer, paper_params, paper_carry = paper_side(seed)
    core, our_params, our_carry, sensitivity = our_side(seed)
    values = drawn(seed)
    xs, weights = inputs(seed)
    gain = input_gain(values)[:, None]

    worst: dict = {}
    for step, x in enumerate(xs):

        def paper_loss(params, carry=paper_carry, x=x):
            _, y = paper_step(layer, params, carry, x)
            return jnp.sum(weights * y)

        def our_loss(params, carry=our_carry, credit=sensitivity, x=x):
            _, y, _ = our_step(core, params, carry, credit, x)
            return jnp.sum(weights * y[0, 0])

        theirs = {
            path[-1]: leaf
            for path, leaf in traverse_util.flatten_dict(
                cast(dict, jax.grad(paper_loss)(paper_params))
            ).items()
        }
        ours = {
            path[-1]: leaf
            for path, leaf in traverse_util.flatten_dict(
                cast(dict, jax.grad(our_loss)(our_params))
            ).items()
        }
        # Theirs is ours for every leaf but the two the missing exponential
        # reaches, and those two are scaled by it rather than excused.
        expected = {
            "nu_log": theirs["nu_log"],
            "theta_log": theirs["theta_log"],
            "gamma_log": theirs["gamma_log"],
            "B_real": theirs["B_real"] * gain,
            "B_imag": theirs["B_img"] * gain,
            "C_real": theirs["C_real"],
            "C_imag": theirs["C_img"],
        }
        watch(worst, {name: ours[name] for name in expected}, expected, f"step={step}")

        paper_carry, _ = paper_step(layer, paper_params, paper_carry, x)
        our_carry, _, sensitivity = our_step(
            core, our_params, our_carry, sensitivity, x
        )
    assert_explained(worst, CREDITED, f"credited gradient seed={seed}")


def test_the_scan_agrees_with_stepping():
    """Ours unrolled at once is ours stepped, to within the reassociation.

    Every comparison above drives one step per call, which is where the
    associative scan has the least to do. This is the one assertion about the
    form the scan actually runs in, and it is made against ours alone because the
    reference has no batched form to answer.
    """

    core, params, carry, sensitivity = our_side(0)
    xs, _ = inputs(0)

    stepped_carry, stepped_credit, ys = carry, sensitivity, []
    for x in xs:
        stepped_carry, y, stepped_credit = our_step(
            core, params, stepped_carry, stepped_credit, x
        )
        ys.append(y[0, 0])

    at_once_carry, at_once_y, at_once_credit = cast(
        tuple[Any, Any, Any],
        core.apply(
            {"params": params},
            xs[None],
            jnp.zeros((1, STEPS), bool),
            carry,
            sensitivity=sensitivity,
            method="local_jacobian",
        ),
    )
    worst: dict = {}
    watch(
        worst,
        flattened(
            {
                "y": at_once_y[0],
                "carry": at_once_carry.state,
                "credit": at_once_credit,
            }
        ),
        flattened(
            {
                "y": jnp.stack(ys),
                "carry": stepped_carry.state,
                "credit": stepped_credit,
            }
        ),
        "one call against five",
    )
    assert_explained(worst, {}, "one call against five", default=UNROLLED)


@pytest.mark.parametrize("seed", range(4))
def test_the_published_arm_needs_no_correction(seed):
    """The arm that reproduces their line agrees with it uncorrected.

    Every comparison above scales their influence matrix for ``B`` by
    ``exp(gamma_log) / gamma_log`` before ours will match it, because ours has
    the exponential their published revision is missing. ``PublishedLRUCell`` is
    ours with that one line put back the way they have it, so it is asserted
    against the same reference with the factor removed -- the same recursion, the
    same reference, one term different, and the term is the defect.

    That is what makes the arm worth running. A learning curve from it is
    comparable to theirs because the quantity it accumulates is theirs, not
    theirs-up-to-a-correction, and the gap between it and our correct arm is then
    the cost of the defect rather than the cost of two implementations differing.
    """

    layer, paper_params, paper_carry = paper_side(seed)
    core, our_params, our_carry, sensitivity = our_side(seed, cell=PublishedLRUCell)
    values = drawn(seed)
    xs, _ = inputs(seed)

    worst: dict = {}
    for step, x in enumerate(xs):
        paper_carry, _ = paper_step(layer, paper_params, paper_carry, x)
        our_carry, _, sensitivity = our_step(
            core, our_params, our_carry, sensitivity, x
        )
        watch(
            worst,
            flattened(jax.tree.map(lambda leaf: leaf[0, 0], sensitivity)),
            flattened(expected_sensitivity(values, paper_carry[1], correcting=False)),
            f"step={step}",
        )
    assert_explained(worst, INFLUENCE, f"published arm seed={seed}")


def test_the_rewritten_arm_forgets_while_crediting_as_though_it_had_not():
    """The other arm, and the disagreement it exists to hold open.

    Their HEAD reshapes to a length-one sequence before an associative scan, so
    the scan is the identity and the previous carry reaches the new state only to
    supply its width. The influence matrices were not rewritten with it and still
    accumulate under ``Lambda``. ``RewrittenLRUCell`` reproduces that by reporting
    a decay of zero from the forward while ``local_jacobian`` keeps reporting
    ``Lambda``, and both halves of that are asserted here, because an arm that
    reproduced only the forgetting would be a different defect than theirs.

    The forgetting is asserted by driving it from two different carries and
    requiring the same state and readout. That is only worth asserting if the
    carry could have mattered, so ours is driven from the same two and required to
    differ. The credit is asserted against our correct cell stepped along the
    rewritten carry sequence: same jacobians, same decay, so bit-exact and not
    approximately, and if the rewrite ever reached the sensitivities that equality
    is what breaks.
    """

    xs, _ = inputs(0)
    core, params, empty, zeros = our_side(0, cell=RewrittenLRUCell)
    correct, _, _, _ = our_side(0)

    elsewhere = LRUCarry(
        state=jnp.full_like(empty.state, 0.5 + 0.25j),
        decay=jnp.full_like(empty.decay, 0.5),
    )

    forgotten: dict = {}
    carry, credit = empty, zeros
    other, other_credit = elsewhere, zeros
    correct_credit = zeros
    remembered, from_empty, from_elsewhere = [], empty, elsewhere
    for step, x in enumerate(xs):
        from_empty, _, _ = our_step(correct, params, from_empty, zeros, x)
        from_elsewhere, _, _ = our_step(correct, params, from_elsewhere, zeros, x)
        remembered.append(
            float(jnp.max(jnp.abs(from_empty.state - from_elsewhere.state)))
        )

        # Ours, at the state theirs is in, credited by the machinery neither arm
        # overrides. Its carry is discarded: the recurrence is in it, and the next
        # step's jacobians have to come from the state the rewrite arrives at.
        _, _, correct_credit = our_step(correct, params, carry, correct_credit, x)

        carry, y, credit = our_step(core, params, carry, credit, x)
        other, other_y, other_credit = our_step(core, params, other, other_credit, x)
        watch(
            forgotten,
            flattened({"h": other.state, "y": other_y}),
            flattened({"h": carry.state, "y": y}),
            f"step={step}",
        )
        assert not deviations(credit, correct_credit), (
            f"step={step}: the rewritten arm's sensitivities are not the ones our "
            "correct cell accumulates from the same states, so the rewrite reached "
            "the credit assignment, which at their HEAD it does not"
        )

    assert_explained(forgotten, {}, "the rewritten arm's state, from two carries")
    # Ours keeps them apart at every step, by a margin that decays as |lambda|^t
    # and has not reached the float noise the comparison above lives in.
    assert min(remembered) > 1e-3, (
        "our LRU driven from those same two carries also ended up in the same "
        f"state, closest {min(remembered):.3g} apart, so requiring the rewritten "
        "arm's states to agree did not establish that it has no memory"
    )


def test_the_missing_exponential_is_a_real_difference():
    """And the correction corrects something.

    ``expected_sensitivity`` divides their ``B`` matrix by ``gamma_log`` and
    multiplies by its exponential. On a draw where those happened to be equal the
    factor would be one, the correction would be doing nothing, and the test that
    applies it would pass whether or not either side had the exponential. So the
    draw is asserted to be a draw that can tell them apart.
    """

    for seed in range(4):
        margin = float(jnp.min(jnp.abs(input_gain(drawn(seed)) - 1.0)))
        assert margin > 0.1, (
            f"seed={seed} drew an input gain within {margin:.3g} of its own "
            "logarithm, so the published influence matrix and ours cannot be "
            "told apart here"
        )


def test_the_skip_connection_would_have_been_seen():
    """And the reference's missing ``D`` is not hiding a difference.

    Every readout comparison injects ``D`` as zeros, because the published layer
    has no skip term. A zero that made no difference would mean the readout
    comparison never depended on it, so this asserts the opposite: with ``D`` set
    to anything, ours stops agreeing with what it agreed with at zero.
    """

    xs, _ = inputs(0)
    readouts = []
    for skip in (0.0, 0.5):
        core, params, carry, sensitivity = our_side(0, skip=skip)
        _, y, _ = our_step(core, params, carry, sensitivity, xs[0])
        readouts.append(y)
    assert deviations({"y": readouts[1]}, {"y": readouts[0]}), (
        "a skip term of one half changed no readout, so injecting zeros for the "
        "term the reference lacks was not what made the two agree"
    )
