"""Our LRU and its exact credit, against the LRU the RTRRL paper published.

There are two references and they are two revisions of one file, transcribed with
their arithmetic untouched: ``upstream_lru`` is ``RTRRL-AAAI25`` at ``b71fd6e``,
the revision the paper was published at, and ``upstream_lru_rewritten`` is the
same file at ``4301943``, their HEAD. Our correct LRU answers to the published
one, since that is the revision with a recurrence in it to agree with, and each of
the two arms answers to the revision it reproduces. Both sides
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

No comparison against the reference initialises either side from a seed. Two
runtimes and two parameter trees spend a key in different orders, so every
parameter is drawn once and injected into both, and ``_inject`` fails if either
side has a parameter the draw does not cover. That is the guard against a
comparison that quietly comes out exact because it compared nothing.

The three arms are the exception, and the last test here is about them. They are
one runtime and one tree, so a seed does buy them a start, and it has to buy all
three the same one or an end-to-end comparison between them is a comparison of
different draws rather than of their arithmetic.

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
from memorax.networks.sequence_models.upstream_lru_rewritten import (
    OnlineLRULayer as RewrittenLayer,
)
from memorax.networks.torso import make_torso

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
READOUT = {"y": 8.0, "h": 8.0}
INFLUENCE = {
    "nu_log": 8.0,
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

# The rewritten arm's forward against their HEAD's, in last bits, and there is
# almost nothing to allow. Three of the four seeds agree on every leaf at every
# step exactly; the fourth is half a last bit in the state and two in the readout,
# from the one place the two still differ in shape -- their state is a matrix
# product over one step and ours is an einsum over a batched time axis with a
# multiplication by a zero decay in front of it. Twice the worst of the four.
REWRITTEN_READOUT = {"h": 4.0, "y": 4.0}
# Their HEAD's gradient is not reachable from here and the table is what says so.
# The five leaves their backward overwrites are about ten million last bits away,
# which is not a rounding of anything, while the three that reach the gradient by
# ordinary backpropagation on both sides are at two. What the arm cannot reproduce
# is written up on the test that xfails.
REWRITTEN_CREDITED = {
    "nu_log": 0.0,
    "theta_log": 0.0,
    "gamma_log": 0.0,
    "B_real": 0.0,
    "B_imag": 0.0,
    "C_real": 4.0,
    "C_imag": 4.0,
    "D": 4.0,
}


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


def inputs(seed: int, *, steps: int = STEPS):
    """One stream of observations, and the weights a scalar reads them out by.

    ``steps`` is longer only where the accumulation along the stream is the thing
    being measured rather than bounded, which is the precision comparison.
    """

    keys = jax.random.split(jax.random.key(seed + 1000), 2)
    xs = jax.random.normal(keys[0], (steps, FEATURES), jnp.float32)
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


def widened(values: dict, dtype) -> dict:
    """The same draw, in the precision it is about to be computed in.

    Widened rather than redrawn, so that the double-precision run is the same
    problem as the single-precision one and the difference between them is the
    rounding and nothing else. A float32 value is exact in float64, so this
    changes no number.
    """

    real = jnp.finfo(dtype).dtype
    return {name: value.astype(real) for name, value in values.items()}


def cold_start(dtype):
    """Our carry and sensitivity before any step, which is zeros and a unit decay.

    The influence matrices being zero here is what makes the whole stream worth
    watching rather than its first step, and it is the reason the module docstring
    gives for accumulating the worst gap instead of stopping early.
    """

    return LRUCarry(
        state=jnp.zeros((1, 1, HIDDEN), dtype),
        decay=jnp.ones((1, 1, HIDDEN), dtype),
    ), {
        "nu_log": jnp.zeros((1, 1, HIDDEN), dtype),
        "theta_log": jnp.zeros((1, 1, HIDDEN), dtype),
        "gamma_log": jnp.zeros((1, 1, HIDDEN), dtype),
        "B_real": jnp.zeros((1, 1, HIDDEN, FEATURES), dtype),
        "B_imag": jnp.zeros((1, 1, HIDDEN, FEATURES), dtype),
    }


def paper_side(seed: int, *, dtype=jnp.complex64):
    """The published layer, its parameters injected, ready to be stepped."""

    layer = OnlineLRULayer(d_hidden=HIDDEN)
    real = jnp.finfo(dtype).dtype
    carry = (
        jnp.zeros((HIDDEN,), dtype),
        (
            jnp.zeros((HIDDEN,), dtype),
            jnp.zeros((HIDDEN,), dtype),
            jnp.zeros((HIDDEN, FEATURES), dtype),
        ),
    )
    shaped = layer.init(jax.random.key(0), carry, jnp.zeros((FEATURES,), real))
    values = widened(drawn(seed), dtype)
    return layer, {"params": _inject(shaped["params"], values, PAPER)}, carry


def paper_step(layer, params, carry, x) -> tuple[Any, Any]:
    """One transition of theirs, which is the only granularity theirs has."""

    return cast(tuple[Any, Any], layer.apply(params, carry, x))


def rewritten_side(seed: int, *, skip: float = 0.0, dtype=jnp.complex64):
    """Their HEAD layer, its parameters injected, ready to be stepped.

    A second reference, because the two revisions compute different things and
    each arm answers to its own. Two settings are chosen rather than defaulted.

    ``plasticity`` is anything other than ``"bptt"``: that string is the branch
    that returns the plain cell and never reaches their online rule, and every
    other value reaches it, including the ``"rflo"`` their own default asks for.

    ``activation`` is off. The same rewrite that removed the recurrence gave the
    layer a ``silu`` and a skip term, and ours reads out linearly. The skip is
    injected on both sides and compared; the nonlinearity is switched off, so what
    these comparisons measure is the recurrence and the credit rather than a
    difference in where an activation sits. That the arm has no ``silu`` either is
    a gap between the arm and their HEAD, and it is a gap in the readout, which is
    the one block here whose behaviour does not depend on it.

    The carry is theirs. The published revision's ``initialize_carry`` sizes the
    influence matrix for ``B`` by the batch axis instead of the input width, which
    is why ``paper_side`` builds one by hand; this revision fixed that, and using
    the fixed one is part of what is being reproduced.
    """

    # Their ``__init__`` annotates none of its parameters, so the default makes
    # ``activation`` read as ``str`` while the field it assigns is ``str | None``.
    # The cast says which of the two is meant, without editing a transcription.
    layer = RewrittenLayer(HIDDEN, plasticity="rtrl", activation=cast(str, None))
    real = jnp.finfo(dtype).dtype
    carry = layer.initialize_carry(jax.random.key(0), (FEATURES,))
    shaped = layer.init(jax.random.key(0), carry, jnp.zeros((FEATURES,), real))
    values = widened(drawn(seed), dtype)
    values["D"] = jnp.full((HIDDEN, FEATURES), skip, real)
    return layer, {"params": _inject(shaped["params"], values, PAPER)}, carry


def rewritten_step(layer, params, carry, x) -> tuple[Any, Any]:
    """One transition of their HEAD, through the rule a gradient would take.

    Differentiated rather than merely applied. Their online rule is a
    ``custom_vjp``, so applying it runs the primal and taking a gradient runs the
    forward, and it is the forward that computes an influence matrix at all. RTRRL
    drives it the second way, and so does this, or the traces this file reasons
    about would never be computed.
    """

    (carry, y), _ = jax.vjp(lambda p: cast(Any, layer.apply(p, carry, x)), params)
    return carry, y


def our_side(seed: int, *, skip: float = 0.0, cell=LRUCell, dtype=jnp.complex64):
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
    real = jnp.finfo(dtype).dtype
    carry, sensitivity = cold_start(dtype)
    values = widened(drawn(seed), dtype)
    values["D"] = jnp.full((HIDDEN, FEATURES), skip, real)
    # Traced through the method that will be applied, not through the plain
    # forward: the two are separate compact methods, and initialising against one
    # while running the other is how a parameter comes out shaped for a call the
    # run never makes.
    shaped = core.init(
        jax.random.key(0),
        jnp.zeros((1, 1, FEATURES), real),
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


@pytest.mark.parametrize("seed", range(4))
def test_their_head_carries_no_credit_from_one_step_to_the_next(seed):
    """What the other arm is shaped by, asserted against their code and not read.

    ``_trace_update`` at their HEAD reads ``Lambda * grad_memory[i] + <this
    step>``, so reading it says the influence matrices accumulate, and the arm was
    built to that reading. They do not. ``grad_memory`` is the carry's, the primal
    returns the traces it was handed unless ``force_trace_compute`` -- a parameter
    with no caller in their repository -- and the traces their forward computes go
    into a residual that one step's gradient consumes and drops. So the carry's
    traces are the zeros their ``initialize_carry`` made, at every step, and the
    accumulation term is always a multiplication by zero.

    Asserted here rather than argued, because the arm's shape follows from it and
    an argument about someone else's control flow is exactly the thing that was
    wrong before. Exactly zero, not nearly: nothing writes to them at all.

    The state is required to be nonzero in the same breath. Traces that stayed at
    zero because the whole layer output zero would satisfy the above and establish
    nothing.
    """

    layer, params, carry = rewritten_side(seed)
    xs, _ = inputs(seed)

    reached = []
    for step, x in enumerate(xs):
        carry, _ = rewritten_step(layer, params, carry, x)
        state, traces = carry
        reached.append(float(jnp.max(jnp.abs(state))))
        for name, trace in zip(("z_lambda", "z_gamma", "z_B"), traces):
            moved = float(jnp.max(jnp.abs(trace)))
            assert moved == 0.0, (
                f"step={step}: their HEAD's {name} left zero, reaching {moved:.3g}. "
                "Its influence matrices accumulate after all, and the arm that "
                "reproduces this revision must accumulate with them"
            )
    assert min(reached) > 1e-3, (
        f"their HEAD's state never left zero either, closest {min(reached):.3g}, "
        "so the traces holding at zero says nothing about what they accumulate"
    )


@pytest.mark.parametrize("seed", range(4))
def test_the_rewritten_arm_is_their_heads_state_and_readout(seed):
    """The arm's forward is the one their rewrite left behind.

    Their HEAD reshapes to a length-one sequence before its associative scan, so
    the scan is the identity, the previous carry reaches the new state only to
    supply its width, and the state is ``B_norm x_t``. The arm reports a decay of
    zero from ``__call__``, which is how a scan that does accumulate is made to
    yield the same thing.

    The skip term is injected nonzero here. It is theirs as much as ours at this
    revision -- the same rewrite added it -- so unlike the published comparison
    there is no reason to zero it, and leaving it in means the readout comparison
    covers the path that carries most of the output once the recurrence is gone.
    """

    layer, their_params, their_carry = rewritten_side(seed, skip=0.5)
    core, our_params, our_carry, sensitivity = our_side(
        seed, skip=0.5, cell=RewrittenLRUCell
    )
    xs, _ = inputs(seed)

    worst: dict = {}
    for step, x in enumerate(xs):
        their_carry, their_y = rewritten_step(layer, their_params, their_carry, x)
        our_carry, our_y, sensitivity = our_step(
            core, our_params, our_carry, sensitivity, x
        )
        watch(
            worst,
            flattened({"h": our_carry.state[0, 0], "y": our_y[0, 0]}),
            flattened({"h": their_carry[0], "y": their_y}),
            f"step={step}",
        )
    assert_explained(worst, REWRITTEN_READOUT, f"rewritten arm's forward seed={seed}")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "their HEAD scales every hidden unit's credit by unit zero's cotangent, "
        "which is a property of how their backward reads a cotangent and not of "
        "any cell, so no arm reaches it; measured at about ten million last bits"
    ),
)
@pytest.mark.parametrize("seed", range(4))
def test_the_rewritten_arm_is_their_heads_gradient(seed):
    """And its credit is one step deep like theirs, and still not their gradient.

    The arm reproduces both halves of what their rewrite did to the cell. The
    forward has no recurrence, asserted above and bit-exact at three of four
    seeds. The credit is one step deep, since their traces never leave the zero
    ``initialize_carry`` made, and the arm reports a decay of zero from
    ``local_jacobian`` to match. No correction is needed on the ``B`` leaves
    either, because ``0dbd780`` gave this revision the exponential its published
    one is missing.

    And the gradient is still ten million last bits away on exactly the five
    leaves their ``bwd`` overwrites, while the three that reach it by ordinary
    backpropagation on both sides are at two. That locates it inside their
    backward, and the line is ``d_output_d_h = y_t[1][0]``. At ``b71fd6e`` the
    primal returned ``(new_carry, new_carry)``, so ``y_t[1]`` was a carry and
    ``[0]`` took the cotangent of the whole hidden state. The rewrite changed the
    primal's second output from that carry to the bare state array and left the
    indexing alone, so ``y_t[1]`` is now an array and ``[0]`` takes hidden unit
    zero's cotangent -- one scalar, which then scales the credit of every unit.

    So this revision has three defects and not two, and the third is not a
    property of a cell. A ``MemoroidCellBase`` says what a state is and what its
    jacobians are; which cotangent weights them is the framework's, and reaching
    this would mean corrupting that for every cell rather than swapping one. The
    comparison is kept and marked rather than deleted, because it is the thing
    that measures the claim, and a claim about their gradient with no failing test
    under it is how the accumulation went unnoticed.

    Strict, so that their gradient becoming reachable is a failure here and not a
    silent pass.
    """

    layer, their_params, their_carry = rewritten_side(seed, skip=0.5)
    core, our_params, our_carry, sensitivity = our_side(
        seed, skip=0.5, cell=RewrittenLRUCell
    )
    xs, weights = inputs(seed)

    worst: dict = {}
    for step, x in enumerate(xs):

        def their_loss(params, carry=their_carry, x=x):
            _, y = cast(Any, layer.apply(params, carry, x))
            return jnp.sum(weights * y)

        def arm_loss(params, carry=our_carry, credit=sensitivity, x=x):
            _, y, _ = our_step(core, params, carry, credit, x)
            return jnp.sum(weights * y[0, 0])

        theirs = {
            path[-1]: leaf
            for path, leaf in traverse_util.flatten_dict(
                cast(dict, jax.grad(their_loss)(their_params))
            ).items()
        }
        expected = {
            "nu_log": theirs["nu_log"],
            "theta_log": theirs["theta_log"],
            "gamma_log": theirs["gamma_log"],
            "B_real": theirs["B_real"],
            "B_imag": theirs["B_img"],
            "C_real": theirs["C_real"],
            "C_imag": theirs["C_img"],
            "D": theirs["D"],
        }
        ours = {
            path[-1]: leaf
            for path, leaf in traverse_util.flatten_dict(
                cast(dict, jax.grad(arm_loss)(our_params))
            ).items()
        }
        watch(worst, {leaf: ours[leaf] for leaf in expected}, expected, f"step={step}")

        their_carry, _ = rewritten_step(layer, their_params, their_carry, x)
        our_carry, _, sensitivity = our_step(
            core, our_params, our_carry, sensitivity, x
        )
    assert_explained(worst, REWRITTEN_CREDITED, f"rewritten arm's credit seed={seed}")


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


# Named as an experiment names them rather than as classes, because a name is
# what an experiment selects and the registry is the part that has to hold.
ARMS = ("lru", "lru_published", "lru_rewritten")


def test_one_seed_buys_every_arm_the_same_start():
    """The three arms differ in their arithmetic and not in where they begin.

    Everything above injects one draw into both sides, because ours and the
    reference are two runtimes whose parameter trees spend a key differently.
    The three arms are not that: they are one runtime and one tree, so for them a
    seed does buy a start, and running them against each other at one seed is a
    comparison of their arithmetic only if it buys all three the same one.

    Today it does, because they subclass one cell and declare no parameters of
    their own. An arm that added a parameter, or declared the existing ones in
    another order, would shift every draw after it and turn the comparison into
    one of three different starting points -- without failing anything, since all
    three would still run and still produce curves. That is the mistake
    ``81d3195f`` found between the two StreamAC kernels, where one seed bought
    two different sets of initial parameters and a thousand-point gap was read as
    a framework difference.
    """

    carry, sensitivity = cold_start(jnp.complex64)
    starts = {}
    for arm in ARMS:
        torso = make_torso(arm, features=FEATURES, hidden_dim=HIDDEN, output_dim=HIDDEN)
        shaped = cast(
            dict,
            torso.init(
                jax.random.key(0),
                jnp.zeros((1, 1, FEATURES), jnp.float32),
                jnp.zeros((1, 1), bool),
                carry,
                sensitivity=sensitivity,
                method="local_jacobian",
            ),
        )
        starts[arm] = flattened(shaped["params"])

    ours = starts["lru"]
    # Eight leaves: five recurrent, two readout matrices and the skip term. A tree
    # that lost most of itself would compare almost nothing and still agree.
    assert len(ours) >= 8, f"only {len(ours)} parameters to compare"
    for arm in ARMS[1:]:
        apart = deviations(starts[arm], ours, 0.0)
        assert not apart, (
            f"{arm} starts somewhere ours does not, so comparing the two end to "
            f"end at one seed would compare two draws: {apart}"
        )
