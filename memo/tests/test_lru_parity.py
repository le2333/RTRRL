"""Our LRU and its exact credit, against the LRU the RTRRL paper published.

The reference is ``memorax.networks.sequence_models.upstream_lru``, which is
``RTRRL-AAAI25`` at ``b71fd6e`` with its arithmetic untouched; why that revision
rather than the repository's HEAD is written at the head of that file. Both sides
are driven one transition at a time, because that is the granularity the
reference computes at: its ``custom_vjp`` is a single-step rule, and comparing a
scan against it would fold our reassociation into every leaf at once instead of
naming the step where a leaf first moves.

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
for every step, and ``INPUT_GAIN`` below applies exactly that factor. A reverse
test asserts the factor is not one, since a draw where it were would make the
correction untestable.

They accumulate ``dh/dLambda`` and chain it to ``nu_log`` and ``theta_log``
afterwards, through a ``vjp`` of ``get_lambda``; we fold ``dLambda/dnu`` into the
Jacobian before the scan. ``dLambda/dnu`` does not depend on the step, so
factoring it out is the same quantity, but it is not the same operation, and the
conjugation convention a ``vjp`` of a real-to-complex function applies is exactly
what those two leaves test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from conftest import assert_within, deviations, flattened
from flax import traverse_util

from memorax.networks.sequence_models.lru import LRUCarry, LRUCell, LRUConfig
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


def _inject(tree: dict, values: dict, spelling: dict) -> dict:
    """Put the drawn parameters into a tree, and account for every leaf.

    Keyed on the last element of each path rather than the path itself, so that
    neither side's module nesting is written down here. Every leaf of the tree
    must be covered and every drawn value must be used, which is what makes a
    parameter appearing or disappearing on either side a failure here rather than
    a comparison that silently skips it.
    """

    named = {spelling.get(name, name): value for name, value in values.items()}
    flat = traverse_util.flatten_dict(tree)
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
    return traverse_util.unflatten_dict(out)


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


def our_side(seed: int, *, skip: float = 0.0):
    """Ours, its parameters injected, with the skip term the reference lacks.

    ``skip`` is the constant every element of ``D`` is set to. It is zero
    everywhere the readout is compared, since the reference has no such term at
    all, and non-zero only in the test that asserts the zero was doing work.
    """

    core = Memoroid(
        cell=LRUCell(
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


def our_step(core, params, carry, sensitivity, x):
    """One transition of ours, at the granularity the reference computes."""

    return core.apply(
        {"params": params},
        x[None, None, :],
        jnp.zeros((1, 1), bool),
        carry,
        sensitivity=sensitivity,
        method="local_jacobian",
    )


def input_gain(values: dict):
    """The factor the published influence matrix for ``B`` is missing.

    It adds the log of the input gain where the derivative calls for the gain, so
    theirs is ours divided by this, per hidden unit, at every step.
    """

    return jnp.exp(values["gamma_log"]) / values["gamma_log"]


def expected_sensitivity(values: dict, traces) -> dict:
    """Our sensitivities as their influence matrices determine them.

    Theirs carry ``dh/dLambda``, ``dh/dgamma`` and ``dh/dB`` and chain the
    parameter derivatives on afterwards. Ours carry the chained quantity already,
    and since none of the three chain factors depends on the step, each is the
    other scaled by a constant.
    """

    lam = jnp.exp(-jnp.exp(values["nu_log"]) + 1j * jnp.exp(values["theta_log"]))
    gain = jnp.exp(values["gamma_log"])
    to_lambda, to_gamma, to_b = traces
    return {
        "nu_log": -jnp.exp(values["nu_log"]) * lam * to_lambda,
        "theta_log": 1j * jnp.exp(values["theta_log"]) * lam * to_lambda,
        "gamma_log": gain * to_gamma,
        "B_real": to_b * input_gain(values)[:, None],
        "B_imag": 1j * to_b * input_gain(values)[:, None],
    }


@pytest.mark.parametrize("seed", range(4))
def test_the_hidden_state_and_readout_are_the_papers(seed):
    """Block: the recurrence, and the readout over it.

    Ours forms the whole input projection and then unrolls it by an associative
    scan; theirs multiplies the carry by the decay and adds the projection, one
    step at a time. Same recurrence, and at one step per call the scan has
    nothing to reassociate, so this is exact or the recurrence differs.
    """

    layer, paper_params, paper_carry = paper_side(seed)
    core, our_params, our_carry, sensitivity = our_side(seed)
    xs, _ = inputs(seed)

    for step, x in enumerate(xs):
        paper_carry, paper_y = layer.apply(paper_params, paper_carry, x)
        our_carry, our_y, sensitivity = our_step(
            core, our_params, our_carry, sensitivity, x
        )
        assert_within(
            flattened({"h": our_carry.state[0, 0], "y": our_y[0, 0]}),
            flattened({"h": paper_carry[0], "y": paper_y}),
            f"hidden state and readout seed={seed} step={step}",
        )


@pytest.mark.parametrize("seed", range(4))
def test_the_influence_matrices_are_the_papers(seed):
    """Block: what the real-time credit accumulates.

    This is the quantity RTRL exists for, and the one place the two
    implementations were always going to be hardest to compare: theirs is three
    matrices carried in the cell's own carry, ours is five sensitivities keyed by
    the parameter each credits. ``expected_sensitivity`` is the map between them,
    and it is arithmetic on their matrices rather than a restatement of ours.
    """

    layer, paper_params, paper_carry = paper_side(seed)
    core, our_params, our_carry, sensitivity = our_side(seed)
    values = drawn(seed)
    xs, _ = inputs(seed)

    for step, x in enumerate(xs):
        paper_carry, _ = layer.apply(paper_params, paper_carry, x)
        our_carry, _, sensitivity = our_step(
            core, our_params, our_carry, sensitivity, x
        )
        assert_within(
            flattened(jax.tree.map(lambda leaf: leaf[0, 0], sensitivity)),
            flattened(expected_sensitivity(values, paper_carry[1])),
            f"influence matrices seed={seed} step={step}",
        )


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

    for step, x in enumerate(xs):

        def paper_loss(params, carry=paper_carry, x=x):
            _, y = layer.apply(params, carry, x)
            return jnp.sum(weights * y)

        def our_loss(params, carry=our_carry, credit=sensitivity, x=x):
            _, y, _ = our_step(core, params, carry, credit, x)
            return jnp.sum(weights * y[0, 0])

        theirs = traverse_util.flatten_dict(jax.grad(paper_loss)(paper_params))
        ours = traverse_util.flatten_dict(jax.grad(our_loss)(our_params))
        theirs = {path[-1]: leaf for path, leaf in theirs.items()}
        ours = {path[-1]: leaf for path, leaf in ours.items()}

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
        assert_within(
            {name: ours[name] for name in expected},
            expected,
            f"credited gradient seed={seed} step={step}",
        )


def test_the_scan_agrees_with_stepping():
    """Ours unrolled at once is ours stepped, or the scan is not the recurrence.

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

    at_once_carry, at_once_y, at_once_credit = core.apply(
        {"params": params},
        xs[None],
        jnp.zeros((1, STEPS), bool),
        carry,
        sensitivity=sensitivity,
        method="local_jacobian",
    )
    assert_within(
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
        carry, y, _ = our_step(core, params, carry, sensitivity, xs[0])
        readouts.append(y)
    assert deviations({"y": readouts[1]}, {"y": readouts[0]}), (
        "a skip term of one half changed no readout, so injecting zeros for the "
        "term the reference lacks was not what made the two agree"
    )
