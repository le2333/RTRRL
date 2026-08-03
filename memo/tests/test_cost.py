"""What a step costs, counted rather than timed.

Timing on a shared runner is noise; the compiler will tell us the arithmetic it
plans to do, and that number is the same on every machine. Counted here because
a recorded hopper run spends about thirty milliseconds per update on work that
is a few thousand parameters wide, which is too much by an order of magnitude
and had to be either explained or fixed.
"""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from test_blocks import ours

from memorax.networks.sequence_models.lru import LRUCarry, LRUCell, LRUConfig
from memorax.networks.sequence_models.memoroid import Memoroid
from memorax.networks.sequence_models.upstream_lru import OnlineLRULayer

STREAMS = (1, 2, 4, 8)


def flops(fn, *args) -> float:
    """What the compiler expects to spend, for one call."""

    analysis = jax.jit(fn).lower(*args).compile().cost_analysis()
    if isinstance(analysis, list):
        analysis = analysis[0]
    if not analysis or "flops" not in analysis:
        pytest.skip("this backend does not report a cost analysis")
    return float(analysis["flops"])


def gradient_cost(streams: int) -> float:
    """One actor gradient, for a kernel running this many parallel streams."""

    mine = ours(num_envs=streams, credit="tbptt")
    state = mine.init(jax.random.key(0))
    action = jax.random.normal(jax.random.key(1), (streams, 2), dtype=jnp.float32)
    delta = jax.random.normal(jax.random.key(2), (streams,), dtype=jnp.float32)

    return flops(
        lambda params: mine._actor_gradient(
            params, state.timestep, state.actor_carry, None, action, delta
        ),
        state.actor_params,
    )


def test_a_gradient_costs_what_the_streams_it_credits_cost():
    """One stream's update should cost what one stream's update costs.

    Streams share parameters but not activations: stream i's ascent direction
    cannot depend on stream j's hidden state, so of the square of blocks a
    Jacobian of a per-stream output computes, only the diagonal can be non-zero.
    Whether autodiff knows that is what this measures. If it does not, the cost
    grows with the square of the streams and most of a training run is spent
    filling in structural zeros.
    """

    measured = {streams: gradient_cost(streams) for streams in STREAMS}
    per_stream = {streams: cost / streams for streams, cost in measured.items()}
    reported = "\n".join(
        f"  {streams:2d} streams  {measured[streams]:12.0f} flops  "
        f"{per_stream[streams]:10.0f} per stream"
        for streams in STREAMS
    )

    cheapest, dearest = min(per_stream.values()), max(per_stream.values())
    assert dearest < 1.5 * cheapest, (
        f"the cost per stream is not flat, so the streams are not being "
        f"credited independently:\n{reported}"
    )


def training_step_cost(credit: str, streams: int) -> float:
    """One training step of the kernel, under one credit setting."""

    agent = ours(num_envs=streams, credit=credit)
    state = agent.init(jax.random.key(0))
    return flops(lambda s: agent.train(jax.random.key(1), s, streams), state)


def test_exact_credit_costs_what_carrying_a_sensitivity_costs(capsys):
    """What the recurrent credit is worth per step, in arithmetic.

    Exact credit carries a sensitivity of the carry with respect to the torso's
    parameters and advances it every step; truncated credit carries nothing. For
    a diagonal recurrence like the RTU that sensitivity is not a dense Jacobian:
    it is one entry per parameter element per stream, and the two input matrices
    dominate it. Advancing it is a rotation and a rescale of every one of those
    entries, so the step's cost is expected to be a multiple of the truncated
    one rather than the same number, and this reports which multiple.

    Asserted loosely on purpose. The claim worth defending is that exact credit
    costs more than truncated and not unboundedly more; the exact ratio depends
    on widths and belongs in the report, where a run's wall clock can be read
    against it.
    """

    streams = 4
    costs = {
        credit: training_step_cost(credit, streams) for credit in ("tbptt", "rtrl")
    }
    ratio = costs["rtrl"] / costs["tbptt"]
    with capsys.disabled():
        print(
            f"\n  {streams} streams, one step:"
            f"\n    tbptt {costs['tbptt']:14.0f} flops"
            f"\n    rtrl  {costs['rtrl']:14.0f} flops  ({ratio:.2f}x)"
        )

    assert 1.0 < ratio < 100.0, (
        f"exact credit costs {ratio:.2f}x truncated, which is outside the range "
        "carrying a per-parameter sensitivity can explain"
    )


# The widths a recorded hopper run used, and a narrower point to read the scaling
# against. At the widths the parity test compares on -- three hidden units -- a
# ratio would be fixed overhead rather than arithmetic.
WIDTHS = ((32, 16), (128, 32))


def _ours_at(hidden: int, features: int):
    """Our LRU under exact credit, one stream, one step, ready to differentiate."""

    core = Memoroid(
        cell=LRUCell(
            config=LRUConfig(features=features, hidden_dim=hidden, output_dim=hidden)
        )
    )
    x = jnp.zeros((1, 1, features), jnp.float32)
    done = jnp.zeros((1, 1), bool)
    carry = LRUCarry(
        state=jnp.zeros((1, 1, hidden), jnp.complex64),
        decay=jnp.ones((1, 1, hidden), jnp.complex64),
    )
    credit = {
        "nu_log": jnp.zeros((1, 1, hidden), jnp.complex64),
        "theta_log": jnp.zeros((1, 1, hidden), jnp.complex64),
        "gamma_log": jnp.zeros((1, 1, hidden), jnp.complex64),
        "B_real": jnp.zeros((1, 1, hidden, features), jnp.complex64),
        "B_imag": jnp.zeros((1, 1, hidden, features), jnp.complex64),
    }
    step = partial(core.apply, method="local_jacobian", sensitivity=credit)
    params = core.init(
        jax.random.key(0), x, done, carry, sensitivity=credit, method="local_jacobian"
    )["params"]

    def stepped(p) -> tuple[Any, Any, Any]:
        return cast(tuple[Any, Any, Any], step({"params": p}, x, done, carry))

    def loss(p):
        _, y, _ = stepped(p)
        return jnp.sum(y.real)

    _, _, advanced = stepped(params)
    return loss, params, advanced


def _theirs_at(hidden: int, features: int):
    """The published LRU, same widths, same one step."""

    layer = OnlineLRULayer(d_hidden=hidden)
    x = jnp.zeros((features,), jnp.float32)
    carry = (
        jnp.zeros((hidden,), jnp.complex64),
        (
            jnp.zeros((hidden,), jnp.complex64),
            jnp.zeros((hidden,), jnp.complex64),
            jnp.zeros((hidden, features), jnp.complex64),
        ),
    )
    params = layer.init(jax.random.key(0), carry, x)

    def stepped(p) -> tuple[Any, Any]:
        return cast(tuple[Any, Any], layer.apply(p, carry, x))

    def loss(p):
        _, y = stepped(p)
        return jnp.sum(y)

    advanced, _ = stepped(params)
    return loss, params, advanced[1]


def _carried(tree) -> int:
    """How many numbers the real-time credit has to carry between steps."""

    return sum(int(leaf.size) for leaf in jax.tree.leaves(tree))


def test_keying_the_sensitivity_by_parameter_costs_a_redundant_accumulator(capsys):
    """What our contract charges for being a contract, measured not argued.

    Both implementations carry the same real-time credit. They key it by
    influence matrix -- ``dh/dLambda``, ``dh/dgamma``, ``dh/dB`` -- and chain each
    parameter's derivative on at the gradient. We key it by parameter, so the
    Memoroid can pair a sensitivity with the parameter it credits without knowing
    which cell produced it, and every cell in the package implements that one
    contract.

    The generality is not free and this measures the bill. Of our five
    accumulators only three are independent: ``nu_log`` and ``theta_log`` both
    accumulate the previous carry against a constant, and ``B_imag`` is ``1j``
    times ``B_real``. So we carry the widest one, ``hidden * features``, twice.

    The bill turned out to be memory alone. We carry 1.94x at hidden 32 and 1.97x
    at hidden 128, and the duplicated accumulator explains all of it. The gradient
    does not follow: 1.08x at the narrow width, and 0.92x at the width a hopper run
    uses, where building each Jacobian already chained saves more than the extra
    accumulator costs. Keying by parameter is cheaper to differentiate at the
    widths that matter, which is the opposite of what the redundancy suggested.

    So the decision this measurement was taken for -- whether to key by influence
    matrix and chain at the phantom instead, dropping the duplication -- was made
    against. It would buy back half the carried state, and cost a change to the
    ``MemoroidCellBase`` contract that every cell implements, for no compute. Left
    here as the reason, so that the question is not reopened without a run that
    shows the carried state is what a configuration is short of.

    Reported rather than tightly asserted either way. The assertion holds only the
    shape of the claim: we carry more, and not unboundedly more.
    """

    lines, ratios = [], []
    for hidden, features in WIDTHS:
        ours_loss, ours_params, ours_credit = _ours_at(hidden, features)
        theirs_loss, theirs_params, theirs_credit = _theirs_at(hidden, features)

        ours_flops = flops(jax.grad(ours_loss), ours_params)
        theirs_flops = flops(jax.grad(theirs_loss), theirs_params)
        ours_state, theirs_state = _carried(ours_credit), _carried(theirs_credit)
        ratios.append(ours_state / theirs_state)

        lines.append(
            f"\n  hidden {hidden:3d}, features {features:2d}:"
            f"\n    carried   ours {ours_state:8d}   theirs {theirs_state:8d}"
            f"   ({ours_state / theirs_state:.2f}x)"
            f"\n    gradient  ours {ours_flops:8.0f}   theirs {theirs_flops:8.0f}"
            f"   ({ours_flops / theirs_flops:.2f}x)"
        )

    with capsys.disabled():
        print("".join(lines))

    assert all(1.0 < ratio < 3.0 for ratio in ratios), (
        "the sensitivity we carry is not between one and three times the one "
        f"they carry, which is outside what a duplicated accumulator explains:"
        f"{''.join(lines)}"
    )
