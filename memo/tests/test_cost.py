"""What a step costs, counted rather than timed.

Timing on a shared runner is noise; the compiler will tell us the arithmetic it
plans to do, and that number is the same on every machine. Counted here because
a recorded hopper run spends about thirty milliseconds per update on work that
is a few thousand parameters wide, which is too much by an order of magnitude
and had to be either explained or fixed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from test_blocks import ours, upstream

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

    state = upstream(num_envs=streams).init(jax.random.key(0))
    mine = ours(num_envs=streams, credit="tbptt")
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
