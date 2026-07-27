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
