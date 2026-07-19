"""Pure eligibility-trace recurrences with explicit boundary timing."""

from __future__ import annotations

from typing import Any

import jax
from flax import struct


@struct.dataclass
class TraceDirections:
    """The trace carried to the next step and the trace used now."""

    carried: Any
    update: Any


def _broadcast_env(values, leaf):
    return values[(slice(None),) + (None,) * (leaf.ndim - 1)]


def make_rtrrl_trace(config):
    """Build RTRRL's three-domain, post-transition trace recurrence."""

    decays = {
        "actor": config.gamma * config.lambda_pi,
        "critic": config.gamma * config.lambda_v,
        "recurrent": config.gamma * config.lambda_rnn,
    }
    use_fresh = config.update_trace_before_td

    def rtrrl_trace(incoming, gradient, *, terminated_after, emphasis):
        carried = {
            domain: jax.tree.map(
                lambda old, grad: (
                    decays[domain] * (1 - _broadcast_env(terminated_after, old)) * old
                    + _broadcast_env(emphasis, grad) * grad
                ),
                incoming[domain],
                gradient[domain],
            )
            for domain in decays
        }
        return TraceDirections(
            carried=carried,
            update=carried if use_fresh else incoming,
        )

    return rtrrl_trace


def make_stream_ac_trace(config):
    """Build StreamAC's pre-forward reset, always-fresh trace recurrence."""

    decay = config.gamma * config.trace_lambda

    def stream_ac_trace(incoming, gradient, *, reset_before):
        carried = jax.tree.map(
            lambda old, grad: (
                decay * (1 - _broadcast_env(reset_before, old)) * old + grad
            ),
            incoming,
            gradient,
        )
        return TraceDirections(carried=carried, update=carried)

    return stream_ac_trace
