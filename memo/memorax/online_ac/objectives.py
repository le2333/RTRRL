"""Pure objective routing for the two online actor-critic algorithms."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ObjectiveDirections:
    """Per-domain traced and direct ascent objectives plus diagnostics."""

    traced_by_domain: Any
    direct_by_domain: Any
    metrics: Any


def make_rtrrl_objective(config):
    """Build RTRRL objective routing without applying TD or recurrent scaling."""

    def rtrrl_objective(
        *,
        log_prob,
        value,
        entropy,
        prediction=None,
        prediction_target=None,
    ):
        actor = config.eta_pi * config.logprob_scale * log_prob
        entropy_direction = config.entropy_rate * config.logprob_scale * entropy
        zero = jnp.zeros_like(value)
        prediction_direction = zero
        if prediction is not None:
            error = prediction - jax.lax.stop_gradient(prediction_target)
            prediction_direction = (
                -config.pred_coeff * 0.5 * jnp.sum(jnp.square(error), axis=-1)
            )

        return ObjectiveDirections(
            traced_by_domain={
                "actor": actor,
                "critic": value,
                "recurrent": actor + value,
                "prediction": zero,
            },
            direct_by_domain={
                "actor": entropy_direction,
                "critic": zero,
                "recurrent": entropy_direction + prediction_direction,
                "prediction": prediction_direction,
            },
            metrics={
                "entropy": entropy,
                "prediction_direction": prediction_direction,
            },
        )

    return rtrrl_objective


def make_stream_ac_objective(config):
    """Build StreamAC's traced actor and critic objectives."""

    def stream_ac_objective(*, log_prob, value, entropy, delta):
        actor = (
            log_prob
            + config.entropy_coefficient
            * jnp.sign(jax.lax.stop_gradient(delta))
            * entropy
        )
        return ObjectiveDirections(
            traced_by_domain={"actor": actor, "critic": value},
            direct_by_domain={
                "actor": jnp.zeros_like(actor),
                "critic": jnp.zeros_like(value),
            },
            metrics={"entropy": entropy},
        )

    return stream_ac_objective
