"""Strict AAAI25 policy and value heads for the modular RTRRL path."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import distrax
import flax.linen as nn
from flax.core import freeze
import jax
import jax.numpy as jnp

from memorax.utils.typing import Array


def _sigmoid_between(
    value: Array, lower: float | Array, upper: float | Array
) -> Array:
    return (upper - lower) * jax.nn.sigmoid(value) + lower


class FADense(nn.Dense):
    """AAAI25 dense layer with optional feedback-alignment input gradients."""

    f_align: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.glorot_normal(
        in_axis=-1, out_axis=-2
    )

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        if self.f_align:
            feedback = self.variable(
                "falign",
                "B",
                self.kernel_init,
                self.make_rng() if self.has_rng("params") else None,
                (jnp.shape(inputs)[-1], self.features),
                self.param_dtype,
            ).value
        else:
            feedback = self.param(
                "kernel",
                self.kernel_init,
                (jnp.shape(inputs)[-1], self.features),
                self.param_dtype,
            )

        def forward(module: FADense, value: Array, matrix: Array) -> Array:
            del matrix
            return nn.Dense.__call__(module, value)

        def forward_vjp(
            module: FADense, value: Array, matrix: Array
        ) -> tuple[Array, tuple[Array, Array]]:
            return nn.Dense.__call__(module, value), (value, matrix)

        def backward_vjp(
            residual: tuple[Array, Array], output_cotangent: Array
        ) -> tuple[dict[str, Any], Array, Array]:
            value, matrix = residual
            parameter_cotangent = {
                "params": {
                    "kernel": jnp.einsum(
                        "...X,...Y->YX", output_cotangent, value
                    )
                }
            }
            if self.use_bias:
                parameter_cotangent["params"]["bias"] = jnp.einsum(
                    "...X->X", output_cotangent
                )
            input_cotangent = jnp.einsum(
                "YX,...X->...Y", matrix, output_cotangent
            )
            return (
                {"params": freeze(parameter_cotangent["params"])},
                input_cotangent,
                jnp.zeros_like(matrix),
            )

        feedback_aligned = nn.custom_vjp(
            forward,
            forward_fn=forward_vjp,
            backward_fn=backward_vjp,
        )
        return feedback_aligned(self, inputs, feedback)


class RTRRLTDHead(nn.Module):
    """Strict linear actor and critic pair used by the AAAI25 RTRRL agent."""

    action_dim: int
    discrete: bool
    f_align: bool = False

    def setup(self) -> None:
        actor_features = self.action_dim if self.discrete else 2 * self.action_dim
        self.actor = FADense(
            actor_features,
            f_align=self.f_align,
            use_bias=False,
            name="actor",
        )
        self.critic = FADense(1, f_align=self.f_align, name="critic")

    def __call__(self, inputs: Array) -> tuple[Array, Array]:
        return self.actor(inputs), self.critic(inputs)


def make_action_distribution(
    actor_output: Array,
    *,
    discrete: bool,
    act_log_bounds: Sequence[float] = (-2.0, 2.0),
    act_bounds: Sequence[float] | None = None,
) -> distrax.Distribution:
    """Build the AAAI25 categorical or per-dimension normal distribution."""
    if discrete:
        return distrax.Categorical(logits=actor_output)

    if actor_output.shape[-1] % 2:
        raise ValueError(
            "continuous actor output must contain loc and log-scale halves"
        )
    if len(act_log_bounds) != 2:
        raise ValueError("act_log_bounds must contain exactly two values")
    if act_bounds is not None and len(act_bounds) != 2:
        raise ValueError("act_bounds must contain exactly two values")

    loc, log_scale = jnp.split(actor_output, 2, axis=-1)
    log_scale = _sigmoid_between(
        log_scale, act_log_bounds[0], act_log_bounds[1]
    )
    if act_bounds is not None:
        loc = _sigmoid_between(loc, act_bounds[0], act_bounds[1])
    return distrax.Normal(loc=loc, scale=jax.nn.softplus(log_scale))


__all__: list[str] = ["FADense", "RTRRLTDHead", "make_action_distribution"]
