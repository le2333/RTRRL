"""Sparse initialisation, rounding the survivors up rather than the zeros.

streaming-drl and memorax both take ``ceil(sparsity * fan_in)`` zeros, which at
a narrow fan-in is every one of them: six inputs at 0.9 leaves an output unit
with nothing to read. Taking ``ceil((1 - sparsity) * fan_in)`` survivors instead
is the same share wherever there is room for it, and never an empty unit. The
two differ by at most one weight per unit.
"""

import math
from typing import Callable

import jax
import jax.numpy as jnp

from memorax.utils.typing import Array, Key


def sparse(sparsity: float = 0.9) -> Callable:

    def init(key: Key, shape: tuple, dtype=jnp.float32) -> Array:
        fan_in = math.prod(shape[:-1])
        fan_out = shape[-1]
        limit = math.sqrt(1.0 / fan_in)

        key, weight_key = jax.random.split(key)
        weights = jax.random.uniform(
            weight_key, shape, dtype, minval=-limit, maxval=limit
        )

        n_zero = fan_in - math.ceil((1.0 - sparsity) * fan_in)
        weights_flat = weights.reshape(fan_in, fan_out)

        perms = jax.vmap(lambda k: jax.random.permutation(k, fan_in))(
            jax.random.split(key, fan_out)
        )
        mask = (perms >= n_zero).astype(dtype).T  # (fan_in, fan_out)

        return (weights_flat * mask).reshape(shape)

    return init
