"""Deterministic cross-environment probe for the preserved RTRRL LRU path."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np


def _arrays(tree: Any, jax: Any) -> list[list[Any]]:
    records = []
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        array = np.asarray(leaf)
        records.append(
            [
                index,
                str(array.dtype),
                list(array.shape),
                np.stack((array.real, array.imag), axis=-1).tolist()
                if np.iscomplexobj(array)
                else array.tolist(),
            ]
        )
    return records


def capture(source_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(source_root))
    import distrax
    import jax
    import jax.numpy as jnp

    online_lru = importlib.import_module("models.online_lru")
    traces = importlib.import_module("traces")

    root_key = jax.random.PRNGKey(7)
    lru_key, _, split_key = jax.random.split(root_key, 3)
    inputs_1 = jnp.asarray([[0.25, -0.5, 0.75, -1.0]], jnp.float32)
    inputs_2 = jnp.asarray([[-0.125, 0.375, 0.5, -0.25]], jnp.float32)
    layer = online_lru.OnlineLRULayer(
        d_output=2,
        d_hidden=2,
        plasticity="rtrl",
        activation="silu",
    )
    carry_0 = layer.initialize_carry(lru_key, inputs_1.shape)
    (carry_1, output_1), variables = layer.init_with_output(
        lru_key,
        carry_0,
        inputs_1,
    )
    native_carry_2, native_output_2 = layer.apply(
        variables,
        carry_1,
        inputs_2,
    )
    counter = iter(range(1, 100))

    def explicit_leaf(leaf):
        start = next(counter) / 100
        return jnp.linspace(
            start,
            start + 0.01 * max(leaf.size - 1, 0),
            leaf.size,
            dtype=leaf.dtype,
        ).reshape(leaf.shape)

    explicit_variables = {
        "params": jax.tree.map(explicit_leaf, variables["params"])
    }
    explicit_carry_1, explicit_output_1 = layer.apply(
        explicit_variables,
        carry_0,
        inputs_1,
    )
    explicit_carry_2, explicit_output_2 = layer.apply(
        explicit_variables,
        explicit_carry_1,
        inputs_2,
    )

    trace_0 = {"weight": jnp.asarray([[0.5, -0.25]], jnp.float32)}
    immediate = {"weight": jnp.asarray([[0.75, 0.125]], jnp.float32)}
    trace_1 = traces.trace_update(
        immediate,
        trace_0,
        gamma_lambda=0.81,
        _I=jnp.asarray(0.9, jnp.float32),
    )
    update = traces.compute_updates(
        trace_1,
        d=jnp.asarray([0.625], jnp.float32),
    )

    loc_0 = jnp.asarray([0.2, -0.4], jnp.float32)
    raw_scale_0 = jnp.asarray([0.3, -0.2], jnp.float32)

    fixed_noise = jnp.asarray([0.625, -1.25], jnp.float32)

    actor_results = {}
    for semantics in ("detached", "reparameterized"):
        def actor_objective(loc, raw_scale):
            distribution = distrax.Normal(loc, jax.nn.softplus(raw_scale))
            action = loc + distribution.scale * fixed_noise
            if semantics == "detached":
                action = jax.lax.stop_gradient(action)
            return distribution.log_prob(action).mean()

        actor_value, actor_grads = jax.value_and_grad(
            actor_objective,
            argnums=(0, 1),
        )(loc_0, raw_scale_0)
        actor_results[semantics] = {
            "objective": float(actor_value),
            "grad_loc": np.asarray(actor_grads[0]).tolist(),
            "grad_raw_scale": np.asarray(actor_grads[1]).tolist(),
        }
    return {
        "schema_version": 1,
        "source_root": str(source_root),
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": metadata.version("jaxlib"),
            "flax": metadata.version("flax"),
            "backend": jax.default_backend(),
        },
        "prng_split": np.asarray(jax.random.split(split_key, 4)).tolist(),
        "native_lru_params": _arrays(variables["params"], jax),
        "lru_carry_0": _arrays(carry_0, jax),
        "native_lru_carry_1": _arrays(carry_1, jax),
        "native_lru_carry_2": _arrays(native_carry_2, jax),
        "native_lru_output_1": np.asarray(output_1).tolist(),
        "native_lru_output_2": np.asarray(native_output_2).tolist(),
        "explicit_lru_params": _arrays(explicit_variables["params"], jax),
        "explicit_lru_carry_1": _arrays(explicit_carry_1, jax),
        "explicit_lru_carry_2": _arrays(explicit_carry_2, jax),
        "explicit_lru_output_1": np.asarray(explicit_output_1).tolist(),
        "explicit_lru_output_2": np.asarray(explicit_output_2).tolist(),
        "trace_1": _arrays(trace_1, jax),
        "trace_update": _arrays(update, jax),
        "actor_results": actor_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            capture(arguments.source_root),
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
