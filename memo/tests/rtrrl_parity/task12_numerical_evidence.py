"""Emit reviewable eager/JIT/oracle measurements for Task 12."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
from typing import Any

import jax
import numpy as np

from memorax.algorithms.rtrrl.program import _register_environment_state_pytree
from memorax.algorithms.rtrrl.state_machine import make_step_fn

from .assertions import _maximum_ulp_distance, flatten_with_paths
from .test_step_parity import (
    _canonical_state_tree,
    _fixture_tree,
    _initialized,
)


def _measurement(path: str, array: np.ndarray, value: float | int) -> dict[str, Any]:
    return {
        "path": path,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "value": value,
    }


def compare_flat_trees(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Measure complete canonical trees without changing acceptance assertions."""

    if actual.keys() != expected.keys():
        raise AssertionError(
            {
                "missing": sorted(expected.keys() - actual.keys()),
                "unexpected": sorted(actual.keys() - expected.keys()),
            }
        )
    result: dict[str, Any] = {
        "leaf_count": len(actual),
        "exact_leaf_count": 0,
        "float_leaf_count": 0,
        "exact_float_leaf_count": 0,
        "max_abs": None,
        "max_rel": None,
        "max_ulp": None,
    }
    maxima: dict[str, tuple[float | int, str, np.ndarray] | None] = {
        "max_abs": None,
        "max_rel": None,
        "max_ulp": None,
    }
    for path in sorted(actual):
        actual_array = np.asarray(actual[path])
        expected_array = np.asarray(expected[path])
        if actual_array.shape != expected_array.shape:
            raise AssertionError(
                f"shape mismatch at {path}: "
                f"{actual_array.shape} != {expected_array.shape}"
            )
        if actual_array.dtype != expected_array.dtype:
            raise AssertionError(
                f"dtype mismatch at {path}: "
                f"{actual_array.dtype} != {expected_array.dtype}"
            )
        if np.array_equal(actual_array, expected_array):
            result["exact_leaf_count"] += 1
        if actual_array.dtype.kind not in "fc":
            continue
        if not (
            np.isfinite(actual_array).all()
            and np.isfinite(expected_array).all()
        ):
            raise AssertionError(f"non-finite floating leaf at {path}")
        result["float_leaf_count"] += 1
        if np.array_equal(actual_array, expected_array):
            result["exact_float_leaf_count"] += 1
        difference = np.abs(actual_array - expected_array)
        max_abs = float(difference.max(initial=0))
        denominator = np.abs(expected_array)
        relative = np.divide(
            difference,
            denominator,
            out=np.where(difference == 0, 0.0, np.inf).astype(np.float64),
            where=denominator != 0,
        )
        max_rel = float(relative.max(initial=0))
        max_ulp = _maximum_ulp_distance(actual_array, expected_array)
        for name, value in (
            ("max_abs", max_abs),
            ("max_rel", max_rel),
            ("max_ulp", max_ulp),
        ):
            previous = maxima[name]
            if previous is None or value > previous[0]:
                maxima[name] = (value, path, actual_array)
    for name, maximum in maxima.items():
        if maximum is not None:
            value, path, array = maximum
            result[name] = _measurement(path, array, value)
    return result


def _canonical_flat(state: Any) -> dict[str, Any]:
    return flatten_with_paths(_canonical_state_tree(state))


def collect_measurements() -> dict[str, Any]:
    arrays, components, config, environment, initial_state, key = _initialized()
    _register_environment_state_pytree(initial_state.environment_state)
    step = make_step_fn(components, config, environment, debug=False)

    eager_one_state, eager_one_key, _ = step(initial_state, key)
    jit_one_state, jit_one_key, _ = jax.jit(step)(initial_state, key)
    jax.block_until_ready((jit_one_state, jit_one_key))

    eager_three_state, eager_three_key = initial_state, key
    for _ in range(3):
        eager_three_state, eager_three_key, _ = step(
            eager_three_state, eager_three_key
        )

    def scan_three(state: Any, scan_key: Any) -> tuple[Any, Any]:
        def body(carry: tuple[Any, Any], _: None) -> tuple[tuple[Any, Any], None]:
            body_state, body_key = carry
            body_state, body_key, _metrics = step(body_state, body_key)
            del _metrics
            return (body_state, body_key), None

        return jax.lax.scan(
            body, (state, scan_key), None, length=3
        )[0]

    jit_three_state, jit_three_key = jax.jit(scan_three)(initial_state, key)
    jax.block_until_ready((jit_three_state, jit_three_key))

    one_oracle = _fixture_tree(arrays, "state_machine/step_1/state")
    three_oracle = _fixture_tree(arrays, "state_machine/step_3/state")
    eager_one = _canonical_flat(eager_one_state)
    jit_one = _canonical_flat(jit_one_state)
    eager_three = _canonical_flat(eager_three_state)
    jit_three = _canonical_flat(jit_three_state)
    return {
        "schema_version": 1,
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": metadata.version("jaxlib"),
            "flax": metadata.version("flax"),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "comparison_semantics": {
            "leaf_count": "all canonical leaves after path/shape/dtype equality",
            "exact_leaf_count": "all leaves satisfying numpy.array_equal",
            "float_leaf_count": "float and complex canonical leaves",
            "exact_float_leaf_count": (
                "float and complex leaves satisfying numpy.array_equal"
            ),
            "acceptance_assertions_modified": False,
        },
        "comparisons": {
            "one_step_eager_vs_oracle": compare_flat_trees(
                eager_one, one_oracle
            ),
            "one_step_jit_vs_oracle": compare_flat_trees(jit_one, one_oracle),
            "one_step_jit_vs_eager": compare_flat_trees(jit_one, eager_one),
            "three_step_eager_vs_oracle": compare_flat_trees(
                eager_three, three_oracle
            ),
            "three_step_jit_vs_oracle": compare_flat_trees(
                jit_three, three_oracle
            ),
            "three_step_jit_vs_eager": compare_flat_trees(
                jit_three, eager_three
            ),
        },
        "keys": {
            "one_step_jit_vs_eager_exact": bool(
                np.array_equal(jit_one_key, eager_one_key)
            ),
            "three_step_jit_vs_eager_exact": bool(
                np.array_equal(jit_three_key, eager_three_key)
            ),
        },
    }


def main() -> None:
    print(json.dumps(collect_measurements(), allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
