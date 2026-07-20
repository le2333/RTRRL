"""Stable pytree comparisons shared by numerical-parity tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np

ComparisonPolicy = str | tuple[float, float] | Mapping[str, int]


def _path_component(key: Any) -> str:
    for attribute in ("key", "idx", "name"):
        if hasattr(key, attribute):
            return str(getattr(key, attribute))
    return str(key)


def flatten_with_paths(tree: Any) -> dict[str, Any]:
    """Flatten a pytree using stable dictionary keys and sequence indices."""
    pairs, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {
        "/".join(_path_component(component) for component in path): leaf
        for path, leaf in pairs
    }


def _ordered_float_bits(values: np.ndarray) -> np.ndarray:
    values = np.where(values == 0, np.zeros((), dtype=values.dtype), values)
    unsigned_dtype = np.dtype(f"u{values.dtype.itemsize}")
    bits = np.ascontiguousarray(values).view(unsigned_dtype)
    sign = np.array(1 << (values.dtype.itemsize * 8 - 1), dtype=unsigned_dtype)
    magnitude = bits & (sign - np.array(1, dtype=unsigned_dtype))
    return np.where(bits & sign, sign - magnitude, sign + magnitude)


def _maximum_ulp_distance(actual: np.ndarray, expected: np.ndarray) -> int:
    if actual.dtype.kind == "c":
        return max(
            _maximum_ulp_distance(np.real(actual), np.real(expected)),
            _maximum_ulp_distance(np.imag(actual), np.imag(expected)),
        )
    if actual.dtype.kind != "f":
        raise TypeError(f"ULP comparison requires float or complex leaves, got {actual.dtype}")
    actual_bits = _ordered_float_bits(actual)
    expected_bits = _ordered_float_bits(expected)
    distance = np.maximum(actual_bits, expected_bits) - np.minimum(
        actual_bits, expected_bits
    )
    return int(distance.max(initial=0))


def assert_tree_close(
    actual: Any,
    expected: Any,
    policy: ComparisonPolicy,
) -> None:
    """Compare pytrees under exact, tolerance, or named per-leaf ULP rules."""
    actual_leaves = flatten_with_paths(actual)
    expected_leaves = flatten_with_paths(expected)
    assert actual_leaves.keys() == expected_leaves.keys(), (
        f"path mismatch: actual={sorted(actual_leaves)}, "
        f"expected={sorted(expected_leaves)}"
    )

    for path in actual_leaves:
        actual_array = np.asarray(actual_leaves[path])
        expected_array = np.asarray(expected_leaves[path])
        assert actual_array.shape == expected_array.shape, (
            f"shape mismatch at {path}: "
            f"{actual_array.shape} != {expected_array.shape}"
        )
        assert actual_array.dtype == expected_array.dtype, (
            f"dtype mismatch at {path}: "
            f"{actual_array.dtype} != {expected_array.dtype}"
        )

        if actual_array.dtype.kind in "fc":
            assert np.isfinite(actual_array).all(), f"non-finite actual leaf at {path}"
            assert np.isfinite(expected_array).all(), (
                f"non-finite expected leaf at {path}"
            )

        if actual_array.dtype.kind in "biu" or policy == "exact":
            np.testing.assert_array_equal(actual_array, expected_array, err_msg=path)
        elif isinstance(policy, tuple):
            rtol, atol = policy
            np.testing.assert_allclose(
                actual_array,
                expected_array,
                rtol=rtol,
                atol=atol,
                err_msg=path,
            )
        elif isinstance(policy, Mapping):
            allowed_ulps = policy.get(path, 0)
            if not isinstance(allowed_ulps, int) or allowed_ulps < 0:
                raise ValueError(f"invalid ULP policy for {path}: {allowed_ulps!r}")
            distance = _maximum_ulp_distance(actual_array, expected_array)
            assert distance <= allowed_ulps, (
                f"ULP mismatch at {path}: {distance} > {allowed_ulps}"
            )
        else:
            raise ValueError(f"unknown comparison policy: {policy!r}")
