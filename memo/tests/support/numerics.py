"""Numerical comparison helpers shared by local and external parity tests."""

from __future__ import annotations

import jax
import numpy as np


def flattened(tree) -> dict:
    """Return a pytree as leaf paths, spelled the way snapshots spell them."""

    pairs, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {
        "/".join(str(getattr(key, "key", getattr(key, "idx", key))) for key in path): (
            np.asarray(leaf)
        )
        for path, leaf in pairs
    }


def _widened(array) -> np.ndarray:
    array = np.asarray(array)
    return array.astype(np.complex128 if np.iscomplexobj(array) else np.float64)


def last_bits(wanted, got) -> float:
    """Measure the maximum gap in float32 last bits at the arrays' scale."""

    wanted, got = _widened(wanted), _widened(got)
    scale = max(float(np.abs(wanted).max()), float(np.abs(got).max()), 1e-6)
    gap = float(np.max(np.abs(got - wanted)))
    return gap / float(np.spacing(np.float32(scale)))


def deviations(actual: dict, expected: dict, allowed: float = 0.0) -> list:
    """Return leaves farther apart than allowed, ordered worst first."""

    missing = sorted(set(expected) - set(actual))
    assert not missing, f"nothing reports {missing}"

    found = []
    for path, wanted in expected.items():
        wanted = np.asarray(wanted)
        got = np.asarray(actual[path])
        assert got.shape == wanted.shape, f"{path}: {got.shape} not {wanted.shape}"
        if wanted.dtype.kind in "biufc" and got.dtype.kind in "biufc":
            bits = last_bits(wanted, got)
            if bits > allowed:
                found.append((bits, path))
        elif not np.array_equal(got, wanted):
            found.append((float("inf"), path))
    return sorted(found, reverse=True, key=lambda entry: entry[0])


def assert_within(
    actual: dict, expected: dict, what: str, *, allowed: float = 0.0
) -> int:
    """Assert every expected leaf is within the allowed float32 last bits."""

    assert expected, f"{what}: there is nothing to compare"
    found = deviations(actual, expected, allowed)
    if found:
        listed = "\n".join(f"  {bits:.1f} last bits  {path}" for bits, path in found)
        raise AssertionError(
            f"{what}: {len(found)} of {len(expected)} leaves are more than "
            f"{allowed:g} last bits apart, worst first:\n{listed}"
        )
    return len(expected)
