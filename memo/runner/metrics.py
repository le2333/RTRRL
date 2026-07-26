"""Reduce a stacked step-metrics struct to the scalars a sink can take.

The rule is about shape, not about names. A field is a metric when it holds
one number per step, optionally per environment; anything wider is a
trajectory and anything that is not an array is structure. That way a new
algorithm reports whatever it declares without this module learning about it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np


def _numeric(value) -> np.ndarray | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        return None
    return array


def scalar_metrics(metrics: Any, *, steps: int, prefix: str = "") -> dict[str, float]:
    """Average every per-step scalar in ``metrics`` over the epoch.

    ``steps`` is the leading axis the caller's scan produced, and is what
    tells a per-step scalar apart from a single number an algorithm reported
    once.
    """

    collected: dict[str, float] = {}
    for name, value in _walk(metrics):
        array = _numeric(value)
        if array is None or array.ndim == 0 or array.shape[0] != steps:
            continue
        if array.ndim > 2:
            continue
        mean = float(np.nanmean(array))
        if np.isfinite(mean):
            collected[f"{prefix}{name}"] = mean
    return collected


def _walk(value: Any, path: str = ""):
    if is_dataclass(value) and not isinstance(value, type):
        for entry in fields(value):
            child = getattr(value, entry.name)
            if child is None:
                continue
            yield from _walk(child, f"{path}{entry.name}/")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if child is None:
                continue
            yield from _walk(child, f"{path}{key}/")
        return
    yield path.rstrip("/"), value
