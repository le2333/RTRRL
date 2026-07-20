"""Compare isolated preserved/oracle probes and emit reviewable measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _record_arrays(records: list[list[Any]]) -> list[np.ndarray]:
    arrays = []
    for _, dtype, shape, values in records:
        array = np.asarray(values)
        if dtype.startswith("complex"):
            array = array[..., 0] + 1j * array[..., 1]
        arrays.append(array.astype(dtype).reshape(shape))
    return arrays


def _maximum(left: Any, right: Any) -> float:
    def as_arrays(value: Any) -> list[np.ndarray]:
        is_records = (
            isinstance(value, list)
            and bool(value)
            and isinstance(value[0], list)
            and len(value[0]) == 4
            and isinstance(value[0][0], int)
            and isinstance(value[0][1], str)
        )
        return _record_arrays(value) if is_records else [np.asarray(value)]

    left_arrays = as_arrays(left)
    right_arrays = as_arrays(right)
    if len(left_arrays) != len(right_arrays):
        raise AssertionError("probe leaf counts differ")
    maxima = []
    for left_array, right_array in zip(left_arrays, right_arrays, strict=True):
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            raise AssertionError(
                f"probe metadata differs: {left_array.shape}/{left_array.dtype} "
                f"!= {right_array.shape}/{right_array.dtype}"
            )
        maxima.append(float(np.max(np.abs(left_array - right_array), initial=0)))
    return max(maxima, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preserved", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    arguments = parser.parse_args()
    preserved = json.loads(arguments.preserved.read_text())
    oracle = json.loads(arguments.oracle.read_text())

    comparable = (
        "explicit_lru_params",
        "lru_carry_0",
        "explicit_lru_carry_1",
        "explicit_lru_carry_2",
        "explicit_lru_output_1",
        "explicit_lru_output_2",
        "trace_1",
        "trace_update",
        "actor_objective",
    )
    maxima = {name: _maximum(preserved[name], oracle[name]) for name in comparable}
    runtime_sensitive_maxima = {
        name: _maximum(preserved[name], oracle[name])
        for name in (
            "prng_split",
            "native_lru_params",
            "native_lru_carry_1",
            "native_lru_carry_2",
            "native_lru_output_1",
            "native_lru_output_2",
        )
    }
    actor_grad_maxima = {
        name: _maximum(preserved[name], oracle[name])
        for name in ("actor_grad_loc", "actor_grad_raw_scale")
    }
    if any(value > 2e-6 for value in maxima.values()):
        raise AssertionError(f"unexpected forward/trace mismatch: {maxima}")
    if not any(value > 1e-4 for value in actor_grad_maxima.values()):
        raise AssertionError(
            f"expected preserved actor-gradient mismatch: {actor_grad_maxima}"
        )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "preserved_runtime": preserved["runtime"],
                "oracle_runtime": oracle["runtime"],
                "explicit_forward_trace_max_abs": maxima,
                "runtime_sensitive_max_abs": runtime_sensitive_maxima,
                "actor_gradient_max_abs": actor_grad_maxima,
                "verdict": (
                    "explicit forward/trace matches; native PRNG/init and "
                    "actor gradient differ"
                ),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
