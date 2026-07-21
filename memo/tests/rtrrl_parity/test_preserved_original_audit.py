"""Contracts for the preserved-source audit and controlled actor probe."""

from __future__ import annotations

import os
from pathlib import Path

from .preserved_original_compare import compare_probes
from .preserved_original_source_audit import audit_sources


REPOSITORY_ROOT = Path(__file__).parents[3]
ORACLE_ROOT = Path(
    os.environ.get("RTRRL_AAAI25_ROOT", "/home/ubuntu/trainer/RTRRL-AAAI25")
)


def test_source_audit_records_precise_ast_normalization_and_actor_structure():
    audit = audit_sources(REPOSITORY_ROOT / "rtrrl", ORACLE_ROOT)

    assert audit["ast_normalization"]["traces.py"] == {
        "equal_after_docstring_removal": True,
        "equal_after_all_standalone_string_removal": True,
    }
    assert audit["ast_normalization"]["models/online_lru.py"] == {
        "equal_after_docstring_removal": False,
        "equal_after_all_standalone_string_removal": True,
    }
    assert audit["actor_log_prob_structure"] == {
        "preserved_argument": "stop_gradient(action)",
        "oracle_argument": "action",
        "preserved_detaches_action": True,
        "oracle_detaches_action": False,
    }


def _probe(actor_results):
    scalar = [[0, "float32", [], 0.0]]
    return {
        "runtime": {"jax": "test"},
        "explicit_lru_params": scalar,
        "lru_carry_0": scalar,
        "explicit_lru_carry_1": scalar,
        "explicit_lru_carry_2": scalar,
        "explicit_lru_output_1": 0.0,
        "explicit_lru_output_2": 0.0,
        "trace_1": scalar,
        "trace_update": scalar,
        "prng_split": [0],
        "native_lru_params": scalar,
        "native_lru_carry_1": scalar,
        "native_lru_carry_2": scalar,
        "native_lru_output_1": 0.0,
        "native_lru_output_2": 0.0,
        "actor_results": actor_results,
    }


def test_two_by_two_actor_control_separates_semantics_from_runtime():
    actor_results = {
        "detached": {
            "objective": -0.5,
            "grad_loc": [1.0, -2.0],
            "grad_raw_scale": [0.5, 0.25],
        },
        "reparameterized": {
            "objective": -0.5,
            "grad_loc": [0.0, 0.0],
            "grad_raw_scale": [-0.5, -0.5],
        },
    }

    comparison = compare_probes(_probe(actor_results), _probe(actor_results))

    assert comparison["within_runtime_semantic_max_abs"] == {
        "preserved_runtime": {
            "grad_loc": 2.0,
            "grad_raw_scale": 1.0,
        },
        "oracle_runtime": {
            "grad_loc": 2.0,
            "grad_raw_scale": 1.0,
        },
    }
    assert comparison["cross_runtime_same_semantic_max_abs"] == {
        "detached": {
            "objective": 0.0,
            "grad_loc": 0.0,
            "grad_raw_scale": 0.0,
        },
        "reparameterized": {
            "objective": 0.0,
            "grad_loc": 0.0,
            "grad_raw_scale": 0.0,
        },
    }
