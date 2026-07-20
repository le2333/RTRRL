import numpy as np
import pytest

from .assertions import assert_tree_close, flatten_with_paths
from .oracle_capture import load_oracle


def test_oracle_manifest_pins_source_and_runtime():
    arrays, manifest = load_oracle()
    assert manifest["source"] == "RTRRL-AAAI25"
    assert manifest["commit"] == "4301943c349171d828d0fcf3e40944c286451415"
    assert manifest["algorithm"] == "lru"
    assert manifest["seed"] == 7
    assert manifest["dtype_policy"] == "float32-complex64"
    assert sorted(arrays) == manifest["leaf_paths"]


def test_oracle_fixture_has_required_sections():
    arrays, _ = load_oracle()
    required = {
        "heads/input",
        "heads/actor_loc",
        "heads/actor_scale",
        "heads/value",
        "lru/input",
        "lru/carry_before",
        "lru/carry_after",
        "lru/output",
        "credit/after_step_1",
        "credit/after_step_2",
        "init/action",
        "init/value",
        "step/td_error",
    }
    assert required <= arrays.keys()


def test_flatten_with_paths_uses_keys_and_sequence_indices():
    leaves = flatten_with_paths({"state": (np.array([1]), [np.array([2])])})

    assert sorted(leaves) == ["state/0", "state/1/0"]


def test_assert_tree_close_checks_shape_and_dtype_before_values():
    with pytest.raises(AssertionError, match="shape mismatch at value"):
        assert_tree_close(
            {"value": np.ones((1,), dtype=np.float32)},
            {"value": np.ones((2,), dtype=np.float32)},
            "exact",
        )
    with pytest.raises(AssertionError, match="dtype mismatch at value"):
        assert_tree_close(
            {"value": np.ones((1,), dtype=np.float32)},
            {"value": np.ones((1,), dtype=np.float64)},
            "exact",
        )


def test_assert_tree_close_supports_exact_tolerance_and_leaf_ulps():
    baseline = np.array([1.0, -0.0], dtype=np.float32)
    one_ulp = np.nextafter(baseline, np.float32(np.inf))

    with pytest.raises(AssertionError):
        assert_tree_close({"value": one_ulp}, {"value": baseline}, "exact")
    assert_tree_close({"value": one_ulp}, {"value": baseline}, (1e-6, 1e-44))
    assert_tree_close({"value": one_ulp}, {"value": baseline}, {"value": 1})
    assert_tree_close(
        {"value": np.array([0.0], dtype=np.float32)},
        {"value": np.array([-0.0], dtype=np.float32)},
        {"value": 0},
    )


def test_assert_tree_close_rejects_non_finite_and_compares_integers_exactly():
    with pytest.raises(AssertionError, match="non-finite"):
        assert_tree_close(
            {"value": np.array([np.inf], dtype=np.float32)},
            {"value": np.array([np.inf], dtype=np.float32)},
            "exact",
        )
    with pytest.raises(AssertionError):
        assert_tree_close(
            {"value": np.array([2], dtype=np.int32)},
            {"value": np.array([1], dtype=np.int32)},
            (1.0, 1.0),
        )
