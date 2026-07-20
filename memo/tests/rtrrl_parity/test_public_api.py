import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from .assertions import assert_tree_close, flatten_with_paths
from . import oracle_capture
from .oracle_capture import SOURCE_COMMIT, load_oracle


def test_rtrrl_public_exports_remain_stable():
    from memorax.algorithms import RTRRL, RTRRLConfig, RTRRLState
    from memorax.algorithms.rtrrl import _find_leaf, _tree_norm

    online_ac_conftest = Path(__file__).parents[1] / "online_ac" / "conftest.py"
    build_rtrrl_agent = runpy.run_path(str(online_ac_conftest))["build_rtrrl_agent"]
    agent = build_rtrrl_agent(fresh_trace=True)

    assert RTRRL.__name__ == "RTRRL"
    assert RTRRLConfig.__name__ == "RTRRLConfig"
    assert RTRRLState.__name__ == "RTRRLState"
    assert callable(_find_leaf)
    assert callable(_tree_norm)
    assert isinstance(agent, RTRRL)


def test_oracle_manifest_pins_source_and_runtime():
    arrays, manifest = load_oracle()
    assert manifest["source"] == "RTRRL-AAAI25"
    assert manifest["commit"] == "4301943c349171d828d0fcf3e40944c286451415"
    assert manifest["algorithm"] == "lru"
    assert manifest["seed"] == 7
    assert manifest["dtype_policy"] == "float32-complex64"
    assert sorted(arrays) == manifest["leaf_paths"]


def test_oracle_fixture_has_required_sections():
    arrays, manifest = load_oracle()
    required_heads = {
        "heads/input",
        "heads/actor_output",
        "heads/actor_loc",
        "heads/actor_scale",
        "heads/value",
        "heads/sample_key",
        "heads/sampled_action",
        "heads/log_prob",
        "heads/entropy",
        "heads/log_prob_mean",
        "heads/entropy_mean",
        "heads/params/actor/kernel",
        "heads/params/critic/kernel",
        "heads/params/critic/bias",
        "heads/falign/actor/B",
        "heads/falign/critic/B",
        "heads/vjp/cotangent/actor",
        "heads/vjp/cotangent/value",
        "heads/vjp/input",
        "heads/vjp/params/actor/kernel",
        "heads/vjp/params/critic/kernel",
        "heads/vjp/params/critic/bias",
        "heads/vjp/falign/actor/B",
        "heads/vjp/falign/critic/B",
    }
    required_other = {
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
    assert {path for path in arrays if path.startswith("heads/")} == required_heads
    assert required_other <= arrays.keys()
    assert manifest["head_vjp"] == {
        "function": "(actor_raw, value) = strict_linear_heads(heads/input)",
        "cotangent_order": ["actor_raw", "value"],
        "cotangent_leaves": [
            "heads/vjp/cotangent/actor",
            "heads/vjp/cotangent/value",
        ],
        "input_vjp_leaf": "heads/vjp/input",
        "variable_collections": ["params", "falign"],
    }


def test_oracle_fixture_matches_all_manifest_leaf_metadata():
    arrays, manifest = load_oracle()

    assert sorted(manifest["leaves"]) == manifest["leaf_paths"]
    for path in manifest["leaf_paths"]:
        array = arrays[path]
        assert list(array.shape) == manifest["leaves"][path]["shape"], path
        assert str(array.dtype) == manifest["leaves"][path]["dtype"], path
        if array.dtype.kind in "fc":
            assert np.isfinite(array).all(), path


def test_load_oracle_rejects_invalid_leaf_metadata(tmp_path, monkeypatch):
    manifest = {
        "leaf_paths": ["value"],
        "leaves": {"value": {"shape": [2], "dtype": "float32"}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    np.savez(tmp_path / "aaai25_lru.npz", value=np.array([np.inf], np.float32))
    monkeypatch.setattr(oracle_capture, "GOLDEN_DIR", tmp_path)

    with pytest.raises(ValueError, match="shape mismatch"):
        load_oracle()


def test_source_commit_rejects_dirty_oracle(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("clean = True\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.py"], check=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Oracle Test",
        "GIT_AUTHOR_EMAIL": "oracle@example.invalid",
        "GIT_COMMITTER_NAME": "Oracle Test",
        "GIT_COMMITTER_EMAIL": "oracle@example.invalid",
    }
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "fixture"],
        check=True,
        env=env,
    )
    tracked.write_text("clean = False\n")

    with pytest.raises(RuntimeError, match="dirty"):
        oracle_capture._source_commit(tmp_path)


def test_failed_overwrite_preserves_existing_fixture(tmp_path, monkeypatch):
    archive = tmp_path / "aaai25_lru.npz"
    manifest = tmp_path / "manifest.json"
    archive.write_bytes(b"existing archive")
    manifest.write_bytes(b"existing manifest")
    monkeypatch.setattr(oracle_capture, "_source_commit", lambda _: SOURCE_COMMIT)

    def fail_capture(_):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(oracle_capture, "_capture_arrays", fail_capture)

    with pytest.raises(RuntimeError, match="capture failed"):
        oracle_capture.main(tmp_path, tmp_path, overwrite=True)
    assert archive.read_bytes() == b"existing archive"
    assert manifest.read_bytes() == b"existing manifest"


def test_legacy_golden_import_does_not_change_sys_path():
    memo_root = Path(__file__).parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(memo_root / "tests" / "online_ac"), str(memo_root)]
        ),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; before=list(sys.path); import golden; "
            "assert sys.path == before, (before, sys.path)",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


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
