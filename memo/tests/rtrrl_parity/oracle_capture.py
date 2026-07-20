"""Capture and load small numerical fixtures from the AAAI25 implementation."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

GOLDEN_DIR = Path(__file__).with_name("golden")
ARCHIVE_NAME = "aaai25_lru.npz"
MANIFEST_NAME = "manifest.json"
SOURCE_COMMIT = "4301943c349171d828d0fcf3e40944c286451415"


def load_oracle() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load the committed fixture without importing the oracle repository."""
    manifest = json.loads((GOLDEN_DIR / MANIFEST_NAME).read_text())
    with np.load(GOLDEN_DIR / ARCHIVE_NAME, allow_pickle=False) as payload:
        arrays = {path: np.array(payload[path]) for path in payload.files}
    return arrays, manifest


def _source_commit(oracle_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(oracle_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _credit_vector(carry: Any, jax: Any) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(carry[1:])
    return np.concatenate(
        [np.asarray(leaf, dtype=np.complex64).reshape(-1) for leaf in leaves]
    )


def _capture_arrays(seed: int) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    import jax
    import jax.numpy as jnp

    rtrrl = importlib.import_module("rtrrl")
    online_lru = importlib.import_module("models.online_lru")
    traces = importlib.import_module("traces")
    optimizers = importlib.import_module("optimizers")

    RNNActorCritic = rtrrl.RNNActorCritic
    OnlineLRULayer = online_lru.OnlineLRULayer
    init_trace = traces.init_trace
    trace_update = traces.trace_update
    compute_updates = traces.compute_updates
    make_optimizer = optimizers.make_optimizer

    root_key = jax.random.PRNGKey(seed)
    lru_key, model_key, action_key = jax.random.split(root_key, 3)
    lru_input = jnp.array([[0.25, -0.5, 0.75, -1.0]], dtype=jnp.float32)
    lru = OnlineLRULayer(
        d_output=2,
        d_hidden=2,
        plasticity="rtrl",
        activation=None,
    )
    lru_carry_before = lru.initialize_carry(lru_key, lru_input.shape)
    (lru_carry_after_1, lru_output), lru_variables = lru.init_with_output(
        lru_key,
        lru_carry_before,
        lru_input,
    )
    lru_input_2 = jnp.array([[-0.125, 0.375, 0.5, -0.25]], dtype=jnp.float32)
    lru_carry_after_2, _ = lru.apply(
        lru_variables,
        lru_carry_after_1,
        lru_input_2,
    )

    model = RNNActorCritic(
        a_dim=2,
        discrete=False,
        obs_dim=4,
        batch_shape=(1,),
        hidden_size=2,
        rnn_model="lru",
        gradient_mode="rtrl",
        f_align=False,
        pass_obs=False,
        mlp_actor=False,
        layer_norm=False,
    )
    model_carry = model.initialize_carry(model_key, lru_input.shape)
    (model_carry, (distribution, value, hidden)), model_variables = (
        model.init_with_output(model_key, model_carry, lru_input)
    )
    initial_action = distribution.sample(seed=action_key)
    next_input = jnp.array([[0.1, initial_action[0, 0], -0.2, 0.3]], dtype=jnp.float32)
    _, (_, next_value, _) = model.apply(model_variables, model_carry, next_input)
    reward = jnp.array([0.625], dtype=jnp.float32)
    done = jnp.array([False])
    td_error = reward + jnp.float32(0.99) * (1 - done) * next_value.squeeze() - value.squeeze()

    # Exercise the exact oracle helpers imported by this standalone capture.
    toy_trace = init_trace({"weight": jnp.ones((2,), dtype=jnp.float32)})
    toy_trace = trace_update(
        {"weight": jnp.array([0.25, -0.5], dtype=jnp.float32)},
        toy_trace,
        gamma_lambda=0.9,
    )
    compute_updates(toy_trace, d=jnp.array(0.5, dtype=jnp.float32))
    make_optimizer(direction="max").init({"weight": jnp.ones((2,), dtype=jnp.float32)})

    arrays = {
        "heads/input": np.asarray(hidden),
        "heads/actor_loc": np.asarray(distribution.loc),
        "heads/actor_scale": np.asarray(distribution.scale),
        "heads/value": np.asarray(value),
        "lru/input": np.asarray(lru_input),
        "lru/carry_before": np.asarray(lru_carry_before[0]),
        "lru/carry_after": np.asarray(lru_carry_after_1[0]),
        "lru/output": np.asarray(lru_output),
        "credit/after_step_1": _credit_vector(lru_carry_after_1, jax),
        "credit/after_step_2": _credit_vector(lru_carry_after_2, jax),
        "init/action": np.asarray(initial_action),
        "init/value": np.asarray(value),
        "step/td_error": np.asarray(td_error),
    }
    versions = {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "backend": jax.default_backend(),
    }
    return arrays, versions


def main(
    oracle_root: Path,
    output_dir: Path,
    seed: int = 7,
) -> None:
    """Generate the deterministic fixture from an explicitly selected oracle."""
    oracle_root = oracle_root.resolve()
    output_dir = output_dir.resolve()
    archive_path = output_dir / ARCHIVE_NAME
    manifest_path = output_dir / MANIFEST_NAME
    existing = [path for path in (archive_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing fixture(s): "
            + ", ".join(str(path) for path in existing)
        )
    commit = _source_commit(oracle_root)
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"oracle commit mismatch: {commit} != {SOURCE_COMMIT}")

    sys.path.insert(0, str(oracle_root))
    try:
        arrays, versions = _capture_arrays(seed)
    finally:
        if sys.path[0] == str(oracle_root):
            sys.path.pop(0)

    for path, array in arrays.items():
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"oracle produced non-finite leaf: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    leaf_paths = sorted(arrays)
    arrays_to_save: Any = {path: arrays[path] for path in leaf_paths}
    np.savez(archive_path, **arrays_to_save)
    manifest = {
        "source": "RTRRL-AAAI25",
        "commit": commit,
        "algorithm": "lru",
        "seed": seed,
        "dtype_policy": "float32-complex64",
        **versions,
        "dimensions": {
            "hidden_size": 2,
            "input_size": 4,
            "action_size": 2,
            "batch_size": 1,
        },
        "transitions": "deterministic-explicit-mock",
        "leaf_paths": leaf_paths,
        "leaves": {
            path: {
                "dtype": str(arrays[path].dtype),
                "shape": list(arrays[path].shape),
            }
            for path in leaf_paths
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.overwrite:
        for filename in (ARCHIVE_NAME, MANIFEST_NAME):
            (arguments.output_dir / filename).unlink(missing_ok=True)
    main(arguments.oracle_root, arguments.output_dir, arguments.seed)
