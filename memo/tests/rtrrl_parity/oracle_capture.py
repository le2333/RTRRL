"""Capture and load small numerical fixtures from the AAAI25 implementation."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
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
    leaf_paths = manifest["leaf_paths"]
    if sorted(arrays) != leaf_paths or sorted(manifest["leaves"]) != leaf_paths:
        raise ValueError("fixture leaf paths do not match manifest")
    for path in leaf_paths:
        array = arrays[path]
        metadata = manifest["leaves"][path]
        if list(array.shape) != metadata["shape"]:
            raise ValueError(
                f"shape mismatch at {path}: {list(array.shape)} != {metadata['shape']}"
            )
        if str(array.dtype) != metadata["dtype"]:
            raise ValueError(
                f"dtype mismatch at {path}: {array.dtype} != {metadata['dtype']}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"non-finite fixture leaf: {path}")
    return arrays, manifest


def _source_commit(oracle_root: Path) -> str:
    status = subprocess.run(
        [
            "git",
            "-C",
            str(oracle_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(f"oracle worktree is dirty:\n{status.rstrip()}")
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


def _repack_forced_credit_carry(carry: Any) -> tuple[Any, tuple[Any, Any, Any]]:
    """Restore the pinned initializer layout after force_trace_compute flattens it."""
    hidden, lambda_sensitivity, gamma_sensitivity, B_sensitivity = carry
    return hidden, (lambda_sensitivity, gamma_sensitivity, B_sensitivity)


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
        activation="silu",
    )
    lru_carry_before = lru.initialize_carry(lru_key, lru_input.shape)
    (lru_carry_after_1, lru_output), lru_variables = lru.init_with_output(
        lru_key,
        lru_carry_before,
        lru_input,
    )
    lru_input_2 = jnp.array([[-0.125, 0.375, 0.5, -0.25]], dtype=jnp.float32)
    lru_carry_after_2, lru_output_2 = lru.apply(
        lru_variables,
        lru_carry_after_1,
        lru_input_2,
    )
    lru_reset_carry, lru_reset_output = lru.apply(
        lru_variables,
        lru_carry_after_1,
        lru_input_2,
        resets=jnp.array([True]),
    )
    lru_params = lru_variables["params"]
    lru_cell_params = lru_params["OnlineLRUCell_0"]["LRUCell_0"]
    lru_lambda = online_lru.get_lambda(
        lru_cell_params["nu_log"], lru_cell_params["theta_log"]
    )
    lru_normalized_B = online_lru.get_B_norm(
        lru_cell_params["B_real"],
        lru_cell_params["B_img"],
        lru_cell_params["gamma_log"],
    )
    lru_C = lru_params["C_real"] + 1j * lru_params["C_img"]
    lru_projection = (lru_carry_after_1[0] @ lru_C.transpose()).real
    lru_skip = lru_input @ lru_params["D"].transpose()
    lru_preactivation = lru_projection + lru_skip
    lru_unbatched_input = lru_input[0]
    lru_unbatched_carry_before = lru.initialize_carry(
        lru_key, lru_unbatched_input.shape
    )
    lru_unbatched_carry_after, lru_unbatched_output = lru.apply(
        lru_variables,
        lru_unbatched_carry_before,
        lru_unbatched_input,
    )
    lru_unbatched_projection = (
        lru_unbatched_carry_after[0] @ lru_C.transpose()
    ).real
    lru_unbatched_skip = lru_unbatched_input @ lru_params["D"].transpose()
    lru_unbatched_preactivation = (
        lru_unbatched_projection + lru_unbatched_skip
    )
    lru_unbatched_input_2 = lru_input_2[0]
    credit_carry_after_1_flat, _ = lru.apply(
        lru_variables,
        lru_unbatched_carry_before,
        lru_unbatched_input,
        force_trace_compute=True,
    )
    credit_carry_after_1 = _repack_forced_credit_carry(credit_carry_after_1_flat)
    credit_carry_after_2_flat, _ = lru.apply(
        lru_variables,
        credit_carry_after_1,
        lru_unbatched_input_2,
        force_trace_compute=True,
    )
    credit_carry_after_2 = _repack_forced_credit_carry(credit_carry_after_2_flat)
    credit_cotangent_1 = jnp.array([0.625, -0.375], dtype=jnp.float32)
    credit_cotangent_2 = jnp.array([-0.25, 0.75], dtype=jnp.float32)

    def apply_lru_output(variables, carry, inputs):
        return lru.apply(variables, carry, inputs)[1]

    _, credit_pullback_1 = jax.vjp(
        apply_lru_output,
        lru_variables,
        lru_unbatched_carry_before,
        lru_unbatched_input,
    )
    credit_variables_grad_1, _, _ = credit_pullback_1(credit_cotangent_1)
    _, credit_pullback_2 = jax.vjp(
        apply_lru_output,
        lru_variables,
        credit_carry_after_1,
        lru_unbatched_input_2,
    )
    credit_variables_grad_2, _, _ = credit_pullback_2(credit_cotangent_2)
    credit_cell_grad_1 = credit_variables_grad_1["params"]["OnlineLRUCell_0"][
        "LRUCell_0"
    ]
    credit_cell_grad_2 = credit_variables_grad_2["params"]["OnlineLRUCell_0"][
        "LRUCell_0"
    ]

    def lru_readout(hidden, inputs):
        preactivation = (hidden @ lru_C.transpose()).real
        preactivation = preactivation + inputs @ lru_params["D"].transpose()
        return jax.nn.silu(preactivation)

    _, hidden_pullback_1 = jax.vjp(
        lambda hidden: lru_readout(hidden, lru_unbatched_input),
        credit_carry_after_1[0],
    )
    credit_hidden_cotangent_1 = hidden_pullback_1(credit_cotangent_1)[0][0]
    _, hidden_pullback_2 = jax.vjp(
        lambda hidden: lru_readout(hidden, lru_unbatched_input_2),
        credit_carry_after_2[0],
    )
    credit_hidden_cotangent_2 = hidden_pullback_2(credit_cotangent_2)[0][0]

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

    # Capture an independently initialized strict head with feedback alignment.
    # Keeping this separate preserves the original end-to-end fixture leaves while
    # exposing every parameter and cotangent needed to replay the head in Memorax.
    head_key = jax.random.fold_in(root_key, 4)
    head_sample_key = jax.random.fold_in(root_key, 5)
    head_model = RNNActorCritic(
        a_dim=2,
        discrete=False,
        obs_dim=4,
        batch_shape=(1,),
        hidden_size=2,
        rnn_model=None,
        f_align=True,
        pass_obs=False,
        mlp_actor=False,
        layer_norm=False,
    )
    _, head_variables = head_model.init_with_output(head_key, None, hidden)

    def raw_heads(module, head_input):
        return module.td.actor(head_input), module.td.critic(head_input)

    def apply_raw_heads(variables, head_input):
        return head_model.apply(variables, head_input, method=raw_heads)

    (head_actor_output, head_value), head_pullback = jax.vjp(
        apply_raw_heads, head_variables, hidden
    )
    head_distribution = head_model.apply(
        head_variables, hidden, method=head_model.action_dist
    )
    head_sampled_action = head_distribution.sample(seed=head_sample_key)
    head_log_prob = head_distribution.log_prob(head_sampled_action)
    head_entropy = head_distribution.entropy()
    actor_cotangent = jnp.array(
        [[0.25, -0.5, 0.75, -1.25]], dtype=jnp.float32
    )
    value_cotangent = jnp.array([[0.625]], dtype=jnp.float32)
    head_variables_vjp, head_input_vjp = head_pullback(
        (actor_cotangent, value_cotangent)
    )

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
        "heads/actor_output": np.asarray(head_actor_output),
        "heads/actor_loc": np.asarray(head_distribution.loc),
        "heads/actor_scale": np.asarray(head_distribution.scale),
        "heads/value": np.asarray(head_value),
        "heads/sample_key": np.asarray(head_sample_key),
        "heads/sampled_action": np.asarray(head_sampled_action),
        "heads/log_prob": np.asarray(head_log_prob),
        "heads/entropy": np.asarray(head_entropy),
        "heads/log_prob_mean": np.asarray(head_log_prob.mean()),
        "heads/entropy_mean": np.asarray(head_entropy.mean()),
        "heads/params/actor/kernel": np.asarray(
            head_variables["params"]["td"]["actor"]["kernel"]
        ),
        "heads/params/critic/kernel": np.asarray(
            head_variables["params"]["td"]["critic"]["kernel"]
        ),
        "heads/params/critic/bias": np.asarray(
            head_variables["params"]["td"]["critic"]["bias"]
        ),
        "heads/falign/actor/B": np.asarray(
            head_variables["falign"]["td"]["actor"]["B"]
        ),
        "heads/falign/critic/B": np.asarray(
            head_variables["falign"]["td"]["critic"]["B"]
        ),
        "heads/vjp/cotangent/actor": np.asarray(actor_cotangent),
        "heads/vjp/cotangent/value": np.asarray(value_cotangent),
        "heads/vjp/input": np.asarray(head_input_vjp),
        "heads/vjp/params/actor/kernel": np.asarray(
            head_variables_vjp["params"]["td"]["actor"]["kernel"]
        ),
        "heads/vjp/params/critic/kernel": np.asarray(
            head_variables_vjp["params"]["td"]["critic"]["kernel"]
        ),
        "heads/vjp/params/critic/bias": np.asarray(
            head_variables_vjp["params"]["td"]["critic"]["bias"]
        ),
        "heads/vjp/falign/actor/B": np.asarray(
            head_variables_vjp["falign"]["td"]["actor"]["B"]
        ),
        "heads/vjp/falign/critic/B": np.asarray(
            head_variables_vjp["falign"]["td"]["critic"]["B"]
        ),
        "lru/input": np.asarray(lru_input),
        "lru/init_key": np.asarray(lru_key),
        "lru/carry_before": np.asarray(lru_carry_before[0]),
        "lru/carry_after": np.asarray(lru_carry_after_1[0]),
        "lru/lambda": np.asarray(lru_lambda),
        "lru/normalized_B": np.asarray(lru_normalized_B),
        "lru/params/nu_log": np.asarray(lru_cell_params["nu_log"]),
        "lru/params/theta_log": np.asarray(lru_cell_params["theta_log"]),
        "lru/params/gamma_log": np.asarray(lru_cell_params["gamma_log"]),
        "lru/params/B_real": np.asarray(lru_cell_params["B_real"]),
        "lru/params/B_img": np.asarray(lru_cell_params["B_img"]),
        "lru/params/C_real": np.asarray(lru_params["C_real"]),
        "lru/params/C_img": np.asarray(lru_params["C_img"]),
        "lru/params/D": np.asarray(lru_params["D"]),
        "lru/projection": np.asarray(lru_projection),
        "lru/skip": np.asarray(lru_skip),
        "lru/preactivation": np.asarray(lru_preactivation),
        "lru/output": np.asarray(lru_output),
        "lru/next/input": np.asarray(lru_input_2),
        "lru/next/carry_after": np.asarray(lru_carry_after_2[0]),
        "lru/next/output": np.asarray(lru_output_2),
        "lru/reset/input": np.asarray(lru_input_2),
        "lru/reset/carry_after": np.asarray(lru_reset_carry[0]),
        "lru/reset/output": np.asarray(lru_reset_output),
        "lru/unbatched/input": np.asarray(lru_unbatched_input),
        "lru/unbatched/carry_before": np.asarray(
            lru_unbatched_carry_before[0]
        ),
        "lru/unbatched/carry_after": np.asarray(lru_unbatched_carry_after[0]),
        "lru/unbatched/projection": np.asarray(lru_unbatched_projection),
        "lru/unbatched/skip": np.asarray(lru_unbatched_skip),
        "lru/unbatched/preactivation": np.asarray(
            lru_unbatched_preactivation
        ),
        "lru/unbatched/output": np.asarray(lru_unbatched_output),
        "credit/after_step_1": _credit_vector(credit_carry_after_1, jax),
        "credit/after_step_2": _credit_vector(credit_carry_after_2, jax),
        "credit/step_1/lambda_sensitivity": np.asarray(
            credit_carry_after_1[1][0]
        ),
        "credit/step_1/gamma_sensitivity": np.asarray(
            credit_carry_after_1[1][1]
        ),
        "credit/step_1/B_sensitivity": np.asarray(credit_carry_after_1[1][2]),
        "credit/step_1/cotangent": np.asarray(credit_cotangent_1),
        "credit/step_1/hidden_cotangent": np.asarray(
            credit_hidden_cotangent_1
        ),
        "credit/step_2/lambda_sensitivity": np.asarray(
            credit_carry_after_2[1][0]
        ),
        "credit/step_2/gamma_sensitivity": np.asarray(
            credit_carry_after_2[1][1]
        ),
        "credit/step_2/B_sensitivity": np.asarray(credit_carry_after_2[1][2]),
        "credit/step_2/cotangent": np.asarray(credit_cotangent_2),
        "credit/step_2/hidden_cotangent": np.asarray(
            credit_hidden_cotangent_2
        ),
        **{
            f"credit/step_1/grad/{name}": np.asarray(credit_cell_grad_1[name])
            for name in ("nu_log", "theta_log", "gamma_log", "B_real", "B_img")
        },
        **{
            f"credit/step_2/grad/{name}": np.asarray(credit_cell_grad_2[name])
            for name in ("nu_log", "theta_log", "gamma_log", "B_real", "B_img")
        },
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
    *,
    overwrite: bool = False,
) -> None:
    """Generate the deterministic fixture from an explicitly selected oracle."""
    oracle_root = oracle_root.resolve()
    output_dir = output_dir.resolve()
    archive_path = output_dir / ARCHIVE_NAME
    manifest_path = output_dir / MANIFEST_NAME
    existing = [path for path in (archive_path, manifest_path) if path.exists()]
    if existing and not overwrite:
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

    leaf_paths = sorted(arrays)
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
        "head_vjp": {
            "function": "(actor_raw, value) = strict_linear_heads(heads/input)",
            "cotangent_order": ["actor_raw", "value"],
            "cotangent_leaves": [
                "heads/vjp/cotangent/actor",
                "heads/vjp/cotangent/value",
            ],
            "input_vjp_leaf": "heads/vjp/input",
            "variable_collections": ["params", "falign"],
        },
        "lru_forward": {
            "activation": "silu",
            "parameter_leaves": [
                "nu_log",
                "theta_log",
                "gamma_log",
                "B_real",
                "B_img",
                "C_real",
                "C_img",
                "D",
            ],
            "reset": (
                "one-step output ignores the previous hidden state for both "
                "boolean reset values"
            ),
        },
        "lru_credit": {
            "state_layout": [
                "lambda_sensitivity",
                "gamma_sensitivity",
                "B_sensitivity",
            ],
            "gradient_leaves": [
                "nu_log",
                "theta_log",
                "gamma_log",
                "B_real",
                "B_img",
            ],
            "cotangents": [
                "credit/step_1/cotangent",
                "credit/step_2/cotangent",
            ],
            "B_img_rule": "negative-imaginary-complex-B-contraction",
            "force_trace_layout": (
                "pinned force_trace_compute returns a flat four-tuple; capture "
                "repacks it to the initializer's nested carry before step two"
            ),
        },
        "leaf_paths": leaf_paths,
        "leaves": {
            path: {
                "dtype": str(arrays[path].dtype),
                "shape": list(arrays[path].shape),
            }
            for path in leaf_paths
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        temporary_dir = Path(temporary)
        staged_archive = temporary_dir / ARCHIVE_NAME
        staged_manifest = temporary_dir / MANIFEST_NAME
        arrays_to_save: Any = {path: arrays[path] for path in leaf_paths}
        np.savez(staged_archive, **arrays_to_save)
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staged_archive, archive_path)
        os.replace(staged_manifest, manifest_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    main(
        arguments.oracle_root,
        arguments.output_dir,
        arguments.seed,
        overwrite=arguments.overwrite,
    )
