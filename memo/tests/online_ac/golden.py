"""Versioned, leaf-exact characterization snapshots for legacy online AC."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import jaxlib
import lox
import numpy as np

from memorax.utils import Timestep
from memorax.utils.axes import remove_feature_axis, remove_time_axis

GOLDEN_DIR = Path(__file__).with_name("golden")


def flatten_with_paths(tree):
    pairs, treedef = jax.tree_util.tree_flatten_with_path(tree)
    return {
        "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path): leaf
        for path, leaf in pairs
    }, treedef


@dataclass(frozen=True)
class GoldenSnapshot:
    leaves: Mapping[str, np.ndarray]


def assert_tree_allclose(actual, expected, *, rtol=1e-6, atol=1e-7):
    actual_leaves = flatten_with_paths(actual)[0]
    expected_leaves = (
        dict(expected.leaves)
        if isinstance(expected, GoldenSnapshot)
        else flatten_with_paths(expected)[0]
    )
    assert actual_leaves.keys() == expected_leaves.keys()
    for path in actual_leaves:
        actual_array = np.asarray(actual_leaves[path])
        expected_array = np.asarray(expected_leaves[path])
        if actual_array.dtype.kind in "biufc" and expected_array.dtype.kind in "biufc":
            np.testing.assert_allclose(
                actual_array,
                expected_array,
                rtol=rtol,
                atol=atol,
                err_msg=path,
            )
        else:
            np.testing.assert_array_equal(actual_array, expected_array, err_msg=path)


def save_golden(name, tree, metadata):
    """Write an NPZ payload and JSON manifest; callers opt in explicitly."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = tree if isinstance(tree, dict) else {"snapshot": tree}
    arrays: dict[str, np.ndarray] = {}
    snapshot_names: list[str] = []
    for snapshot_name, snapshot in snapshots.items():
        snapshot_names.append(snapshot_name)
        leaves, _ = flatten_with_paths(snapshot)
        for leaf_path, leaf in leaves.items():
            arrays[f"{snapshot_name}/{leaf_path}"] = np.asarray(leaf)

    leaf_paths = sorted(arrays)
    manifest = {
        **metadata,
        "leaf_paths": leaf_paths,
        "snapshots": snapshot_names,
        "leaves": {
            path: {"dtype": str(arrays[path].dtype), "shape": list(arrays[path].shape)}
            for path in leaf_paths
        },
    }
    arrays_to_save: Any = {p: arrays[p] for p in leaf_paths}
    np.savez(GOLDEN_DIR / f"{name}.npz", **arrays_to_save)
    (GOLDEN_DIR / f"{name}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def load_golden(name):
    manifest = json.loads((GOLDEN_DIR / f"{name}.json").read_text())
    with np.load(GOLDEN_DIR / f"{name}.npz", allow_pickle=False) as payload:
        snapshots = {}
        for snapshot_name in manifest["snapshots"]:
            prefix = f"{snapshot_name}/"
            snapshots[snapshot_name] = GoldenSnapshot(
                {
                    path[len(prefix) :]: np.array(payload[path])
                    for path in manifest["leaf_paths"]
                    if path.startswith(prefix)
                }
            )
    if manifest["snapshots"] == ["snapshot"]:
        return snapshots["snapshot"], manifest
    return snapshots, manifest


def _metric_schema(logs):
    leaves, _ = flatten_with_paths(logs)
    return {
        "keys": np.asarray(sorted(leaves), dtype=np.str_),
        "dtypes": np.asarray(
            [str(np.asarray(leaves[path]).dtype) for path in sorted(leaves)],
            dtype=np.str_,
        ),
    }


def _spooled_update(agent, state, key):
    (updated_state, aux), logs = lox.spool(agent._update_step)(state, key)
    assert aux is None
    return updated_state, logs


def _rtrrl_observables(agent, state, key):
    action_key, step_key = jax.random.split(key)
    obs, done, ts_action, reward = state.timestep.to_sequence()
    grad_params = agent._grad_params(state.params, state.slow_torso)
    (carry, sensitivity), (dist, value, _) = agent._forward(
        grad_params,
        obs,
        ts_action,
        reward,
        done,
        state.carry,
        state.sensitivity,
    )
    sampled_action, logprob = dist.sample_and_log_prob(seed=action_key)
    sampled_action = remove_time_axis(sampled_action)
    logprob = remove_time_axis(logprob)
    value = remove_feature_axis(remove_time_axis(value))
    clip = agent.cfg.act_clip
    env_action = jnp.clip(sampled_action, -clip, clip) if clip else sampled_action
    step_keys = jax.random.split(step_key, agent.cfg.num_envs)
    next_obs, _, next_reward, next_done, _ = jax.vmap(
        agent.env.step, in_axes=(0, 0, 0, None)
    )(step_keys, state.env_state, env_action, agent.env_params)
    next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
        obs=next_obs,
        action=env_action,
        reward=next_reward,
        done=next_done,
    ).to_sequence()
    _, (_, next_value, _) = agent._forward(
        jax.lax.stop_gradient(grad_params),
        next_obs_s,
        next_action_s,
        next_reward_s,
        next_done_s,
        jax.lax.stop_gradient(carry),
        jax.lax.stop_gradient(sensitivity),
    )
    next_value = remove_feature_axis(remove_time_axis(next_value))
    td = next_reward + agent.cfg.gamma * (1 - next_done) * next_value - value
    return {
        "sampled_action": sampled_action,
        "logprob_action": sampled_action,
        "env_action": env_action,
        "feedback_action": env_action,
        "logprob": logprob,
        "value": value,
        "next_value": next_value,
        "td": td,
    }


def _stream_observables(agent, state, key):
    action_key, step_key = jax.random.split(key)
    obs, done, ts_action, reward = state.timestep.to_sequence()
    (actor_carry, actor_sensitivity), (dist, _) = agent._rtrl_forward(
        agent.actor_network,
        state.actor_params,
        obs,
        ts_action,
        reward,
        done,
        state.actor_carry,
        state.actor_sensitivity,
    )
    sampled_action, logprob = dist.sample_and_log_prob(seed=action_key)
    sampled_action = remove_time_axis(sampled_action)
    logprob = remove_time_axis(logprob)
    (critic_carry, critic_sensitivity), (value, _) = agent._rtrl_forward(
        agent.critic_network,
        state.critic_params,
        obs,
        ts_action,
        reward,
        done,
        state.critic_carry,
        state.critic_sensitivity,
    )
    value = remove_feature_axis(remove_time_axis(value))
    step_keys = jax.random.split(step_key, agent.cfg.num_envs)
    next_obs, _, next_reward, next_done, _ = jax.vmap(
        agent.env.step, in_axes=(0, 0, 0, None)
    )(step_keys, state.env_state, sampled_action, agent.env_params)
    next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
        obs=next_obs,
        action=sampled_action,
        reward=next_reward,
        done=next_done,
    ).to_sequence()
    _, (next_value, _) = agent._rtrl_forward(
        agent.critic_network,
        jax.lax.stop_gradient(state.critic_params),
        next_obs_s,
        next_action_s,
        next_reward_s,
        next_done_s,
        jax.lax.stop_gradient(critic_carry),
        jax.lax.stop_gradient(critic_sensitivity),
    )
    next_value = remove_feature_axis(remove_time_axis(next_value))
    td = next_reward + agent.cfg.gamma * (1 - next_done) * next_value - value
    return {
        "sampled_action": sampled_action,
        "logprob_action": sampled_action,
        "env_action": sampled_action,
        "feedback_action": sampled_action,
        "logprob": logprob,
        "value": value,
        "next_value": next_value,
        "td": td,
        "actor_carry_after_action": actor_carry,
        "actor_sensitivity_after_action": actor_sensitivity,
    }


def _snapshot(agent, observer, seed, steps):
    base_key = jax.random.key(seed)
    init_state = agent.init(jax.random.fold_in(base_key, 0))
    update_key = jax.random.fold_in(base_key, 1)
    observables = observer(agent, init_state, update_key)
    one_step, logs = _spooled_update(agent, init_state, update_key)
    trained = agent.train(jax.random.fold_in(base_key, 2), init_state, num_steps=steps)
    evaluated = agent.evaluate(
        jax.random.fold_in(base_key, 3), trained, num_steps=steps
    )
    return {
        "init": init_state,
        "one_step": one_step,
        "train": trained,
        "evaluate": evaluated,
        "observables": observables,
        "metrics": _metric_schema(logs),
    }


def snapshot_rtrrl(agent, *, seed, steps):
    return _snapshot(agent, _rtrrl_observables, seed, steps)


def snapshot_stream_ac(agent, *, seed, steps):
    return _snapshot(agent, _stream_observables, seed, steps)


def _struct_config(value):
    """Serialize every declared config field with stable JSON scalar values."""
    result = {}
    for field in fields(value):
        item = getattr(value, field.name)
        if item is None or isinstance(item, (bool, int, float, str)):
            result[field.name] = item
        elif callable(item):
            result[field.name] = getattr(item, "__name__", type(item).__name__)
        else:
            try:
                result[field.name] = np.dtype(item).name
            except TypeError:
                raise TypeError(
                    f"config field {field.name!r} is not JSON-serializable: {item!r}"
                ) from None
    return result


def _snapshot_protocol(steps):
    return {
        "seed": 7,
        "steps": steps,
        "num_envs": 1,
        "init_key": "fold_in(seed, 0)",
        "one_step_key": "fold_in(seed, 1)",
        "train_key": "fold_in(seed, 2)",
        "evaluate_key": "fold_in(seed, 3)",
        "state_sections": [
            "init",
            "one_step",
            "train",
            "evaluate",
            "observables",
            "metrics",
        ],
    }


def _rtrrl_effective_config(incoming_agent, fresh_agent, steps):
    return {
        "algorithm": {
            "type": "RTRRL",
            "variants": {
                "trace_timing/incoming": _struct_config(incoming_agent.cfg),
                "trace_timing/fresh": _struct_config(fresh_agent.cfg),
            },
        },
        "network": {
            "feature_extractor": {
                "type": "FeatureExtractor",
                "concatenate_order": ["observation", "action", "reward"],
                "observation": {"layers": ["Dense(3)", "tanh"]},
                "action": {"layers": ["Dense(3)", "tanh"]},
                "reward": {"layers": ["Dense(3)", "tanh"]},
                "done": None,
                "dense_kernel_init": "lecun_normal",
                "dense_bias_init": "zeros",
                "dense_param_dtype": "float32",
            },
            "torso": {
                "type": "Memoroid",
                "cell": "LRUCell",
                **_struct_config(incoming_agent.torso.cell.config),
            },
            "post_torso_activation": "silu",
            "actor_head": {
                "type": "Gaussian",
                "action_dim": 2,
                "bound": False,
                "loc_bounds": [-1.0, 1.0],
                "log_std_bounds": [-2.0, 2.0],
                "kernel_init": "lecun_normal",
                "bias_init": "zeros",
                "log_std_init": "zeros",
            },
            "critic_head": {
                "type": "VNetwork",
                "output_dim": 1,
                "kernel_init": "lecun_normal",
                "bias_init": "zeros",
            },
            "prediction_head": None,
        },
        "environment": {
            "type": "TinyContinuousEnv",
            "params": {"horizon": 3},
            "observation_shape": [2],
            "reset_observation": [0.25, -0.5],
            "transition": "obs + action + [0.15, 0.45]",
            "reward": "0.4 + 0.35 * step_count",
            "terminal_transition": 3,
        },
        "action": {
            "space": "Box",
            "shape": [2],
            "dtype": "float32",
            "low": -2.0,
            "high": 2.0,
            "sampling": "MultivariateNormalDiag.sample_and_log_prob",
            "environment_clip": incoming_agent.cfg.act_clip,
            "feedback_action": "environment_action",
            "logprob_action": "unclipped_sample",
            "logprob_reduction_scale": incoming_agent.cfg.logprob_scale,
        },
        "evaluation": {
            "policy": "deterministic_mode",
            "steps": steps,
            "reset_environment": True,
            "reset_carry": True,
            "reset_sensitivity": True,
            "updates_parameters": False,
        },
        "snapshot_protocol": _snapshot_protocol(steps),
    }


def _stream_ac_effective_config(non_adaptive_agent, adaptive_agent, steps):
    torso_config = _struct_config(non_adaptive_agent.actor_network.torso.cell.config)
    return {
        "algorithm": {
            "type": "StreamACRtrl",
            "variants": {
                "obgd/non_adaptive": _struct_config(non_adaptive_agent.cfg),
                "obgd/adaptive": _struct_config(adaptive_agent.cfg),
            },
        },
        "network": {
            "separate_actor_critic": True,
            "feature_extractor": {
                "type": "FeatureExtractor",
                "concatenate_order": ["observation"],
                "observation": {"layers": ["Dense(3)", "tanh"]},
                "action": None,
                "reward": None,
                "done": None,
                "dense_kernel_init": "lecun_normal",
                "dense_bias_init": "zeros",
                "dense_param_dtype": "float32",
            },
            "torso": {"type": "RNN", "cell": "RTUCell", **torso_config},
            "actor": {
                "head": {"type": "Categorical", "action_dim": 2},
                "head_kernel_init": "lecun_normal",
                "head_bias_init": "zeros",
            },
            "critic": {
                "head": {"type": "VNetwork", "output_dim": 1},
                "head_kernel_init": "lecun_normal",
                "head_bias_init": "zeros",
            },
        },
        "environment": {
            "type": "TinyDiscreteEnv",
            "params": {"horizon": 3},
            "observation_shape": [2],
            "reset_observation": [-0.25, 0.5],
            "transition": "obs + [-0.2|0.2, 0.05|0.15] selected by action",
            "reward": "0.25 * direction(action) + 0.1 * step_count",
            "terminal_transition": 3,
        },
        "action": {
            "space": "Discrete",
            "count": 2,
            "dtype": "int32_effective_from_gymnax_int64_with_x64_disabled",
            "sampling": "Categorical.sample_and_log_prob",
            "environment_action": "sampled_action",
            "feedback_action": "sampled_action",
            "logprob_action": "sampled_action",
        },
        "evaluation": {
            "policy": "deterministic_argmax",
            "steps": steps,
            "reset_environment": True,
            "reset_actor_carry": True,
            "reset_critic_carry": True,
            "reset_actor_sensitivity": True,
            "reset_critic_sensitivity": True,
            "updates_parameters": False,
        },
        "snapshot_protocol": _snapshot_protocol(steps),
    }


def generate_all():
    from conftest import build_rtrrl_agent, build_stream_ac_agent

    steps = 3
    common = {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,  # pyright: ignore[reportPrivateImportUsage]
        "backend": jax.default_backend(),
        "seed": 7,
        "steps": steps,
    }
    incoming_rtrrl = build_rtrrl_agent(fresh_trace=False)
    fresh_rtrrl = build_rtrrl_agent(fresh_trace=True)
    save_golden(
        "rtrrl_lru",
        {
            "trace_timing/incoming": snapshot_rtrrl(
                incoming_rtrrl, seed=7, steps=steps
            ),
            "trace_timing/fresh": snapshot_rtrrl(fresh_rtrrl, seed=7, steps=steps),
        },
        {
            **common,
            "algorithm": "rtrrl_lru",
            "config": _rtrrl_effective_config(incoming_rtrrl, fresh_rtrrl, steps),
        },
    )
    non_adaptive_stream = build_stream_ac_agent(adaptive=False)
    adaptive_stream = build_stream_ac_agent(adaptive=True)
    save_golden(
        "stream_ac_rtu",
        {
            "obgd/non_adaptive": snapshot_stream_ac(
                non_adaptive_stream, seed=7, steps=steps
            ),
            "obgd/adaptive": snapshot_stream_ac(adaptive_stream, seed=7, steps=steps),
        },
        {
            **common,
            "algorithm": "stream_ac_rtu",
            "config": _stream_ac_effective_config(
                non_adaptive_stream, adaptive_stream, steps
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if not args.generate:
        parser.error("golden files are only written with --generate")
    generate_all()
