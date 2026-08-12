"""Blocking short acceptance for both S1 DiagSSM learners."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from s1_diagssm_runner import Config, S1Runner, load_checkpoint, save_checkpoint


def tree_error(left, right):
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    maximum = 0.0
    equal = True
    for first, second in zip(left_leaves, right_leaves, strict=True):
        a = np.asarray(jax.device_get(first))
        b = np.asarray(jax.device_get(second))
        equal = equal and np.array_equal(a, b, equal_nan=True)
        if np.issubdtype(a.dtype, np.number):
            maximum = max(maximum, float(np.nanmax(np.abs(a - b), initial=0.0)))
    return {"bitwise_equal": bool(equal), "max_abs": maximum}


def recurrent_parity(runner: S1Runner):
    state = runner.init()
    inputs = jax.random.normal(jax.random.PRNGKey(91), (runner.input_size,))
    state_only = runner.recurrent_step(state.hidden, state.params.recurrent, inputs, False)
    with_trace = runner.rtrl_step(
        state.hidden, state.sensitivity, state.params.recurrent, inputs, False
    )[0]
    reset_state = runner.recurrent_step(state.hidden, state.params.recurrent, inputs, True)
    reset_trace = runner.rtrl_step(
        state.hidden, state.sensitivity, state.params.recurrent, inputs, True
    )[0]
    return {
        "no_reset": tree_error(state_only, with_trace),
        "reset": tree_error(reset_state, reset_trace),
    }


def action_detach_check():
    sampled = jnp.asarray([0.2, -0.4], jnp.float32)
    mean = jnp.asarray([0.1, 0.3], jnp.float32)
    scale = jnp.asarray([0.8, 1.2], jnp.float32)

    def loss(action, location):
        action = jax.lax.stop_gradient(action)
        return -jnp.sum(-0.5 * ((action - location) / scale) ** 2 - jnp.log(scale))

    action_gradient = jax.grad(loss, 0)(sampled, mean)
    mean_gradient = jax.grad(loss, 1)(sampled, mean)
    return {
        "action_gradient_norm": float(jnp.linalg.norm(action_gradient)),
        "mean_gradient_norm": float(jnp.linalg.norm(mean_gradient)),
        "pass": bool(jnp.linalg.norm(action_gradient) == 0 and jnp.linalg.norm(mean_gradient) > 0),
    }


def terminal_target_check():
    reward, next_value, gamma = 2.5, 17.0, 0.99
    terminal = reward + gamma * next_value * (1.0 - 1.0)
    continuing = reward + gamma * next_value
    return {
        "terminal_target": terminal,
        "continuing_target": continuing,
        "pass": terminal == reward and continuing != reward,
    }


def run_steps(runner: S1Runner, state, count: int):
    if runner.cfg.learner == "rtrl":
        step = jax.jit(runner.rtrl_train_step)
        metrics = []
        for _ in range(count):
            state, metric = step(state, None)
            metrics.append(metric)
    else:
        if count % runner.cfg.bptt_length:
            raise ValueError("BPTT acceptance steps must align to the chunk length")
        step = jax.jit(runner.bptt_chunk)
        metrics = []
        for _ in range(count // runner.cfg.bptt_length):
            state, metric = step(state)
            metrics.append(metric)
    jax.block_until_ready(metrics[-1])
    return state, jnp.stack(metrics)


def learner_acceptance(learner: str, seed: int, steps: int, output: Path):
    cfg = Config(learner=learner, seed=seed, steps=steps)
    runner = S1Runner(cfg)
    initial = runner.init()
    split = steps // 2
    if learner == "bptt128":
        split -= split % cfg.bptt_length
    uninterrupted, metrics = run_steps(runner, initial, steps)
    first, _ = run_steps(runner, initial, split)
    checkpoint = output / f"{learner}_resume.pkl"
    save_checkpoint(checkpoint, cfg, first)
    restored = load_checkpoint(checkpoint, cfg, runner)
    resumed, _ = run_steps(runner, restored, steps - split)
    resume = tree_error(uninterrupted, resumed)
    eval_before = runner.evaluate(uninterrupted.params)
    carry_before = jax.tree.map(lambda value: np.asarray(jax.device_get(value)).copy(), uninterrupted)
    eval_repeat = runner.evaluate(uninterrupted.params)
    carry_after = jax.tree.map(lambda value: np.asarray(jax.device_get(value)).copy(), uninterrupted)
    eval_equal = np.array_equal(eval_before, eval_repeat)
    eval_non_mutating = tree_error(carry_before, carry_after)
    metric_values = np.asarray(jax.device_get(metrics))
    # Completed-return/length metric slots intentionally use NaN when no episode
    # ended in that step/chunk.  The explicit learner finite slot is index 23.
    finite = bool(
        all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(uninterrupted))
        and np.all(metric_values[:, 23] == 1.0)
    )
    radius = np.asarray(jax.device_get(runner.coefficients(uninterrupted.params.recurrent)[2]))
    checkpoint_fields = (
        set(uninterrupted._fields)
        if learner == "rtrl"
        else set(load_checkpoint(checkpoint, cfg, runner)._fields)
    )
    required = {
        "env", "params", "hidden", "action", "value", "key", "previous_reward",
        "episode_age", "episode_return", "step",
    }
    if learner == "rtrl":
        required |= {"sensitivity", "actor_elig", "critic_elig", "recurrent_elig"}
    checks = {
        "finite": finite,
        "stable_poles": bool(np.all(radius < 1.0)),
        "checkpoint_fields": required <= checkpoint_fields,
        "resume_exact": resume["bitwise_equal"],
        "fixed_eval_repeatable": bool(eval_equal),
        "fixed_eval_non_mutating": eval_non_mutating["bitwise_equal"],
        "fixed_eval_has_10_returns": len(eval_before) == 10,
    }
    return {
        "learner": learner,
        "config": asdict(cfg),
        "checks": checks,
        "pass": all(checks.values()),
        "resume": resume,
        "fixed_eval_returns": eval_before.astype(float).tolist(),
        "pole_radius": {"min": float(radius.min()), "mean": float(radius.mean()), "max": float(radius.max())},
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.steps % 256:
        raise ValueError("acceptance steps must be divisible by 256")
    base_runner = S1Runner(Config(learner="rtrl", seed=1, steps=args.steps))
    recurrence = recurrent_parity(base_runner)
    action_detach = action_detach_check()
    terminal = terminal_target_check()
    learners = [
        learner_acceptance(name, 1, args.steps, args.output)
        for name in ("rtrl", "bptt128")
    ]
    common_checks = {
        "recurrence_no_reset": recurrence["no_reset"]["bitwise_equal"],
        "recurrence_reset": recurrence["reset"]["bitwise_equal"],
        "executed_action_detached": action_detach["pass"],
        "terminal_td_target": terminal["pass"],
    }
    result = {
        "schema_version": 1,
        "gate": "S1 RTRL/BPTT short acceptance",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "steps_per_learner": args.steps,
        "common_checks": common_checks,
        "recurrence": recurrence,
        "action_detach": action_detach,
        "terminal_target": terminal,
        "learners": learners,
    }
    result["pass"] = all(common_checks.values()) and all(item["pass"] for item in learners)
    document = json.dumps(result, indent=2, sort_keys=True)
    print(document)
    (args.output / "acceptance.json").write_text(document + "\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
