"""Run and continuously archive the handoff CTRNN BP+RFLO continuation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runner(round_root: Path):
    runner_path = next(
        (round_root / "artifacts/hcbase").glob(
            "rtrrl_halfcheetah_instability_probe_*/rtrrl_halfcheetah_runner_v1_20260809.py"
        )
    )
    spec = importlib.util.spec_from_file_location("hc_archived_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import archived runner: {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner, runner_path


def run_prepared(root: Path, config_path: Path) -> int:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import random

    config = json.loads(config_path.read_text())
    round_root = root / "round/hc_bp_round_20260810"
    sys.path.insert(0, str(round_root / "work"))
    from teaching_runner_lib import make_runner_class

    runner, runner_path = load_runner(round_root)
    continuation = config["continuation"]
    training = config["training"]
    evaluation = config["evaluation"]
    checkpoint = round_root / "results/exp3_bp_fromscratch_3seed/batch_checkpoints/step_0300000.pkl"
    actual_checkpoint_sha = sha256_file(checkpoint)
    if actual_checkpoint_sha != continuation["checkpoint_sha256"]:
        raise RuntimeError(
            f"checkpoint checksum mismatch: {actual_checkpoint_sha} != "
            f"{continuation['checkpoint_sha256']}"
        )
    carry, payload = runner.load_checkpoint(checkpoint)
    start = int(payload["step"])
    if start != int(continuation["start_step"]):
        raise RuntimeError(f"checkpoint starts at {start}, expected {continuation['start_step']}")

    target = int(os.environ.get("RTRRL_HC_TARGET_STEP", continuation["target_step"]))
    chunk_steps = int(training["chunk_steps"])
    if target <= start or (target - start) % chunk_steps:
        raise ValueError("target must be above start and aligned to chunk_steps")

    checkpoint_config = dict(payload["config"])
    algorithm = config["algorithm"]
    expected = {
        "H": algorithm["hidden_size"],
        "gamma": algorithm["gamma"],
        "lambda": algorithm["lambda_rnn"],
        "lrA": algorithm["actor_lr"],
        "lrC": algorithm["critic_lr"],
        "lrR": algorithm["rnn_lr"],
        "clip": algorithm["update_clip"],
        "backend": config["environment"]["backend"],
        "mask": config["environment"]["observation_mask"],
    }
    mismatches = {
        key: {"checkpoint": checkpoint_config.get(key), "requested": value}
        for key, value in expected.items()
        if checkpoint_config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"checkpoint/config mismatch: {json.dumps(mismatches, sort_keys=True)}")

    runner_class = make_runner_class(
        runner.Runner,
        "bp",
        "bp",
        float(algorithm["actor_teaching_scale"]),
        float(algorithm["critic_teaching_scale"]),
    )
    learner = runner_class(seed=1, config=checkpoint_config)

    def batched_step(batch_carry, _):
        return jax.vmap(lambda item: learner.step(item, None))(batch_carry)

    train_chunk = jax.jit(lambda batch_carry: jax.lax.scan(
        batched_step, batch_carry, None, length=chunk_steps
    ))

    eval_count = int(evaluation["episodes"])
    eval_horizon = int(evaluation["horizon"])
    env_keys = random.split(random.PRNGKey(int(evaluation["environment_seed"])), eval_count)
    action_keys = random.split(random.PRNGKey(int(evaluation["action_noise_seed"])), eval_count)

    def evaluate_one(W, tau, W_actor):
        env = learner.env
        states = jax.vmap(env.reset)(env_keys)
        hidden = jnp.zeros((eval_count, learner.H), jnp.float32)
        x = jnp.concatenate(
            [states.obs[:, learner.obs_mask], jnp.zeros((eval_count, learner.A + 1), jnp.float32)],
            axis=1,
        )

        def recurrent(h, inputs):
            features = jnp.concatenate(
                [inputs, h, jnp.ones((eval_count, 1), jnp.float32)], axis=1
            )
            activation = jnp.tanh(features @ W.T)
            return h + (activation - h) / tau

        def sample(h, keys):
            output = h @ W_actor
            location, raw_scale = jnp.split(output, 2, axis=-1)
            log_scale = -2 + 4 * jax.nn.sigmoid(raw_scale)
            scale = jax.nn.softplus(log_scale)
            noise = jax.vmap(lambda key: random.normal(key, (learner.A,)))(keys)
            return location + scale * noise

        hidden = recurrent(hidden, x)
        split = jax.vmap(lambda key: random.split(key))(action_keys)
        keys = split[:, 0, :]
        action = sample(hidden, split[:, 1, :])
        returns = jnp.zeros((eval_count,), jnp.float32)

        def eval_step(state, _):
            env_state, h, previous_action, keys, episode_return = state
            clipped_action = jnp.clip(previous_action, -1, 1)
            next_state = jax.vmap(env.step)(env_state, clipped_action)
            episode_return = episode_return + next_state.reward
            inputs = jnp.concatenate(
                [next_state.obs[:, learner.obs_mask], clipped_action, next_state.reward[:, None]],
                axis=1,
            )
            h = recurrent(h, inputs)
            split_keys = jax.vmap(lambda key: random.split(key))(keys)
            keys = split_keys[:, 0, :]
            action = sample(h, split_keys[:, 1, :])
            return (next_state, h, action, keys, episode_return), None

        (*_, returns), _ = jax.lax.scan(
            eval_step,
            (states, hidden, action, keys, returns),
            None,
            length=eval_horizon,
        )
        return returns

    evaluate = jax.jit(evaluate_one)
    output = root / "output"
    checkpoint_dir = output / "checkpoints"
    evaluation_dir = output / "fixed_eval"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "algorithm": "ctrnn-bp-rflo",
        "start_step": start,
        "target_step": target,
        "seeds": continuation["seeds"],
        "source_checkpoint_sha256": actual_checkpoint_sha,
        "runner_sha256": sha256_file(runner_path),
        "config": config,
        "host": {"platform": platform.platform(), "processor": platform.processor()},
        "checkpoints": {},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    current = start
    while current < target:
        before = time.time()
        carry, (metrics, finite) = train_chunk(carry)
        jax.block_until_ready(metrics)
        current += chunk_steps
        finite_host = np.asarray(jax.device_get(finite))
        record = {
            "start": current - chunk_steps,
            "step": current,
            "steps": chunk_steps,
            "wall_sec": time.time() - before,
            "finite_fraction": float(finite_host.mean()),
            "seeds": [],
        }
        for index, seed in enumerate(continuation["seeds"]):
            summary = runner.summarize_metrics(metrics[:, index, :], finite[:, index])
            record["seeds"].append({"seed": seed, **summary})
        with (output / "train_chunks.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

        checkpoint_path = checkpoint_dir / f"step_{current:07d}.pkl"
        saved_config = dict(
            checkpoint_config,
            teaching_actor="bp",
            teaching_critic="bp",
            actor_teaching_scale=algorithm["actor_teaching_scale"],
            critic_teaching_scale=algorithm["critic_teaching_scale"],
            bp_definition="exact current-step output-to-hidden teaching; RFLO through time; no BPTT",
        )
        runner.save_checkpoint(
            checkpoint_path,
            carry,
            current,
            saved_config,
            sha256_file(Path(__file__)),
        )
        manifest["checkpoints"][str(current)] = {
            "file": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
        }

        if current % int(evaluation["every_steps"]) == 0:
            evaluation_record = {"step": current, "seeds": []}
            for index, seed in enumerate(continuation["seeds"]):
                seed_carry = jax.tree.map(lambda value: value[index], carry)
                returns = np.asarray(
                    jax.device_get(evaluate(seed_carry[1], seed_carry[2], seed_carry[3]))
                )
                evaluation_record["seeds"].append(
                    {
                        "seed": seed,
                        "mean": float(returns.mean()),
                        "median": float(np.median(returns)),
                        "std": float(returns.std()),
                        "returns": returns.tolist(),
                    }
                )
            evaluation_path = evaluation_dir / f"step_{current:07d}.json"
            evaluation_path.write_text(json.dumps(evaluation_record, indent=2))
            with (output / "fixed_eval.jsonl").open("a") as stream:
                stream.write(json.dumps(evaluation_record) + "\n")

        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(json.dumps({"event": "chunk_complete", **record}), flush=True)
        if training["stop_on_nonfinite"] and record["finite_fraction"] != 1.0:
            raise RuntimeError(f"non-finite learner state at step {current}")
    return 0


def upload_changed(s3, bucket: str, prefix: str, output: Path, uploaded: dict[str, str]) -> None:
    if not output.exists():
        return
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        digest = sha256_file(path)
        if uploaded.get(relative) == digest:
            continue
        s3.upload_file(str(path), bucket, f"{prefix}/{relative}")
        uploaded[relative] = digest
        print(json.dumps({"event": "artifact_uploaded", "path": relative, "sha256": digest}), flush=True)


def main() -> int:
    prepared_root = os.environ.get("RTRRL_HC_PREPARED_ROOT")
    prepared_config = os.environ.get("RTRRL_HC_CONFIG_JSON")
    if prepared_root and prepared_config:
        return run_prepared(Path(prepared_root), Path(prepared_config))

    import boto3
    import yaml

    from hc_cpu_repro_gate import create_exact_environment, download_and_extract, prepare_workspace

    config_yaml = Path(os.environ.get("RTRRL_HC_CONFIG", "/opt/rtrrl/hc_ctrnn_continuation.yml"))
    config = yaml.safe_load(config_yaml.read_text())
    bucket, key_prefix = config["artifacts"]["output_prefix"].removeprefix("s3://").split("/", 1)
    override = os.environ.get("RTRRL_HC_OUTPUT_PREFIX")
    if override:
        bucket, key_prefix = override.removeprefix("s3://").split("/", 1)

    root = Path(tempfile.mkdtemp(prefix="rtrrl-hc-ctrnn-"))
    extracted = download_and_extract(root)
    prepare_workspace(root, extracted)
    exact_python = create_exact_environment(root, extracted["wheels"])
    config_json = root / "config.json"
    config_json.write_text(json.dumps(config, sort_keys=True))
    environment = os.environ.copy()
    environment["RTRRL_HC_PREPARED_ROOT"] = str(root)
    environment["RTRRL_HC_CONFIG_JSON"] = str(config_json)
    process = subprocess.Popen([str(exact_python), str(Path(__file__).resolve())], env=environment)
    s3 = boto3.client("s3", region_name="eu-north-1")
    uploaded: dict[str, str] = {}
    while process.poll() is None:
        upload_changed(s3, bucket, key_prefix, root / "output", uploaded)
        time.sleep(5)
    upload_changed(s3, bucket, key_prefix, root / "output", uploaded)
    return int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
