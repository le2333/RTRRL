"""What the GPU image aborts on, one variant at a time.

The image dies on ``free(): double free detected in tcache 2`` inside
``jax/_src/compiler.py:backend_compile_and_load`` while compiling ``train``.
Everything before that is fine: the L4 is found, a small matmul runs, and
``init`` compiles in nine seconds. So the double free is XLA's own, in the CUDA
plugin's compilation of this particular module, and what is worth knowing is
which part of the module provokes it.

This compiles ``train`` under one setting, named by the environment, and exits
zero if it survived. The driver beside it runs one process per setting so that
an abort takes down only its own variant, which is the only way to get a table
out of a crash in one job.

Needs no S3, no Aim and no manifest: the entry's ``build`` is called directly.
"""

from __future__ import annotations

import faulthandler
import os
import time
from types import SimpleNamespace

faulthandler.enable()

START = time.time()


def mark(what: str) -> None:
    print(f"  [{time.time() - START:6.1f}s] {what}", flush=True)


# The settings the CPU probe ran to completion on, so a difference here is the
# platform and not the configuration.
ENVIRONMENT = SimpleNamespace(
    id="brax::hopper",
    backend="spring",
    observed=(0, 1, 2, 3, 4),
    num_envs=16,
)

PARAMS = {
    "backbone": "rtu",
    "credit": "rtrl",
    "hidden_dim": 128,
    "feature_dim": 32,
    "meta_rl": True,
    "normalize_observation": True,
    "normalize_reward": True,
    "gamma": 0.99,
    "trace_lambda": 0.9060749966425912,
    "actor_lr": 0.532246925974616,
    "critic_lr": 0.11990103985952048,
    "actor_kappa": 2.722352816476039,
    "critic_kappa": 1.4401857247174306,
    "entropy_coefficient": 0.000271109456772485,
    "bounded_rule": "obgd",
    "beta2": 0.999,
    "eps": 1.0e-8,
}


def main() -> int:
    params = dict(PARAMS)
    params["credit"] = os.environ.get("PROBE_CREDIT", params["credit"])
    params["backbone"] = os.environ.get("PROBE_BACKBONE", params["backbone"])
    environment = SimpleNamespace(
        **vars(ENVIRONMENT)
        | {"num_envs": int(os.environ.get("PROBE_ENVS", ENVIRONMENT.num_envs))}
    )
    steps = int(os.environ.get("PROBE_STEPS", "16"))

    import jax

    mark(f"jax {jax.__version__} on {jax.default_backend()} {jax.devices()}")
    mark(f"credit={params['credit']} backbone={params['backbone']} "
         f"envs={environment.num_envs} steps={steps}")

    from entries import stream_ac

    agent = stream_ac.build(params, environment)
    mark("agent built")

    state = jax.block_until_ready(jax.jit(agent.init)(jax.random.key(0)))
    mark("init compiled")

    state, _ = jax.block_until_ready(
        jax.jit(agent.train, static_argnums=2)(jax.random.key(1), state, steps)
    )
    mark("train compiled and run")

    jax.block_until_ready(
        jax.jit(agent.evaluate, static_argnums=2)(jax.random.key(2), state, 8)
    )
    mark("evaluate compiled and run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
