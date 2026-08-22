"""What the GPU image compiles, and what it aborts on, one variant at a time.

The question this answers is whether XLA can compile this project's training
graph on a GPU. The last time anyone asked, the answer was a double free inside
the CUDA plugin's compilation of ``train`` -- ``free(): double free detected in
tcache 2``, no Python frame, because the aborting thread never entered the
interpreter -- while ``init`` compiled fine and a small matmul ran. That was a
different image and a different algorithm: both have been rebuilt since, so the
abort is not so much unfixed as unasked.

DRQN rather than the streaming learner. A GPU is paid for by arithmetic wide
enough to fill it, and DRQN's update is a batch of windows drawn from replay and
differentiated at once, where the streaming learner's parallel environments make
kernels too small to matter. If the GPU is worth anything here it is worth it on
this entry, and if it aborts here that is the abort worth reading.

The parameters below are ``experiments/drqn acceptance.yaml``'s space with its
single-element lists flattened -- a configuration that exists and is maintained,
rather than one assembled to suit a probe. The schema is flat: assembly reads
dotted paths, not nested mappings.

Each variant is named by the environment so that the driver can run one process
per setting and an abort takes down only its own, which is the only way to get a
table out of a crash in one job. ``init`` is compiled before ``train`` on
purpose: it is the control. An abort at ``init`` is the plugin or the driver, an
abort at ``train`` and not at ``init`` is this graph.

Needs no S3, no Aim and no manifest: assembly is called directly.
"""

from __future__ import annotations

import faulthandler
import os
import time

import jax

from memorax.algorithms.drqn import DRQN
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble

faulthandler.enable()

START = time.time()


def mark(what: str) -> None:
    print(f"  [{time.time() - START:6.1f}s] {what}", flush=True)


# `experiments/drqn acceptance.yaml`, environment block.
ENVIRONMENT = EnvironmentSpec(
    id="gymnax::CartPole-v1",
    backend=None,  # gymnax ships one implementation per environment
    observed=None,
    episode_length=32,
)

# The same file's space, flattened onto the dotted paths assembly reads.
PARAMETERS: dict[str, object] = {
    "core.kind": "lru",
    "core.lru.hidden_dim": 64,
    "core.lru.feature_dim": 64,
    "learning.kind": "truncated",
    "learning.truncated.length": 10,
    "optimizer.kind": "adadelta",
    "optimizer.adadelta.lr": 0.1,
    "optimizer.adadelta.rho": 0.95,
    "optimizer.adadelta.eps": 1e-8,
    "grad_clip": 10.0,
    "replay.capacity": 8192,
    "replay.minimum_size": 256,
    "replay.batch_size": 8,
    "target.update_period": 100,
    "exploration.epsilon_start": 1.0,
    "exploration.epsilon_end": 0.05,
    "exploration.epsilon_decay_steps": 10000,
    "exploration.evaluation_epsilon": 0.0,
    "gamma": 0.99,
}

# What one variant may move, and where it lands. Batch size and truncation are
# the two that change the shape of the differentiated window -- the arithmetic a
# GPU is here for, and the arithmetic an emitter has to fuse -- so they are the
# axes worth bisecting on. The core is here because an abort that follows the
# recurrent kernel rather than the batch is a different defect.
OVERRIDES = {
    "PROBE_BATCH": ("replay.batch_size", int),
    "PROBE_TRUNCATION": ("learning.truncated.length", int),
    "PROBE_CORE": ("core.kind", str),
}


def main() -> int:
    parameters = dict(PARAMETERS)
    for variable, (path, cast) in OVERRIDES.items():
        value = os.environ.get(variable)
        if value is not None:
            parameters[path] = cast(value)

    num_envs = int(os.environ.get("PROBE_ENVS", "1"))
    steps = int(os.environ.get("PROBE_STEPS", "512"))

    mark(f"jax {jax.__version__} on {jax.default_backend()} {jax.devices()}")
    mark(
        f"batch={parameters['replay.batch_size']} "
        f"truncation={parameters['learning.truncated.length']} "
        f"core={parameters['core.kind']} envs={num_envs} steps={steps}"
    )

    built = assemble(
        DRQN,
        BuildRequest(
            parameters=parameters, environment=ENVIRONMENT, num_envs=num_envs
        ),
    )
    mark("assembled")

    program = built.program
    state = jax.block_until_ready(jax.jit(program.init)(jax.random.key(0)))
    mark("init compiled")

    # Enough steps to pass replay's minimum size, so the update is compiled with
    # a batch actually drawn rather than skipped: a scan that never enters the
    # learning branch is not the graph this is asking about.
    state, _ = jax.block_until_ready(
        jax.jit(program.train, static_argnums=2)(jax.random.key(1), state, steps)
    )
    mark("train compiled and run")

    evaluation = jax.block_until_ready(
        jax.jit(program.open_evaluation)(jax.random.key(2), state)
    )
    mark("evaluation opened")

    jax.block_until_ready(
        jax.jit(program.evaluate, static_argnums=2)(jax.random.key(3), evaluation, 64)
    )
    mark("evaluate compiled and run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
