"""What the GPU image compiles, and what it aborts on, one variant at a time.

The question this answers is whether XLA can compile this project's training
graphs on a GPU. The last time anyone asked, the answer was a double free inside
the CUDA plugin's compilation of ``train`` -- ``free(): double free detected in
tcache 2``, no Python frame, because the aborting thread never entered the
interpreter -- while ``init`` compiled fine and a small matmul ran.

That reading was taken on RTRRL, whose torso differentiates by carrying a
jacobian through a scan, and it is the graph a rebuilt image has to be asked
about before the defect can be called gone. DRQN compiling is not evidence about
it: different core, different credit assignment, different fusions.

Each subject's parameters are an experiment file's own space with its
single-element lists flattened, named in ``source`` -- a configuration that
exists and is maintained, rather than one assembled to suit a probe. The schema
is flat: assembly reads dotted paths, not nested mappings.

``init`` is compiled before ``train`` on purpose. It is the control: an abort at
``init`` is the plugin or the driver, an abort at ``train`` and not at ``init``
is the graph. Each variant is named by the environment so a driver can run one
process per setting and an abort takes down only its own, which is the only way
to get a table out of a crash in one job.

Needs no S3, no Aim and no manifest: assembly is called directly.

    PROBE_ENTRY=rtrrl PROBE_DIFFERENTIATION=tbptt python gpu-abort-probe.py

``PROBE_DIFFERENTIATION`` is the one worth walking first on RTRRL: it selects
between the jacobian-in-scan that aborted and truncated backpropagation, so a
crash on one and not the other names the defect rather than guessing at it.
"""

from __future__ import annotations

import faulthandler
import importlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import jax

from memorax.assembly import BuildRequest, EnvironmentSpec, assemble

faulthandler.enable()

START = time.time()


def mark(what: str) -> None:
    print(f"  [{time.time() - START:6.1f}s] {what}", flush=True)


@dataclass(frozen=True)
class Subject:
    """One entry's graph, as an experiment file already spells it."""

    module: str
    attribute: str
    source: str
    environment: EnvironmentSpec
    parameters: dict[str, Any]
    # Environment variable -> the dotted path it moves and how to read it. Only
    # what changes the shape of the compiled arithmetic is here; a learning rate
    # cannot provoke a fusion the graph did not already have.
    axes: dict[str, tuple[str, Callable[[str], Any]]] = field(default_factory=dict)

    def definition(self) -> Any:
        return getattr(importlib.import_module(self.module), self.attribute)


SUBJECTS: dict[str, Subject] = {
    "drqn": Subject(
        module="memorax.algorithms.drqn",
        attribute="DRQN",
        source="experiments/drqn acceptance.yaml",
        environment=EnvironmentSpec(
            id="gymnax::CartPole-v1",
            backend=None,  # gymnax ships one implementation per environment
            observed=None,
            episode_length=32,
        ),
        parameters={
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
        },
        axes={
            # Batch size and truncation are the shape of the differentiated
            # window: the arithmetic a GPU is here for, and the arithmetic an
            # emitter has to fuse.
            "PROBE_BATCH": ("replay.batch_size", int),
            "PROBE_TRUNCATION": ("learning.truncated.length", int),
            "PROBE_CORE": ("core.kind", str),
            "PROBE_HIDDEN": ("core.lru.hidden_dim", int),
        },
    ),
    "rtrrl": Subject(
        module="memorax.algorithms.rtrrl_aaai",
        attribute="RTRRL",
        source="experiments/protocol brax smoke.yaml",
        environment=EnvironmentSpec(
            id="brax::halfcheetah",
            backend="spring",
            observed=None,
            episode_length=64,
        ),
        parameters={
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 32,
            "torso.backbone.lru.hidden_dim": 32,
            "torso.backbone.lru.differentiation.kind": "exact_rtrl",
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 0.001,
            # Adam's three constants are here and absent from the space
            # file: the search declares them as single-valued, so the HPO
            # layer materializes them and a file that picks nothing need
            # not name them. Assembly is below that layer and requires
            # every leaf, so the probe carries the schema's own values.
            "torso.optimizer.adam.b1": 0.9,
            "torso.optimizer.adam.b2": 0.999,
            "torso.optimizer.adam.eps": 1e-8,
            "torso.grad_clip": 1.0,
            "torso.follow": 1.0,
            "actor.optimizer.kind": "adam",
            "actor.optimizer.adam.lr": 0.001,
            "actor.optimizer.adam.b1": 0.9,
            "actor.optimizer.adam.b2": 0.999,
            "actor.optimizer.adam.eps": 1e-8,
            "actor.head.kind": "state_std",
            "actor.head.state_std.initialization.kind": "lecun",
            "critic.optimizer.kind": "adam",
            "critic.optimizer.adam.lr": 0.001,
            "critic.optimizer.adam.b1": 0.9,
            "critic.optimizer.adam.b2": 0.999,
            "critic.optimizer.adam.eps": 1e-8,
            "critic.head.kind": "value",
            "critic.head.value.initialization.kind": "lecun",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "gamma": 0.99,
            "lambda_pi": 0.99,
            "lambda_v": 0.99,
            "lambda_rnn": 0.99,
            "eta_pi": 1.0,
            "eta_f": 1.0,
            "entropy_rate": 0.00001,
            "meta_rl": False,
        },
        axes={
            # The one that matters: exact_rtrl carries a jacobian through the
            # scan and tbptt does not, so this is the axis the July abort sat
            # on. Hidden width is here because the jacobian is square in it.
            "PROBE_DIFFERENTIATION": (
                "torso.backbone.lru.differentiation.kind",
                str,
            ),
            "PROBE_HIDDEN": ("torso.backbone.lru.hidden_dim", int),
            "PROBE_FEATURE": ("torso.backbone.lru.feature_dim", int),
            "PROBE_CORE": ("torso.backbone.kind", str),
        },
    ),
}


def main() -> int:
    name = os.environ.get("PROBE_ENTRY", "drqn")
    if name not in SUBJECTS:
        raise SystemExit(
            f"PROBE_ENTRY={name!r} is not one of {sorted(SUBJECTS)}"
        )
    subject = SUBJECTS[name]

    parameters = dict(subject.parameters)
    moved = []
    for variable, (path, cast) in subject.axes.items():
        value = os.environ.get(variable)
        if value is not None:
            parameters[path] = cast(value)
            moved.append(f"{path}={parameters[path]}")

    num_envs = int(os.environ.get("PROBE_ENVS", "1"))
    steps = int(os.environ.get("PROBE_STEPS", "512"))

    backend = jax.default_backend()
    devices = jax.devices()
    mark(f"jax {jax.__version__} on {backend} {devices}")
    # A CUDA plugin that failed to initialize does not stop the process: it
    # falls back, and this runs to completion on the CPU of a GPU instance
    # while reporting that nothing aborted. That reading is worse than a
    # crash, because it is wrong and it looks like an answer. The only run
    # worth paying an instance for is one that reached the device.
    if backend != "gpu" and not os.environ.get("PROBE_ALLOW_CPU"):
        raise SystemExit(
            f"probe landed on {backend!r}, not a GPU: {devices}. "
            "Set PROBE_ALLOW_CPU=1 to compile this graph on the CPU on purpose."
        )

    mark(f"entry {name} ({subject.source})")
    mark(f"envs={num_envs} steps={steps} moved={moved or 'nothing'}")

    built = assemble(
        subject.definition(),
        BuildRequest(
            parameters=parameters,
            environment=subject.environment,
            num_envs=num_envs,
        ),
    )
    mark("assembled")

    program = built.program
    state = jax.block_until_ready(jax.jit(program.init)(jax.random.key(0)))
    mark("init compiled")

    # Enough steps that a learner gated on a warmup -- DRQN's replay minimum --
    # has actually entered its update branch: a scan compiled without it is not
    # the graph this is asking about.
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
