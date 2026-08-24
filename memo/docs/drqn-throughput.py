"""What a DRQN step costs, with replay in it -- the learner, not the buffer.

`episode-sampling-throughput.py` measures replay on its own, which is the right
instrument for a change to replay and the wrong one for the question the R1.1
sweep actually asks: how many steps a second does the run manage at
`replay.capacity: 400000`. This assembles the `drqn` entry through
`memorax.assembly` and times its `train`, so the answer includes the network,
the environment and the solver, and a difference between two checkouts is a
difference in the whole step.

The second pass is the one timed: the first compiles, and it also crosses the
replay warmup, where the update branch has not been taken yet.

Run it from a `memo` directory with `PYTHONPATH=.`, and read the first line it
prints. A script outside the tree leads `sys.path` with its own directory, so
without that it measures whichever memorax is installed rather than the
checkout it was run from -- which, between two worktrees, is the difference the
run was for.

    BENCH_CAPACITY=400000 BENCH_STEPS=3000 PYTHONPATH=. python docs/drqn-throughput.py

`gymnax::CartPole-v1` stands in for the sweep's `StatelessCartPoleEasy`, which
needs the `popjym` extra: the two differ in the observation the network reads
and in nothing that decides what a step costs.
"""

from __future__ import annotations

import os
import time

import jax

import memorax
from memorax.algorithms.drqn import DRQN
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble

CAPACITY = int(os.environ.get("BENCH_CAPACITY", "400000"))
STEPS = int(os.environ.get("BENCH_STEPS", "3000"))
EPISODE_LENGTH = int(os.environ.get("BENCH_EPISODE", "200"))

parameters = {
    "core.kind": "lstm",
    "core.lstm.hidden_dim": 32,
    "learning.kind": "truncated",
    "learning.truncated.length": 10,
    "optimizer.kind": "adadelta",
    "optimizer.adadelta.lr": 0.1,
    "optimizer.adadelta.rho": 0.95,
    "optimizer.adadelta.eps": 1e-8,
    "grad_clip": 10.0,
    "replay.capacity": CAPACITY,
    "replay.minimum_size": 256,
    "replay.batch_size": 8,
    "target.update_period": 100,
    "exploration.epsilon_start": 1.0,
    "exploration.epsilon_end": 0.05,
    "exploration.epsilon_decay_steps": 10000,
    "exploration.evaluation_epsilon": 0.0,
    "gamma": 0.99,
}

print(f"memorax: {memorax.__file__}", flush=True)
print(f"capacity={CAPACITY} steps={STEPS} episode_length={EPISODE_LENGTH}", flush=True)

built = assemble(
    DRQN,
    BuildRequest(
        parameters=parameters,
        environment=EnvironmentSpec(
            id="gymnax::CartPole-v1",
            backend=None,
            observed=None,
            episode_length=EPISODE_LENGTH,
        ),
        num_envs=1,
    ),
)
program = built.program

state = jax.block_until_ready(jax.jit(program.init)(jax.random.key(0)))
train = jax.jit(program.train, static_argnums=2)

start = time.perf_counter()
state, _ = jax.block_until_ready(train(jax.random.key(1), state, STEPS))
first = time.perf_counter() - start
print(f"compile+warmup pass: {first:8.2f} s", flush=True)

start = time.perf_counter()
state, _ = jax.block_until_ready(train(jax.random.key(2), state, STEPS))
steady = time.perf_counter() - start
print(f"steady pass:         {steady:8.2f} s  ->  {STEPS / steady:8.1f} steps/s")
print(f"per step:            {1e6 * steady / STEPS:8.1f} us")
