"""What a DRQN update spends on choosing which episodes to replay.

DRQN updates once per environment transition, so every part of the replay
buffer's per-update work is multiplied by the length of the run. The question
this answers is which parts of it grow with replay's *capacity*, which is the
one number a long run raises -- the R1.1 throughput sweep asks for 400k, fifty
times the acceptance file's 8192 -- and whether the growth lands on
compilation, on the steps before learning starts, or on the steady state,
because those are three different failures and only the last is a slow run.

Everything is measured inside a ``lax.scan`` over transitions, which is where
the learner runs it. Timing one jitted ``add`` per call would measure XLA
copying the whole ring out to a fresh buffer, an O(capacity) cost the training
loop does not pay because its state is a scan carry that is updated in place;
it would report the buffer as fifty times slower at fifty times the capacity
and none of it would be real. Python dispatch is out of the numbers for the
same reason -- one call per pass, not one per transition.

Four measurements per capacity:

``compile``       tracing and compiling the pass. Reported on its own because
                  a large operation inside the learner's ``lax.cond`` lands
                  here before it lands on any step, and a benchmark that has
                  not reached its first evaluation may be waiting on this
                  rather than running slowly.
``warmup``        ``add`` per transition, which is the whole of replay's cost
                  before learning starts and stays underneath it after.
``steady state``  ``add``, ``can_sample`` and a draw under the ``cond`` that
                  reads it, on a wrapped buffer: the per-transition replay
                  cost of the published one-update-per-step loop.
``selection``     the two ways of choosing B episodes, timed apart from the
                  gather around them so the change is visible on its own:
                  ``scored`` is the sampler this replaced -- a Gumbel per
                  episode over the whole index, then ``top_k`` -- and
                  ``ranked`` is Floyd's algorithm over the count of eligible
                  episodes, which is what the buffer does now.

Run it from ``memo`` so the checkout under test is the one imported -- a
script's own directory leads ``sys.path``, so ``python docs/...`` from
anywhere else measures whichever memorax is installed:

    PYTHONPATH=. python docs/episode-sampling-throughput.py

``SAMPLING_CAPACITIES`` overrides the capacities and ``SAMPLING_STEPS`` the
transitions per pass. CPU is the platform that matters here: this is index
arithmetic, and the sweep it was written for runs the learner on CPU.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import struct

from memorax.buffers import make_uniform_episode_window_buffer
from memorax.utils.typing import Array, Key

EPISODE_LENGTH = 100
WINDOW = 10
BATCH = 8


class Step(struct.PyTreeNode):
    observation: Array
    done: Array


@dataclass(frozen=True)
class Timing:
    compile_seconds: float
    step_microseconds: float


def timed(function: Callable[..., Any], *arguments: Any, steps: int) -> Timing:
    """Compilation once, then a rate over the transitions of one pass."""

    compiled = jax.jit(function)
    first = time.perf_counter()
    jax.block_until_ready(compiled(*arguments))
    compilation = time.perf_counter() - first

    start = time.perf_counter()
    jax.block_until_ready(compiled(*arguments))
    return Timing(compilation, 1e6 * (time.perf_counter() - start) / steps)


def stream_of(transitions: int) -> Step:
    """Episodes of a fixed length, laid end to end on one stream."""

    steps = jnp.arange(transitions, dtype=jnp.int32)
    return Step(
        observation=jnp.zeros((transitions, 1, 4), jnp.float32),
        done=((steps + 1) % EPISODE_LENGTH == 0)[:, None],
    )


def stored(buffer, transitions: int):
    """A buffer past its own wrap, so eviction is live and the index is full."""

    state = buffer.init(
        Step(observation=jnp.zeros((4,), jnp.float32), done=jnp.asarray(False))
    )
    return jax.lax.scan(
        lambda carry, step: (buffer.add(carry, step), None),
        state,
        stream_of(transitions),
    )[0]


def warming(buffer, state, stream):
    """Storing transitions and nothing else: what a run does before it learns."""

    return jax.lax.scan(
        lambda carry, step: (buffer.add(carry, step), None), state, stream
    )[0].written


def updating(buffer, state, stream, keys):
    """One transition stored and one minibatch drawn, as the published loop has it.

    The draw sits under the same ``lax.cond`` the learner reads ``can_sample``
    through, and only a scalar leaves the scan, so what is timed is the replay
    path rather than a copy of the buffer.
    """

    def step(carry, item):
        state, total = carry
        transition, key = item
        state = buffer.add(state, transition)
        drawn = jax.lax.cond(
            buffer.can_sample(state),
            lambda: jnp.sum(buffer.sample(state, key).experience.observation),
            lambda: jnp.asarray(0.0),
        )
        return (state, total + drawn), None

    return jax.lax.scan(step, (state, jnp.asarray(0.0)), (stream, keys))[0][1]


def scored(key: Key, eligible: Array, batch: int) -> Array:
    """The previous selection: a score per episode in the index, then ``top_k``."""

    scores = jnp.where(eligible, jax.random.gumbel(key, eligible.shape), -jnp.inf)
    return jax.lax.top_k(scores, batch)[1]


def ranked(key: Key, total: Array, batch: int) -> Array:
    """The present selection: Floyd's algorithm, in the size of the minibatch."""

    def take(carry, step):
        held, filled = carry
        position, step_key = step
        bound = total - batch + position
        candidate = jax.random.randint(
            step_key, (), 0, jnp.maximum(bound + 1, 1), dtype=jnp.int32
        )
        seen = jnp.any(filled & (held == candidate))
        return (
            held.at[position].set(jnp.where(seen, bound, candidate)),
            filled.at[position].set(bound >= 0),
        ), None

    return jax.lax.scan(
        take,
        (jnp.zeros((batch,), jnp.int32), jnp.zeros((batch,), bool)),
        (jnp.arange(batch, dtype=jnp.int32), jax.random.split(key, batch)),
    )[0][0]


def selecting(draw, argument, steps: int):
    """One selection per transition, so it is comparable with the columns above."""

    return jax.lax.scan(
        lambda total, key: (total + jnp.sum(draw(key, argument)), None),
        jnp.asarray(0, jnp.int32),
        jax.random.split(jax.random.key(0), steps),
    )[0]


def report(capacities: list[int], steps: int) -> None:
    rows = []
    for capacity in capacities:
        buffer = make_uniform_episode_window_buffer(
            max_length=capacity,
            min_length=capacity // 4,
            sample_batch_size=BATCH,
            sample_sequence_length=WINDOW,
            add_batch_size=1,
            max_episode_length=EPISODE_LENGTH,
        )
        state = stored(buffer, capacity + 2 * EPISODE_LENGTH)
        stream = stream_of(steps)
        keys = jax.random.split(jax.random.key(0), steps)
        # As many eligible episodes as the capacity holds, which is what the
        # sampler this replaced would have had to score.
        index = jnp.arange(capacity + EPISODE_LENGTH) < (capacity // EPISODE_LENGTH)
        total = jnp.asarray(capacity // EPISODE_LENGTH, jnp.int32)

        rows.append(
            (
                capacity,
                timed(lambda s: warming(buffer, s, stream), state, steps=steps),
                timed(lambda s: updating(buffer, s, stream, keys), state, steps=steps),
                timed(
                    lambda eligible: selecting(
                        lambda key, mask: scored(key, mask, BATCH), eligible, steps
                    ),
                    index,
                    steps=steps,
                ),
                timed(
                    lambda count: selecting(
                        lambda key, size: ranked(key, size, BATCH), count, steps
                    ),
                    total,
                    steps=steps,
                ),
            )
        )

    print(f"device: {jax.devices()[0].platform}, {steps} transitions per pass")
    print(f"episodes of {EPISODE_LENGTH}, window {WINDOW}, minibatch {BATCH}\n")
    heads = ("warmup", "steady state", "scored", "ranked")
    print("| capacity | " + " | ".join(f"{head} us/step" for head in heads) + " |")
    print("| ---: | " + " | ".join(["---:"] * len(heads)) + " |")
    for capacity, *timings in rows:
        cells = " | ".join(f"{timing.step_microseconds:.2f}" for timing in timings)
        print(f"| {capacity} | {cells} |")

    print("\n| capacity | " + " | ".join(f"{head} compile s" for head in heads) + " |")
    print("| ---: | " + " | ".join(["---:"] * len(heads)) + " |")
    for capacity, *timings in rows:
        cells = " | ".join(f"{timing.compile_seconds:.2f}" for timing in timings)
        print(f"| {capacity} | {cells} |")


if __name__ == "__main__":
    declared = os.environ.get("SAMPLING_CAPACITIES", "8192,65536,400000")
    report(
        [int(capacity) for capacity in declared.split(",")],
        int(os.environ.get("SAMPLING_STEPS", "2000")),
    )
