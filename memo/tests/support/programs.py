"""A deterministic JAX program for Runtime scheduling and rollout tests."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

NUM_ENVS = 2
EPOCH_STEPS = 8
TOTAL_STEPS = 16

TRAIN_REWARD = jnp.array([[1.0, 5.0], [3.0, 5.0], [7.0, 5.0], [9.0, 5.0]])
TRAIN_DONE = jnp.array([[False, False], [True, False], [False, True], [False, False]])
TRAIN_LOSS = jnp.array([[0.0, 0.0], [2.0, 1.0], [4.0, 1.0], [6.0, 1.0]])
# A run that walks its training episodes keeps the transition either side of
# each step, which is the shape the schema's trajectory paths name.
TRAIN_OBSERVATION = jnp.arange(4 * NUM_ENVS, dtype=jnp.float32).reshape(4, NUM_ENVS)[
    ..., None
]

EVAL_REWARD = jnp.array([[2.0, 0.0], [4.0, 0.0], [0.0, 0.0]])
EVAL_DONE = jnp.array([[False, False], [True, False], [False, False]])
EVAL_ROWS = EVAL_DONE.shape[0]
# One evaluation call's step budget, which is the whole rollout here: the one
# episode the arithmetic ends lands inside it, so nothing depends on how the
# driver would have chunked a longer one.
EVAL_CHUNK_STEPS = EVAL_ROWS * NUM_ENVS
# Only stream zero reaches an ending, so a checkpoint scored on more than one
# episode would never finish against this arithmetic.
EVAL_EPISODES = 1

# One vectorized transition, with no leading time axis, that ends both streams.
INTERACT_REWARD = jnp.array([11.0, 13.0])
INTERACT_DONE = jnp.array([True, True])
INTERACT_LOSS = jnp.array([8.0, 9.0])

SERIES = ("loss", "by_part.torso")


class InteractionMetrics(NamedTuple):
    reward: Any
    done: Any
    terminal: Any = None
    observation: Any = None
    next_observation: Any = None
    action: Any = None
    paid: Any = None


class Metrics(NamedTuple):
    interaction: InteractionMetrics
    loss: Any = None
    by_part: Any = None
    only_sometimes: Any = None


def arithmetic_program():
    """Return the five program arrows without an algorithm dependency."""

    def init_fn(key):
        del key
        return jnp.asarray(0.0)

    def open_evaluation_fn(key, state):
        """A rollout of its own, which is all the driver may reach through."""

        del key, state
        return jnp.asarray(0.0)

    def train_fn(key, state, num_steps):
        del key, num_steps
        return state + EPOCH_STEPS, Metrics(
            interaction=InteractionMetrics(
                observation=TRAIN_OBSERVATION,
                next_observation=TRAIN_OBSERVATION + 1,
                action=TRAIN_OBSERVATION,
                reward=TRAIN_REWARD,
                done=TRAIN_DONE,
                terminal=TRAIN_DONE,
                paid=TRAIN_REWARD / 10,
            ),
            loss=TRAIN_LOSS,
            by_part={"torso": TRAIN_LOSS},
        )

    def evaluate_fn(key, state, num_steps):
        """Advance the opened rollout, in environment steps as training is."""

        del key
        rows = num_steps // NUM_ENVS
        column = jnp.arange(rows * NUM_ENVS, dtype=jnp.float32)
        grid = column.reshape(rows, NUM_ENVS)
        return state + num_steps, Metrics(
            interaction=InteractionMetrics(
                observation=grid[..., None],
                next_observation=grid[..., None] + 1,
                action=grid[..., None],
                reward=EVAL_REWARD[:rows],
                done=EVAL_DONE[:rows],
                terminal=EVAL_DONE[:rows],
                paid=EVAL_REWARD[:rows] / 10,
            ),
        )

    def interact_fn(key, state):
        """One transition that learns nothing, so the step budget stays put."""

        del key
        return state, Metrics(
            interaction=InteractionMetrics(
                reward=INTERACT_REWARD,
                done=INTERACT_DONE,
                terminal=INTERACT_DONE,
                paid=INTERACT_REWARD / 10,
            ),
            loss=INTERACT_LOSS,
            by_part={"torso": INTERACT_LOSS},
        )

    return init_fn, train_fn, open_evaluation_fn, evaluate_fn, interact_fn
