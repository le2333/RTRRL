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

EVAL_REWARD = jnp.array([[2.0, 0.0], [4.0, 0.0], [0.0, 0.0]])
EVAL_DONE = jnp.array([[False, False], [True, False], [False, False]])
EVAL_STEPS = EVAL_DONE.shape[0]

SERIES = ("loss", "by_part.torso", "only_sometimes")


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
    """Return init/train/evaluate functions without an algorithm dependency."""

    def init_fn(key):
        del key
        return jnp.asarray(0.0)

    def train_fn(key, state, num_steps):
        del key, num_steps
        return state + EPOCH_STEPS, Metrics(
            interaction=InteractionMetrics(
                reward=TRAIN_REWARD,
                done=TRAIN_DONE,
                terminal=TRAIN_DONE,
                paid=TRAIN_REWARD / 10,
            ),
            loss=TRAIN_LOSS,
            by_part={"torso": TRAIN_LOSS},
        )

    def evaluate_fn(key, state, num_steps):
        del key
        column = jnp.arange(num_steps * NUM_ENVS, dtype=jnp.float32)
        grid = column.reshape(num_steps, NUM_ENVS)
        return state, Metrics(
            interaction=InteractionMetrics(
                observation=grid[..., None],
                next_observation=grid[..., None] + 1,
                action=grid[..., None],
                reward=EVAL_REWARD,
                done=EVAL_DONE,
                terminal=EVAL_DONE,
                paid=EVAL_REWARD / 10,
            )
        )

    return init_fn, train_fn, evaluate_fn
