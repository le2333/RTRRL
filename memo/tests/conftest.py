"""The environments and the numerical judgement the tests here share.

Two tests compare arrays that should agree to their last bits: one answers a
recorded snapshot, the other answers upstream's own arithmetic. They agree on
what "apart" means, and it is measured here rather than twice.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces


def flattened(tree) -> dict:
    """A pytree as leaf paths, spelled the way a snapshot spells them."""

    pairs, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {
        "/".join(str(getattr(key, "key", getattr(key, "idx", key))) for key in path): (
            np.asarray(leaf)
        )
        for path, leaf in pairs
    }


def last_bits(wanted, got) -> float:
    """How many float32 last bits apart two arrays are, at their own scale."""

    scale = max(float(np.abs(wanted).max()), float(np.abs(got).max()), 1e-6)
    gap = float(np.max(np.abs(got.astype(np.float64) - wanted.astype(np.float64))))
    return gap / float(np.spacing(np.float32(scale)))


def deviations(actual: dict, expected: dict, allowed: float = 0.0) -> list:
    """Every leaf further apart than allowed, worst first, in last bits."""

    missing = sorted(set(expected) - set(actual))
    assert not missing, f"nothing reports {missing}"

    found = []
    for path, wanted in expected.items():
        wanted = np.asarray(wanted)
        got = np.asarray(actual[path])
        assert got.shape == wanted.shape, f"{path}: {got.shape} not {wanted.shape}"
        if wanted.dtype.kind in "biufc" and got.dtype.kind in "biufc":
            bits = last_bits(wanted, got)
            if bits > allowed:
                found.append((bits, path))
        elif not np.array_equal(got, wanted):
            found.append((float("inf"), path))
    return sorted(found, reverse=True, key=lambda entry: entry[0])


def assert_within(
    actual: dict, expected: dict, what: str, *, allowed: float = 0.0
) -> int:
    """Every leaf of expected is within allowed last bits of actual's."""

    assert expected, f"{what}: there is nothing to compare"
    found = deviations(actual, expected, allowed)
    if found:
        listed = "\n".join(f"  {bits:.1f} last bits  {path}" for bits, path in found)
        raise AssertionError(
            f"{what}: {len(found)} of {len(expected)} leaves are more than "
            f"{allowed:g} last bits apart, worst first:\n{listed}"
        )
    return len(expected)


@struct.dataclass
class TinyEnvState:
    step_count: jnp.ndarray
    observation: jnp.ndarray


@struct.dataclass
class TinyEnvParams:
    horizon: int = struct.field(pytree_node=False, default=3)


@dataclass(frozen=True)
class TinyContinuousEnv:
    """A two-dimensional environment short enough to terminate during a test."""

    @property
    def default_params(self):
        return TinyEnvParams()

    def reset(self, key, params):
        del key, params
        obs = jnp.array([0.25, -0.5], dtype=jnp.float32)
        return obs, TinyEnvState(jnp.asarray(0, jnp.int32), obs)

    def step(self, key, state, action, params):
        del key
        action = jnp.asarray(action, jnp.float32)
        obs = state.observation + action + jnp.array([0.15, 0.45], jnp.float32)
        step_count = state.step_count + 1
        reward = jnp.asarray(0.4, jnp.float32) + 0.35 * step_count
        done = step_count >= params.horizon
        return (
            obs,
            TinyEnvState(step_count, obs),
            reward,
            done,
            {"step_count": step_count},
        )

    def action_space(self, params):
        del params
        return spaces.Box(-2.0, 2.0, (2,), dtype=jnp.float32)

    def observation_space(self, params):
        del params
        return spaces.Box(-10.0, 10.0, (2,), dtype=jnp.float32)


@dataclass(frozen=True)
class TinyDiscreteEnv:
    """The environment the recorded StreamAC snapshot was taken against.

    Its dynamics are part of that recording, so nothing here may change while
    the snapshot is still being answered against.
    """

    @property
    def default_params(self):
        return TinyEnvParams()

    def reset(self, key, params):
        del key, params
        obs = jnp.array([-0.25, 0.5], dtype=jnp.float32)
        return obs, TinyEnvState(jnp.asarray(0, jnp.int32), obs)

    def step(self, key, state, action, params):
        del key
        direction = jnp.where(action == 0, -1.0, 1.0)
        delta = jnp.array([0.2 * direction, 0.1 + 0.05 * direction], jnp.float32)
        obs = state.observation + delta
        step_count = state.step_count + 1
        reward = jnp.asarray(0.25 * direction + 0.1 * step_count, jnp.float32)
        done = step_count >= params.horizon
        return (
            obs,
            TinyEnvState(step_count, obs),
            reward,
            done,
            {"step_count": step_count},
        )

    def action_space(self, params):
        del params
        return spaces.Discrete(2)

    def observation_space(self, params):
        del params
        return spaces.Box(-10.0, 10.0, (2,), dtype=jnp.float32)
