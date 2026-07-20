"""Import-compatible RTRRL façade backed by one composed ``AgentProgram``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import flax.linen as nn
from flax import core, struct
import jax
import jax.numpy as jnp

from memorax.networks.sequence_models.memoroid import Memoroid
from memorax.utils import Timestep
from memorax.utils.typing import (
    Array,
    Carry,
    EnvParams,
    EnvState,
    Environment,
)


def _tree_norm(tree) -> Array:
    """L2 norm over all leaves of a possibly complex pytree."""

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves))


def _find_leaf(tree, name):
    """Return the first leaf whose flatten path contains ``name``."""

    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if any(getattr(key, "key", None) == name for key in path):
            return leaf
    return None


@struct.dataclass(frozen=True)
class RTRRLConfig:
    """Historical constructor schema retained for existing Memorax callers."""

    num_envs: int
    gamma: float = 0.95
    lambda_pi: float = 0.97
    lambda_v: float = 0.9
    lambda_rnn: float = 0.945
    td_lr: float = 3e-5
    rnn_lr: float = 2e-6
    eta_pi: float = 0.38
    eta_f: float = 0.5
    entropy_rate: float = 3e-5
    update_period: float = 0.1
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    rnn_grad_clip: float = 1.0
    act_clip: float = 0.0
    freeze_gamma: bool = False
    update_trace_before_td: bool = True
    logprob_scale: float = 1.0
    pred_obs: bool = False
    pred_coeff: float = 1.0
    normalize_obs: bool = False
    normalize_reward: bool = False


@struct.dataclass(frozen=True)
class RTRRLState:
    """Historical state name retained for import and annotation compatibility."""

    step: int
    update_step: int
    timestep: Timestep
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    slow_torso: core.FrozenDict[str, Any]
    traces: core.FrozenDict[str, Any]
    opt_state: Any
    carry: Carry
    sensitivity: Any
    I: Array  # noqa: E741 - historical public state field


@dataclass
class RTRRL:
    """Legacy lifecycle methods delegating to one lazily composed program."""

    cfg: RTRRLConfig
    env: Environment
    env_params: EnvParams
    feature_extractor: nn.Module
    torso: Memoroid
    actor_head: nn.Module
    critic_head: nn.Module
    pred_head: nn.Module | None = None
    activation: Callable = jax.nn.silu
    program_normalization: Any = None
    strip_environment_normalization: bool = False
    _delegate: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from memorax.online_ac import (
            LegacyProgram,
            MetaProgramConfig,
            build_meta_program,
            legacy_env_adapter,
        )

        config = MetaProgramConfig.from_legacy_parts(
            self,
            normalization=self.program_normalization,
        )
        adapter = legacy_env_adapter(
            self.env,
            self.env_params,
            strip_normalization=self.strip_environment_normalization,
        )
        self._delegate = LegacyProgram(
            build_meta_program(config, adapter),
            config,
        )

    def as_legacy_program(self):
        """Return the one program façade selected during construction."""

        return self._delegate

    def init(self, key):
        return self._delegate.init(key)

    def warmup(self, key, state, num_steps):
        return self._delegate.warmup(key, state, num_steps)

    def train(self, key, state, num_steps):
        return self._delegate.train(key, state, num_steps)

    def evaluate(self, key, state, num_steps):
        return self._delegate.evaluate(key, state, num_steps)

    def _update_step(self, state, key):
        """Compatibility shim; the update itself remains owned by the program."""

        return self._delegate.train(key, state, 1), None


__all__ = [
    "RTRRL",
    "RTRRLConfig",
    "RTRRLState",
    "_find_leaf",
    "_tree_norm",
]
