"""Import-compatible RTRRL façade backed by one composed ``AgentProgram``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp

from memorax.networks.sequence_models.memoroid import Memoroid
from memorax.utils.typing import (
    Array,
    EnvParams,
    Environment,
)

from .types import RTRRLState


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
    act_magnitude_factor: float = 0.0
    profile: str = "memo_experimental"


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
    program: Any = field(default=None, init=False, repr=False)
    profile: str = field(default="memo_experimental", init=False)
    num_envs: int = field(default=1, init=False)
    runtime_config: Any = field(default=None, init=False, repr=False)
    render_evaluation: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.profile = getattr(self.cfg, "profile", "memo_experimental")
        if self.profile != "memo_experimental":
            raise ValueError(
                "strict RTRRL must be constructed from build_rtrrl_program"
            )
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
        self.program = self._delegate.program
        self.num_envs = self.cfg.num_envs
        self.runtime_config = self.cfg

    @classmethod
    def from_program(
        cls,
        program,
        *,
        profile: str,
        num_envs: int,
        runtime_config: Any = None,
        render_evaluation: Any = None,
    ):
        """Construct the public strict façade around one closed program."""

        if profile != "aaai25_strict_lru":
            raise ValueError("from_program is reserved for the strict profile")
        instance = cls.__new__(cls)
        instance.__dict__.update(
            cfg=runtime_config,
            env=None,
            env_params=None,
            feature_extractor=None,
            torso=None,
            actor_head=None,
            critic_head=None,
            pred_head=None,
            activation=jax.nn.silu,
            program_normalization=None,
            strip_environment_normalization=False,
            _delegate=None,
            program=program,
            profile=profile,
            num_envs=num_envs,
            runtime_config=runtime_config,
            render_evaluation=render_evaluation,
        )
        return instance

    def as_legacy_program(self):
        """Return the one program façade selected during construction."""

        return self if self.profile == "aaai25_strict_lru" else self._delegate

    def init(self, key):
        if self.profile == "aaai25_strict_lru":
            return self.program.init_fn(key)
        return self._delegate.init(key)

    def warmup(self, key, state, num_steps):
        if self.profile == "aaai25_strict_lru":
            del key, num_steps
            return state
        return self._delegate.warmup(key, state, num_steps)

    def train(self, key, state, num_steps):
        if self.profile == "aaai25_strict_lru":
            return self.program.train_epoch_fn(key, state, num_steps)[0]
        return self._delegate.train(key, state, num_steps)

    def evaluate(self, key, state, num_steps):
        if self.profile == "aaai25_strict_lru":
            return self.program.evaluate_fn(key, state, num_steps)[0]
        return self._delegate.evaluate(key, state, num_steps)

    def evaluate_summary(self, key, state, num_steps):
        """Return delegated state and stable evaluation information."""

        return self.program.evaluate_fn(key, state, num_steps)

    def _update_step(self, state, key):
        """Compatibility shim; the update itself remains owned by the program."""

        if self.profile == "aaai25_strict_lru":
            next_state, _ = self.program.train_epoch_fn(key, state, 1)
            return next_state, None
        return self._delegate.train(key, state, self.num_envs), None


__all__ = [
    "RTRRL",
    "RTRRLConfig",
    "RTRRLState",
    "_find_leaf",
    "_tree_norm",
]
