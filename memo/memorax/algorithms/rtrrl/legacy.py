"""Import-compatible RTRRL façade backed by one composed ``AgentProgram``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp

from memorax.utils.typing import (
    Array,
    EnvParams,
    Environment,
)

from .components import RecurrentComponent
from .types import RTRRLState


@dataclass(frozen=True)
class _MemoraxParts:
    env: Any
    env_params: Any
    feature_extractor: Any
    torso: Any
    actor_head: Any
    critic_head: Any
    pred_head: Any
    activation: Any


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
    torso: RecurrentComponent
    actor_head: nn.Module
    critic_head: nn.Module
    pred_head: nn.Module | None = None
    activation: Callable = jax.nn.silu
    program_normalization: Any = None
    strip_environment_normalization: bool = False
    effective_config: Any = None
    compatibility_parts: Any = None
    _delegate: Any = field(default=None, init=False, repr=False)
    _debug_interface: Any = field(default=None, init=False, repr=False)
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
            NormalizationConfig,
            legacy_env_adapter,
        )
        from memorax.online_ac.meta import make_meta_program

        config = MetaProgramConfig.from_legacy_parts(
            self,
            normalization=self.program_normalization,
        )
        adapter = legacy_env_adapter(
            self.env,
            self.env_params,
            strip_normalization=self.strip_environment_normalization,
        )
        parts = _MemoraxParts(
            env=adapter.build_context["env"],
            env_params=adapter.env_params,
            feature_extractor=self.feature_extractor,
            torso=self.torso,
            actor_head=self.actor_head,
            critic_head=self.critic_head,
            pred_head=self.pred_head,
            activation=self.activation,
        )
        debug_sink = []
        self.program = make_meta_program(
            parts,
            config.static_config,
            normalization_config=(
                config.normalization or NormalizationConfig()
            ),
            reset_on_start=config.evaluation.reset_on_start,
            update_during_eval=config.evaluation.update_during_eval,
            _debug_sink=debug_sink,
        )
        self._debug_interface = debug_sink[0]
        self._delegate = LegacyProgram(self.program, config)
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
        effective_config: Any = None,
        compatibility_parts: Any = None,
    ):
        """Construct the public strict façade around one closed program."""

        if profile not in {"aaai25_strict_lru", "memo_experimental"}:
            raise ValueError(f"unsupported RTRRL profile: {profile!r}")
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
            effective_config=effective_config,
            compatibility_parts=compatibility_parts,
        )
        return instance

    def as_legacy_program(self):
        """Return the one program façade selected during construction."""

        return self

    def __getattr__(self, name):
        parts = self.__dict__.get("compatibility_parts")
        if parts is not None:
            return getattr(parts, name)
        debug = self.__dict__.get("_debug_interface")
        debug_names = {
            "_forward": "forward",
            "_grad_params": "grad_params",
            "optimizer": "optimizer",
            "_program_step": "step",
        }
        if debug is not None and name in debug_names:
            return getattr(debug, debug_names[name])
        delegate = self.__dict__.get("_delegate")
        if delegate is not None:
            return getattr(delegate, name)
        raise AttributeError(name)

    def init(self, key):
        return self.program.init_fn(key)

    def warmup(self, key, state, num_steps):
        del key, num_steps
        return state

    def train(self, key, state, num_steps):
        if self._delegate is not None:
            return self._delegate.train(key, state, num_steps)
        return self.program.train_epoch_fn(key, state, num_steps)[0]

    def evaluate(self, key, state, num_steps):
        if self._delegate is not None:
            return self._delegate.evaluate(key, state, num_steps)
        return self.program.evaluate_fn(key, state, num_steps)[0]

    def evaluate_summary(self, key, state, num_steps):
        """Return delegated state and stable evaluation information."""

        return self.program.evaluate_fn(key, state, num_steps)

    def _update_step(self, state, key):
        """Compatibility shim; the update itself remains owned by the program."""

        if self.compatibility_parts is not None:
            return self.compatibility_parts._update_step(state, key)
        if self.profile == "aaai25_strict_lru":
            next_state, _ = self.program.train_epoch_fn(key, state, 1)
            return next_state, None
        debug = self._debug_interface
        if debug is not None:
            return debug.step(state, key)
        return self._delegate.train(key, state, self.num_envs), None


__all__ = [
    "RTRRL",
    "RTRRLConfig",
    "RTRRLState",
    "_find_leaf",
    "_tree_norm",
]
