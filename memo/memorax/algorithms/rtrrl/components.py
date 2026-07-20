"""Construction-time adapters for strict and retained Memorax components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import flax.linen as nn
import jax
import jax.numpy as jnp

from memorax.networks import (
    FeatureExtractor,
    LRUCell,
    LRUConfig,
    Memoroid,
    RNN,
    RTUCell,
    RTUConfig,
    heads,
)

from .compatibility import (
    LegacyRTRRLConfig,
    RTRRLComponentConfig,
    to_component_config,
)


@runtime_checkable
class RecurrentComponent(Protocol):
    """A recurrent component whose changing values are explicit arguments."""

    def initialize(self, key: Any, input_shape: tuple[int, ...]) -> Any: ...

    def forward(
        self, params: Any, carry: Any, inputs: Any, reset: Any
    ) -> Any: ...

    def credit(
        self,
        params: Any,
        credit_state: Any,
        carry: Any,
        inputs: Any,
        cotangent: Any,
    ) -> Any: ...


def _identity(value):
    return value


@dataclass(frozen=True)
class MemoraxRecurrentAdapter:
    """Expose an existing Memorax torso through explicit dynamic arguments."""

    module: Memoroid | RNN

    def initialize(self, key: Any, input_shape: tuple[int, ...]) -> Any:
        carry_shape = cast(Any, (*input_shape[:-1], None))
        carry_key, params_key, sensitivity_key = jax.random.split(key, 3)
        carry = self.module.initialize_carry(carry_key, carry_shape)
        inputs = jnp.zeros(input_shape, dtype=jnp.float32)
        reset = jnp.ones(input_shape[:-1], dtype=jnp.bool_)
        variables = self.module.init(
            {"params": params_key}, inputs, reset, initial_carry=carry
        )
        sensitivity = self.module.initialize_sensitivity(
            sensitivity_key, carry_shape
        )
        return variables["params"], carry, sensitivity

    def forward(
        self, params: Any, carry: Any, inputs: Any, reset: Any
    ) -> Any:
        return self.module.apply(
            {"params": params}, inputs, reset, initial_carry=carry
        )

    def credit(
        self,
        params: Any,
        credit_state: Any,
        carry: Any,
        inputs: Any,
        cotangent: Any,
    ) -> Any:
        del cotangent
        return self.module.apply(
            {"params": params},
            inputs,
            False,
            carry,
            sensitivity=credit_state,
            method="local_jacobian",
        )


@dataclass(frozen=True)
class MemoraxComponentSelection:
    """Fixed experimental recipe selected before any JIT closure is built."""

    effective_config: RTRRLComponentConfig
    feature_extractor: FeatureExtractor
    recurrent: Memoroid | RNN
    recurrent_adapter: MemoraxRecurrentAdapter
    actor_head: nn.Module
    critic_head: nn.Module
    prediction_head: nn.Module | None
    activation: Any


def select_memorax_components(
    config: LegacyRTRRLConfig,
    *,
    observation_dim: int,
    action_dim: int,
    topology: str | None = None,
) -> MemoraxComponentSelection:
    """Select retained Memorax branches without tracing or duplicating math."""

    if config.profile != "memo_experimental":
        raise ValueError("Memorax adapters require profile='memo_experimental'")
    topology = topology or config.rtrrl_topology
    if topology not in {"shared", "independent"}:
        raise ValueError(f"unsupported RTRRL topology: {topology!r}")
    effective = to_component_config(config)
    effective = RTRRLComponentConfig(
        **{
            **effective.__dict__,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "topology": topology,
        }
    )

    if config.use_encoder:
        feature_width = config.encoder_dim
        observation_extractor = nn.Sequential(
            (nn.Dense(feature_width), nn.relu)
        )
        action_extractor = (
            nn.Sequential((nn.Dense(feature_width), nn.relu))
            if config.meta_rl
            else None
        )
        reward_extractor = (
            nn.Sequential((nn.Dense(feature_width), nn.relu))
            if config.meta_rl
            else None
        )
        input_dim = feature_width * (3 if config.meta_rl else 1)
        output_dim = None
    else:
        observation_extractor = _identity
        action_extractor = _identity if config.meta_rl else None
        reward_extractor = _identity if config.meta_rl else None
        input_dim = observation_dim + (action_dim + 1 if config.meta_rl else 0)
        output_dim = config.lru_output_dim or config.hidden_dim
    feature_extractor = FeatureExtractor(
        observation_extractor=observation_extractor,
        action_extractor=action_extractor,
        reward_extractor=reward_extractor,
    )
    if config.backbone == "rtu":
        recurrent: Memoroid | RNN = RNN(
            cell=RTUCell(
                config=RTUConfig(
                    features=input_dim, hidden_dim=config.hidden_dim
                )
            )
        )
    elif config.backbone == "lru":
        recurrent = Memoroid(
            cell=LRUCell(
                config=LRUConfig(
                    features=input_dim,
                    hidden_dim=config.hidden_dim,
                    output_dim=output_dim,
                )
            )
        )
    else:
        raise ValueError(
            f"backbone {config.backbone!r} not supported; use 'lru' or 'rtu'"
        )
    prediction_head = (
        heads.Regressor(out_dim=observation_dim + 1)
        if config.pred_obs
        else None
    )
    return MemoraxComponentSelection(
        effective_config=effective,
        feature_extractor=feature_extractor,
        recurrent=recurrent,
        recurrent_adapter=MemoraxRecurrentAdapter(recurrent),
        actor_head=heads.Gaussian(
            action_dim=action_dim, bound=config.bound_actor
        ),
        critic_head=heads.VNetwork(),
        prediction_head=prediction_head,
        activation=jax.nn.silu,
    )


__all__ = [
    "MemoraxComponentSelection",
    "MemoraxRecurrentAdapter",
    "RecurrentComponent",
    "select_memorax_components",
]
