"""Memorax: a unified framework for memory-augmented reinforcement learning."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memorax.algorithms import (
        DQN as DQN,
        MAPPO as MAPPO,
        PPO as PPO,
        PQN as PQN,
        R2D2 as R2D2,
        RTRRLParts as RTRRLParts,
        SAC as SAC,
        DQNConfig as DQNConfig,
        DQNState as DQNState,
        GradientPPO as GradientPPO,
        GradientPPOConfig as GradientPPOConfig,
        GradientPPOState as GradientPPOState,
        IndependentRTRRL as IndependentRTRRL,
        IndependentRTRRLConfig as IndependentRTRRLConfig,
        IndependentRTRRLState as IndependentRTRRLState,
        MAPPOConfig as MAPPOConfig,
        MAPPOState as MAPPOState,
        PPOConfig as PPOConfig,
        PPOState as PPOState,
        PQNConfig as PQNConfig,
        PQNState as PQNState,
        R2D2Config as R2D2Config,
        R2D2State as R2D2State,
        RTRRLConfig as RTRRLConfig,
        RTRRLState as RTRRLState,
        SACConfig as SACConfig,
        SACState as SACState,
        StreamACRTRLConfig as StreamACRTRLConfig,
        StreamACRTRLParts as StreamACRTRLParts,
        StreamACRTRLState as StreamACRTRLState,
        build_rtrrl as build_rtrrl,
        build_stream_ac_rtrl as build_stream_ac_rtrl,
    )
    from memorax.environments import make as make
    from memorax.networks import (
        FeatureExtractor as FeatureExtractor,
        Network as Network,
        SequenceModel as SequenceModel,
        SequenceModelWrapper as SequenceModelWrapper,
    )


__version__ = "1.0.1"

_ALGORITHM_EXPORTS = {
    "DQN",
    "DQNConfig",
    "DQNState",
    "GradientPPO",
    "GradientPPOConfig",
    "GradientPPOState",
    "MAPPO",
    "MAPPOConfig",
    "MAPPOState",
    "PPO",
    "PPOConfig",
    "PPOState",
    "PQN",
    "PQNConfig",
    "PQNState",
    "R2D2",
    "R2D2Config",
    "R2D2State",
    "SAC",
    "SACConfig",
    "SACState",
    "RTRRLConfig",
    "RTRRLParts",
    "RTRRLState",
    "IndependentRTRRL",
    "IndependentRTRRLConfig",
    "IndependentRTRRLState",
    "StreamACRTRLConfig",
    "StreamACRTRLParts",
    "StreamACRTRLState",
    "build_rtrrl",
    "build_stream_ac_rtrl",
}
_NETWORK_EXPORTS = {
    "FeatureExtractor",
    "Network",
    "SequenceModel",
    "SequenceModelWrapper",
}

__all__ = [
    "__version__",
    "DQN",
    "DQNConfig",
    "DQNState",
    "GradientPPO",
    "GradientPPOConfig",
    "GradientPPOState",
    "MAPPO",
    "MAPPOConfig",
    "MAPPOState",
    "PPO",
    "PPOConfig",
    "PPOState",
    "PQN",
    "PQNConfig",
    "PQNState",
    "R2D2",
    "R2D2Config",
    "R2D2State",
    "SAC",
    "SACConfig",
    "SACState",
    "RTRRLConfig",
    "RTRRLParts",
    "RTRRLState",
    "IndependentRTRRL",
    "IndependentRTRRLConfig",
    "IndependentRTRRLState",
    "StreamACRTRLConfig",
    "StreamACRTRLParts",
    "StreamACRTRLState",
    "build_rtrrl",
    "build_stream_ac_rtrl",
    "make",
    "FeatureExtractor",
    "Network",
    "SequenceModel",
    "SequenceModelWrapper",
]


def __getattr__(name):
    if name in _ALGORITHM_EXPORTS:
        module = import_module("memorax.algorithms")
    elif name in _NETWORK_EXPORTS:
        module = import_module("memorax.networks")
    elif name == "make":
        module = import_module("memorax.environments")
    else:
        raise AttributeError(name)
    value = getattr(module, name)
    globals()[name] = value
    return value
