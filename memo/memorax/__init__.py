"""Memorax: a unified framework for memory-augmented reinforcement learning."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memorax.algorithms import DQN as DQN
    from memorax.algorithms import MAPPO as MAPPO
    from memorax.algorithms import PPO as PPO
    from memorax.algorithms import PQN as PQN
    from memorax.algorithms import R2D2 as R2D2
    from memorax.algorithms import RTRRL as RTRRL
    from memorax.algorithms import SAC as SAC
    from memorax.algorithms import DQNConfig as DQNConfig
    from memorax.algorithms import DQNState as DQNState
    from memorax.algorithms import GradientPPO as GradientPPO
    from memorax.algorithms import GradientPPOConfig as GradientPPOConfig
    from memorax.algorithms import GradientPPOState as GradientPPOState
    from memorax.algorithms import IndependentRTRRL as IndependentRTRRL
    from memorax.algorithms import IndependentRTRRLConfig as IndependentRTRRLConfig
    from memorax.algorithms import IndependentRTRRLState as IndependentRTRRLState
    from memorax.algorithms import MAPPOConfig as MAPPOConfig
    from memorax.algorithms import MAPPOState as MAPPOState
    from memorax.algorithms import PPOConfig as PPOConfig
    from memorax.algorithms import PPOState as PPOState
    from memorax.algorithms import PQNConfig as PQNConfig
    from memorax.algorithms import PQNState as PQNState
    from memorax.algorithms import R2D2Config as R2D2Config
    from memorax.algorithms import R2D2State as R2D2State
    from memorax.algorithms import RTRRLConfig as RTRRLConfig
    from memorax.algorithms import RTRRLState as RTRRLState
    from memorax.algorithms import SACConfig as SACConfig
    from memorax.algorithms import SACState as SACState
    from memorax.algorithms import StreamAC as StreamAC
    from memorax.algorithms import StreamACConfig as StreamACConfig
    from memorax.algorithms import StreamACRTRL as StreamACRTRL
    from memorax.algorithms import StreamACRTRLConfig as StreamACRTRLConfig
    from memorax.algorithms import StreamACRTRLState as StreamACRTRLState
    from memorax.algorithms import StreamACState as StreamACState
    from memorax.environments import make as make
    from memorax.networks import FeatureExtractor as FeatureExtractor
    from memorax.networks import Network as Network
    from memorax.networks import SequenceModel as SequenceModel
    from memorax.networks import SequenceModelWrapper as SequenceModelWrapper


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
    "RTRRL",
    "RTRRLConfig",
    "RTRRLState",
    "IndependentRTRRL",
    "IndependentRTRRLConfig",
    "IndependentRTRRLState",
    "StreamAC",
    "StreamACConfig",
    "StreamACState",
    "StreamACRTRL",
    "StreamACRTRLConfig",
    "StreamACRTRLState",
}
_NETWORK_EXPORTS = {
    "FeatureExtractor",
    "Network",
    "SequenceModel",
    "SequenceModelWrapper",
}

__all__ = [
    "DQN",
    "MAPPO",
    "PPO",
    "PQN",
    "R2D2",
    "RTRRL",
    "SAC",
    "DQNConfig",
    "DQNState",
    "FeatureExtractor",
    "GradientPPO",
    "GradientPPOConfig",
    "GradientPPOState",
    "IndependentRTRRL",
    "IndependentRTRRLConfig",
    "IndependentRTRRLState",
    "MAPPOConfig",
    "MAPPOState",
    "Network",
    "PPOConfig",
    "PPOState",
    "PQNConfig",
    "PQNState",
    "R2D2Config",
    "R2D2State",
    "RTRRLConfig",
    "RTRRLState",
    "SACConfig",
    "SACState",
    "SequenceModel",
    "SequenceModelWrapper",
    "StreamAC",
    "StreamACConfig",
    "StreamACRTRL",
    "StreamACRTRLConfig",
    "StreamACRTRLState",
    "StreamACState",
    "__version__",
    "make",
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
