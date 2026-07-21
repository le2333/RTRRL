"""Memorax: a unified framework for memory-augmented reinforcement learning."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memorax.algorithms import (
        DQN as DQN,
        DQNConfig as DQNConfig,
        DQNState as DQNState,
        GradientPPO as GradientPPO,
        GradientPPOConfig as GradientPPOConfig,
        GradientPPOState as GradientPPOState,
        MAPPO as MAPPO,
        MAPPOConfig as MAPPOConfig,
        MAPPOState as MAPPOState,
        PPO as PPO,
        PPOConfig as PPOConfig,
        PPOState as PPOState,
        PQN as PQN,
        PQNConfig as PQNConfig,
        PQNState as PQNState,
        R2D2 as R2D2,
        R2D2Config as R2D2Config,
        R2D2State as R2D2State,
        SAC as SAC,
        SACConfig as SACConfig,
        SACState as SACState,
        StreamAC as StreamAC,
        StreamACConfig as StreamACConfig,
        StreamACState as StreamACState,
    )
    from memorax.environments import make as make
    from memorax.loggers import (
        CheckpointLogger as CheckpointLogger,
        DashboardLogger as DashboardLogger,
        FileLogger as FileLogger,
        Logger as Logger,
        MultiLogger as MultiLogger,
        TensorBoardLogger as TensorBoardLogger,
        WandbLogger as WandbLogger,
    )
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
    "StreamAC",
    "StreamACConfig",
    "StreamACState",
}
_NETWORK_EXPORTS = {
    "FeatureExtractor",
    "Network",
    "SequenceModel",
    "SequenceModelWrapper",
}
_LOGGER_EXPORTS = {
    "CheckpointLogger",
    "DashboardLogger",
    "FileLogger",
    "Logger",
    "MultiLogger",
    "TensorBoardLogger",
    "WandbLogger",
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
    "StreamAC",
    "StreamACConfig",
    "StreamACState",
    "make",
    "FeatureExtractor",
    "Network",
    "SequenceModel",
    "SequenceModelWrapper",
    "CheckpointLogger",
    "DashboardLogger",
    "FileLogger",
    "Logger",
    "MultiLogger",
    "TensorBoardLogger",
    "WandbLogger",
]


def __getattr__(name):
    if name in _ALGORITHM_EXPORTS:
        module = import_module("memorax.algorithms")
    elif name in _NETWORK_EXPORTS:
        module = import_module("memorax.networks")
    elif name in _LOGGER_EXPORTS:
        module = import_module("memorax.loggers")
    elif name == "make":
        module = import_module("memorax.environments")
    else:
        raise AttributeError(name)
    value = getattr(module, name)
    globals()[name] = value
    return value
