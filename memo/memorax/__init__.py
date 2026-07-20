"""Memorax: a unified framework for memory-augmented reinforcement learning."""

from importlib import import_module


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
    *_ALGORITHM_EXPORTS,
    "make",
    *_NETWORK_EXPORTS,
    *_LOGGER_EXPORTS,
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
