"""Memorax: a unified framework for memory-augmented reinforcement learning."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memorax.algorithms import (
        RTRRL as RTRRL,
        IndependentRTRRL as IndependentRTRRL,
        IndependentRTRRLConfig as IndependentRTRRLConfig,
        IndependentRTRRLState as IndependentRTRRLState,
        RTRRLConfig as RTRRLConfig,
        RTRRLState as RTRRLState,
        StreamAC as StreamAC,
        StreamACConfig as StreamACConfig,
        StreamACRtrl as StreamACRtrl,
        StreamACRtrlState as StreamACRtrlState,
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
    "RTRRL",
    "RTRRLConfig",
    "RTRRLState",
    "IndependentRTRRL",
    "IndependentRTRRLConfig",
    "IndependentRTRRLState",
    "StreamAC",
    "StreamACConfig",
    "StreamACState",
    "StreamACRtrl",
    "StreamACRtrlState",
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
    "RTRRL",
    "RTRRLConfig",
    "RTRRLState",
    "IndependentRTRRL",
    "IndependentRTRRLConfig",
    "IndependentRTRRLState",
    "StreamAC",
    "StreamACConfig",
    "StreamACState",
    "StreamACRtrl",
    "StreamACRtrlState",
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
