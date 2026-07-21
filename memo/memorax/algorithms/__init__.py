"""Public algorithm exports, loaded lazily for parse-only command paths."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .algorithm import Algorithm as Algorithm, State as State
    from .dqn import DQN as DQN, DQNConfig as DQNConfig, DQNState as DQNState
    from .gradient_ppo import (
        GradientPPO as GradientPPO,
        GradientPPOConfig as GradientPPOConfig,
        GradientPPOState as GradientPPOState,
    )
    from .independent_rtrrl import (
        IndependentRTRRL as IndependentRTRRL,
        IndependentRTRRLConfig as IndependentRTRRLConfig,
        IndependentRTRRLState as IndependentRTRRLState,
    )
    from .mappo import (
        MAPPO as MAPPO,
        MAPPOConfig as MAPPOConfig,
        MAPPOState as MAPPOState,
    )
    from .ppo import PPO as PPO, PPOConfig as PPOConfig, PPOState as PPOState
    from .pqn import PQN as PQN, PQNConfig as PQNConfig, PQNState as PQNState
    from .qrc import QRC as QRC, QRCConfig as QRCConfig, QRCState as QRCState
    from .qrc_rtrl import QRCRtrl as QRCRtrl, QRCRtrlState as QRCRtrlState
    from .r2d2 import R2D2 as R2D2, R2D2Config as R2D2Config, R2D2State as R2D2State
    from .rtrrl import RTRRL as RTRRL, RTRRLConfig as RTRRLConfig, RTRRLState as RTRRLState
    from .sac import SAC as SAC, SACConfig as SACConfig, SACState as SACState
    from .stream_ac import (
        StreamAC as StreamAC,
        StreamACConfig as StreamACConfig,
        StreamACState as StreamACState,
    )
    from .stream_ac_rtrl import (
        StreamACRtrl as StreamACRtrl,
        StreamACRtrlState as StreamACRtrlState,
    )


_EXPORTS = {
    "Algorithm": (".algorithm", "Algorithm"),
    "State": (".algorithm", "State"),
    "DQN": (".dqn", "DQN"),
    "DQNConfig": (".dqn", "DQNConfig"),
    "DQNState": (".dqn", "DQNState"),
    "GradientPPO": (".gradient_ppo", "GradientPPO"),
    "GradientPPOConfig": (".gradient_ppo", "GradientPPOConfig"),
    "GradientPPOState": (".gradient_ppo", "GradientPPOState"),
    "IndependentRTRRL": (".independent_rtrrl", "IndependentRTRRL"),
    "IndependentRTRRLConfig": (
        ".independent_rtrrl",
        "IndependentRTRRLConfig",
    ),
    "IndependentRTRRLState": (
        ".independent_rtrrl",
        "IndependentRTRRLState",
    ),
    "MAPPO": (".mappo", "MAPPO"),
    "MAPPOConfig": (".mappo", "MAPPOConfig"),
    "MAPPOState": (".mappo", "MAPPOState"),
    "PPO": (".ppo", "PPO"),
    "PPOConfig": (".ppo", "PPOConfig"),
    "PPOState": (".ppo", "PPOState"),
    "PQN": (".pqn", "PQN"),
    "PQNConfig": (".pqn", "PQNConfig"),
    "PQNState": (".pqn", "PQNState"),
    "QRC": (".qrc", "QRC"),
    "QRCConfig": (".qrc", "QRCConfig"),
    "QRCState": (".qrc", "QRCState"),
    "QRCRtrl": (".qrc_rtrl", "QRCRtrl"),
    "QRCRtrlState": (".qrc_rtrl", "QRCRtrlState"),
    "R2D2": (".r2d2", "R2D2"),
    "R2D2Config": (".r2d2", "R2D2Config"),
    "R2D2State": (".r2d2", "R2D2State"),
    "RTRRL": (".rtrrl", "RTRRL"),
    "RTRRLConfig": (".rtrrl", "RTRRLConfig"),
    "RTRRLState": (".rtrrl", "RTRRLState"),
    "SAC": (".sac", "SAC"),
    "SACConfig": (".sac", "SACConfig"),
    "SACState": (".sac", "SACState"),
    "StreamAC": (".stream_ac", "StreamAC"),
    "StreamACConfig": (".stream_ac", "StreamACConfig"),
    "StreamACState": (".stream_ac", "StreamACState"),
    "StreamACRtrl": (".stream_ac_rtrl", "StreamACRtrl"),
    "StreamACRtrlState": (".stream_ac_rtrl", "StreamACRtrlState"),
}

__all__ = [
    "Algorithm",
    "State",
    "DQN",
    "DQNConfig",
    "DQNState",
    "GradientPPO",
    "GradientPPOConfig",
    "GradientPPOState",
    "IndependentRTRRL",
    "IndependentRTRRLConfig",
    "IndependentRTRRLState",
    "MAPPO",
    "MAPPOConfig",
    "MAPPOState",
    "PPO",
    "PPOConfig",
    "PPOState",
    "PQN",
    "PQNConfig",
    "PQNState",
    "QRC",
    "QRCConfig",
    "QRCState",
    "QRCRtrl",
    "QRCRtrlState",
    "R2D2",
    "R2D2Config",
    "R2D2State",
    "RTRRL",
    "RTRRLConfig",
    "RTRRLState",
    "SAC",
    "SACConfig",
    "SACState",
    "StreamAC",
    "StreamACConfig",
    "StreamACState",
    "StreamACRtrl",
    "StreamACRtrlState",
]


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
