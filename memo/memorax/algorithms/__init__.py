"""Public algorithm exports, loaded lazily for parse-only command paths."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contract import ActionDecision as ActionDecision
    from .contract import EvaluationConfig as EvaluationConfig
    from .dqn import DQN as DQN
    from .dqn import DQNConfig as DQNConfig
    from .dqn import DQNState as DQNState
    from .drqn import DRQN as DRQN
    from .drqn import DRQNConfig as DRQNConfig
    from .drqn import DRQNState as DRQNState
    from .gradient_ppo import GradientPPO as GradientPPO
    from .gradient_ppo import GradientPPOConfig as GradientPPOConfig
    from .gradient_ppo import GradientPPOState as GradientPPOState
    from .mappo import MAPPO as MAPPO
    from .mappo import MAPPOConfig as MAPPOConfig
    from .mappo import MAPPOState as MAPPOState
    from .ppo import PPO as PPO
    from .ppo import PPOConfig as PPOConfig
    from .ppo import PPOState as PPOState
    from .pqn import PQN as PQN
    from .pqn import PQNConfig as PQNConfig
    from .pqn import PQNState as PQNState
    from .r2d2 import R2D2 as R2D2
    from .r2d2 import R2D2Config as R2D2Config
    from .r2d2 import R2D2State as R2D2State
    from .rtrrl_aaai import RTRRL as RTRRL
    from .rtrrl_aaai import RTRRLConfig as RTRRLConfig
    from .rtrrl_aaai import RTRRLState as RTRRLState
    from .sac import SAC as SAC
    from .sac import SACConfig as SACConfig
    from .sac import SACState as SACState
    from .stream_ac import StreamAC as StreamAC
    from .stream_ac import StreamACConfig as StreamACConfig
    from .stream_ac import StreamACState as StreamACState


_EXPORTS = {
    "ActionDecision": (".contract", "ActionDecision"),
    "EvaluationConfig": (".contract", "EvaluationConfig"),
    "DQN": (".dqn", "DQN"),
    "DQNConfig": (".dqn", "DQNConfig"),
    "DQNState": (".dqn", "DQNState"),
    "DRQN": (".drqn", "DRQN"),
    "DRQNConfig": (".drqn", "DRQNConfig"),
    "DRQNState": (".drqn", "DRQNState"),
    "GradientPPO": (".gradient_ppo", "GradientPPO"),
    "GradientPPOConfig": (".gradient_ppo", "GradientPPOConfig"),
    "GradientPPOState": (".gradient_ppo", "GradientPPOState"),
    "MAPPO": (".mappo", "MAPPO"),
    "MAPPOConfig": (".mappo", "MAPPOConfig"),
    "MAPPOState": (".mappo", "MAPPOState"),
    "PPO": (".ppo", "PPO"),
    "PPOConfig": (".ppo", "PPOConfig"),
    "PPOState": (".ppo", "PPOState"),
    "PQN": (".pqn", "PQN"),
    "PQNConfig": (".pqn", "PQNConfig"),
    "PQNState": (".pqn", "PQNState"),
    "R2D2": (".r2d2", "R2D2"),
    "R2D2Config": (".r2d2", "R2D2Config"),
    "R2D2State": (".r2d2", "R2D2State"),
    "SAC": (".sac", "SAC"),
    "SACConfig": (".sac", "SACConfig"),
    "SACState": (".sac", "SACState"),
    "RTRRL": (".rtrrl_aaai", "RTRRL"),
    "RTRRLConfig": (".rtrrl_aaai", "RTRRLConfig"),
    "RTRRLState": (".rtrrl_aaai", "RTRRLState"),
    "StreamAC": (".stream_ac", "StreamAC"),
    "StreamACConfig": (".stream_ac", "StreamACConfig"),
    "StreamACState": (".stream_ac", "StreamACState"),
}

__all__ = [
    "DQN",
    "DRQN",
    "MAPPO",
    "PPO",
    "PQN",
    "R2D2",
    "RTRRL",
    "SAC",
    "ActionDecision",
    "DQNConfig",
    "DQNState",
    "DRQNConfig",
    "DRQNState",
    "EvaluationConfig",
    "GradientPPO",
    "GradientPPOConfig",
    "GradientPPOState",
    "MAPPOConfig",
    "MAPPOState",
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
    "StreamAC",
    "StreamACConfig",
    "StreamACState",
]


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
