"""Public algorithm exports, loaded lazily for parse-only command paths."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contract import (
        ActionDecision as ActionDecision,
        AgentProgram as AgentProgram,
        EvalSummary as EvalSummary,
        EvaluationConfig as EvaluationConfig,
        Transition as Transition,
    )
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
    from .r2d2 import R2D2 as R2D2, R2D2Config as R2D2Config, R2D2State as R2D2State
    from .rtrrl import (
        RTRRL as RTRRL,
        RTRRLConfig as RTRRLConfig,
        RTRRLState as RTRRLState,
    )
    from .sac import SAC as SAC, SACConfig as SACConfig, SACState as SACState
    from .stream_ac_rtrl import (
        StreamACRTRL as StreamACRTRL,
        StreamACRTRLConfig as StreamACRTRLConfig,
        StreamACRTRLState as StreamACRTRLState,
    )


_EXPORTS = {
    "ActionDecision": (".contract", "ActionDecision"),
    "AgentProgram": (".contract", "AgentProgram"),
    "EvalSummary": (".contract", "EvalSummary"),
    "EvaluationConfig": (".contract", "EvaluationConfig"),
    "Transition": (".contract", "Transition"),
    "DQN": (".dqn", "DQN"),
    "DQNConfig": (".dqn", "DQNConfig"),
    "DQNState": (".dqn", "DQNState"),
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
    "IndependentRTRRL": (".independent_rtrrl", "IndependentRTRRL"),
    "IndependentRTRRLConfig": (
        ".independent_rtrrl",
        "IndependentRTRRLConfig",
    ),
    "IndependentRTRRLState": (
        ".independent_rtrrl",
        "IndependentRTRRLState",
    ),
    "RTRRL": (".rtrrl", "RTRRL"),
    "RTRRLConfig": (".rtrrl", "RTRRLConfig"),
    "RTRRLState": (".rtrrl", "RTRRLState"),
    "StreamACRTRL": (".stream_ac_rtrl", "StreamACRTRL"),
    "StreamACRTRLConfig": (".stream_ac_rtrl", "StreamACRTRLConfig"),
    "StreamACRTRLState": (".stream_ac_rtrl", "StreamACRTRLState"),
}

__all__ = [
    "ActionDecision",
    "AgentProgram",
    "EvalSummary",
    "EvaluationConfig",
    "Transition",
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
    "IndependentRTRRL",
    "IndependentRTRRLConfig",
    "IndependentRTRRLState",
    "RTRRL",
    "RTRRLConfig",
    "RTRRLState",
    "StreamACRTRL",
    "StreamACRTRLConfig",
    "StreamACRTRLState",
]


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
