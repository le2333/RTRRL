"""Public algorithm exports, loaded lazily for parse-only command paths."""

from importlib import import_module


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

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
