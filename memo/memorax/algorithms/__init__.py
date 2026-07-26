"""Public algorithm exports, loaded lazily for parse-only command paths."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .algorithm import Algorithm as Algorithm, State as State
    from .independent_rtrrl import (
        IndependentRTRRL as IndependentRTRRL,
        IndependentRTRRLConfig as IndependentRTRRLConfig,
        IndependentRTRRLState as IndependentRTRRLState,
    )
    from .qrc import QRC as QRC, QRCConfig as QRCConfig, QRCState as QRCState
    from .qrc_rtrl import QRCRtrl as QRCRtrl, QRCRtrlState as QRCRtrlState
    from .rtrrl import RTRRL as RTRRL, RTRRLConfig as RTRRLConfig, RTRRLState as RTRRLState
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
    "IndependentRTRRL": (".independent_rtrrl", "IndependentRTRRL"),
    "IndependentRTRRLConfig": (
        ".independent_rtrrl",
        "IndependentRTRRLConfig",
    ),
    "IndependentRTRRLState": (
        ".independent_rtrrl",
        "IndependentRTRRLState",
    ),
    "QRC": (".qrc", "QRC"),
    "QRCConfig": (".qrc", "QRCConfig"),
    "QRCState": (".qrc", "QRCState"),
    "QRCRtrl": (".qrc_rtrl", "QRCRtrl"),
    "QRCRtrlState": (".qrc_rtrl", "QRCRtrlState"),
    "RTRRL": (".rtrrl", "RTRRL"),
    "RTRRLConfig": (".rtrrl", "RTRRLConfig"),
    "RTRRLState": (".rtrrl", "RTRRLState"),
    "StreamAC": (".stream_ac", "StreamAC"),
    "StreamACConfig": (".stream_ac", "StreamACConfig"),
    "StreamACState": (".stream_ac", "StreamACState"),
    "StreamACRtrl": (".stream_ac_rtrl", "StreamACRtrl"),
    "StreamACRtrlState": (".stream_ac_rtrl", "StreamACRtrlState"),
}

__all__ = [
    "Algorithm",
    "State",
    "IndependentRTRRL",
    "IndependentRTRRLConfig",
    "IndependentRTRRLState",
    "QRC",
    "QRCConfig",
    "QRCState",
    "QRCRtrl",
    "QRCRtrlState",
    "RTRRL",
    "RTRRLConfig",
    "RTRRLState",
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
