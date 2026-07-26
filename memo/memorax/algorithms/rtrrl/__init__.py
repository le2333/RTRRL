from typing import TYPE_CHECKING

from .compatibility import (
    InvalidRTRRLConfig,
    LegacyRTRRLConfig,
    RTRRLComponentConfig,
    UnknownRTRRLField,
    UnsupportedRTRRLBranch,
    normalize_legacy_config,
    to_component_config,
)

if TYPE_CHECKING:
    from .legacy import (
        RTRRL as RTRRL,
        RTRRLConfig as RTRRLConfig,
        _find_leaf as _find_leaf,
        _tree_norm as _tree_norm,
    )
    from .types import RTRRLComponents as RTRRLComponents, RTRRLState as RTRRLState


_LAZY_EXPORTS = {
    "RTRRL": (".legacy", "RTRRL"),
    "RTRRLConfig": (".legacy", "RTRRLConfig"),
    "_find_leaf": (".legacy", "_find_leaf"),
    "_tree_norm": (".legacy", "_tree_norm"),
    "RTRRLComponents": (".types", "RTRRLComponents"),
    "RTRRLState": (".types", "RTRRLState"),
}


def __getattr__(name):
    """Keep parse-only compatibility imports free of JAX initialization."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "RTRRL",
    "LegacyRTRRLConfig",
    "RTRRLComponentConfig",
    "RTRRLConfig",
    "RTRRLEpochSummary",
    "RTRRLState",
    "RTRRLComponents",
    "InvalidRTRRLConfig",
    "UnsupportedRTRRLBranch",
    "UnknownRTRRLField",
    "aggregate_epoch_summary",
    "build_rtrrl_program",
    "normalize_legacy_config",
    "make_init_fn",
    "make_step_fn",
    "to_component_config",
]
