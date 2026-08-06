"""Driving an algorithm to its budget, and the program it is driven through.

What an entry assembles and starts. It knows an algorithm only by its three
arrows, so a kernel becomes runnable by having them rather than by being
registered anywhere.
"""

from .driver import (
    DEFAULT_REWARD,
    EPISODE_FIELDS,
    TRANSITIONS,
    Destination,
    Runtime,
    drive,
    whole_epochs,
)
from .program import INIT_NAMES, AgentProgram, program_of

__all__ = [
    "DEFAULT_REWARD",
    "EPISODE_FIELDS",
    "INIT_NAMES",
    "TRANSITIONS",
    "AgentProgram",
    "Destination",
    "Runtime",
    "drive",
    "program_of",
    "whole_epochs",
]
