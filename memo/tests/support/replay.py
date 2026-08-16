"""Start-flag rules in the terms the replay buffer takes them.

The buffer stores an opaque tree and asks a rule which stored positions may be
sampled from. R2D2's rules are written for the transition it stores, which is
narrower than what the buffer promises to hand back, so the rule a buffer is
given is typed by what the buffer knows. ``SelectedLearning.start_flags`` says
the same thing on the production side; this is where the tests say it.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from memorax.utils.typing import Array


def start_flags(rule, **options) -> Callable[[Any], Array]:
    """``rule`` with its horizons fixed, as the buffer will call it."""

    return partial(rule, **options)
