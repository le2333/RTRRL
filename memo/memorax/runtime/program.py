"""The three arrows the runtime drives, taken from whatever is holding them.

An algorithm owns its own loop and hands over three functions. What it calls
them is its own business -- the kernel written against the current interface
spells the first one ``reset`` and the older ones spell it ``init`` -- and a
runtime that insisted on one spelling would have made the name part of the
contract rather than the arrow.

Nothing is registered and nothing subclasses anything. An algorithm becomes
drivable by having the three methods, which every kernel in this package
already had before this file existed.
"""

from __future__ import annotations

from typing import Any

from memorax.algorithms.contract import AgentProgram

# In order of preference, so that a kernel offering both is driven through the
# one it means. Nothing offers both today; the order is written down so that a
# kernel keeping an old name as an alias does not become ambiguous.
INIT_NAMES: tuple[str, ...] = ("reset", "init")


def _arrow(algorithm: Any, names: tuple[str, ...]):
    for name in names:
        found = getattr(algorithm, name, None)
        if callable(found):
            return found
    raise TypeError(
        f"{type(algorithm).__name__} has no "
        f"{' or '.join(names)}; a program is three arrows and this is not one"
    )


def program_of(algorithm: Any) -> AgentProgram:
    """The program an algorithm presents, however it presents it.

    Already a program, able to build one, or an object carrying the three
    methods: all three arrive here and leave as the same thing, which is what
    lets one runtime drive kernels that share no base class.
    """

    if isinstance(algorithm, AgentProgram):
        return algorithm
    build = getattr(algorithm, "as_program", None)
    if callable(build):
        return build()
    return AgentProgram(
        init_fn=_arrow(algorithm, INIT_NAMES),
        train_epoch_fn=_arrow(algorithm, ("train",)),
        evaluate_fn=_arrow(algorithm, ("evaluate",)),
        # Only a kernel that declares them has them. The runtime reads neither,
        # so an algorithm that never wrote them down is still drivable; what
        # they are for is a host that wants to allocate before the first step.
        state_schema=getattr(algorithm, "state_schema", None),
        metric_schema=getattr(algorithm, "metric_schema", None),
    )
