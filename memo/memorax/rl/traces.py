"""The eligibility recurrence, owned by the algorithm rather than a rule.

An eligibility trace is what an algorithm says a parameter is still credited
for. It belongs beside the objective that produced the derivative, not inside
whichever rule happens to be stepping this week: two algorithms that trace the
same derivative differently are two algorithms, and an optimizer that quietly
accumulated its own would make that difference unreadable from the outside.

Two recurrences, and the difference between them is *when* the update reads:

``emphasized``
    ``z_t = decay * (1 - reset) * z_{t-1} + m_t * p_t``, and the update steps
    along ``z_{t-1}``. RTRRL's, including the followed-trace emphasis ``m_t``.
    Its first transition moves nothing, because the trace the update reads has
    not been written yet.

``accumulating``
    ``z_t = decay * (1 - reset) * z_{t-1} + p_t``, and the update steps along
    ``z_t``. StreamAC's, and the one the intentional update is derived
    against. Its first transition already moves.

That ordering is not a detail of when a line runs. It decides whether this
step's derivative is in the trace this step spends, and any rule reading the
trace inherits the answer -- which is why it is declared here, once, rather
than discovered from whichever rule is downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .interaction import broadcast_stream

CARRIED = "carried"
CURRENT = "current"
READINGS: tuple[str, ...] = (CARRIED, CURRENT)


@dataclass(frozen=True)
class Trace:
    """One parameter group's eligibility recurrence.

    ``decay`` is the algorithm's ``gamma * lambda`` for this group. ``reads``
    says which trace an update steps along, ``carried`` or ``current``.
    ``emphasized`` says whether the incoming derivative is weighted by the
    algorithm's emphasis before it joins; an algorithm with no emphasis passes
    ones and gets the same answer either way, but declaring it means the
    published recurrence can be asked for by name.
    """

    decay: float
    reads: str = CARRIED
    emphasized: bool = True

    def __post_init__(self) -> None:
        if self.reads not in READINGS:
            raise ValueError(
                f"{self.reads!r} is not one of the traces an update can step "
                f"along ({', '.join(READINGS)})"
            )

    def initial(self, params, streams: int):
        """One trace per parameter per stream, which is what online means."""

        return jax.tree.map(
            lambda parameter: jnp.zeros((streams, *parameter.shape)), params
        )

    def advance(self, carried, derivative, *, reset, emphasis):
        """The trace after this transition's derivative has joined it."""

        weight = None if not self.emphasized else emphasis
        return jax.tree.map(
            lambda old, incoming: (
                self.decay * (1 - broadcast_stream(reset, old)) * old
                + (
                    incoming
                    if weight is None
                    else broadcast_stream(weight, incoming) * incoming
                )
            ),
            carried,
            derivative,
        )

    def stepped(self, carried, derivative, *, reset, emphasis):
        """What this update steps along, and what the next transition carries.

        Both, from one call, because the two answers differ only in the reading
        and nothing outside this component should have to know which.
        """

        advanced = self.advance(carried, derivative, reset=reset, emphasis=emphasis)
        return (carried if self.reads == CARRIED else advanced), advanced
