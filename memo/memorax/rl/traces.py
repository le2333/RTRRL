"""The eligibility recurrence, owned by the algorithm rather than a rule.

An eligibility trace is what an algorithm says a parameter is still credited
for. It belongs beside the objective that produced the derivative, not inside
whichever rule happens to be stepping this week: two algorithms that trace the
same derivative differently are two algorithms, and an optimizer that quietly
accumulated its own would make that difference unreadable from the outside.

Two independent choices, and keeping them apart is the point of this module.

**The recurrence** is what joins the trace:

``emphasized``
    ``z_t = decay * (1 - reset) * z_{t-1} + m_t * p_t``. RTRRL's, including the
    followed-trace emphasis ``m_t``.

``accumulating``
    ``z_t = decay * (1 - reset) * z_{t-1} + p_t``, with no emphasis. StreamAC's,
    and the recurrence the intentional update is derived against.

**The reading** is which of the two traces a step is taken along, ``carried``
(``z_{t-1}``) or ``current`` (``z_t``), and it is a fact about the *algorithm's
ordering* rather than about the rule:

- an algorithm that computes ``V(s)`` and ``V(s')`` in one update -- StreamAC,
  and the intentional update's published implementation -- has just accumulated
  the derivative its TD error measures, so it reads ``current``;
- an algorithm that runs one forward per transition and carries the previous
  value -- RTRRL -- has an error whose transition *ended* here and whose
  derivative joined the trace last step, so it reads ``carried``. Its first
  transition moves nothing, because the trace the update reads has not been
  written yet.

Both are the same trace read at the index the error is indexed at. Pairing a
rule's recurrence with the *other* algorithm's reading is what issue 87 was:
the intentional branches of RTRRL took ``accumulating`` and ``current``
together, and ``current`` put the bootstrap state's derivative at the head of
the trace, where the TD error carries it with the opposite sign.

Neither choice is a rule's to make, which is why both are declared here, once,
rather than discovered from whichever rule is downstream.
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
