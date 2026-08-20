"""Pure temporal-difference target primitives."""

from __future__ import annotations


def make_td0():
    """Build TD(0) over the ending that decides whether there is a future.

    The mask is the whole of what TD says about an ending, so it is formed here
    rather than by the caller: an episode cut off at its step limit was about to
    go on earning and the value of where it stopped is the best statement anyone
    has about the rest, while one that failed has nothing after it. Handed a
    discount already multiplied by something, this could not tell the two apart
    and neither could its callers.
    """

    def td0(*, reward, value, next_value, terminal, gamma):
        return reward + gamma * (1 - terminal) * next_value - value

    return td0
