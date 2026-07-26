"""Pure temporal-difference target primitives."""

from __future__ import annotations


def make_td0():
    """Build TD(0) with an explicit, caller-owned bootstrap discount."""

    def td0(*, reward, value, next_value, bootstrap_discount):
        return reward + bootstrap_discount * next_value - value

    return td0
