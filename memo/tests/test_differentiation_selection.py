"""Recurrent differentiation is selected inside the recurrent kernel branch.

Structures are fixed for a study and carry no ``search``; parameters are what a
trial samples. The two settings do not build the same state and do not
initialise the recurrent cell through the same method, so trials that differed
in it would not be trials of one space.
"""

from __future__ import annotations

import jax
from test_blocks import ours

from entries import stream_ac
from memorax.parameters import KIND

KINDS = ("exact_rtrl", "tbptt")


def test_the_two_settings_do_not_build_the_same_state():
    key = jax.random.key(0)
    shapes = {
        name: jax.tree.structure(
            ours(differentiation=name).init(key).actor.recurrence.differentiation_state
        )
        for name in KINDS
    }

    assert shapes["exact_rtrl"] != shapes["tbptt"]


def test_the_two_settings_do_not_initialise_through_the_same_method():
    """Structured RTRL and TBPTT initialize through their executed methods."""

    contexts = {
        name: type(
            ours(differentiation=name).core.actor.block.differentiation.initialization()
        ).__name__
        for name in KINDS
    }

    assert contexts["exact_rtrl"] != contexts["tbptt"]


def test_differentiation_is_declared_inside_the_rtu_structure():
    node = stream_ac.PARAMETERS["backbone"]["rtu"]["differentiation"]

    assert set(node[KIND].valid.values) == set(KINDS)
