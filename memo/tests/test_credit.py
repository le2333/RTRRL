"""``credit`` picks a compute graph, so it is declared as a structure.

Structures are fixed for a study and carry no ``search``; parameters are what a
trial samples. The two settings do not build the same state and do not
initialise the recurrent cell through the same method, so trials that differed
in it would not be trials of one space.
"""

from __future__ import annotations

import jax
from test_blocks import ours
from training_sdk.contract import StructureSpec

from entries import stream_ac
from memorax.rl.credit import CREDITS


def test_the_two_settings_do_not_build_the_same_state():
    key = jax.random.key(0)
    shapes = {
        name: jax.tree.structure(ours(credit=name).init(key).actor_sensitivity)
        for name in CREDITS
    }

    assert shapes["rtrl"] != shapes["tbptt"]


def test_the_two_settings_do_not_initialise_through_the_same_method():
    """Exact credit traces ``local_jacobian``; truncated credit traces the forward."""

    contexts = {
        name: type(ours(credit=name).actor_credit.initialization()).__name__
        for name in CREDITS
    }

    assert contexts["rtrl"] != contexts["tbptt"]


def test_credit_is_declared_as_a_structure():
    node = stream_ac.PARAMETERS["credit"]

    assert isinstance(node, StructureSpec)
    assert set(node.branches) == set(CREDITS)
