"""A declaration and what a kernel produces, checked against each other.

Every name published for a run has to be one the run emits, and every reading
emitted has to be one that was declared. Neither direction held before: the
driver drops a series it cannot find without saying so, and the shipped entry
declared eighteen readings against a kernel that emitted none.
"""

from __future__ import annotations

import importlib

import jax
import pytest
from test_rtrrl import ENVS, build

from memorax.networks.sequence import PLACES
from memorax.readings import declared, reading, readings, taken
from tests.support.numerics import flattened

layered = importlib.import_module("memorax.algorithms.stream_ac")
rtrrl = importlib.import_module("memorax.algorithms.rtrrl_aaai")


def emitted(metrics) -> set[str]:
    """Every path a step's readings actually carry, minus the transition."""

    found = flattened({"forward": metrics.forward, "update": metrics.update})
    return {name.lstrip(".").replace("/.", ".").replace("/", ".") for name in found}


def test_a_declaration_names_what_it_can_offer():
    assert declared(layered.Reports)[:2] == (
        "forward.actor.log_prob",
        "forward.actor.entropy",
    )
    assert "update.actor.step_size" in declared(layered.Reports)
    # RTRRL steps a block at a time now, so each of the three offers the step
    # size it took and there is no group left to offer one instead.
    assert "update.actor.step_size" in declared(rtrrl.Reports)
    assert "update.critic.step_size" in declared(rtrrl.Reports)
    assert not [name for name in declared(rtrrl.Reports) if "_step." in name]


def test_a_split_reading_becomes_one_name_per_part():
    plain = taken(layered.Reports())
    split = taken(layered.Reports(), parts=PLACES)
    assert len(split) == len(plain) + 2 * 2 * (len(PLACES) - 1)
    assert "update.actor.grad_norm.recurrence" in split


def test_taking_less_publishes_less():
    everything = set(taken(rtrrl.Reports()))
    less = set(
        taken(
            rtrrl.Reports(entropy=False, torso=rtrrl.BlockReports(False, False, False))
        )
    )
    assert less < everything
    assert "forward.actor.entropy" in everything - less
    assert "update.torso.grad_norm" in everything - less


def test_what_rtrrl_declares_is_what_rtrrl_emits():
    """Both directions: nothing published that is absent, nothing absent that is published."""

    agent = build()
    state = agent.init(jax.random.key(0))
    _, metrics = agent.train(jax.random.key(1), state, 2 * ENVS)

    published = set(taken(rtrrl.Reports(), parts=PLACES))
    produced = emitted(metrics)
    assert produced == published


def test_a_field_that_is_not_declared_is_refused():
    """The declaration is the only way in, so a plain field cannot slip through."""

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Sloppy:
        good: bool = reading(at="update.good")
        bad: bool = True

    with pytest.raises(ValueError, match="not declared with"):
        taken(Sloppy())


def test_a_nested_declaration_hangs_off_one_path():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Inner:
        one: bool = reading(at="one")

    @dataclass(frozen=True)
    class Outer:
        inner: Inner = readings(of=Inner, at="update.block")

    assert taken(Outer()) == ("update.block.one",)
