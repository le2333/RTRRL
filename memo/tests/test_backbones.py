"""Each backbone contributes the components it puts into a sequence.

``mlp`` is streaming-drl's, checked against ``stream_ac_continuous.py``:
``Linear(128) -> layer_norm -> leaky_relu`` twice, then the head.

``rtu`` is the cell and nothing else. The paper its settings come from publishes
no code, and memorax defines an ``RTUCell`` but composes nothing around it --
neither of its StreamAC examples uses one. An encoder in front would be this
repository's invention, so there is not one.
"""

from __future__ import annotations

import pytest

from memorax.networks.backbones import backbone
from memorax.networks.components import FFN, LayerNorm, LeakyReLU
from memorax.networks.sequence_models import RNN

FEATURES = 8
HIDDEN = 16

RTU = (RNN,)

# ``Linear(128) -> layer_norm -> leaky_relu``, twice.
MLP = (FFN, LayerNorm, LeakyReLU, FFN, LayerNorm, LeakyReLU)


def kinds(name: str) -> tuple[type, ...]:
    built = backbone(name, features=FEATURES, hidden_dim=HIDDEN, output_dim=FEATURES)
    return tuple(type(component) for component in built)


def test_rtu_is_the_recurrent_cell_and_nothing_else():
    assert kinds("rtu") == RTU


def test_the_rtu_cell_takes_the_input_width_it_was_given():
    (recurrence,) = backbone("rtu", features=FEATURES, hidden_dim=HIDDEN)

    assert recurrence.cell.config.features == FEATURES
    assert recurrence.cell.config.hidden_dim == HIDDEN


def test_mlp_is_two_feedforward_blocks():
    assert kinds("mlp") == MLP


def test_both_mlp_blocks_are_the_hidden_width():
    built = backbone("mlp", features=FEATURES, hidden_dim=HIDDEN, output_dim=FEATURES)
    widths = [one.features for one in built if isinstance(one, FFN)]

    assert widths == [HIDDEN, HIDDEN]


def test_an_unknown_backbone_is_refused():
    with pytest.raises(ValueError, match="unknown backbone"):
        backbone("nonesuch", features=FEATURES, hidden_dim=HIDDEN)
