"""Each backbone contributes the sequence its source specifies.

The compositions asserted here are the ones written down in
``2026-07-30-configuration-surface-design.md`` §4, one row per branch. Neither
source can be driven from this repository: the RTU settings are a paper with no
code here, and streaming-drl's file is fetched from a checkout ``STREAMING_DRL``
points at, which ``test_paper_parity`` skips without. So these compare the
composition against the recipe and not against a running reference.
"""

from __future__ import annotations

import dataclasses

import pytest

from memorax.networks.backbones import backbone
from memorax.networks.components import FFN, LayerNorm, LeakyReLU, Tanh
from memorax.networks.sequence_models import RNN, LRUConfig, RTUConfig

FEATURES = 8
HIDDEN = 16

# ``FFN(64) -> LayerNorm -> tanh -> RTU(hidden 192)``. The head that follows is
# the caller's, not the backbone's.
RTU = (FFN, LayerNorm, Tanh, RNN)

# ``FFN(128) -> LayerNorm -> LeakyReLU`` twice.
MLP = (FFN, LayerNorm, LeakyReLU, FFN, LayerNorm, LeakyReLU)


def kinds(name: str) -> tuple[type, ...]:
    built = backbone(name, features=FEATURES, hidden_dim=HIDDEN, output_dim=FEATURES)
    return tuple(type(component) for component in built)


def test_rtu_is_an_encoder_a_normalisation_an_activation_and_one_recurrent_layer():
    assert kinds("rtu") == RTU


def test_the_rtu_encoder_is_the_width_it_was_asked_for():
    encoder, *_ = backbone(
        "rtu", features=FEATURES, hidden_dim=HIDDEN, output_dim=FEATURES
    )

    assert encoder.features == FEATURES


def test_mlp_is_two_feedforward_blocks():
    assert kinds("mlp") == MLP


def test_both_mlp_blocks_are_the_hidden_width():
    built = backbone("mlp", features=FEATURES, hidden_dim=HIDDEN, output_dim=FEATURES)
    widths = [one.features for one in built if isinstance(one, FFN)]

    assert widths == [HIDDEN, HIDDEN]


@pytest.mark.parametrize("config", (RTUConfig, LRUConfig))
def test_a_recurrent_cell_does_not_offer_a_choice_of_activation(config):
    """RTU's nonlinearity is inside its recurrence and LRU has none."""

    named = [field.name for field in dataclasses.fields(config)]

    assert "activation_fn" not in named


def test_an_unknown_backbone_is_refused():
    with pytest.raises(ValueError, match="unknown backbone"):
        backbone("nonesuch", features=FEATURES, hidden_dim=HIDDEN)
