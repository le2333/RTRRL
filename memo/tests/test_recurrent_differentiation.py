"""The recurrent kernel and the algorithm differentiating it share one boundary.

The state behind that boundary is deliberately opaque: structured RTRL, full
RTRL, RFLO, and truncation do not have to agree on a representation.  What a
sequence consumer needs is only initialization, the parameter-initialization
context, and one differentiated recurrent step.
"""

from __future__ import annotations

from contextlib import nullcontext
from importlib import import_module


class ExampleDifferentiation:
    """A non-RTRL implementation proving the contract is structural."""

    def initialize(self, key, input_shape):
        return {"key": key, "input_shape": input_shape}

    def initialization(self):
        return nullcontext()

    def __call__(self, params, inputs, done, carry, state):
        del params, done
        return carry + inputs, inputs, {"previous": state}


def test_recurrent_differentiation_has_one_public_structural_contract():
    differentiation = import_module("memorax.networks.differentiation")
    contract = differentiation.RecurrentDifferentiation
    implementation = ExampleDifferentiation()

    assert isinstance(implementation, contract)

    state = implementation.initialize("key", (2, 3))
    with implementation.initialization():
        next_carry, output, next_state = implementation(
            {}, 2, False, 5, state
        )

    assert (next_carry, output) == (7, 2)
    assert next_state == {"previous": state}


def test_truncated_bptt_is_a_general_implementation_of_the_same_contract():
    differentiation = import_module("memorax.networks.differentiation")

    class Core:
        def apply(self, variables, inputs, done, carry):
            assert variables == {"params": "parameters"}
            assert done is False
            return carry + inputs, inputs * 3

    implementation = differentiation.TruncatedBPTT(Core())

    assert isinstance(implementation, differentiation.RecurrentDifferentiation)
    assert implementation.initialize("key", (2, 3)) is None
    with implementation.initialization():
        result = implementation("parameters", 2, False, 5, {"ignored": True})

    assert result == (7, 6, None)


def test_generic_recurrent_wrappers_do_not_own_kernel_differentiation():
    sequence_model = import_module(
        "memorax.networks.sequence_models.sequence_model"
    ).SequenceModel
    rnn = import_module("memorax.networks.sequence_models.rnn")
    memoroid = import_module("memorax.networks.sequence_models.memoroid")

    for generic in (
        sequence_model,
        rnn.RNNCellBase,
        rnn.RNN,
        memoroid.MemoroidCellBase,
        memoroid.Memoroid,
    ):
        assert "local_jacobian" not in generic.__dict__
        assert "initialize_sensitivity" not in generic.__dict__
