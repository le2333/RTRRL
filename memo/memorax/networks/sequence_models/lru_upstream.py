"""Our LRU computing what the RTRRL authors' two revisions compute.

``upstream_lru.py`` beside this is their file, transcribed, and it answers one
question: does our arithmetic match theirs. It cannot answer the other one --
whether our framework reproduces their *performance* -- because it is their
module in their shape, and the kernel that would have to run it is ours.

These two are that instead. Each is our ``LRUCell`` with one thing put back the
way one of their revisions has it, so the RTRRL kernel can be pointed at it and a
learning curve comes out that is comparable to theirs by construction: same
parameter tree, same paths, same initialiser spending the same key, because
nothing here touches ``setup``. What differs between an arm and ours is only the
line named in its docstring.

Which is the point. Comparing our correct LRU against their published one and
finding a gap would attribute to nothing -- it could be our framework failing to
reproduce them, or it could be the defect. An arm that reproduces the defect
separates those: if it matches their run, the framework is faithful, and whatever
their defect costs is then measurable against our correct arm rather than mixed
into it.

Neither of these is a thing to train with. They exist to be compared against.
"""

from __future__ import annotations

import jax.numpy as jnp

from memorax.utils.typing import Array, Carry

from .lru import LRUCell


class PublishedLRUCell(LRUCell):
    """Their ``b71fd6e``: the recurrence, and the input gain's logarithm.

    The revision the paper was published at accumulates the influence matrix for
    ``B`` as ``Lambda * grad + outer(gamma_log, x)``, using the log of the input
    gain where the derivative calls for the gain itself. Every other appearance
    of that parameter, there and here, exponentiates it.

    The consequence is not a small one and is worth stating in the file that
    reproduces it. ``gamma_log`` is initialised at ``log(sqrt(1 - |lambda|^2))``,
    which is negative, while ``exp(gamma_log)`` is positive. So this influence
    matrix does not merely have the wrong scale, it has the wrong sign in every
    element, and the gradient it produces for ``B`` points the other way. Their
    commit ``0dbd780``, fourteen months after publication, changed the line to
    ``jnp.exp(self.gamma_log)``.

    Only the two ``B`` Jacobians move. The recurrence, the readout, the other
    three Jacobians and every parameter are our own.
    """

    def local_jacobian(self, carry: Carry, z: Carry, inputs: Array, **kwargs):
        decay, jacobians = super().local_jacobian(carry, z, inputs, **kwargs)
        published = jnp.einsum("h,btf->bthf", self.gamma_log, inputs)
        return decay, {
            **jacobians,
            "B_real": published,
            "B_imag": 1j * published,
        }


class RewrittenLRUCell(LRUCell):
    """Their ``4301943``: the input gain, and no recurrence at all.

    After publication, ``8d27f18`` rewrote their forward pass around
    ``jax.lax.associative_scan`` with a ``reshape(-1, input_dim)`` before it.
    RTRRL steps one transition at a time, so that reshape makes a sequence of
    length one, the scan over it is the identity, and ``h_tminus1`` survives only
    to supply ``hidden_dim``. Their state is ``B_norm x_t`` with no history in it.

    The influence matrices were not rewritten with it. They still accumulate
    under ``Lambda``, as though the carry they credit still decayed into the next
    one. So this arm is a forward pass with no memory and a credit assignment
    that believes there is memory -- which is reproduced here the only way it can
    be, by having the two disagree: ``__call__`` reports a decay of zero, so the
    scan yields ``B_norm x_t``, while ``local_jacobian`` keeps reporting
    ``Lambda`` and the sensitivities accumulate under it.

    Zero and not one. A decay of one would accumulate the states undamped, which
    is a third thing that neither implementation computes.

    ``0dbd780`` added the missing exponential on top of this, so this revision has
    the input gain right and the recurrence gone. It is their HEAD.
    """

    def __call__(self, x: Array, **kwargs) -> Carry:
        carry = super().__call__(x, **kwargs)
        return carry.replace(decay=jnp.zeros_like(carry.decay))
