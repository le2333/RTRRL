"""The LRU the RTRRL paper published, for one comparison and nothing else.

Our ``lru.py`` computes the same recurrence and the same real-time credit, but it
computes them differently: the recurrence is unrolled by an associative scan
instead of stepped, and the sensitivity reaches the gradient through a phantom
term instead of a hand-written ``custom_vjp``. Taking arithmetic apart that way
is where it gets quietly changed, so this file is the version ours answers to,
driven beside it and compared leaf by leaf in ``tests/test_lru_parity.py``.

Source: ``RTRRL-AAAI25`` commit ``b71fd6e``, file ``models/online_lru.py`` --
the revision the paper was published at. Their HEAD (``4301943``) is transcribed
separately, in ``upstream_lru_rewritten.py``, because the two revisions compute
different things and one file cannot be faithful to both: ``8d27f18`` after
publication rewrote the forward pass around ``jax.lax.associative_scan``, and
since RTRRL steps one transition at a time, ``reshape(-1, input_dim)`` there
makes a length-one sequence, the scan becomes the identity, and ``h_tminus1``
survives only to supply ``hidden_dim``. So the recurrence ``Lambda * h_tminus1``
below is absent there, and so -- for a different reason, given in that file -- is
the accumulation in the influence matrices.

What is below therefore accumulates and their HEAD does not. Both are reproduced,
by an arm each in ``lru_upstream.py``, and each arm is checked against its own
transcription rather than against a reading of the other's.

The arithmetic is untouched, down to the spelling of ``B_img`` and the order of
every multiplication, and the layout is upstream's rather than this repository's
so that ``git show b71fd6e:models/online_lru.py`` diffs against it in a few
lines. Those lines are the imports, this docstring, and the removal of the
``__main__`` demonstration. Nothing here may be tidied: a reassociated product
is a changed number, and this file exists to be the number that did not change.

One consequence of that faithfulness is worth naming in advance, because a test
asserts its exact shape rather than tolerating it. ``new_grad_B`` below adds
``jnp.outer(self.gamma_log, inputs)``, using the log of the input gain where the
derivative calls for the gain itself. Every other appearance of that parameter
exponentiates it. The commit ``0dbd780``, also after publication, changed this
line to ``jnp.exp(self.gamma_log)``; ours has always computed the exponential.
So the published influence matrix for ``B`` is ours scaled, per hidden unit, by
``gamma_log / exp(gamma_log)``, and the test asserts that factor exactly.
"""

from functools import partial
from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp
from flax import linen as nn

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


def nu_log_init(key, shape, r_max=1, r_min=0):
    u1 = jax.random.uniform(key, shape=shape)
    nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
    return nu_log


def theta_log_init(key, shape, max_phase=6.28):
    u2 = jax.random.uniform(key, shape=shape)
    theta_log = jnp.log(max_phase * u2)
    return theta_log


def gamma_log_init(key, shape, nu_log, theta_log):
    nu = jnp.exp(nu_log)
    theta = jnp.exp(theta_log)
    diag_lambda = jnp.exp(-nu + 1j * theta)
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


# Glorot initialization
def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


def get_lambda(nu_log, theta_log):
    Lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
    return Lambda


class OnlineLRU(nn.Module):
    d_hidden: int
    r_max: jnp.float32 = 1.0
    r_min: jnp.float32 = 0.0
    max_phase: jnp.float32 = 6.28
    """
    grad memory: dh_{t-1}/d lambda, dh_{t-1}/d gamma #1,
                 dh_c_{t-1}/d B #2
    """

    def setup(self):
        self.nu_log = self.param(
            "nu_log", nu_log_init, (self.d_hidden,), self.r_max, self.r_min
        )
        self.theta_log = self.param(
            "theta_log", theta_log_init, (self.d_hidden,), self.max_phase
        )
        self.gamma_log = self.param(
            "gamma_log", gamma_log_init, (self.d_hidden,), self.nu_log, self.theta_log
        )

    @nn.compact
    def __call__(self, carry, inputs):
        h_tminus1, grad_memory = carry
        input_dim = inputs.shape[-1]
        hidden_dim = h_tminus1.shape[-1]

        B_real = self.param(
            "B_real",
            partial(matrix_init, normalization=jnp.sqrt(2 * input_dim)),
            (hidden_dim, input_dim),
        )

        B_img = self.param(
            "B_img",
            partial(matrix_init, normalization=jnp.sqrt(2 * input_dim)),
            (hidden_dim, input_dim),
        )

        Lambda = get_lambda(self.nu_log, self.theta_log)
        B = B_real + 1j * B_img

        B_norm = B * jnp.exp(jnp.expand_dims(self.gamma_log, axis=-1))

        h_t = (Lambda * h_tminus1) + (inputs @ B_norm.squeeze().transpose())

        new_grad_lambda = Lambda * grad_memory[0] + h_tminus1

        new_grad_gamma = (
            Lambda * grad_memory[1] + (inputs @ jnp.swapaxes(B, -1, -2)).squeeze()
        )

        new_grad_B = (jnp.expand_dims(Lambda, axis=-1)) * grad_memory[2] + jnp.outer(
            self.gamma_log, inputs
        )

        new_grad_memory = (
            new_grad_lambda,
            new_grad_gamma,
            new_grad_B,
        )
        new_carry = (h_t, new_grad_memory)
        return new_carry, new_carry

    def to_lambda(self, x):
        return get_lambda(self.nu_log, self.theta_log)


class OnlineLRUCell(nn.RNNCellBase):
    d_hidden: int

    @nn.compact
    def __call__(self, carry, x_t):
        def f(mdl, carry, x_t):
            return mdl(carry, x_t)

        def fwd(mdl: OnlineLRU, carry, x_t):
            f_out, vjp_func = nn.vjp(f, mdl, carry, x_t)
            _, vjp_to_lambda = nn.vjp(lambda m, x: m.to_lambda(x), mdl, x_t)
            return f_out, (
                vjp_func,
                f_out[1][1],
                vjp_to_lambda,
                mdl.gamma_log,
            )  # output, residual

        def bwd(residuals, y_t):
            # y_t =(partial{output}/partial{h_{t}},ignore the rest
            # grad_memory = \partial{h_{t-1}} \partial{lambda},
            # \partial{h_{t-1},c1} \partial{gamma},\partial{h_{t-1}} \partial{B}
            vjp_func, new_grad_memory, vjp_to_lambda, gamma_log = residuals
            params_t, *inputs_t = vjp_func(y_t)
            d_output_d_h = y_t[1][0]

            d_output_d_lambda = d_output_d_h * new_grad_memory[0]
            d_params_rec = vjp_to_lambda(d_output_d_lambda)[0]
            correct_nu_log, correct_theta_log = (
                d_params_rec["params"]["nu_log"],
                d_params_rec["params"]["theta_log"],
            )

            correct_gamma_log = (d_output_d_h * new_grad_memory[1]).real * jnp.exp(
                gamma_log
            )
            grad_B = jnp.expand_dims(d_output_d_h, -1) * new_grad_memory[2]
            params_t1 = flax.core.unfreeze(params_t)
            params_t1["params"]["nu_log"] = correct_nu_log
            params_t1["params"]["theta_log"] = correct_theta_log
            params_t1["params"]["gamma_log"] = correct_gamma_log.real
            params_t1["params"]["B_real"] = grad_B.real
            params_t1["params"]["B_img"] = -grad_B.imag
            return (params_t1, *inputs_t)

        online_lru_cell_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        model_fn = OnlineLRU(self.d_hidden)
        (h_t, new_grad_memory), (h_t, new_grad_memory) = online_lru_cell_grad(
            model_fn, carry, x_t
        )
        return (h_t, new_grad_memory), (h_t)  # carry, output


class OnlineLRULayer(nn.RNNCellBase):
    d_hidden: int

    @nn.compact
    def __call__(self, carry, x_t):
        h_tminus1, _ = carry
        hidden_dim = h_tminus1.shape[-1]

        C_real = self.param(
            "C_real",
            partial(matrix_init, normalization=jnp.sqrt(hidden_dim)),
            (self.d_hidden, hidden_dim),
        )

        C_img = self.param(
            "C_img",
            partial(matrix_init, normalization=jnp.sqrt(hidden_dim)),
            (self.d_hidden, hidden_dim),
        )

        online_lru = OnlineLRUCell(self.d_hidden)
        carry, h_t = online_lru(carry, x_t)
        C = C_real + 1j * C_img
        y_t = (h_t @ C.transpose()).real

        return carry, y_t  # carry, output

    def initialize_carry(self, rng, input_shape):
        batch_size = input_shape[0:1] if len(input_shape) > 1 else ()
        d_input = input_shape[0]
        hidden_init = jnp.zeros((*batch_size, self.d_hidden), dtype=jnp.complex64)
        memory_grad_init = (
            jnp.zeros((*batch_size, self.d_hidden), dtype=jnp.complex64),
            jnp.zeros((*batch_size, self.d_hidden), dtype=jnp.complex64),
            jnp.zeros((*batch_size, self.d_hidden, d_input), dtype=jnp.complex64),
        )
        return (hidden_init, memory_grad_init)
