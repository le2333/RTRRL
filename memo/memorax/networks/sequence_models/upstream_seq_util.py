"""The scan operators ``upstream_lru_rewritten.py`` is transcribed against.

Source: ``RTRRL-AAAI25`` commit ``4301943``, file ``models/seq_util.py``, whole
and unedited but for this docstring. Their ``online_lru.py`` imports two of these
three, and a transcription that reached for our own scan operator instead would
be comparing our associativity to theirs while claiming to compare cells.

``binary_operator_diag_spatial`` has no caller there or here. It is kept so the
file diffs against theirs in one place.
"""

import jax


@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.

    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)

    Returns:
        new element ( A_out, Bu_out ).
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


@jax.vmap
def binary_operator_diag_spatial(q_i, q_j):
    """Operator for parallel scan as above but stop the gradient for the recurrent connection."""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, jax.lax.stop_gradient(A_j * b_i) + b_j


@jax.vmap
def binary_operator_reset(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.

    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)

    Returns:
        new element ( A_out, Bu_out ).
    """
    A_i, b_i, c_i = q_i
    A_j, b_j, c_j = q_j
    return (
        (A_j * A_i) * (1 - c_j) + A_j * c_j,
        (A_j * b_i + b_j) * (1 - c_j) + b_j * c_j,
        c_i * (1 - c_j) + c_j,
    )
