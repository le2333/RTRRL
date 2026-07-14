"""parallel scan 工具。"""

import jax


@jax.vmap
def binary_operator(q_i, q_j):
    """线性递归 parallel scan 的二元算子。假设 A 为对角矩阵。

    Args:
        q_i: 位置 i 处 (A_i, Bu_i) 组成的元组       (P,), (P,)
        q_j: 位置 j 处 (A_j, Bu_j) 组成的元组       (P,), (P,)

    Returns:
        新元素 ( A_out, Bu_out )。
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


@jax.vmap
def binary_operator_diag_spatial(q_i, q_j):
    """同上的 parallel scan 算子,但对循环连接停止梯度。"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, jax.lax.stop_gradient(A_j * b_i) + b_j


@jax.vmap
def binary_operator_reset(q_i, q_j):
    """线性递归 parallel scan 的二元算子。假设 A 为对角矩阵。

    Args:
        q_i: 位置 i 处 (A_i, Bu_i) 组成的元组       (P,), (P,)
        q_j: 位置 j 处 (A_j, Bu_j) 组成的元组       (P,), (P,)

    Returns:
        新元素 ( A_out, Bu_out )。
    """
    A_i, b_i, c_i = q_i
    A_j, b_j, c_j = q_j
    return (
        (A_j * A_i) * (1 - c_j) + A_j * c_j,
        (A_j * b_i + b_j) * (1 - c_j) + b_j * c_j,
        c_i * (1 - c_j) + c_j,
    )
