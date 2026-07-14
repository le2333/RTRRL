"""ObGD(Online Gradient Descent)流式优化器,JAX 移植版。

参考: streaming-drl/optim.py (PyTorch 原始实现, arXiv 2410.14606)。

设计要点(见 docs/superpowers/specs/2026-07-04-jax-rl-component-library-design.md §4):
- eligibility trace (e) 由外部 jax_rl/traces/ 维护,ObGD 不内部维护 trace。
  这与 PyTorch 原版不同(原版内部维护 e),目的是让 trace 成为可复用的
  独立组件,ObGD 只保留其核心贡献——限幅。
- ObGD 作为 optax.GradientTransformation,但 update 带 extra args
  (delta, z_sum / z),因为 optax 标准 (updates, state, params) 签名
  缺少这些标量。因此 ObGD 不能放进 optax.chain 由标准 optax.update 调用,
  需由算法侧直接调用:
      updates, state = obgd.update(updates, state, delta=..., z_sum=...)
- 限幅: step_size = lr / max(|delta|_bar · z_sum · lr · kappa, 1),
  保证每步更新被限制在有界范围内。

提供两个变换:
- obgd:          标准版,无状态,只做限幅 + SGD scale。
- adaptive_obgd: 自适应版,维护二阶矩 v (RMSProp 式),需内部状态。
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


def obgd(lr: float = 1.0, kappa: float = 2.0) -> optax.GradientTransformation:
    """标准 ObGD 限幅变换(trace 外部维护,无状态)。

    调用方需外部完成:
      - trace 更新:   z <- gamma * lambda * z + grad   (jax_rl/traces/)
      - updates 组合:  updates = delta * z
      - z_sum 计算:    z_sum = sum |z|                  (jax_rl/utils.tree_abs_sum)
    再调用:
      updates, state = obgd.update(updates, state, delta=delta, z_sum=z_sum)

    Args:
        lr:    基础学习率(ObGD 论文默认 1.0)。
        kappa: 限幅系数(默认 2.0)。
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None, *, delta, z_sum):
        del params
        # delta_bar = max(|delta|, 1),避免对小 delta 过度放大
        delta_bar = jnp.maximum(jnp.abs(delta), 1.0)
        dot_product = delta_bar * z_sum * lr * kappa
        # 限幅:dot_product > 1 时缩小 step_size,否则用 lr
        step_size = jnp.where(dot_product > 1.0, lr / dot_product, lr)
        updates = jax.tree.map(lambda u: step_size * u, updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


class AdaptiveObGDState(NamedTuple):
    """AdaptiveObGD 状态:二阶矩 v 与步数计数。"""

    v: Any  # 与参数同构的二阶矩
    count: jnp.ndarray  # 步数(用于 bias correction)


def adaptive_obgd(
    lr: float = 1.0,
    kappa: float = 2.0,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """自适应 ObGD:维护二阶矩 v(RMSProp 式),对更新做逐元素自适应缩放。

    与标准版的区别:
      - 维护 v <- beta2 * v + (1-beta2) * (delta * z)^2,bias correction 得 v_hat。
      - z_sum 用自适应形式: z_sum = sum |z / sqrt(v_hat + eps)|。
        该量依赖内部 v_hat,故由本变换内部计算(不接收外部 z_sum)。
      - 更新: updates <- step_size * (delta * z) / sqrt(v_hat + eps)。

    调用方需外部完成 trace 更新与 updates 组合,再调用:
      updates, state = adaptive_obgd.update(updates, state, delta=delta, z=z)

    Args:
        lr:    基础学习率。
        kappa: 限幅系数。
        beta2: 二阶矩衰减率(默认 0.999)。
        eps:   分母稳定项(默认 1e-8)。
    """

    def init_fn(params):
        v = jax.tree.map(jnp.zeros_like, params)
        return AdaptiveObGDState(v=v, count=jnp.zeros((), jnp.int32))

    def update_fn(updates, state, params=None, *, delta, z):
        del params
        count = state.count + 1
        # v <- beta2 * v + (1-beta2) * updates^2  (updates = delta * z)
        v = jax.tree.map(
            lambda vi, u: beta2 * vi + (1.0 - beta2) * jnp.square(u),
            state.v,
            updates,
        )
        # bias correction
        bias = 1.0 - jnp.power(beta2, count)
        v_hat = jax.tree.map(lambda vi: vi / bias, v)
        # 自适应 z_sum = sum |z / sqrt(v_hat + eps)|
        per_leaf = jax.tree.map(
            lambda zi, vh: jnp.sum(jnp.abs(zi / jnp.sqrt(vh + eps))),
            z,
            v_hat,
        )
        z_sum = jax.tree_util.tree_reduce(jnp.add, per_leaf, 0.0)
        # 限幅
        delta_bar = jnp.maximum(jnp.abs(delta), 1.0)
        dot_product = delta_bar * z_sum * lr * kappa
        step_size = jnp.where(dot_product > 1.0, lr / dot_product, lr)
        # 自适应更新
        updates = jax.tree.map(
            lambda u, vh: step_size * u / jnp.sqrt(vh + eps),
            updates,
            v_hat,
        )
        return updates, AdaptiveObGDState(v=v, count=count)

    return optax.GradientTransformation(init_fn, update_fn)
