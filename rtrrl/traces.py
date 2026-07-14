"""资格迹工具。"""
from functools import partial
from flax import linen as nn
import jax
import jax.numpy as jnp

from models.neural_networks import FADense
import numpy as np


@partial(jax.jit, static_argnames=('trace_mode'))
def trace_update(grads, z, gamma_lambda, trace_mode: str = 'accumulate', alpha=None, _I=1):
    """更新资格迹;若给定 d 则同时计算梯度。

    参考 Sutton & Barto, 1998, p. 275
    Args:
        grads: 损失对模型参数的即时梯度
        z: 资格迹
        I: 累积 gamma,用于 episodic 任务
        trace_mode: "accumulate" 或 "dutch" 之一
        gamma_lambda: 折扣因子
        alpha: 学习率,用于 dutch trace
    """

    def accumulate(_z, _g):
        """累积资格迹。

        z ← λz + α I @ ∇f。
        """
        return gamma_lambda * _z + (_I * _g.T).T

    # def acc_tanh(_z, _g):
    #     """带 tanh 压缩的累积资格迹。"""
    #     return jnp.tanh(accumulate(_z, _g))

    def dutch(_z, _g):
        """Dutch 资格迹。

        z ← γλz + α[1 - γλ e.T ∇f] ∇f
        """
        return gamma_lambda * _z + (1 - alpha * gamma_lambda * (_z.T @ _g)) * _g

    return jax.tree.map(locals()[trace_mode], z, grads)


@partial(jax.jit, static_argnames=('trace_mode',))
def compute_updates(z, trace_mode: str = 'accumulate', d=None, dutch_diff=None, alpha=None, grads=None):
    """根据资格迹计算梯度。"""
    # 资格迹乘 TD 误差。
    grads = jax.tree.map(lambda t: (d.T * t.T).T,  z)
    if trace_mode == 'dutch':
        grads = jax.tree.map(lambda _z, _g: _g + alpha * (dutch_diff.T * (_z-_g).T).T, z, grads)
    return grads


def init_trace(params, batch_shape=()):
    """初始化资格迹:zθ ← 0(d' 维资格迹向量)。"""
    return jax.tree.map(lambda x: jnp.zeros(batch_shape+x.shape), params)


class TraceModel(nn.Module):
    """预测资格迹的模型。"""

    flat_shapes: list
    f_align: bool = True

    def setup(self):
        """为扁平化示例 trace 的每个分量各初始化一个模型。"""
        self.models = [FADense(np.prod(s), f_align=self.f_align) for s in self.flat_shapes]

    def __call__(self, obs):
        """将给定 pytree 扁平化,并用线性函数分别预测每个分量。"""
        return [m(obs).reshape(s) for m, s in zip(self.models, self.flat_shapes)]
