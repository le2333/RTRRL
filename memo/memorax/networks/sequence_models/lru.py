"""LRU（Linear Recurrent Unit）线性循环单元模块。

LRU 是一种对角复数线性状态空间模型：隐状态按复数特征值 λ = exp(-e^ν + i·e^θ)
逐步衰减/旋转，输入经复数矩阵 B 投影入状态，输出由复数矩阵 C 读出并取实部，
外加输入直连项 D。其线性递推天然满足结合律，可用 Memoroid 的并行扫描高效展开。
本文件还实现了 RTRL 所需的局部雅可比与敏感度初始化。
"""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from flax.typing import Dtype

from memorax.building import ComponentFamily
from memorax.networks.differentiation import TruncatedBPTT
from memorax.utils.axes import add_feature_axis, broadcast_done, init, last
from memorax.utils.typing import Array, Carry, Key

from .memoroid import MemoroidCellBase


@dataclass(frozen=True)
class LRUStructuredRTRL:
    """The exact structured sensitivity recurrence implemented by LRU/Memoroid."""

    core: Any

    def initialize(self, key, input_shape):
        return self.core.cell.initialize_sensitivity(key, input_shape)

    def initialization(self):
        return nullcontext()

    def __call__(self, params, inputs, done, carry, state):
        return self.core.apply(
            {"params": params},
            inputs,
            done,
            carry,
            state,
            method=_structured_rtrl,
        )


def _construct_differentiation(selection, builder, *, core):
    del builder
    if selection.kind == "exact_rtrl":
        return LRUStructuredRTRL(core)
    return TruncatedBPTT(core)


LRU_DIFFERENTIATION_FAMILY = ComponentFamily(
    branches={"exact_rtrl": (), "tbptt": ()},
    construct=_construct_differentiation,
)


def _propagate_sensitivities(decay, jacobians, state, done):
    batch, steps, hidden = decay.shape
    done = add_feature_axis(done)
    propagated = {}

    @jax.vmap
    def binary_operator(a, b):
        state_i, decay_i = a
        state_j, decay_j = b
        return decay_j * state_i + state_j, decay_j * decay_i

    for name in sorted(jacobians):
        jacobian = jacobians[name]
        previous = state[name]
        param_shape = jacobian.shape[3:]
        param_size = math.prod(param_shape)

        jacobian = jacobian.reshape(batch, steps, hidden * param_size)
        previous = previous.reshape(batch, 1, hidden * param_size)
        decay_for_parameter = jnp.where(
            done, 0, jnp.repeat(decay, param_size, axis=-1)
        )

        values = jnp.concatenate([previous, jacobian], axis=1)
        decays = jnp.concatenate(
            [jnp.ones_like(previous), decay_for_parameter], axis=1
        )
        values, _ = jax.lax.associative_scan(
            binary_operator, (values, decays), axis=1
        )
        propagated[name] = last(values).reshape(batch, 1, hidden, *param_shape)
    return propagated


def _structured_rtrl(core, inputs, done, carry, state, **kwargs):
    """Run the LRU/Memoroid forward and its diagonal sensitivity recurrence."""

    z = core.cell(inputs, **kwargs)
    phantom = core.cell.compute_phantom(state)
    carry = core.cell.inject_phantom(carry, phantom)

    hidden, next_carry = core.scan_fn(z, carry, done)
    output = core.cell.read(hidden, inputs, **kwargs)
    previous_carry = jax.tree.map(
        lambda initial, history: jnp.concatenate([initial, init(history)], axis=1),
        carry,
        hidden,
    )
    reset = add_feature_axis(done)
    previous_carry = jax.tree.map(
        lambda value: jnp.where(broadcast_done(reset, value), 0, value),
        previous_carry,
    )
    decay, jacobians = core.cell.local_jacobian(previous_carry, z, inputs)
    next_state = _propagate_sensitivities(decay, jacobians, state, done)
    return next_carry, output, next_state


def _nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    """初始化 ν=log(-0.5·log|λ|²)，使特征值模长 |λ| 落在 [r_min, r_max] 环内。"""
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def _theta_init(key, shape, max_phase, dtype=jnp.float32):
    """初始化相位参数 θ=log(θ_max·u)，控制特征值的旋转角度。"""
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def _gamma_log_init(key, lamb):
    """初始化归一化增益 γ=sqrt(1-|λ|²)（取对数），补偿状态衰减带来的能量损失。"""
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


def _matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    """按给定归一化系数缩放的正态初始化，用于 B/C/D 等矩阵。"""
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


@struct.dataclass
class LRUConfig:
    """LRU 配置。

    Args:
        features: 输入特征维度 i（output_dim 为 None 时也作为输出维度）。
        hidden_dim: 复数隐状态维度 h。
        output_dim: 输出维度 o。None 时退化为 features（向后兼容旧行为：
            C 形状 (features, hidden)、D 为逐元素直连 (features,)）。
            指定为整数时解耦读出：C 形状 (output_dim, hidden)、D 为满阵
            (output_dim, features)，与原版 OnlineLRULayer 的 d_output 对齐。
        expose_hidden: 为 True 时旁路 C/D 读出，直接输出复数状态的实部与虚部
            拼接 [Re h, Im h]（2*hidden 维，与 RTU 输出语义一致），由下游头自行读出。
        r_min / r_max: 初始化特征值模长的下/上界。
        max_phase: 初始化特征值相位的上界。
    """

    features: int
    hidden_dim: int
    output_dim: int | None = None
    expose_hidden: bool = False
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32


@struct.dataclass
class LRUCarry:
    """LRU 的 carry：复数隐状态 state 与逐步衰减因子 decay（即 λ）。"""

    state: Array
    decay: Array


class LRUCell(MemoroidCellBase):
    """LRU 记忆单元：对角复数线性递推，配合 Memoroid 结合律扫描。"""

    config: LRUConfig

    def setup(self):
        # 特征值以极坐标参数化：θ_log 控相位、ν_log 控模长，均取对数保证正性。
        self.theta_log = self.param(
            "theta_log",
            partial(_theta_init, max_phase=self.config.max_phase),
            (self.config.hidden_dim,),
        )
        self.nu_log = self.param(
            "nu_log",
            partial(_nu_init, r_min=self.config.r_min, r_max=self.config.r_max),
            (self.config.hidden_dim,),
        )
        # γ 为输入归一化增益，依赖 ν/θ 初始化以稳定各步能量。
        self.gamma_log = self.param(
            "gamma_log", _gamma_log_init, (self.nu_log, self.theta_log)
        )

        # B/C 拆成实部与虚部分别作为可学习参数（复数投影矩阵）。
        self.B_real = self.param(
            "B_real",
            partial(_matrix_init, normalization=jnp.sqrt(2 * self.config.features)),
            (self.config.hidden_dim, self.config.features),
        )
        self.B_imag = self.param(
            "B_imag",
            partial(_matrix_init, normalization=jnp.sqrt(2 * self.config.features)),
            (self.config.hidden_dim, self.config.features),
        )
        # 读出矩阵 C/D。expose_hidden 时旁路读出（直接暴露状态），故不创建。
        if not self.config.expose_hidden:
            out_dim = (
                self.config.features
                if self.config.output_dim is None
                else self.config.output_dim
            )
            self.C_real = self.param(
                "C_real",
                partial(_matrix_init, normalization=jnp.sqrt(self.config.hidden_dim)),
                (out_dim, self.config.hidden_dim),
            )
            self.C_imag = self.param(
                "C_imag",
                partial(_matrix_init, normalization=jnp.sqrt(self.config.hidden_dim)),
                (out_dim, self.config.hidden_dim),
            )
            if self.config.output_dim is None:
                # 向后兼容：输出维=输入维，D 为逐元素输入直连。
                self.D = self.param("D", _matrix_init, (self.config.features,))
            else:
                # 输出维解耦：D 为满阵 (output_dim, features)，对齐原版 D 的 (d_output, input)。
                self.D = self.param("D", _matrix_init, (out_dim, self.config.features))

    def __call__(self, x: Array, **kwargs) -> Carry:
        """将输入投影入复数状态，并给出每步衰减因子 λ。

        Returns:
            LRUCarry(state, decay)，其中 state=B_norm·x，decay 为广播的 λ。
        """
        B, T, _ = x.shape

        # 复对角特征值 λ = exp(-e^ν + i·e^θ)：模长<1 保证稳定衰减。
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        # 归一化后的复数输入矩阵 B（含 γ 增益）。
        B_norm = (self.B_real + 1j * self.B_imag) * jnp.exp(self.gamma_log)[:, None]

        # 每个时间步的衰减都相同（时不变系统），广播到 (B, T, hidden)。
        decay = jnp.broadcast_to(diag_lambda, (B, T, self.config.hidden_dim))

        # 输入投影到隐空间：state[b,t,i] = Σ_j B_norm[i,j]·x[b,t,j]。
        state = jnp.einsum('ij,btj->bti', B_norm, x)

        return LRUCarry(state=state, decay=decay)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        """线性递推的结合律算子：h_b = λ_b·h_a + x_b，衰减相乘累积。"""
        return LRUCarry(
            state=b.decay * a.state + b.state,
            decay=b.decay * a.decay,
        )

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        """从复数状态读出实数输出。

        expose_hidden 时旁路 C/D，直接返回 [Re h, Im h]（2*hidden 维）；
        否则 y = Re(C·h) + 输入直连项。直连项在 output_dim 为 None 时为逐元素
        D⊙x（(features,)），output_dim 解耦时为满阵投影 x·Dᵀ（(output_dim, features)）。
        """
        if self.config.expose_hidden:
            return jnp.concatenate([h.state.real, h.state.imag], axis=-1)
        C = jax.lax.complex(self.C_real, self.C_imag)
        y = jnp.einsum('ij,btj->bti', C, h.state).real
        if self.config.output_dim is None:
            return y + self.D * x
        return y + jnp.einsum('of,btf->bto', self.D, x)

    def initialize_carry(self, key: jax.Array, input_shape: tuple[int, ...]) -> Carry:
        """零复数状态、单位衰减的初始 carry。"""
        *batch_dims, _ = input_shape
        state = jnp.zeros((*batch_dims, 1, self.config.hidden_dim), dtype=jnp.complex64)
        decay = jnp.ones((*batch_dims, 1, self.config.hidden_dim), dtype=jnp.complex64)
        return LRUCarry(state=state, decay=decay)

    def compute_phantom(self, sensitivity: dict) -> Array:
        phantom = 0
        for name, value in sensitivity.items():
            parameter = self.variables["params"][name]
            difference = parameter - jax.lax.stop_gradient(parameter)
            phantom = phantom + jnp.sum(
                value * difference, axis=tuple(range(3, value.ndim))
            )
        return phantom

    def inject_phantom(self, carry: Carry, phantom: Array) -> Carry:
        """把 phantom 注入复数状态（RTRL 用），对原状态截断梯度。"""
        return carry.replace(state=jax.lax.stop_gradient(carry.state) + phantom)

    def local_jacobian(self, carry, z, inputs, **kwargs) -> tuple[Array, dict]:
        """给出状态对各参数的局部雅可比，供敏感度传播（RTRL）。

        Returns:
            (decay, jacobians)：decay 为每步 λ；jacobians 为各参数对应的
            ∂state/∂param 项（nu_log/theta_log/gamma_log/B_real/B_imag）。
        """
        lam = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        gamma_exp = jnp.exp(self.gamma_log)

        B, T = inputs.shape[:2]
        decay_3d = jnp.broadcast_to(lam, (B, T, self.config.hidden_dim))

        # 依据 λ 与 state/input 的链式法则给出各参数的瞬时雅可比。
        return decay_3d, {
            "nu_log": -jnp.exp(self.nu_log) * lam * carry.state,
            "theta_log": 1j * jnp.exp(self.theta_log) * lam * carry.state,
            "gamma_log": z.state,
            "B_real": jnp.einsum('h,btf->bthf', gamma_exp, inputs),
            "B_imag": 1j * jnp.einsum('h,btf->bthf', gamma_exp, inputs),
        }

    def initialize_sensitivity(self, key: Key, input_shape: tuple) -> dict:
        """按各参数形状构造零敏感度字典（在线学习起点）。"""
        *batch_dims, _ = input_shape
        H = self.config.hidden_dim
        def z(*shape):
            return jnp.zeros((*batch_dims, 1, *shape), dtype=jnp.complex64)

        sensitivity = {
            "nu_log": z(H),
            "theta_log": z(H),
            "gamma_log": z(H),
            "B_real": z(H, self.config.features),
            "B_imag": z(H, self.config.features),
        }
        return sensitivity
