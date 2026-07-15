"""网络输出头（head）集合模块。

汇集各类强化学习输出头：价值/Q 函数（标量回归、分布式 C51、HL-Gauss）、
策略分布（离散 Categorical、连续 Gaussian、SquashedGaussian）、温度参数
（Alpha/Beta），以及通用价值函数（GVF/Horde）与后继特征等。
每个头统一约定 ``__call__`` 返回 ``(输出, 辅助字典)``，并可提供 ``loss`` 计算损失。
"""

from typing import Callable

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant

from memorax.utils.axes import add_feature_axis, remove_feature_axis
from memorax.utils.typing import Array, PyTree


def sigmoid_between(x: Array, lower: float, upper: float) -> Array:
    """将输入经 sigmoid 映射到 [lower, upper] 区间(复刻 streaming-rtrrl 的动作 bound)。"""
    return (upper - lower) * jax.nn.sigmoid(x) + lower


class DiscreteQNetwork(nn.Module):
    """离散动作 Q 网络：输出每个动作的标量 Q 值。

    Args:
        action_dim: 离散动作个数（输出维度）。
    """

    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> tuple[Array, dict]:
        """线性映射到各动作 Q 值，返回 (q_values, 空辅助字典)。"""
        q_values = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        return q_values, {}

    def loss(
        self, output: Array, aux: dict, targets: Array, **kwargs
    ) -> Array:
        """标准 1/2 均方误差（TD 回归损失）。"""
        return 0.5 * jnp.square(output - targets)


class ContinuousQNetwork(nn.Module):
    """连续动作 Q 网络：拼接状态与动作后回归标量 Q 值。"""

    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(
        self, x: Array, *, action: Array, **kwargs
    ) -> tuple[Array, dict]:
        """将状态特征与动作在特征维拼接，映射为单个 Q 值。"""
        # 拼接 [状态特征, 动作] 后线性到 1 维，再压掉尾部单位维。
        q_values = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init)(
            jnp.concatenate([x, action], axis=-1)
        )
        return jnp.squeeze(q_values, -1), {}

    def loss(
        self, output: Array, aux: dict, targets: Array, **kwargs
    ) -> Array:
        """1/2 均方误差损失。"""
        return 0.5 * jnp.square(output - targets)


class TwinContinuousQNetwork(nn.Module):
    """双 Q（twin/clipped double Q）网络：并行输出两个独立的 Q 估计。

    常用于 TD3/SAC，通过取较小者缓解 Q 值高估。
    """

    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(
        self, x: Array, *, action: Array, **kwargs
    ) -> tuple[tuple[Array, Array], dict]:
        """返回 ((q1, q2), 辅助字典)，两支共享输入但参数独立。"""
        inp = jnp.concatenate([x, action], axis=-1)
        # 两个命名独立的 Dense 头，各自估计一个 Q 值。
        q1 = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init, name="q1")(inp)
        q2 = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init, name="q2")(inp)
        return (jnp.squeeze(q1, -1), jnp.squeeze(q2, -1)), {}

    def loss(
        self, output: Array, aux: dict, targets: Array, **kwargs
    ) -> Array:
        """1/2 均方误差损失（对两支输出逐元素生效）。"""
        return 0.5 * jnp.square(output - targets)


class VNetwork(nn.Module):
    """状态价值网络 V(s)：回归单个标量价值。"""

    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> tuple[Array, dict]:
        """线性映射到 1 维价值输出。"""
        v_value = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return v_value, {}

    def loss(
        self, output: Array, aux: dict, targets: Array, **kwargs
    ) -> Array:
        """1/2 均方误差损失。"""
        return 0.5 * jnp.square(output - targets)


class HLGaussVNetwork(nn.Module):
    """HL-Gauss value head with two-hot cross-entropy loss.

    HL-Gauss 价值头：将标量价值回归转化为在固定支撑点上的分类问题。
    网络输出各 bin 的 logits，价值为“概率 × bin 中心”的期望；
    训练时用 two-hot（相邻两 bin 加权）的交叉熵作为损失，更稳健。

    Args:
        num_bins: 支撑点（bin）数量。
        v_min / v_max: 价值支撑区间的下界与上界。
    """

    num_bins: int = 101
    v_min: float = -10.0
    v_max: float = 10.0
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    def setup(self):
        # 预计算 bin 宽度与各 bin 的中心坐标（等距划分 [v_min, v_max]）。
        self.bin_width = (self.v_max - self.v_min) / (self.num_bins - 1)
        self.bin_centers = jnp.linspace(self.v_min, self.v_max, self.num_bins)

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> tuple[Array, dict]:
        """输出各 bin 概率的期望作为标量价值，logits 存入辅助字典供损失使用。"""
        logits = nn.Dense(
            self.num_bins, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        probs = jax.nn.softmax(logits, axis=-1)
        # 价值 = Σ p_i * center_i，即分类分布在支撑点上的期望。
        value = jnp.sum(probs * self.bin_centers, axis=-1, keepdims=True)
        return value, {"logits": logits}

    @nn.nowrap
    def loss(
        self, output: Array, aux: dict, targets: Array, **kwargs
    ) -> Array:
        """Two-hot cross-entropy loss.

        two-hot 交叉熵：把连续目标投影到相邻两个 bin 上（按距离加权），
        再对这两个 bin 的对数概率取加权负和。
        """
        logits = aux["logits"]

        targets = remove_feature_axis(targets)
        # 目标裁剪到支撑区间，避免落在 bin 之外。
        targets = jnp.clip(targets, self.v_min, self.v_max)

        # 定位目标所在的下方 bin 索引，并保证 upper_idx 不越界。
        lower_idx = ((targets - self.v_min) / self.bin_width).astype(jnp.int32)
        lower_idx = jnp.clip(lower_idx, 0, self.num_bins - 2)
        upper_idx = lower_idx + 1

        # 依据目标到下方 bin 中心的距离，线性分配到相邻两个 bin 的权重。
        upper_weight = (
            targets - (self.v_min + lower_idx * self.bin_width)
        ) / self.bin_width
        lower_weight = 1.0 - upper_weight

        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # 取出目标两个 bin 对应的对数概率。
        lower_log_prob = jnp.take_along_axis(
            log_probs, lower_idx[..., None].astype(jnp.int32), axis=-1
        ).squeeze(-1)
        upper_log_prob = jnp.take_along_axis(
            log_probs, upper_idx[..., None].astype(jnp.int32), axis=-1
        ).squeeze(-1)

        # 交叉熵：两个 bin 的加权负对数概率。
        loss = -(lower_weight * lower_log_prob + upper_weight * upper_log_prob)
        return loss


class C51QNetwork(nn.Module):
    """C51 分布式 Q 网络：为每个动作学习价值分布（固定 atom 支撑）。

    每个动作输出 ``num_atoms`` 个 logits，softmax 得到概率，
    与 atom 坐标做期望即为该动作的 Q 值。

    Args:
        action_dim: 动作数量。
        num_atoms: 每个动作的支撑原子（atom）数量。
        v_min / v_max: 价值支撑区间。
    """

    action_dim: int
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    def setup(self):
        # atom 间距与各 atom 坐标（等距划分价值区间）。
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.atoms = jnp.linspace(self.v_min, self.v_max, self.num_atoms)

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> tuple[Array, dict]:
        """输出各动作的价值分布，并以分布期望给出 Q 值。"""
        logits = nn.Dense(
            self.action_dim * self.num_atoms,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)

        # 将扁平输出重排为 (..., action_dim, num_atoms)。
        batch_shape = logits.shape[:-1]
        logits = logits.reshape(*batch_shape, self.action_dim, self.num_atoms)

        probs = jax.nn.softmax(logits, axis=-1)
        # 每个动作的 Q 值 = 该动作分布在 atom 支撑上的期望。
        q_values = jnp.sum(probs * self.atoms, axis=-1)

        return q_values, {"logits": logits, "probs": probs}

    @nn.nowrap
    def loss(
        self, output: Array, aux: dict, targets: Array, **kwargs
    ) -> Array:
        """two-hot 交叉熵损失（与 HL-Gauss 类似，作用在 atom 支撑上）。"""
        logits = aux["logits"]

        targets = jnp.clip(targets, self.v_min, self.v_max)

        # 目标投影到相邻两个 atom，并按距离加权。
        lower_idx = ((targets - self.v_min) / self.delta_z).astype(jnp.int32)
        lower_idx = jnp.clip(lower_idx, 0, self.num_atoms - 2)
        upper_idx = lower_idx + 1

        upper_weight = (
            targets - (self.v_min + lower_idx * self.delta_z)
        ) / self.delta_z
        lower_weight = 1.0 - upper_weight

        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # 取出目标相邻两个 atom 的对数概率。
        lower_log_prob = jnp.take_along_axis(
            log_probs, lower_idx[..., None].astype(jnp.int32), axis=-1
        ).squeeze(-1)
        upper_log_prob = jnp.take_along_axis(
            log_probs, upper_idx[..., None].astype(jnp.int32), axis=-1
        ).squeeze(-1)

        loss = -(lower_weight * lower_log_prob + upper_weight * upper_log_prob)
        return loss


class Categorical(nn.Module):
    """离散策略头：输出 ``distrax.Categorical`` 分布。

    Args:
        action_dim: 动作类别数（logits 维度）。
    """

    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> tuple[distrax.Categorical, dict]:
        """线性得到各动作 logits，包装为 Categorical 分布。"""
        logits = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)

        return distrax.Categorical(logits=logits), {}


class Gaussian(nn.Module):
    """连续策略头：对角高斯分布。

    两种参数化(由 ``bound`` 选择):

    * ``bound=False``(默认,保持既有行为):均值由状态线性映射给出,log_std 为
      与状态无关的全局可学习标量向量,std=exp(log_std)(无界)。
    * ``bound=True``(复刻 streaming-rtrrl 的 RNNActorCritic):网络输出
      ``2*action_dim``,拆成 loc 与 log_scale;loc 经 ``sigmoid_between`` bound 到
      ``loc_bounds``、log_scale bound 到 ``log_std_bounds``,再 ``std=softplus(log_scale)``。
      这把均值与标准差都限制在有界区间内,切断"动作头无界放大→循环状态发散"的正反馈臂。

    Args:
        action_dim: 连续动作维度。
        bound: 是否启用有界(状态相关 loc+log_scale)参数化。
        loc_bounds: bound 时 loc 的 sigmoid 区间(hopper 动作 ∈ [-1,1])。
        log_std_bounds: bound 时 log_scale 的 sigmoid 区间(softplus 前);
            [-2,2] 对应 std∈[softplus(-2),softplus(2)]≈[0.13,2.13]。
    """

    action_dim: int
    bound: bool = False
    loc_bounds: tuple[float, float] = (-1.0, 1.0)
    log_std_bounds: tuple[float, float] = (-2.0, 2.0)
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(
        self, x: Array, **kwargs
    ) -> tuple[distrax.MultivariateNormalDiag, dict]:
        """返回对角高斯分布;bound 决定 loc/std 是否被限制在有界区间。"""
        if self.bound:
            # 状态相关的 loc 与 log_scale(单个 Dense 输出 2*action_dim 后拆分)。
            out = nn.Dense(
                2 * self.action_dim,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x)
            loc, log_scale = jnp.split(out, 2, axis=-1)
            loc = sigmoid_between(loc, *self.loc_bounds)
            log_scale = sigmoid_between(log_scale, *self.log_std_bounds)
            std = jax.nn.softplus(log_scale)
            return distrax.MultivariateNormalDiag(loc=loc, scale_diag=std), {}

        mean = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)

        # log_std 为与状态无关的可学习参数，指数化保证 std 为正。
        log_std = self.param("log_std", nn.initializers.zeros, self.action_dim)
        std = jnp.exp(log_std)

        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std), {}


class SquashedGaussian(nn.Module):
    """SAC 常用的 tanh 压缩高斯策略头。

    均值与 log_std 均由状态映射得到，采样后经 tanh 压缩到 (-1, 1)；
    ``temperature`` 可在推理时缩放标准差以调节探索。

    Args:
        action_dim: 连续动作维度。
    """

    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    # log_std 的裁剪范围，防止方差过大或过小导致数值不稳定。
    LOG_STD_MIN = -10
    LOG_STD_MAX = 2

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> tuple[distrax.Transformed, dict]:
        """构造对角高斯并叠加 tanh 变换，返回压缩后的分布。"""
        temperature = kwargs.get("temperature", 1.0)

        mean = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)

        log_std = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        log_std = jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = jnp.exp(log_std)

        # 基础对角高斯，再用逐维 tanh（Block 保证对最后一维联合变换）压缩。
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std * temperature)
        return distrax.Transformed(dist, distrax.Block(distrax.Tanh(), ndims=1)), {}


class Alpha(nn.Module):
    """SAC 温度系数 α 的对数参数化（返回 log α，指数化后恒正）。"""

    initial_alpha: float

    @nn.compact
    def __call__(self) -> Array:
        """返回可学习的 log 温度参数，初始化为 log(initial_alpha)。"""
        log_alpha = self.param(
            "log_temp",
            constant(jnp.log(self.initial_alpha)),
            (),
        )
        return log_alpha


class Beta(nn.Module):
    """对数参数化的标量系数 β（同 Alpha，用于其它温度/权重项）。"""

    initial_beta: float

    @nn.compact
    def __call__(self) -> Array:
        """返回可学习的 log β 参数，初始化为 log(initial_beta)。"""
        log_beta = self.param(
            "log_temp",
            constant(jnp.log(self.initial_beta)),
            (),
        )
        return log_beta



class GVF(nn.Module):
    """通用价值函数（General Value Function）封装。

    在给定输出头之上，用自定义 cumulant（即时“伪奖励”）与折扣 gamma
    构造 TD 目标，从而预测任意信号的折扣累计量。

    Args:
        head: 底层价值头。
        gamma: 折扣因子。
        cumulant: 从转移中提取 cumulant（伪奖励）的函数。
    """

    head: nn.Module
    gamma: float
    cumulant: Callable

    def __call__(self, x: Array, **kwargs) -> tuple[Array, dict]:
        """直接委托底层 head 进行前向计算。"""
        return self.head(x, **kwargs)

    @nn.nowrap
    def get_target(self, transition: PyTree, next_value: Array) -> Array:
        """构造 TD(0) 目标：cumulant + γ·(1-done)·next_value。"""
        # 对 bootstrap 的下一状态价值截断梯度，仅作为目标常量。
        next_value = jax.lax.stop_gradient(next_value)
        return (
            self.cumulant(transition)
            + self.gamma * (1 - transition.second.done) * next_value
        )

    @nn.nowrap
    def loss(self, output: Array, aux: dict, targets: Array, **kwargs) -> Array:
        """损失委托给底层 head 计算。"""
        return self.head.loss(output, aux, targets, **kwargs)


class Horde(nn.Module):
    """Horde 架构：主输出头 + 一组并行的 GVF “demon” 预测。

    主 head 负责主任务，多个 demon 各自预测一个 GVF；
    总损失为主损失与各 demon 损失之和。

    Args:
        head: 主输出头。
        demons: 名称到 GVF demon 的字典。
    """

    head: nn.Module
    demons: dict[str, nn.Module]

    def __call__(self, x: Array, **kwargs) -> tuple[Array, dict]:
        """主 head 与所有 demon 各自前向，demon 输出汇入辅助字典。"""
        output, aux = self.head(x, **kwargs)
        demons = {}
        for name, demon in self.demons.items():
            demons[name] = demon(x, **kwargs)
        return output, {**aux, "demons": demons}

    @nn.nowrap
    def loss(self, output: Array, aux: dict, targets: Array, **kwargs) -> Array:
        """累加主损失与各 demon 的自举 GVF 损失。"""
        loss = self.head.loss(output, aux, targets, **kwargs)
        transitions = kwargs.get("transitions")
        for name, demon in self.demons.items():
            values, _ = aux["demons"][name]
            # 沿时间轴左移一位（末尾补 0）得到 next_value，用于自举目标。
            padding = ((0, 0, 0),) + ((-1, 1, 0),) + ((0, 0, 0),) * (values.ndim - 2)
            next_values = jax.lax.pad(values, 0.0, padding)
            demon_targets = demon.get_target(
                transitions, remove_feature_axis(next_values)
            )
            loss = loss + demon.loss(
                *aux["demons"][name],
                add_feature_axis(demon_targets),
                transitions=transitions,
            )
        return loss


class PredecessorHead(nn.Module):
    """后继/前驱特征头：并行输出特征表示 phi 与反向后继表示 psi_back。

    Args:
        features: 特征维度。
    """

    features: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> tuple[tuple[Array, Array], dict]:
        """分别线性映射得到 phi 与 psi_back 两组特征。"""
        phi = nn.Dense(
            self.features, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        psi_back = nn.Dense(
            self.features, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        return (phi, psi_back), {}
