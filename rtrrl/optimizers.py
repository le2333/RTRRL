"""optax 优化器构造工具。"""

from dataclasses import dataclass, field
from simple_parsing.helpers import dict_field
import json
import optax

# 优化器配置
@dataclass(frozen=True)
class OptimizerConfig:
    """优化器参数配置。"""

    opt_name: str = "adam"  # 优化器名称。
    learning_rate: float = 1e-3  # 基础学习率。
    kwargs: dict = dict_field(
        hash=False, type=json.loads
    )  # 优化器的额外关键字参数。
    decay_type: str | None = None  # 学习率衰减类型。
    lr_kwargs: dict = dict_field(
        hash=False, type=json.loads
    )  # 学习率衰减的额外参数。
    weight_decay: float = 0.0  # 权重衰减。
    gradient_clip: float | None = None  # 梯度裁剪。
    multi_step: int | None = None  # 梯度累积步数。

# 构建优化器
def make_optimizer(
    config=OptimizerConfig(), direction="min"
) -> optax.GradientTransformation:
    """构造 optax 优化器。

    该装饰器允许从优化器状态读取按计划变化的学习率。

    Parameters
    ----------
    learning_rate : float
        初始学习率
    direction : str, optional
        min 或 max,默认 "min"
    opt_name : str, optional
        优化器名称,默认 'sgd'
    gradient_clip : int, optional
        梯度范数裁剪,默认 0
    lr_decay : int, optional
        指数学习率衰减,默认 1
    optimizer_params : dict, optional
        传给优化器的额外参数,默认 {}
    Returns
    -------
        optax 优化器
    """
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    if direction in ["max", "maximize"]:
        learning_rate = -learning_rate
    else:
        weight_decay = -weight_decay

    # def decay_mask(tree):
    #     mask = jax.tree.map(lambda _: False, tree)  # 全部初始化为 False
    #
    #     # 只为 RNN 的部分叶子置 True
    #     if isinstance(tree, BaseRNNCell):
    #         mask = eqx.tree_at(lambda x: x.w,  # 在此选择要衰减的叶子
    #                            mask, True)
    #     # elif isinstance(tree, Linear):
    #     #     mask = eqx.tree_at(lambda x: x.W,  # 在此选择要衰减的叶子
    #     #                        mask, True)
    #     return mask

    if config.decay_type == "cosine_warmup":
        """参数:
            init_value: 待退火标量的初始值。
            peak_value: warmup 结束时达到的峰值。
            warmup_steps: 正整数,线性 warmup 的长度。
            decay_steps: 正整数,调度总长度。注意它包含 warmup 时间,
                因此余弦退火应用的步数为 ``decay_steps - warmup_steps``。
            end_value: 待退火标量的终值。
            exponent: 浮点数。默认退火为 ``0.5 * (1 + cos(pi t/T))``,
                其中 ``t`` 是当前步,``T`` 是 ``decay_steps``。
                exponent 将其改为 ``(0.5 * (1 + cos(pi * t/T))) ** exponent``。
                默认 1.0。
        """
        learning_rate = optax.warmup_cosine_decay_schedule(
            learning_rate * config.lr_kwargs.get("initial_multiplier", 0.0),
            peak_value=learning_rate,
            end_value=learning_rate * config.lr_kwargs.get("end_multiplier", 0.01),
            decay_steps=config.lr_kwargs.get("decay_steps", 1e6),
            warmup_steps=config.lr_kwargs.get("warmup_steps", 1e4),
        )
    elif config.decay_type == "cosine":
        """参数:
            init_value: 学习率初始值。
            decay_steps: 正整数,应用退火的步数。
            alpha: 调整学习率所用的乘子下限,默认 0.0。
            exponent:  默认退火为 ``0.5 * (1 + cos(pi t/T))``,其中
                ``t`` 是当前步,``T`` 为 ``decay_steps``。
                exponent 将其改为 ``(0.5 * (1 + cos(pi * t/T))) ** exponent``。
                默认 1.0。
        """
        learning_rate = optax.cosine_decay_schedule(
            learning_rate,
            decay_steps=config.lr_kwargs.get("decay_steps", 1e6),
            alpha=config.lr_kwargs.get("alpha", learning_rate * 0.01),
        )
    elif config.decay_type == "exponential":
        """参数:
            init_value: 初始学习率。
            transition_steps: 必须为正,见上方退火计算。
            decay_rate: 不可为零,衰减率。
            transition_begin: 必须为正,多少步后开始退火
                (在此步数之前标量固定为 `init_value`)。
            staircase: 若为 `True`,在离散区间上退火。
            end_value: 指数衰减停止处的值。当 `decay_rate` < 1 时,
                `end_value` 视为下界,否则视为上界。`decay_rate` = 0 时无效果。
        """
        learning_rate = optax.exponential_decay(
            learning_rate,
            config.lr_kwargs["transition_steps"],
            config.lr_kwargs["decay_rate"],
            config.lr_kwargs.get("warmup_steps", 0),
            config.lr_kwargs.get("staircase", False),
            config.lr_kwargs.get("end_value", None),
        )
    elif config.decay_type is not None:
        raise ValueError(f"Decay type {config.decay_type} unknown.")

    @optax.inject_hyperparams
    def _make_opt(learning_rate):
        # 从 optax chain 构造优化器
        optimizer = optax.chain(
            # 权重衰减
            optax.add_decayed_weights(weight_decay),  # , mask=decay_mask
            # 梯度裁剪
            optax.clip_by_global_norm(config.gradient_clip)
            if config.gradient_clip
            else optax.identity(),
            # 优化器
            getattr(optax, config.opt_name)(learning_rate, **config.kwargs),
        )
        if config.multi_step:
            optimizer = optax.MultiSteps(optimizer, every_k_schedule=config.multi_step)
        return optimizer

    return _make_opt(learning_rate=learning_rate)

# 从优化器状态读取当前学习率
def get_current_lrs(opt_state, opt_config: OptimizerConfig | None = None):
    """从优化器状态读取当前学习率。"""
    lrs = {}
    _reduce_on_plateau = False if opt_config is None else opt_config.reduce_on_plateau
    if hasattr(opt_state, "inner_states"):
        for k, s in opt_state.inner_states.items():
            reduce_on_plateau_lr = s[0][3][3].scale if _reduce_on_plateau else 1
            lrs["lr_" + k] = s[0][1]["learning_rate"] * reduce_on_plateau_lr
    else:
        reduce_on_plateau_lr = opt_state[3][3].scale if _reduce_on_plateau else 1
        lrs["learning_rate"] = opt_state[1]["learning_rate"] * reduce_on_plateau_lr
    return lrs