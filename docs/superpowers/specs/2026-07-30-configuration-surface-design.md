# 配置面设计

范围:实验配置文件的分段、观测的指定方式、算法侧的参数声明、结构的表达与采样、OBGD 的分解。

不在本范围:数值偏差判据(见 `2026-07-29-numerical-testbench-design.md`)、金快照重录、具体实验的取值选择。

取代 `2026-07-26-algorithm-config-contract-design.md`。

## 1. 配置文件的分段

实验 YAML 顶层分三段:

```yaml
environment:
  id: brax::hopper
  backend: spring
  observed: [0, 1, 2, 3, 4]
  num_envs: 1

budget:
  total_steps: 2000000
  epoch_steps: 100000
  eval_steps: 1000

space: {}
```

`environment` 与 `budget` 的取值定义任务与开销本身,不同取值之间的 trial 不可比,因此不进入搜索。控制面直接读取它们,不经过采样器。`score.window_steps` 与 `budget.total_steps` 直接比较,`minimum_total_steps`(`space.py:50`)不再需要。

`space` 只放两类东西:要固定的结构选择,以及要覆盖默认搜索范围的参数。常态是空的。

## 2. 观测的指定与实现

`observed` 是**保留**的观测维度下标。命名为 `observed` 而非 `mask`,因为"掩码数组"未指明列出的是留下的还是去掉的。省略该键表示全观测。与 AAAI 实现的 `obs_mask` 同义(`RTRRL-AAAI25/envs/environments.py:34,54-67`)。

实现为删维度而非置零:包装器在 reset 与 step 中做 `o[..., observed]`,并覆写 `observation_space` 报告缩减后的形状。首层 Dense 的扇入随之从完整观测维数降到保留维数。

改动面三处:`memo/memorax/environments/wrappers/mask_observation.py`(现状在 `:24,37` 做 `o * m`)、`memo/memorax/environments/brax.py:82` 的唯一调用点、`memo/tests/test_masking.py`。`brax.py:9-30` 的 F/P/V 表删除。`mask_rate` 属性在仓库内无消费者,删除。`test_masking.py` 的三个测试都是在论证置零与删维度等价,失去对象,删除。

## 3. 参数声明

算法侧用 dataclass 声明,每个参数三个字段:

```python
lambda_v: float = param(valid=(0.0, 1.0),     search=(0.5, 0.99),  default=0.9)
meta_rl:  bool  = param(valid=[False, True],  search=[False, True], default=True)
seed:     int   = param(valid=(0, None),                            default=0)
```

- **`valid`** 是硬边界,只用于校验。二元组为数值边界,`None` 表示该侧无界;列表为允许取值的集合。实验请求的单值或范围越界时 preflight 拒绝,并指出参数名与越界的一侧。
- **`search`** 是默认搜索空间。二元组为连续区间,列表为离散候选集。省略表示该参数默认不搜索。
- **`default`** 是单值,该参数不被搜索时的取值。

`search` 与 `default` 都必须落在 `valid` 内,导出 catalog 时检查。`log=True` 时 `valid` 的下界必须严格大于零。

catalog 从这些声明导出,不手写字面量,覆盖算法的全部配置面。实验 YAML 未提及的参数按其 `search` 搜索;`search` 缺省的取 `default`。

manifest 携带全部参数,算法一律 `params["x"]` 取值。禁止 `params.get("x", v)` 这类 Python 端兜底,缺键即报错。

## 4. 结构与条件空间

结构是计算图的可替换部分:backbone、归一化的开关、梯度门控、优化器的界与底。声明为一个选择字段加若干分支,每个分支可携带自己的子参数:

```python
optimizer_bound: Structure = structure(
    default="ob",
    branches={
        "none": (),
        "ob": ObBound,
        "adaptive_ob": AdaptiveObBound,
        "adaptive_ob_fixed": AdaptiveObBound,
    },
)
```

catalog 导出这棵树。控制面按树逐层采样:先定结构分支,再只对该分支下的参数取值。

未激活分支下的参数**搜索空间收缩为单点 `default`**,不是从参数面上消失。它仍然存在,仍然带着该值写进 manifest,只是不再构成一个搜索维度。

在 YAML 中把某个结构写成单值即为钉死该结构。此时若同时为未选中分支下的参数给出搜索范围,preflight 拒绝并指出该参数属于哪个未选中的分支。结构未被钉死时不存在这种拒绝——分支在部分 trial 中激活、在其余 trial 中不激活,正是条件采样本身。

没有分支特有参数的结构(两个归一化开关、两个梯度门控)在机制上与普通离散参数一致,不特判。

### 采样的改动

现状 `ask_round` 调用 `study.ask(dict(distributions))`,一次给出固定分布集(`study.py:79`),`distributions` 由整个 space 一次性构造(`space.py:34`)。改为 `study.ask()` 取得 trial 后按结构树逐层调用 `trial.suggest_*`。条件性在控制面解析完毕,作业提交前参数已全部确定,worker 不受影响。

TPE 支持条件空间。`GridSampler` 要求预先给出完整网格,给不出条件空间,`check_sampler`(`study.py:14`)增加一条:空间含未钉死的结构时拒绝 grid。

## 5. OBGD 的分解

现状 `make_obgd_rule` 一个函数同时做四件事:算步长界、按界与 TD 误差加权迹、按二阶矩归一化、对环境轴取均值(`memo/memorax/rl/updates.py:109-222`)。Adam 路径则是 `make_optax_rule` 包一个 optax 变换(`updates.py:82-103`)。两条路径不对称,且"OB 界 + Adam 底"无法表达。

拆成两条独立的结构轴:

```
optimizer.bound: none | ob | adaptive_ob | adaptive_ob_fixed
optimizer.base:  sgd  | adam
```

更新链固定为三段:**bound → 环境轴均值 → base**。

- `bound=none` 时第一段是 `delta * trace`。
- `bound=ob` 时第一段在此基础上乘以步长界 `lr / max(1, |δ̄| · Σ|z̃| · lr · κ)`。
- 自适应变体在第一段内部除以二阶矩分母。该分母同时进入界的计算与最终上升方向,拆成两个变换会迫使二者共享状态,因此留在同一段内。两个自适应分支的区别只在 eps 相对平方根的位置。

接口用 `optax.GradientTransformationExtraArgs`(optax 0.2.8 已提供,`memo/uv.lock`):迹作为 `updates`,TD 误差作为额外参数传入。界的公式需要 δ 与迹分别可见,还需要一次跨整棵树的 L1 求和,把 δ 预先乘进迹之后界无法还原,所以单树接口不够用。

学习率由链中唯一消费它的那一段应用,不重复应用:`bound` 不为 `none` 时它已在界的公式内部,base 段不再乘;`bound=none` 时由 base 段应用,即现状 `optax.scale(config.td_lr)` 所处的位置(`memorax/algorithms/rtrrl.py:244-249`)。

分解后的数值结果不要求与现状逐位一致。`bound=ob, base=sgd` 是已发表的 OBGD,`bound=none, base=adam` 是现状的 Adam 路径,`bound=ob, base=adam` 是新增组合,没有对照实现。

`freeze_gamma` 在有界时报错的现状(`memorax/algorithms/rtrrl.py:200-205`)保留:界按组整体缩放,无法单独按住一个叶子。

## 6. 迁移

`EntryDescriptor` 去掉 `source_hash` 字段,连同三个 catalog 构建脚本里的计算(`memo/runner/catalog.py:38-56`、`rtrrl/scripts/build_catalog.py`、`rtrrl/infra/mock-trainer/scripts/build_catalog.py`)、`preflight.py:135` 的比对、`launch.py` 与 `loop.py` 写进 study 属性的引用一并删除。镜像 digest 已经回答"跑的是哪个镜像"。

`CONTRACT_VERSION` 递增。catalog 结构改变,旧镜像的 catalog 不再被接受,memo 与 rtrrl 两个镜像都要重建。

`experiments/` 下现存的 20 个 streamac YAML 与 5 个 rtrrl YAML 一并迁移到新格式,旧格式不保留,控制面不同时接受两种语义。

实施分三个阶段,顺序固定,每阶段自身可验证:

1. 环境与预算分段、`observed` 与删维度。不触碰参数声明。
2. `param()` 三件套、结构树、条件采样、移除 `source_hash`。
3. OBGD 分解为 bound 与 base 两轴。

阶段 3 依赖阶段 2 的结构树来表达两条轴,阶段 2 依赖阶段 1 腾空 space。
