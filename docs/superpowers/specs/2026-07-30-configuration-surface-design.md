# 配置面设计

范围:实验配置文件的分段、观测的指定方式、算法侧的参数声明、结构的表达与采样、OBGD 的分解。

不在本范围:数值偏差判据(见 `2026-07-29-numerical-testbench-design.md`)、金快照重录、具体实验的取值选择。

## 1. 与 2026-07-26 契约设计的关系

`2026-07-26-algorithm-config-contract-design.md` 覆盖同一片区域且从未实施——代码中没有 `param()`、没有 `bound`,`docs/superpowers/plans/` 下没有对应计划。本文取代它。三处沿用,两处推翻。

沿用:

- **catalog 从声明导出**(旧 §3)。不手写字面量,覆盖算法全部配置面。
- **省略即默认**(旧 §4)。实验 YAML 只写要改的,未出现的参数取 catalog 的默认。
- **`source_hash` 只对入口自身模块求哈希**(旧 §3)。现状是对 `memorax/`、`runner/`、`entries/` 下全部 `.py` 的字节求和(`memo/runner/catalog.py:38-56`),改一个入口的一行注释会让同镜像所有入口的哈希一起变,而 preflight 逐入口比对镜像与仓库的哈希(`preflight.py:135`),于是纯注释改动也要求重建镜像。

推翻第一处:**契约必须支持条件参数**。

旧 §6 的判据是"该变体是否改变参数面或 state 形状",改变的提升为独立入口,并据此断言"任何单个入口的空间内都不存在无效组合,契约无需支持条件参数"。

该判据在"结构 = 计算图"的定义下不可用。`memo/memorax/networks/sequence_models/__init__.py` 导出十余个 cell;再乘 update rule、两个梯度门控、两个归一化开关、`meta_rl`(多两个 encoder)、`bound_actor`(多一层 `sigmoid_between`),组合数在三千以上。入口按结构穷举不成立,所以条件依赖必须由契约表达。

推翻第二处:**`default` 与"搜索范围"分开**。

旧 §2 让 `default` 一字多义:标量为固定值,二元组为默认搜索区间,列表为默认候选集。一个参数因此只能二选一——要么默认固定,要么默认搜索。引入条件结构后二者同时需要:结构激活时该参数应当有默认搜索范围,结构未激活时该参数需要一个具体值写进 manifest。见 §5。

另有一处调整:旧 §89 把 `total_steps` 留在 space 内。本文将环境与预算整体移出 space,见 §3。

## 2. 配置文件的分段

实验 YAML 顶层增加 `environment` 与 `budget` 两段,`space` 只保留待搜索或待覆盖的算法参数:

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

分出来的判据是:这些量定义任务与开销本身,不同取值之间的 trial 不可比。搜索环境等于让同一个 study 里的 trial 跑不同任务,搜索预算等于让它们花不同的钱;`score.window_steps` 也只有在预算固定时才有确定含义。

`minimum_total_steps`(`space.py:50`)随之简化为读取 `budget.total_steps`,preflight 对评分窗口的检查变为与该值直接比较。

`space: {}` 是常态。写进 space 的只有两类:要固定的结构选择,以及要覆盖默认范围的参数。

## 3. 观测的指定与实现

`observed` 是**保留**的观测维度下标,与 AAAI 实现的 `obs_mask` 同义(`RTRRL-AAAI25/envs/environments.py:34,54-67`)。命名为 `observed` 而非 `mask`,因为"掩码数组"未指明列出的是留下的还是去掉的。省略该键表示全观测。

实现从置零改为删维度。现状 `MaskObservationWrapper` 在 reset 与 step 中做 `o * m`(`memo/memorax/environments/wrappers/mask_observation.py:24,37`),改为 `o[..., observed]` 并覆写 `observation_space` 报告缩减后的形状。

改动面已核实为三处:该包装器、`memo/memorax/environments/brax.py:82` 的唯一调用点、`memo/tests/test_masking.py`。`brax.py:9-30` 的 F/P/V 表随之删除。`mask_rate` 属性在仓库内无消费者,一并删除。

两项后果:首层 Dense 的扇入由 11 降到 5,参数与乘加相应减少;`test_masking.py` 记录的 `sqrt(11/5)` 初始化尺度差随之消失,该文件的三个测试全部失去对象,删除。

## 4. 参数声明

算法侧用 dataclass 声明,每个参数三个字段:

```python
lambda_v: float = param(valid=(0.0, 1.0), search=(0.5, 0.99), default=0.9)
seed:     int   = param(valid=(0, None),  default=0)
```

- **`valid`** 是硬边界,只用于校验。实验请求的单值或范围越界时 preflight 拒绝并指出参数名与越界的一侧。数值参数用二元组,`None` 表示该侧无界;分类参数用允许取值的列表。
- **`search`** 是该参数所属结构激活、且实验未覆盖时交给采样器的范围。省略 `search` 表示该参数默认不搜索,取 `default`。
- **`default`** 是单值。用于两种情形:该参数所属结构未激活时写进 manifest 的取值;以及有 `valid` 无 `search` 时的固定取值。

`search` 与 `default` 都必须落在 `valid` 内,导出 catalog 时检查。`log=True` 时 `valid` 的下界必须严格大于零。

manifest 携带全部参数,包括未激活结构下的参数(取其 `default`)。算法一律 `params["x"]` 取值,禁止 `params.get("x", v)` 这类 Python 端兜底,缺键即报错。这一条沿用旧 §5。

## 5. 结构与条件空间

结构是计算图的可替换部分。声明为一个选择字段加若干分支,每个分支可携带自己的子参数:

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

catalog 导出这棵树。控制面按树逐层采样:先定结构分支,再只对该分支下的参数取值。未激活分支的参数不进入搜索空间;实验 YAML 若命名了未激活分支下的参数,preflight 拒绝并指出该参数属于哪个未选中的分支。

没有分支特有参数的结构(如两个归一化开关、两个梯度门控)退化为普通分类参数,机制上不特判。

在 YAML 中把某个结构写成单元素列表即为钉死该结构,此时只有一支激活。五臂对比实验会把全部结构钉死,结构搜索是本设计提供的能力而非该实验使用的功能。

### 采样的改动

现状 `ask_round` 调用 `study.ask(dict(distributions))`,一次给出固定分布集(`study.py:79`),`distributions` 由整个 space 一次性构造(`space.py:34`)。改为 `study.ask()` 取得 trial 后按结构树逐层调用 `trial.suggest_*`。条件性在控制面解析完毕,作业提交前参数已全部确定,worker 不受影响。

TPE 支持条件空间。`GridSampler` 要求预先给出完整网格,给不出条件空间,`check_sampler`(`study.py:14`)增加一条:空间含未钉死的结构时拒绝 grid。

## 6. OBGD 的分解

现状 `make_obgd_rule` 一个函数同时做四件事:算步长界、按界与 TD 误差加权迹、按二阶矩归一化、对环境轴取均值(`memo/memorax/rl/updates.py:109-222`)。Adam 路径则是 `make_optax_rule` 包一个 optax 变换(`updates.py:82-103`)。两条路径不对称,且"OB 界 + Adam 底"无法表达。

拆成两条独立的结构轴:

```
optimizer.bound: none | ob | adaptive_ob | adaptive_ob_fixed
optimizer.base:  sgd  | adam
```

更新链固定为三段:**bound → 环境轴均值 → base**。

- `bound=none` 时第一段是 `delta * trace`,即现状 `_combine` 所做的。
- `bound=ob` 时第一段是 `(step_size * delta) * trace`,即现状 OBGD 所做的,乘法次序不变。
- 自适应变体在第一段内部除以二阶矩分母,因为该分母同时进入界的计算与最终上升方向,拆成两个变换会迫使二者共享状态。

接口用 `optax.GradientTransformationExtraArgs`(optax 0.2.8 已提供,`memo/uv.lock`):迹作为 `updates`,TD 误差作为额外参数传入。这不是为了绕过 optax 的单树接口而做的妥协——界的公式 `lr / max(1, |δ̄| · Σ|z̃| · lr · κ)` 需要 δ 与迹分别可见,且需要一次跨整棵树的 L1 求和,把 δ 预先乘进迹之后界无法还原。

学习率是优化器整体的参数,由链中唯一一个消费它的段应用,不重复应用:

- `bound` 不为 `none` 时,界的公式内部就含学习率,该段输出的已是最终步长下的上升方向,base 段不再乘学习率。
- `bound=none` 时,第一段只做 `delta * trace`,学习率由 base 段应用,即现状 `optax.scale(config.td_lr)` 所处的位置(`memorax/algorithms/rtrrl.py:244-249`)。

据此 `bound=ob, base=sgd` 逐位复现现状 OBGD,`bound=none, base=adam` 逐位复现现状 Adam 路径;`bound=ob, base=adam` 是本设计新增的组合,没有对照实现,也没有已知的发表结果。

`freeze_gamma` 在 OBGD 下报错的现状(`memorax/algorithms/rtrrl.py:200-205`)保留:界按组整体缩放,无法单独按住一个叶子。

### 与金快照的关系

`test_stream_ac_golden.py` 当前有五个失败,原因是 main 的 81d3195 把 StreamAC 的种子从三个 key 改为七个,而快照记录的是三 key 的初始化;人已决定留着,重录排在 testbench 计划之后(`.superpowers/sdd/progress.md:31-48`)。`updates.py:188-193` 记录的"不复用 `_combine` 以免改变乘法次序"是针对该快照的历史决策。本设计不依赖该约束成立,但也不需要它:如上,链的第一段保持原次序。

## 7. 迁移

`CONTRACT_VERSION` 递增。catalog 结构改变(新增 `valid`,`default` 语义变更,新增结构树),旧镜像的 catalog 不再被接受,memo 与 rtrrl 两个镜像都要重建。

`experiments/` 下现存的 20 个 streamac YAML 与 5 个 rtrrl YAML 一并迁移到新格式,旧格式不保留,控制面不同时接受两种语义。

实施分三个阶段,顺序固定,每阶段自身可验证:

1. 环境与预算分段、`observed` 与删维度。不触碰参数声明。
2. `param()` 三件套、结构树、条件采样、`source_hash` 改为按入口模块。
3. OBGD 分解为 bound 与 base 两轴。

阶段 3 依赖阶段 2 的结构树来表达两条轴,阶段 2 依赖阶段 1 腾空 space。
