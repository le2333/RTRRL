# 两个未做的重构

`docs/sdk-split.md` 记的是已完成的拆包（T1–T5）。这份记的是接下来的两件，以及它们之间的次序。

## 系统的目标形状

| | ctrler（`infra/`） | worker（`memo/`） |
| --- | --- | --- |
| 职责 | HPO、task 分发、托管 aim server、aim/rerun 可视化 | 跑若干指定配置、回报结果 |
| 依赖 | optuna, pyyaml | 训练框架, aim(客户端), rerun-sdk, numpy, boto3, pydantic |
| 之间 | **只有 S3 上的 JSON**，无共享包，无 API | |

infra 通过 AWS Batch 唤起 worker 容器。worker **持有的是进程不是对象**——一个 run 崩了不带走 manifest 里后面的 run，显存随进程退出释放。这条边界不动。

三个契约，三个所有者，各自维护自己的小东西：

| 契约 | 提供方 | 内容 |
| --- | --- | --- |
| **catalog** | worker | "我接受什么"——镜像构建时扫 `entries/` 生成，贴在镜像标签上，绑 digest |
| **实验 YAML** | infra | 外围参数 + 搜索域覆盖 |
| **run config** | infra | 协调字段 + 外围透传 + 采样值 |

---

## R1 — infra 契约重构

### 问题

infra 的 `_configurations` 产出 `{experiment, trial, algorithm, run, objective, loggers, parameters}`，worker 的 `RunConfig` 要 `{contract, run_id, entry, digest, environment, training, evaluation, params, logging, score}`。**两者字段几乎没有交集**——infra 说 `algorithm`，worker 说 `entry`；infra 说 `parameters`，worker 说 `params`。

这不是回归：拉进来的 infra（来自 `rewrite/temp`）本来配的是另一套 worker。两侧同处一分支后才第一次可见。

### 运行配置的分解

```
运行配置 = 实验配置的外围参数（透传）
         + 采样( catalog 的搜索域 ⊕ 实验配置的覆盖 )
```

| 部分 | 谁定形状 | 谁供值 | 缺了怎么办 |
| --- | --- | --- | --- |
| 搜索参数 | worker（catalog） | infra 采样 | catalog 的搜索域兜底 |
| 外围参数 | — 见下 | 实验配置，**只能透传** | 报错，无人能救 |

catalog 能声明的是**空间**，而空间需要采样器。外围参数没有空间，只有一个值，而那个值只有实验配置有。所以在 catalog 里声明外围参数等于声明一个永远填不上的洞——**这条路试过并否决了**。

### 决定：infra 的实验配置形状对齐 worker 的运行配置

于是没有两个形状，也就没有漂移发生的地方：

- 人漏了 `epoch_steps` → **infra 校验自己的输入时就失败**，零个容器起来（否则一轮 HPO 起 N 个全死）
- infra 校验的是"我的输入格式完整吗"，不是"worker 会不会喜欢这个值"——它不需要理解任何名字的语义，只做集合判断
- 不需要 catalog 做在线检测。`requires` 名单那个方案**否决了**：它解的是"两个形状各自维护"才有的漂移

代价是 infra 的 schema 必须与 worker 的形状相等。两侧在同一分支，改动是一个提交的事；保证相等的唯一机制是往返测试（见下）。

### 往返测试是唯一的同步机制

```
runner.catalog.build_catalog()      ← worker 声明的空间
        ↓
trainer_infra 真实的解析与采样
        ↓
worker.contract.RunConfig 校验       ← 接收方校验
        ↓
entry.build(...) 并跑一步
```

`memo/tests/test_template.py` 已经做了它的一半——它的 `manifest()` docstring 写着"采样器会交给 entry 的东西"，即**在模拟 infra**，因为当时 infra 不在本分支。现在在了，把模拟换成真链路即可。**这个文件不要删**。

链绿 = 两份拷贝仍然相等；红 = 有人只改了一边。

### 组装器需要什么（这是契约唯一的难点）

`entries/stream_ac.py` 的 `build(params, environment, training)` 追到底只用三样：

| 来源 | 字段 | 决定图的什么 |
| --- | --- | --- |
| `params` | 全部 | 组件选择 + 超参 |
| `environment` | `id` `backend` `observed` `episode_length` | 造 env，由它得 `action_dim` 和观测宽度 |
| `training` | **`num_envs`** | 每个 carry / trace / sensitivity 的第一维 |

`environment.seed` 不进图。其余全是预算、上报、协调。`num_envs` 是唯一从预算块穿进组装器的字段，建图时钉死。

顺这条看清了 `evaluation.num_envs` 为什么删得最硬：`evaluate` 用的是 `cfg.num_envs`，即训练那张图的宽度。按字面实现需要第二张图——它不是"没实现"，是**与单图假设矛盾**。

### 目标形状已经写好了，在 `experiments/streamac template.yaml`

那个文件的结构与 `RunConfig` 几乎逐字对应。R1b 不是设计新 schema，是**让 infra 读这个文件**：

```
透传:  experiment entry environment training evaluation logging  score(除 s3)
生成:  contract  run_id  launch_id  trial
       digest ← image        score.s3 ← storage
采样:  params ← catalog 搜索域 ⊕ space
自用:  name description compute hpo
```

`image` / `storage` 早在文件里但今天无人消费——它们正是"缺的协调字段"的原料，不用新增。

### 修订后的次序（adapter 不改，等 R2）

infra 的 `adapter.py` 判叶子用 `"search" in node`，那**正是 R2 之后**的 `Parameter(valid, search)` 形状；今天的 catalog 是 `kind`/`branches`，它认不了。现在改等于改错再改回来。往返测试同理——写今天的形状，R2 时整个重写。所以缝切在同一处：

| | 内容 | 依赖 |
| --- | --- | --- |
| **R1a** ✅ | `RunConfig` 删 4 个死字段，`CONTRACT_VERSION` 6→7 | 无 |
| **R1b** ✅ | infra 读模板 YAML 的名字，`_configurations` 产出 `RunConfig` 形状 | 无 |
| **R1c** ✅ | 往返测试**上半**：infra 产出 → `RunConfig` 校验；`params` 手工给 | R1a+R1b |
| **R2a** | 参数树重构，catalog 产出嵌套 `{valid, search}`；adapter 自动变对 | — |
| **R1d** ✅ | 往返测试**下半**：真 catalog → 采样 → `build` | R2a |

### R1a ✅ — 已完成

删掉 `training.chunk_steps`、`training.early_stop_patience`、`evaluation.num_envs`、顶层 `name`（四个都零读取；aim 的 run 名用 `run_id`，实验文件的 `name` 是 optuna study 名，属 infra）。`CONTRACT_VERSION` 6→7，测试改成引用它而不是字面量。模板 YAML 按去向重排并标注。289 passed 不变。

### R1b ✅ — 已完成

`experiment.py` 改读模板的名字（`entry` / `space` / `score.direction` / `catalog["entries"]`），`_configurations` 产出 `RunConfig` 的十三个字段。五个块原样透传，`PASSED_THROUGH` 是那份清单。

协调字段全在 infra 侧生成，实验文件不问：`launch_id`（UTC 时间戳，可用 `--launch-id` 固定）、`run_id = {name}-{launch}-t{trial}`、`digest ← image`、`score.s3` 和 `logging.rerun_s3 ← storage/{experiment}/{launch}/{run_id}/`。`contract` **抄 catalog 的**而不是自己声明——运行配置该claim的是那个将要读它的镜像实现的版本。

三个 fail-fast 都在起容器之前：文件缺字段（`REQUIRED`）、`space` 里有 catalog 没声明的名字（`SpaceError`）、`image` 没钉 digest。

`adapter._collect` 改成按扁平点名查覆盖（`overrides.get(path)`），与 R2 无关的那一半。判叶子的 `"search" in node` **一行没动**，等 R2a。

infra 10 → 19 passed。测试的实验文件和 catalog 收进 `tests/conftest.py`，因为一份形状的意义就在于两侧共用一份，每个文件抄一遍正是它失效的方式。

**未接的一环**：真 catalog（`kind`/`branches`）走不通 `adapter`，所以往返只验到协调与外围。R1d 补。

### R1c ✅ — 已完成

`infra/tests/test_round_trip.py`，**跑在 infra 侧**。落点当时在犹豫，因为我按"从 memo import 任何东西都拖进 jax"估的代价——查了是错的：`worker.contract` → `memorax.parameters` → 只到 pydantic，216 个模块，零 numpy 零 aim。`memorax/__init__.py` 惰性，`worker/__init__.py` 只有 docstring。所以 `pythonpath = ["src", "../memo"]` 按源码 import，不安装（安装会连带 jax/aimrocks，后者没有 Windows wheel），pydantic 进 infra 的 dev 组。

十个断言，两个是真正防漂移的：

- **块清单两侧都读出来**，不在测试里列。列了就对"新增一个块"失明——新块不在列表里，比较的两边都把它过滤掉了
- **worker 每个块的必填字段 ⊆ infra 要求的字段**（`score.s3` 除外，那是 infra 生成的）。方向是包含不是相等：infra 可以更严（它就要求 `episode_length`，而 worker 有默认值）

其余：真模板文件本身过校验（不是它的副本）、`RunConfig` 十三个字段无一靠默认值到货、ragged 预算由 worker 拒绝而非 infra（分工正确的样子：infra 查"文件说了没有"，worker 查"说的有没有意义"）。

infra 19 → 33 passed。

---

## R2 — 组件重构

### 目标

一个组件同时拥有**参数表**和**计算图**。今天加一个 backbone 要动四处：cell 文件、`backbones.py` 的声明 dataclass、`backbone()` 工厂、entry 的 `BACKBONE_BRANCHES`。

形状是带 `build()` 的 frozen dataclass：

```python
@dataclass(frozen=True)
class Rtu:
    hidden_dim: int = param(valid=..., search=...)
    def build(self, *, features: int, output_dim: int) -> tuple[nn.Module, ...]:
        """任务给的形状从这里进来，参数表里的值已经在 self 上。"""
```

组件**拥有**计算图（知道怎么造），而不是**本身就是**计算图。看构造器知道什么可搜，看 `build` 的签名知道什么是任务给定的。

### 已经查实的约束

- **`nn.Module` 不能兼作声明类**：flax 往子类自己的 `__annotations__` 里写 `parent` 和 `name`（实测过，不是继承来的），而 `describe_parameters` 要求每个字段都用 `param()`/`structure()` 声明，走到它们会 raise。过滤它们只能让契约层认识 flax
- **`params` 保持平的**：每个前向都读它，包一层只是给最热的路径加一跳
- **多递归组件不在需求内**。`sequence.py` 拒绝第二个递归组件是因为跨层敏感度需要 dense cross-layer Jacobian——**嵌套容器给的是容器，不是链式法则**，这个重构解决不了它，也不需要解决

### 规格已经写过一次，在 `rewrite/temp`

`rewrite/temp:docs/reference/legacy-code-reuse-assessment.md` 第 58 行起，是一份**否决清单**：它点名今天这套实现的哪五处与那轮设计冲突，所以它比"该长成什么样"更有用——它说了不该长成什么样。

| # | 今天的实现 | 那轮设计 |
| --- | --- | --- |
| 1 | 数值参数和结构参数是**两类节点**（`ParameterSpec` / `StructureSpec`） | 统一树节点 |
| 2 | `StructureSpec` 在**声明阶段展开所有 branch 的参数树** | 不展开 |
| 3 | `flatten` / `walk` 遍历所有分支，用 `placeholder` 填未选分支 | 只走选中的那条 |
| 4 | 参数声明绑定 **dataclass metadata** | 脱钩 |
| 5 | — | 按实验配置里的 `kind` **从入口递归路由**，**仅发现作用域内显式注册的组件** |

点名不复用的三个文件：SDK 的 `contract.py`（`ParameterSpec`/`StructureSpec`/`ParameterTree`）、SDK 的 `parameters.py`、control-plane 的 `space.py`。前两个的内容今天在 `memorax/parameters.py` 里。

**保留的约束意图**（这些是要留住的，不是要删的）：范围必须非空、上下界有序、实验覆盖不得超出组件合法范围、单元素选项表示固定值、未知字段必须报错。

#### 第 5 条回答了作用域

命名撞车——`ob` 在 actor 和 critic 各一份、`sparse` 在 6 处 initialization、`running` 在两处 normalization、`sgd`/`adam` 在两个角色——**只有在存在一个全局命名空间时才是撞车**。递归路由下不存在这个空间：actor 的 bound 只在 actor 的作用域里查它显式注册的分支表，critic 查自己的，两者从来不是同一个节点。

所以"实验配置扁平"指的是**每一层之内平铺**，用缩进分组表达层级，不是全局单层。`params` 的点名路径（`actor.ob.kappa`）是树路径的渲染，天然唯一。

#### 第 2、3 条是今天 63 个参数名的来源

`flatten(PARAMETERS)` 产出 63 个名字、最长 52 字符（`actor_head.global_std.initialization.sparse.sparsity`），而任何一次运行只用到其中约 20 个——其余是**未选分支被 placeholder 填出来的**。删掉展开，长度问题自己消失一大半。

### 参数契约的三个角色

`param(valid=, search=, placeholder=)` 的三个参数属于三个不同的知情者：

- **`valid`** = 什么值有意义 → 组件知道（γ ∈ [0,1] 是算法性质）
- **`search`** = 默认搜索域 → 组件给一个建议，**实验覆盖它**（`resolve_parameter_ranges` 里 `overrides.get(name, node["search"])` 就是这个语义）
- **`placeholder`** = **可弃用**。它是为条件结构问题设计的，而结构永远被钉死到一个分支（`test_template.py` 强制），条件空间在运行期不出现。而且 `param()` 的实现里 `field(default=placeholder)` 已经把它存了第二遍

### `structure` 应当删除

`BACKBONE_BRANCHES` 这个字典被喂了两次——一次给 `structure(branches=...)` 描述空间，一次给 `read_branch(...)` 重建组件。**描述侧的 branches 是消费侧注册表的副本。** 替代形状不需要新类型：

```python
"backbone": Parameter(valid=Choice(["rtu", "mlp"]), search=Choice(["rtu"])),
"rtu":  {"hidden_dim": Parameter(...)},
"mlp":  {"depth": Parameter(...)},
```

唯一损失的静态检查（默认值必须是真实分支）在消费侧已经有了：`read_branch` 遇到不认识的分支名会 raise。

**唯一会让这个决定反转的未来需求**：真的要搜 `backbone ∈ {rtu, mlp}`。那时两分支子空间不同，需要条件采样，enum + 平铺嵌套表达不了。今天 `test_template.py` 明确禁着搜结构。

### 参考实现

`rewrite/temp` 的 `algorithms/StreamAC/parameters.py`——**词汇表就是全部，没有一行遍历逻辑**：

```python
@dataclass(frozen=True)
class Choice:      values: tuple[Scalar, ...]
@dataclass(frozen=True)
class FloatRange:  low: float | None; high: float | None; log: bool = False
@dataclass(frozen=True)
class IntRange:    low: int | None; high: int | None; step: int = 1; log: bool = False

Range: TypeAlias = Choice | FloatRange | IntRange

@dataclass(frozen=True)
class Parameter:
    valid: Range
    search: Range

ParameterNode: TypeAlias = Parameter | Mapping[str, "ParameterNode"]
ParameterTree: TypeAlias = Mapping[str, ParameterNode]
```

对照今天的 `memorax/parameters.py`（391 行、11 个 pydantic 模型、`walk`/`expand`/`flatten`/`_check_*`）：那边**没有 pydantic**、没有 `kind` 判别式、没有 placeholder、没有校验函数。节点要么是 `Parameter` 要么是一组节点，`isinstance` 就是判别。

`scripts/build_catalog.py` 的序列化也只有三个函数、四十行，`_parameters` 递归、`_parameter` 出 `{valid, search}`、`_range` 出 `{type, ...}`。**infra 的 `adapter._collect` 判叶子用 `"search" in node`，正是对着这个形状写的**——这就是它今天认不了我们 catalog 的原因，也是它在 R2a 之后不用改的原因。

但注意：那边的 StreamAC 参数树是**六个平铺标量**，没有可选分支、没有组件。所以它是**词汇表的**参考实现，不是**路由的**——路由那部分只有上面那份否决清单的第 5 条写了意图，代码从未存在。

另有本分支归档时丢失的两个未提交模块（只剩 `infra/src/trainer_infra/__pycache__/` 里的字节码）：`parameters.py` 声明 infra **自己的** `Choice`/`FloatRange`/`IntRange`，`parameter_adapter.py` 的 `resolve_parameters` 对着它们解析。那份工作已经在做"两侧各自声明、不共享包"。

### R2 落地后 R1 的 schema 会消失

外围配置也建模成组件之后，条件性要求（`enable_rerun` 为真才要 `rerun_s3`）就是**沿着选中的分支走一遍树**——因为结构永远被钉死，不需要条件采样，只需要条件要求。infra 的校验从"对着一份 schema"变成"走一遍树"，R1 那份 schema 自己消失。

**所以 R1 不是 R2 的临时铺垫，两步都在删东西。**

### R2 的动作

拆成 a/b/c，因为 R1d 只卡在 a。

**R2a — 词汇表与树**（R1d 的前置）
1. `memorax/parameters.py` 换成 `rewrite/temp` 那份词汇表：`Choice`/`FloatRange`/`IntRange`/`Parameter(valid, search)`，`ParameterNode = Parameter | Mapping`。删 `ParameterSpec`/`StructureSpec`/`kind`/`placeholder`
2. 保留的约束移到构造时校验：范围非空、上下界有序、search ⊆ valid、单元素选项即固定值
3. 声明从 dataclass metadata 脱钩——参数表是模块级的 `ParameterTree` 字面量，不是 `field(metadata=...)`
4. catalog 序列化成嵌套 `{valid, search}`；infra 的 `adapter` **一行不改**就认得
5. 实验 YAML 的 `space` 改成缩进；`adapter._collect` 的覆盖查找改回并行走树（R1b 那次改成扁平点名查找，是我把这一步误判成与 R2 无关）
6. R1d：往返测试接上真 catalog → 采样 → `build`

### R1d ✅ — 已完成，R1 到此结束

`memo/tests/test_round_trip.py`，跑在 memo 侧（这一半要 jax）。`pytest.ini` 的 `pythonpath` 加 `../infra/src`（**不是** pyproject——memo 有 `pytest.ini`，它优先），dev 组加 optuna（实测只多一个 colorlog）。

链路：真 catalog（`build_catalog()`）→ `ExperimentRunner` 读真模板 → optuna 真采样 → `RunConfig` 校验 → `stream_ac.build` → 走两步。

**两个 walk 必须到达同一组名字**是这里最要紧的断言：`memorax.parameters` 那个从 mapping 里读已选分支，`trainer_infra.hpo` 那个在打开分支前一行抽它。两份是刻意分开写的（两侧不共享包），除了这条没有别的东西保证它们是同一个 walk。

写的时候先写错了一版：断言两个 trial 的参数集相同。它们不相同——`initialization.kind` 现在被搜，抽到 `sparse` 的 trial 有 `sparse.sparsity`，抽到 `lecun` 的没有。**条件空间在工作的样子就是这个**，不变量是逐配置自洽（`expand(PARAMETERS, params)` 复现 `params` 本身），不是跨 trial 相等。

两个 CI 的触发路径都放宽了：memo-ci 加 `infra/src/**` 和 `experiments/**`，tests.yml 加 `memo/worker/contract.py`、`memo/memorax/parameters.py`、`experiments/**`。缝两侧现在互相依赖，只盯自己那半边会漏。

memo 289 → 303 passed。

**R2b — 递归路由**
7. ✅ **采样侧**：`resolve_parameter_ranges` 返回树而不是扁平表，`sample_parameters` 走树——先抽 `kind`，只下降抽中的那条
8. 消费侧：作用域内只发现显式注册的组件；`read_branch` 保留但改成在作用域内查

#### 第 7 步 ✅ — 采样也要递归，不只是消费

两侧是同一个 walk，差别只在 `kind` 从哪来：worker 从运行配置**读**，采样器在打开分支前一行**抽**。

之前 `sample_parameters` 收的是一张扁平范围表，而表表达不了"这一项只在 `kind == ob` 的试验里存在"。所以实验把 `actor_optimizer_bound.kind` 钉成 `none`，`ob.kappa` 仍在被采样——TPE 要为没人读的维度建模，运行配置还会带着它，看起来像有人选过。

真模板上的效果：

| | 之前 | 现在 |
| --- | --- | --- |
| catalog 声明的叶子 | 63 | 63 |
| 这个实验到达的 | 63 | 34 |
| **真正要搜的维度** | **27** | **11** |

**顺带解锁了搜结构。** roadmap 原来写着"唯一会让删除 `structure` 这个决定反转的未来需求，是真的要搜 `backbone ∈ {rtu, mlp}`，那时两分支子空间不同、需要条件采样、平铺表达不了"。递归采样就是条件采样，树本来就是条件结构——不需要发明任何新东西。今天真模板上已经有两个结构在被搜（两处 `initialization.kind`）。`test_template` 那条"结构必须钉死"的断言可以在需要时去掉。

**R2c0 — 无选择的分组**（R2c 的前置）

参数树今天只能表达两种字段：一个值（`param`）和一条选择（`structure`）。**没有词汇表达"这是一层作用域，不是一个选择点"**——`describe_parameters` 遇到纯嵌套字段直接 raise（实测）。而遍历侧早就支持了：`walk` / `_draw` 在没有 `kind` 时无条件下降所有组，infra 那边还有一条专门的测试。所以缺的只有声明和读回。

`group(of=X)`：发 `describe_parameters(X)`，不发 `kind`。**契约层一个节点类型都不多**——树仍然只有 `Parameter | Mapping`，`group` 和 `structure` 的区别只是发不发那个 `kind` 叶子、分支名占不占一层。`of=` 显式传类，因为声明文件都有 `from __future__ import annotations`，`item.type` 是字符串。

*什么时候用哪个*

| | 例子 | 判据 |
| --- | --- | --- |
| **group** | `actor` `critic` `optimizer` `normalization` | 用**角色**命名，回答"谁的"。将来不会出现第二个候选 |
| **structure** | `backbone` `head` `bound` `base` `credit` `initialization` | 用**变化的东西**命名，回答"哪个" |

今天七个分支表全部落在第二类。`critic_head` 只有一个分支（`value`）却仍是 structure——它是选择点，只是今天恰好一个候选，早晚有分布式 critic。

*为什么默认走 group*

group 提升成 structure 时，它底下每个名字多一段（`actor.optimizer.bound.*` → `actor.optimizer.standard.bound.*`）；而单分支 structure 长出第二个分支时名字一个不动。所以严格说 group 是贵的那一侧。

但贵的部分已经付过了：**搜索空间一变就绑到新 digest，新镜像、新 study**，两种情况一样。差别只剩一次机械编辑（一个 YAML 块缩进一层），而且 `test_template.py` 会在名字动了 YAML 没跟上时直接红，做不成一半。

一次性成本 < 永久噪声（给永不选择的层级挂一个单值 `kind`，它会出现在每份运行配置、每个 trial、每次 catalog 里）。所以**拿不准时才用单分支 structure 对冲**，不是反过来。

*StreamAC 的新形状*

```python
@dataclass(frozen=True)
class Optimizer:                      # 一份声明，两个角色用
    bound: str = structure(branches=BOUND_BRANCHES)
    base:  str = structure(branches=BASE_BRANCHES)

@dataclass(frozen=True)
class Actor:
    head:      str       = structure(branches=ACTOR_HEAD_BRANCHES)
    optimizer: Optimizer = group(of=Optimizer)

@dataclass(frozen=True)
class StreamACParameters:
    actor:  Actor  = group(of=Actor)
    critic: Critic = group(of=Critic)
    normalization: Normalization = group(of=Normalization)
    backbone: str = structure(branches=BACKBONE_BRANCHES)
    ...
```

**名字不会变短**——`actor.optimizer.bound.ob.kappa` 和今天的 `actor_optimizer_bound.ob.kappa` 都是 30 字符。收益是结构的：同一个分支表今天写了四遍靠手写前缀区分，之后是一个 `Optimizer` 声明用两次。加第三个角色不再要抄一遍。

*动作*

1. `group(of=)`；`describe_parameters` 加一个分支；`read_parameters` 加一个递归
2. `StreamACParameters` 收成 `Optimizer` / `Actor` / `Critic` / `Normalization`
3. entry 的 `build()` 读取点跟着走（`read_branch` 本来就收 `prefix`）
4. 模板 YAML 的 `space` 重新缩进
5. 验证：memo 全套、infra 全套、往返、catalog

**R2c — 组件拥有计算图**
9. 一个组件先合并成声明 + `build()`（建议 `Rtu`：有 carry、有 sensitivity、有 `local_jacobian`，最复杂），`backbone()` 工厂留作兼容层
10. 其余组件跟进
11. 外围配置并入组件树，R1 的 schema 删除

---

## 验证方式

**在 WSL 里跑**，不要只在 Windows 上：`aimrocks` 无 Windows wheel、moto server 绑 `0.0.0.0` 而 Windows 连不上、`uv lock` 要构建 `mettagrid`（import `fcntl`）。这三样都只在 Linux 上成立，而且都压在 worker 侧最要紧的地方。

```bash
wsl -e bash -lc 'cd /mnt/c/.../memo && export PATH=$HOME/.local/bin:$PATH \
  && export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/memo-linux \
  && uv lock && uv sync --frozen --group development'

wsl -e bash -lc 'cd /mnt/c/.../memo && $HOME/.venvs/memo-linux/bin/python -m pytest tests -p no:warnings'
```

infra 的环境在 `~/.venvs/infra`（Windows 侧即可，只有 optuna + pyyaml）。

---

# R3 — 算法层结构（设计终稿）

这一节是 StreamAC 和 RTRRL 共同的目标结构。它不是推演出来的：`StreamAC.py`
先按四层实现了一遍，实现过程否掉了几条原本看着合理的划分，下面每条结论后面都
记着它是被什么否掉的。

## 参考实现在哪

| | 路径 | 说明 |
| --- | --- | --- |
| RTRRL 发表版 | `../RTRRL-AAAI25/`（与 `streaming-rtrrl` **同级**） | `rtrrl.py` `traces.py` `models/` `optimizers.py`。**没有 obgd.py** |
| StreamAC 发表版 | `../streaming-drl/` | |
| RTRRL 工作副本 | `streaming-rtrrl/rtrrl/` | 后来的，**不是蓝本**，里面的 `obgd.py` 是后加的 |

**实现 RTRRL 时以 AAAI 版为蓝本重构**，不沿用 `memorax/algorithms/rtrrl.py` 里的
旧实现，也不直接复用 memorax 现有组件；但**边界要和库里组件保持一致**，这样才
能逐个替换和对测。

## 四层

```
Flow  (StreamAC / RTRRL)   环境、顺序、scan、上报
  EnvTransformer             环境 + 观测/回报两个尺度
  Core                       这个算法的完整封装
    Network×N                前向 / carry / sensitivity / params
    Head×2                   量 + 要上升的标量
```

## Head 持有多大的 Network：独占即持有，共享即上浮

Head **按需持有到最大边界**——只属于它自己的那部分参数，全部归它。一旦某块被两个
Head 用到，它上浮成一个兄弟组件，由 Core 持有。

| | StreamAC | RTRRL |
| --- | --- | --- |
| 共享的部分 | 无 | `feature_extractor` + `torso` |
| Actor 持有 | **整条序列** | 只有它的输出变换 |
| Core 持有的兄弟组件 | 无 | bb |
| Core 的厚度 | 薄（只有 delta） | 厚 |

**Core 的厚度反比于共享程度。** 不是两套设计，是同一套在两个极端上的样子。

### 这条规则对上了方向域的数量

RTRRL 三个方向域 `actor` / `critic` / `recurrent`，各有自己的 λ（`lambda_pi` 0.97 /
`lambda_v` 0.9 / `lambda_rnn` 0.945）。数独占参数块：actor 输出变换、critic 输出变换、
共享 torso——**也是三个**。StreamAC 两个域、两个独占块。

**方向域数 = 独占参数块数。** 之前看着像算法细节的那个"三"，就是"有几块参数被独占"。
λ 归块、迹归块，全部对上。

所以 `Network` 的定义是：**一块被独占的参数，带自己的 λ 和迹，能前向、能反向。**
可以退化成一层 Dense（RTRRL 的 head），也可以是整条序列（StreamAC 的 head）。

### 接口不随边界大小改变

`backward(cotangent) -> (grad, upstream)` 两种情况都成立：

- **StreamAC**：边界就是整张网，`upstream` 没人接，丢掉
- **RTRRL**：边界停在 `h`，`upstream` 就是 `dL/dh`，送给 bb

同一个签名，区别只在上游有没有人。这也是 `Actor(Network)` 必须改成组合的原因——
继承把"持有多大"焊死成"全部"。

## 每层拥有的状态

每一层拥有状态的一片，**这是判断一层是否成立的判据**：

| 层 | 拥有的状态 | 通用性 |
| --- | --- | --- |
| Flow | step / update_step / timestep | 两算法同形状 |
| EnvTransformer | env_state + 两个统计量 | **完全通用**，不认识算法 |
| Core | 无（它持组件和接线） | StreamAC 里薄，RTRRL 里厚 |
| Network | params / carry / sensitivity / 迹 | StreamAC 2 个，RTRRL 1 个 |
| Head | 自己那部分 params / 迹 | 两算法都是 2 个 |

## 谁做什么

**组件（Network / Head）全部本地：**

- `forward(params, x, state) -> (y, state')`
- `backward(cotangent) -> (grad, upstream)` —— 本地出 `dL/dθ`，并把 `dL/dx` 往上游送
- `trace(grad, traces, *, reset)` —— **各自的 λ**（RTRRL 有三个：`lambda_pi`
  0.97 / `lambda_v` 0.9 / `lambda_rnn` 0.945）
- 走步 —— 除非选了跨组件读范数的 bound，见下

**Head 不取梯度。** 它交出一个要上升的标量（`log_prob` / `value` / `entropy`），
梯度在 Core 取。理由：RTRRL 的 `actor_to_recurrent=True` 时 actor 的方向要作用到
`torso` 的参数上，而那不是 actor head 的参数——微分必须发生在同时持有标量和完整
参数树的地方。

**Core 做四件事：**

1. **调度 critic 的两次前向** → `delta`，然后广播。当前 timestep 的 `value`，加
   下一个 timestep 上 stop_gradient 的 `next_value`。这是调度不是算术，所以 TD
   不放 head（一开始想放，因为它和采样"同性质"；采样只要 head 当下的输出，TD 要
   两个时刻的，第二个要整张网络再跑一遍）
2. **路由 cotangent，并在这里门控**。`actor_to_recurrent` / `critic_to_recurrent`
   就是把送回的 cotangent 乘 0 或 1
3. **持参数分组表**：`{组名: rule}` + "哪棵参数树归哪组"。原版就是
   `optax.multi_transform` 的 `param_labels`
4. 跨组件的走步（只在需要时，见下）

## 分组是旋钮，不是层边界

RTRRL 发表版用 `optax.multi_transform({"rnn": adam(2e-6), "td": adam(3e-5)})`，
两个组存在的理由是**不同的学习率和裁剪，不是共享范数**。Adam 逐叶，没有跨树
运算。唯一的跨树操作是 `rnn` 组的 `clip_by_global_norm`，而
`rnn = (feature_extractor, torso)` 正好是 bb 一个组件的内部。

所以**走步默认下放到组件**。例外只有一种：某个 bound 读整组的迹范数。StreamAC
的 OBGD 就是这样（`memorax/rl/updates.py:252` 的 `trace_sum` 跨整组所有叶子），
但在 StreamAC 里组=角色=组件，仍然本地。只有当一个组横跨多个组件时才必须提到
Core。

RTRRL 上要不要用 OBGD、范数怎么分组，是未定的研究问题。工作副本的 `obgd.py` 里
普通 obgd 的签名是 `update_fn(..., *, delta, z_sum)`——**`z_sum` 外部注入**，只有
`adaptive_obgd` 因为要用内部 `v_hat` 才自算。也就是说作用域本来就是可配的。

## 两个粒度，在 StreamAC 里重合

| | 方向域（谁收到一个上升方向） | 规则组（谁一起走一步） |
| --- | --- | --- |
| StreamAC | actor, critic | actor, critic |
| RTRRL | **actor, critic, recurrent** | **rnn, td** |

在 StreamAC 上做的任何统一都是猜，因为它两个角色互相独立、三个粒度全部重合。
**抽象只有在 RTRRL 上才被真正测试。**

## 信用分配是组件的性质，不是架构的

`memorax/rl/credit.py` 的两个实现同一个接口：

| | 组件收到 `dL/dh` 后 | 本地 |
| --- | --- | --- |
| `ExactRTRL` | 与携带的 `dh/dθ`（sensitivity）收缩 | ✅ |
| `TruncatedBPTT` | 对自己这一步 VJP，上游 cotangent 丢掉 | ✅ |

所以架构对 TBPTT 完全成立。受限的反而是 RTRL，且只在**叠加两个递归组件**时——
`memorax/networks/sequence.py` 在构造期就拒绝，因为需要稠密跨层 Jacobian。那是
原有限制，不是这次分层引入的。

## 实现时被否掉的划分

1. **统一的四方法签名 `(params, obs, action, reward, done)`** —— 一层都没活下来
2. **Actor 和 Critic 各写一份** —— 两者只在"往哪个方向上升"和"update 被交到什么"
   上不同，其余约 130 行相同
3. **`Actor(Network)` 继承** —— 只在 StreamAC 成立（head 与网络一对一）。RTRRL 的
   head 拿到的是共享的 `h`，自己没有 carry、没有 sensitivity。**必须是组合**
4. **把 TD 误差推进 Critic** —— 它是耦合，属于上级；推下去只是把耦合藏进算它的
   那一半
5. **`StreamACCore` 改名 `Agent`** —— 空改动，两者实现完全一样。改回 `Core`

## StreamAC 现状与待办

`memorax/algorithms/StreamAC.py` 已按四层实现，与未改动的 `memorax/algorithms/
stream_ac.py` 同种子逐位相同（9 个 interaction 叶子、actor 11 个参数叶子、critic
10 个、两边 traces，最大绝对差全 0）。它**不被任何东西 import**（装配走小写的
`stream_ac.py`），零测试覆盖，改动无连带风险。

四步已完成 ✅，改完仍逐位相同：

1. `Actor(Network)` 继承 → **组合**（`self.block`）。持有多大由独占决定
2. `Network` 收紧成"一块被独占的参数"：自己的 λ、自己的迹、作为单位被微分和走步
3. `backward(params, timestep, recurrence, direction)` —— head 只交出 `direction`
   （`log_prob + 熵` / `value`），block 负责微分。head 改成只持输出变换时，变的是
   block 有多大，不是这个签名
4. `Agent` → `Core`

**原计划的第 2、3 条自动消解了**，两条都写在"独占即持有"这条规则定下来之前：

| 原计划 | 实际 |
| --- | --- |
| 梯度提到 Core 路由 | **只对 RTRRL 成立**。StreamAC 的 head 独占整张网，梯度本来就在 head。Core 路由 cotangent 只在有共享块时出现 |
| 走步下放、分组表进 Core | 走步已经在组件里。**分组表只在一个组横跨多个组件时才需要**，StreamAC 没有，不该提前建 |

---

## R3 的下一步：按 AAAI 版重写 RTRRL

蓝本是 `../RTRRL-AAAI25/`，不是 `memorax/algorithms/rtrrl.py` 里的旧实现，也不直接
复用 memorax 现有组件——但边界与库组件一致，便于逐个替换和对测。

这是抽象第一次被真正测试：StreamAC 的两个角色互相独立、三个粒度全部重合，所以它
对"共享"什么都没证明。RTRRL 有共享 torso、三个方向域、两个规则组。

### AAAI 版的结构，以及它怎么映射到四层

读了 `../RTRRL-AAAI25/rtrrl.py`（957 行，算法全在 `train_rtrrl` 一个函数里）。

**网络**：`RNNActorCritic(nn.RNNCellBase)` 一个模块，里面

```
rnn          共享递归 torso          → rnn_step(carry, obs) -> (hidden, carry)
td: TD       一个模块装两个 head      → td.actor, td.critic 都是 FADense
   value(hidden, x)        = td.critic(hidden)
   action_dist(hidden, x)  = td.actor(hidden) → distrax 分布
```

`pass_obs` 打开时 `hidden` 会与原始观测拼接后再进 head。所以**边界与我们设计的一致**：
一个共享块 + 两个只拿 `hidden` 的 head。

**更新**：`grads_step(h, i)`，`jax.vmap` over streams（等价于我们的 `_per_stream`）。

```python
# 共享块的前向，VJP 留着待用；注意用的是 slow_params（目标网络）
hidden, rnn_backwards, rnn_state = jax.vjp(rnn_step, slow_params, has_aux=True)

@partial(jax.grad, has_aux=True, argnums=[0, 1])      # ← 对 head 参数 和 hidden 同时求导
def td_loss(_params, _hidden):
    v_hat       = model.apply(_params, _hidden, i, method=model.value)
    action_dist = model.apply(_params, _hidden, i, method=model.action_dist)
    return actor_loss.mean() * args.eta_pi + v_hat.mean(), aux

(grads_next, hidden_grads), aux = td_loss(slow_params, hidden)
hidden_grads = rnn_backwards(hidden_grads)[0]          # ← 共享块的 backward
grads_next   = tree.map(add, hidden_grads, grads_next) # ← 汇合求和
```

**`argnums=[0, 1]` 的第二个输出就是 `dL/dh`，`rnn_backwards` 就是共享块的 `backward`。**
我们设计的 `backward(cotangent) -> (grad, upstream)` 不是发明，是把这段拆开命名。

**两个目标，一个走迹一个不走**，各自重复一遍上面那三行：

| | 内容 | 去向 |
| --- | --- | --- |
| `td_loss` | `eta_pi * log_prob + v_hat` | 进资格迹 |
| `non_td_loss` | `entropy_rate * entropy`（+ slow-rnn 惩罚 + 动作幅度惩罚） | **直接作用，不进迹**，还乘一个 `actor_scale` |

**TD 误差**：`v_targ = reward + gamma * v_hat * (1 - done)`；`d = v_targ - r_bar - v_prev`。
注意 `v_prev` 是**上一步**算出的值不是重算的，`r_bar` 是平均回报项。这是"transition 之后"
的排法，和 StreamAC 当前 timestep 重新前向一次的排法不同。

**走步**：`compute_updates(z[...], trace_mode=..., d=d, dutch_diff=(v_hat - v_prev),
alpha=critic_lr)`，逐 head 一份，critic 有自己的 lr。

**AAAI 版没有门控。** `actor_to_recurrent` / `critic_to_recurrent` 是后加的（工作副本和
我们的港版里有）。以 AAAI 为蓝本**先不加**，只有一份 `hidden_grads` 送回 torso——但
**复现跑通后的第一步就是把它们加回来**，因为那是这个算法唯一能问"共享表示该听谁的"
的旋钮。

#### 范围：只实现 lru-bp-rtrl 这一条

递归核用 LRU，前馈部分走反向传播，递归部分走 RTRL。AAAI 版里其余的分支都不在范围内：
`ctrnn` / `wirings` 等别的 cell、`f_align`（feedback alignment）、`pred_obs`（观测预测
辅助头）、`mlp_actor`、离散动作分支。它们在 `RTRRLParams` 里都是开关，实现时按这条路径
钉死，不要把开关也搬过来。

#### 映射

| 四层 | AAAI 版对应 |
| --- | --- |
| Flow | `train_rtrrl` 里的 scan + `eval_model` |
| EnvTransformer | `env.step` + `normalize` + 两个 `running_statistics` |
| Core | `grads_step` 的接线：拿 `rnn_backwards`、求和、算 `d`、`compute_updates` |
| Network（共享块） | `rnn`，带 `slow_params` 目标副本 |
| Actor / Critic head | `td.actor` / `td.critic`，各持一个退化 Network（一层 FADense） |

三个独占参数块 = 三个方向域，与前面"方向域数 = 独占块数"对上。

#### 实现时要当心的两处

1. **`slow_params`**：torso 前向用目标副本，`follow_torso` 按 `update_period` 跟随。这是
   StreamAC 没有的状态，`EnvironmentState` / `NetworkState` 之外要多一片
2. **`v_prev`**：TD 误差读上一步的值，所以它是 carry 的一部分，不是本步算出来的。
   StreamAC 是本步重新前向拿 `value`——**两个算法在这里排法不同，别照抄 StreamAC**

---

# R4 — 验证与收口

## R4a 数值比对：`rtrrl_aaai.py` 对 AAAI 版

**现状：没有做过任何数值比对。** 只有 21 条行为性质检查，它们能证明"迹在推进前被
使用"，证不了"和原版算出同一个数"。这是目前唯一未验证的部分。

不能像 StreamAC 那样逐位比的原因：`stream_ac.py` 同仓库、同环境、同一批网络对象；
AAAI 版则用自己的 LRU（`OnlineLRULayer` 把 C/D 和 silu 裹在同一个模块里）、自己的
actor（`FADense` → loc + `sigmoid_between(log_scale)`）、brax 环境、不同的 key 花法。
不移植这些就不可能逐位相同。

能比的是**更新代数**——固定 params 和 hidden，两边算同一件事。

可行性已验证：memo 的解释器里 `models.online_lru`、`models.neural_networks`、
`traces`、`distrax`、`brax` 都能直接 import；只有 `optimizers` 因缺 `simple_parsing`
进不来，那个用 optax 自己拼就行。

### 三级，从最能隔离到最完整

**L1 迹与 TD 代数——不碰网络，必须 0 差**

造一组假的 grads / 迹 / d / done / emphasis，一边跑 AAAI 的 `traces.trace_update` +
`traces.compute_updates`，一边跑 `Network.trace` + `make_optax_rule.apply`，比参数
增量。覆盖：三个 λ 各归其块、emphasis 的递推、结束清迹与更新的先后、`eta_f` 只乘
torso 的 delta、直接项不乘 delta 也不进迹。这一级不依赖任何网络实现，**0 差是硬要求**。

**L2 cotangent 路由——用 AAAI 的网络**

直接建 AAAI 的 `RNNActorCritic`（绕开 `train_rtrrl`），一边跑 `grads_step` 里
`jax.vjp` + `td_loss` + `non_td_loss` 那三段，一边把同一个模块包成 `Torso` /
`Actor` / `Critic` 的接口跑 `Core._per_stream`，比三个块的 grads。需要写适配器：
`Torso` 现在要 `Sequence.walk` 和 `.core`，头要返回 `(output, dict)`。覆盖：VJP 建
一次调两次、两个头的 cotangent 求和、`eta_pi` 落在哪。

**L3 单步全量**

固定一条 transition（reward/done 直接给）和一份 params，比一整步之后的三份参数、
三份迹、`value`、`emphasis`、slow torso。这一级会撞上有意的差异，见下表。

### 有意的差异（比对前必须对齐或显式豁免）

| 项 | AAAI 版 | 这里 | 比对时怎么办 |
| --- | --- | --- | --- |
| log_prob 归约 | 对动作维 `.mean()` | Independent 分布求和 | 差一个 `a_dim`，折进 `eta_pi` |
| 采样 key | 所有流共用一个 | 逐流 split | 强制单流 |
| 结束语义 | 只有 `done` | `terminal` 与 `done` 分开 | 令二者相等 |
| 环境重置 | brax wrapper 自动 | 步首显式 `reset` | 比对绕开环境 |
| actor 头 | `FADense` + `sigmoid_between` | `StateStdGaussian` + softplus | L1/L2 两边用同一个模块规避 |

### 出口条件

L1 全 0 差；L2 全 0 差；L3 每项要么 0 差、要么在上表里有书面理由。

## R4b 组件收口

一条原则贯穿下面几条：**不要按用途做方法，做一个方法、让调用方传那个变化的部分。**

1. **删 `EnvTransformer.opened_for_evaluation`。** train 和 eval 的统计量定义一致，
   eval 读 train 的统计量就是把 train 的 pytree 传进去，不需要第二个分配器。真正该
   拆的是**前向与更新**：估计器答一个值、和估计器见过这个值，是两件事；`update`
   已经是那个开关，把它拆干净即可。
2. **归一化要有自己的 state。** 打包进同一棵树是实现方便，不构成不分离的理由。
   agent 从不需要知道归一化前是多少——它是 env 与 agent 之间的一层，
   `EnvironmentState` 里应当只剩 `env_state`。"结束时一起重置"是生命周期巧合，
   不是所有权。
3. **`Critic.evaluate` / `Critic.bootstrap` 合成一个 `apply`。** 两趟的差别只在传进去
   的东西：参数切不切图、recurrence 切不切图、推进后的 recurrence 留不留。这三件都是
   **耦合的事实**，属于 Core。合完之后 StreamAC 的 Critic 与 RTRRL 的一模一样。
4. **简化 `Network` 的前向侧。** `backward` 下放到 Actor/Critic：RTRRL 完全没用它
   （Core 直接 `jax.vjp` / `jax.grad`，因为 cotangent 要路由、将来要门控），它只在
   "块 = 整个网络"时成立。顺带去掉那两段没有理由的柯里化。
5. **`direction` 改名 `objective`。** 方法返回的是要上升的**标量**，方向是它的梯度——
   名字差了一阶导。而且 `rl/updates.py` 的 `ObjectiveDirections` 已经用 "direction"
   指梯度形状的东西，两处打架。RTRRL 的 `traced` / `direct` 相应改成
   `traced_objective` / `immediate_objective`。
6. **`trace` / `step` 的拆分保留。** 在 StreamAC 里 `trace` 只有一个调用者，看着多余；
   RTRRL 里 Core 直接调 `block.trace`、`Network` 根本没有 `step`，因为规则组横跨块。
   这条缝正是第二个算法需要的。
7. **模块级小函数先不动**（`_broadcast_env` / `_where_done` / `_as_batch` /
   `_from_batch`）。记一笔：`_broadcast_env` 写错过一次——`Torso._input` 里默认
   `done` 是一维，实际是 `(B, T)`——所以它是那个假设被写下来的地方。

## R4c 上一轮实现暴露的事实

- **规则组与方向域不重合**：三个独占块（torso / actor 读出 / critic 读出）= 三个方向域，
  但只有两个规则组（torso 单独裁剪，两个读出一起走）。这个错位就是 `Core` 必须持分组表
  的理由，也是 RTRRL 的 `Network` 没有 `step` 的理由。StreamAC 里三种粒度全重合，证不了。
- **`heads.Gaussian` 的 `log_std` 与状态无关**，于是它的熵不依赖 hidden，
  **熵到共享 torso 的整条路径恒等于零**。要复现 AAAI 的 actor 必须用
  `heads.StateStdGaussian`。
- **`init` 与 `reset` 是两个词**：`init` 分配（Flax、optax、运行契约都这么用），
  `reset` 重新开始（env、`reset_carry` 都这么用）。新文件一度把 `init` 用反了，已改回。
- **Windows 路径不区分大小写**：`RTRRL.py` 和 `rtrrl.py` 是同一个文件。
  `StreamAC.py` / `stream_ac.py` 那套大小写区分的命名在单词模块名上用不了，
  所以新实现叫 `rtrrl_aaai.py`。

## R4d 收口执行结果（已完成）

R4b 的六条全部落在 `memorax/algorithms/StreamAC.py` 上，**全程与未改动的
`stream_ac.py` 逐位相同：4 组场景 × 20 项 + 评估读数 = 82/82，全 0 差**，其中包含
之前从未被验证过的两个估计量、两条评估策略、以及十项读数本身。

| 断言 | 结果 |
| --- | --- |
| `opened_for_evaluation` 冗余 | 真。它逐字是 `init` 加两个开关；`init(key)` 恰好等于 `resets_on_start=True` 时的它 |
| `Critic.evaluate`/`bootstrap` 该合一 | 真。合成 `apply` 后 stop_gradient 出现在 Core 的调用点，"哪一趟被切图"从"调了哪个方法"变成看得见的实参 |
| `Network.backward` 该下放 | 真。搬到 `Actor.gradient` / `Critic.gradient` 后，两段柯里化自己消失 |
| `direction` 该叫 `objective` | 真。方法返回的是要上升的**标量**，方向是它的梯度，名字差了一阶导；且与 `rl/updates.py` 的 `ObjectiveDirections` 打架 |
| 归一化该有自己的 state | 真，但理由见 R4c-2 |

`Environment` 与 `Normalization` 拆成两个组件，状态从"环境(env_state + 两个统计量)"
变成 `env_state` 和 `scales` 两片。`EnvTransformer` 只读过 `num_envs` 一个字段。

---

## R4e 组件契约（定稿）

### 1. 图与状态分离，计算图无状态

**这不是风格选择，是 `lax.scan` 逼出来的**：凡是 kernel 携带的必须是 pytree，凡是不
被携带的就不许变，有状态实例进不了 scan carry。已核对：两个文件里 `self.x =` 全部
落在构造器体内，构造之后没有任何组件改自己；`Normalizer` 是 frozen dataclass。

措辞要精确：无状态指**没有会变的东西**，不是没有数组（`env_params` 是常量）。

### 2. 前向 + 更新，但**更新不是原子的**

RTRRL 暴露出更新分两半：

| | 归属 | 依据 |
| --- | --- | --- |
| `trace` 迹递推 | **永远在组件里** | 每块有自己的 λ |
| `step` 走步 | **在拥有规则组的那一层** | 组可能横跨组件 |

判据：**规则读的东西全在组件内 → 走步就在组件内。** StreamAC 的 OBGD 读整组迹范数
而组=角色=组件，本地；RTRRL 两个读出同组，将来放组级 bound 就横跨两个组件，必须上提。

已知不一致：RTRRL 的 torso 组是单元素，`clip_by_global_norm` 只跨自己，走步本可留在
`Network`；一起上提是为了分组表只有一份。真正强制上提的只有读出那一组。

"更新方法可以为空"改成：**更新在拥有它所读之物的那一层定义，不在的层就没有这个方法**。
空方法比没有方法更糟——读的人过去什么也找不到，而且它不说是谁在做。

### 3. 共享图与状态，不共享有状态实例

1 成立时前半自动。后半的价值远大于"少一个对象"：**它让一整类评估污染变成不可表达的**，
`update_during_eval` 从"我有没有污染训练统计量"降级成"这一步的返回值写不写回去"。

残留的空洞已经堵上：`evaluate` 现在**只返回 metrics**，评估跑的状态在它内部建、内部丢，
调用方拿不到，也就无从写回。

### 4. `init` / `reset` 不合并

关系记一下：`reset` 就是 `init` 保留参数那一路（`reset = init(key, keep=(params, traces))`）。
合并的收益只是少一个名字，代价是一个带可选状态参数和分支的方法。不值。

### 5. 契约还差的两块

- **形状从哪来**：两个 `init` 都要一份样例 `timestep` 才能建状态。契约隐含要求样例输入，
  该写出来，否则"图无状态"会让人以为 `init` 只要 key。
- **只持接线、不持状态的层**：StreamAC 的 `Core` 一片状态都没有，RTRRL 的 `CoreState`
  有 `value`/`emphasis`/`rule`。所以判据要改：

> **一个组件成立，当且仅当它独占一片状态、或独占一段接线。**

`Core` 属于后者：它独占"critic 两次前向 → delta → 广播"和"两个头的 cotangent 求和
→ 回推 torso"，这两段没有别处可放。

---

## R4f metrics 设计（定稿）

### 两级，且算法只能产出一级

| | 谁做 | 形状 |
| --- | --- | --- |
| step 级 | kernel，scan 逐步堆叠 | `(T, N)` 每步每流 |
| `EPISODE_FIELDS` | 底线 `interaction.{reward,done,terminal}` | 没有它们**根本找不到回合** |
| `Runtime.series` | 声明哪些 step 级量按回合归约 | → `statistics()` 出 `name` 与 `name_variance` |
| 真正回合级 | 只有 `length` / `return`，在 `statistics()` 里算 | O(回合数) |

**回合级 metrics 全是 step 级的归约，kernel 一个都产不出**：回合在每条流里结束的步不同，
回合记录是步流的**变长切片**，而 scan 每步只能吐**固定形状**；`Episode` 的字段是
`Sequence`/`float`，根本不是 pytree。切分属于 `memorax/runtime/episode.py`，在 scan 之后
用 Python 做。这个文件欠那一刀的，正好是 `done` 和 `terminal`——两层之间的接口。

### 每层返回 `(state, metrics)`

之前"nothing is observed but the transition"那段立场：**怕的是共享 schema 反过来驱动
算术，这条是对的；结论"因此什么都不发"是错的。** 答案是 metrics 是**各算法自己的类**：
RTRRL 要而 StreamAC 没意见的字段在这里根本不存在，没有签名为它变宽，也没有共享 schema
能伸回算术里。只有返回通道变宽，输入一律不动。

metrics 树**按组件嵌套**，声明名就是路径（`update.actor.step_size`），与参数树同形。

### 副产品 vs 可导出

| 读数 | 来源 | 单独 `metrics()` 方法拿得到吗 |
| --- | --- | --- |
| `step_size` | `rule.apply` 内部 | ❌ 不存任何地方 |
| `grad_norm` | 需要梯度 | ❌ 梯度从未进过 state（进的是迹） |
| `value` / `log_prob` | 前向已在手 | ❌ 事后要重跑前向 |
| `trace_norm` | 迹在 state 里 | ✅ |

三项里只有一项可导出，所以**返回式严格更强**。

### 静态门控 = 不算

只要门控在构造器里定死，`grad_norm=self._norms(g) if self.reports.grad_norm else None`
在 jit 之后 XLA 直接 DCE——**`None` 和"没算"是同一件事**。正合"Everything static is
resolved in the constructor"。

### 声明即门控，用参数树的词汇

`record` 现在是 `frozenset[str]` 点分名字，**两处静默失败**：kernel 侧拼错静默不记，
driver 侧声明了 kernel 产不出的名字静默丢（今天那 18 个就是这么消失的）。

用 `memorax/parameters.py` 那套树词汇声明 metrics，换来：构造期拒绝一个组件给不出的
名字（那 18 个会是构建错误而不是沉默）、catalog 像发布参数树一样发布 metrics 树、
kernel 里没有字符串、声明本身就是门控不需要第二个机制。

---

## R4g 本轮量到 / 撞到的事实

1. **`stream_ac.py` 的 eval 预算 bug**：`evaluate` 直接扫 `num_steps` 轮，而 `train`
   扫 `num_steps // num_envs`。同一个参数一个是环境步数、一个是扫描轮数，评估预算被乘了
   `num_envs` 倍。分层时 `StreamAC.py` 把两者统一了——**一次没人注意到的行为改变**，
   因为没有东西 import 它、也没有测试覆盖 eval。是扩展差分到评估路径才撞出来的。
2. **归一化是三件塞在两个对象里**：`Statistics` 同时装 `mean/M2/count`（估计量）和
   `trace`（把回报累成折扣回报的累加器）；观测那路只有前者，回报那路两者都有。所以
   `observe` 里"先更新再读"**不是约束，就是接线**，只因两件东西不可分别表达而看不见。
   拆不开是**粒度问题，不是算法性质**——这一条推翻了我先前的判断。
3. **18 个声明的 series 会静默消失**：`entries/stream_ac.py` 声明 18 个 `update.*` /
   `forward.*`，分层版一个都不发，driver 的归约 `if ... is not None` 静默丢弃。
   后果不是今天坏掉，而是 **`METRICS` 是 catalog 对外公布的名字表**——契约上写着 18 个
   训练指标，实际一个不出现，没有报错。（本轮已修复。）
4. **"轨迹贵"不是普遍事实**：`timestep` 在向量观测下 0.3 KiB/步 → 0.02 MiB/epoch，
   便宜得很；图像观测（84×84×4 × 16 流）≈ 450 KiB/步 → 27 MiB/epoch。`record` 开关的
   正当性完全取决于观测大小，该按大小说，不该按原则说。
5. **"state 即轨迹"不可行，实测**（16 流 / 128 hidden / 1000 步一 epoch）：

   | 逐步堆什么 | 每步 | 每 epoch |
   | --- | --- | --- |
   | 整个 state | 17067 KiB | **1033 MiB** |
   | 去掉 params/traces/v | 8288 KiB | **502 MiB** |
   | 今天发的 `interaction` | 0.8 KiB | 0.03 MiB |
   | 声明的 18 个 series | 1.1 KiB | **0.07 MiB** |

   **参数只占 1.6%**（266 KiB）。撑爆的是逐流的学习机器：迹是参数的每流一份拷贝（50%），
   carry + RTRL sensitivity（49%）。直觉"参数是大头"在这个算法上是错的。
6. **真正杀死"state 即轨迹"的与大小无关**：声明的读数**根本不在 state 里**。
   `td_error`、`step_size`、`grad_norm`、`value`/`log_prob`/`entropy` 全都在
   `update_parameters` 内部算出、当场丢弃；梯度从未进过 state。哪怕愿意付那 1 GiB，
   一个都拿不到。
7. **Windows 路径不区分大小写**：`RTRRL.py` 和 `rtrrl.py` 是同一个文件，
   `StreamAC.py` / `stream_ac.py` 那套大小写命名在单词模块名上用不了。新实现叫
   `rtrrl_aaai.py`。（`__pycache__` 同样中招。）

## R4h ①③④a 的结果

### 判定的两类，以及为什么必须分开

L1 跑出来两次假失败，都是拿第二类当第一类判的：

| | 是什么 | 判据 |
| --- | --- | --- |
| **同一性** | 同一个量、两个实现各算一遍 | **严格 0**，否则是 bug |
| **恒等式** | 由两个计算结果相减推导出来的等式 | 只可能在舍入内相等，按量级判 ulp |

两次教训：**(1)** 迹的 1 ulp 差是 FMA——发表版的 `trace_update` 带 `@jax.jit`，XLA 把乘加融成 FMA，我这边 eager。两种精度探过，gap 随格式 eps 缩放（f32 1.0 eps / f64 0.33 eps），两边都 jit 后变成 0。**(2)** "delta 加倍只加倍走迹那半"这条恒等式我拿 Adam 验，而 Adam 对它收到的 ascent 非线性，换线性 transform 后成立。

`tests/conftest.py` 里的 `assert_within(..., allowed=)` 以 ulp 为单位，正好是这个区分的连续版：`allowed=0` 是同一性，`allowed=8` 是恒等式。**不要另造判定层。**

### ① L1 — 迹与 TD 代数，20/20

不碰网络。三块各自的 λ、`delta × 迹` 分组（含 `eta_f` 只乘 torso）、`init_trace` 起点，全部 **0 差**；结束清迹的时序、emphasis 递推与复位、直接项不进迹不被加权，全部成立。

### ③ 接口重构 — 自差分 571 叶逐位相同

两趟（环境拆分 + `(state, metrics)`）之后与重构前**逐位不变**，21/21 性质检查。顺带验掉
`sample_and_log_prob` 与 `sample` 同 key 同抽样。

### ④a 整程差分 — 第一步精确 0，12 步 ≤ 3 last bits

基准是**发表版的接线**（vjp 建一次调两次、一个合并目标微分一次、TD 读上一步的值、
沿推进前的迹更新、之后才清迹），跑在 **memorax 的网络和 memo 的环境**上——网络差异按
构造消除，失配只可能来自接线。

同时验掉两件之前只论证过的事：**两个头的 cotangent 相加 ≡ 求和后一次微分**；
**把 sensitivity 从 carry 里拆成 `Recurrence(carry, sensitivity)` 等价**。

12 步的漂移是算子顺序不同带来的舍入（基准先求和再微分，这边先微分再相加）。

## R4i 发表版的 actor 梯度穿过了重参数化采样

`grads_step` 的 `td_loss` 里：

```python
action = action_dist.sample(seed=action_key)   # ← 在被微分的函数里面
actor_loss = action_dist.log_prob(action)
```

`distrax` 的 `.sample()` 是重参数化的，所以 `action` 对 `_params` 可微，**梯度穿过采样**。
在基准里给这一行加 `stop_gradient`，整程差分从"57/61 叶子差 2500 万 ulp"变成 **0/61**——
这一条路径就是全部分歧。

后果量过了（高斯，`x = loc + scale·ε`，于是 `(x-loc)/scale ≡ ε`）：

```
穿过采样:  d/d loc = [0. 0.]           d/d scale = [-0.689 -0.639]
截断采样:  d/d loc = [1.378 -0.411]    d/d scale = [ 0.145 -0.537]
```

**穿过采样时策略均值的梯度精确为零**，scale 的梯度为负（上升即收缩策略）。熵对高斯均值
也没有梯度，所以按发表版那一行写，actor 的均值参数无处可学。`FADense` 的 `custom_vjp`
救不回来：抵消发生在分布那一层，到达 dense 的 cotangent 已经是零。

**这里的实现截断采样**，与 StreamAC 和标准 policy gradient 一致。这是"有意的差异"里最
重要的一条。只验证了这条路径的数学、以及它解释了全部分歧；发表版整个栈里有没有别的机制
没有追。

**这正是整程差分能抓、性质检查抓不到的东西**——21 条性质检查全过，动作和值也完全一致，
只有迹不一致，而且 critic 的迹一致、actor 和 torso 的不一致。

## R4j 保留意见

- **组件是否应各自持有梯度与信用、跨层传递**，而不是组装成 before/recurrence/after 三组。
  当前的回答是：链式传递由 autodiff 完成，块边界只在**有东西要拦截 cotangent** 处存在
  （共享 torso、门控）；迹的衰减与规则组在算法自己的词汇里就是块级的（三个 λ、两个组）；
  唯一真正自持信用的递归单元已经自持，且 `Sequence` 在构造期拒绝第二个递归组件，因为
  跨层稠密 Jacobian 没有实现。**此条未经源码逐行核对，保留。**
- 参考实现应集中一处：**组件级的参考也是参考**，与整程参考同放 `reference/`（⑨）。

## R4k ④b 组件级：memorax 的 LRU 对发表版

同一组参数移植过去、同一段输入、单流。比两件事：读出的值，和 kernel 会从它取的在线梯度。

两边对 sensitivity 的因式分解不同。发表版的单元carry **三项**——`∂h/∂λ`、`∂h/∂γ`、
`∂h/∂B`——求梯度时再经 `vjp_to_lambda` 把 λ 链回 ν 和 θ；memorax carry **五项**，ν 和 θ
已经各自展开。若是同一个递推的两种写法，梯度应当一致。

**它们不一致。** 第 0 步前向相同、读出参数（`C_real`/`C_imag`/`D`）梯度相同，但递归参数
的梯度从第 0 步就分叉。

### 用 BPTT 当裁判

第 0 步没有历史，RTRL 必须等于普通反传。发表版自己带 `plasticity="bptt"` 分支（不走
custom_vjp，纯 autodiff），拿它当第三方：

| 叶子 | 发表版 RTRL vs BPTT | memorax vs BPTT |
| --- | --- | --- |
| `C_real` / `C_imag` / `D` | 0 | 0 |
| `nu_log` / `theta_log` | 0（**无效**，见下） | 0 |
| `B_real` | 9.5e-1 | **0** |
| `B_imag` | 2.4e+0 | **0** |
| `gamma_log` | 3.0e+0 | **0** |

（相对误差，除以该叶子的最大幅值。）

**memorax 在全部八个叶子上与 BPTT 精确相同；发表版的 RTRL 在 B 和 γ 上错。**

`nu_log`/`theta_log` 那两个 0 不算证据：初始 carry 为零时 `∂h₀/∂λ = 0`，两边都是零，
第 0 步根本没有考验到 λ 那条链。要验它需要至少两步，而两步之后 carry 已经因为 B/γ 的
错误而分叉了。

### 错在哪

`gamma_log` 的数值形态是**第 0 个分量对、其余全错**：

```
BPTT      [-0.0378, -0.0158,  0.0910]
发表 RTRL [-0.0378, -0.3519,  0.0236]
```

指向 `models/online_lru.py` 的 `rtrl_gradient`：

```python
d_output_d_h = y_t[1][0]
```

隐状态的 cotangent 形状是 `(H,)`，`[0]` 把整个向量塌成第一个分量再广播到 B 和 γ 的
每一行。`rtrrl.py` 的 `grads_step` 在 `jax.vmap` **内部**调用，carry 是非批处理的，走的
正是这条路径。`nu`/`theta` 逃过一劫是因为它们经 `vjp_to_lambda`——一个真正的 JAX VJP。

只验证了第 0 步、单流、`activation=None`、`output_dim` 指定的配置。根因是从数值形态推
出来的，没有改发表版的代码去证实。

### 对 ④a 的影响

④a 两边共用 memorax 的网络，所以这条与 ④a 的结论无关——④a 验的是接线，而接线在第一步
精确相同。反过来说：**如果 ④a 当初按"AAAI 组件 + 适配器"做，它会因为这个组件级 bug 而
失配，而我会先去怀疑自己的接线。** 先做 ④a 再做 ④b 这个顺序是对的。

## R4l ⑨ 参考迁移与校验改造（已完成）

安全网从 scratchpad 进了仓库。**没有新造任何判定层**——`tests/conftest.py` 里的
`assert_within(..., allowed=)` 以 ulp 为单位，`allowed=0` 是同一性、`allowed=8` 是恒等式，
正是需要的那个区分的连续版。

```
memo/reference/
  __init__.py            位置即规则：包在 memorax 外面，发布代码引不到
  upstream_stream_ac.py  从 memorax/algorithms/ 迁入（相对 import 改绝对）
  rtrrl_aaai25.py        AAAI 接线的参考，checkout 由 RTRRL_AAAI25 环境变量找

memo/tests/
  test_layered_parity.py   StreamAC.py 对 stream_ac.py，4 组场景 77–88 叶，4 项
  test_rtrrl_parity.py     代数 / 接线 / 单元，7 项，没 checkout 就 skip
  test_rtrrl.py            RTRRL 的行为性质，20 项，不依赖任何外部 checkout
```

`upstream_stream_ac.py` 原来靠 docstring 说"不要 import 我来跑东西"，现在靠**目录位置**。
`reference/` 在 `memorax` 包外面，发布路径想引都引不到；测试能找到是因为 `pytest.ini`
把 checkout 根放在了路径上。

### 两个模块兼作报告脚本

```
pytest tests/test_rtrrl_parity.py    →  判定（7 passed，或 7 skipped）
python tests/test_rtrrl_parity.py    →  数字
```

判定是给 CI 的，数字是给人的。同一份代码、同一批数——这补上了"test 只能判定不能报告"
那个缺口，而不需要 `record_property` 或第二套机制。

### 两处写死在测试里的判断

1. **`nu_log` / `theta_log` 在单元比对里被显式排除**，并写明理由：初始 carry 为零时
   `∂h₀/∂λ = 0`，两边都是零，那两个 0 不是证据。留着会让人以为覆盖了 λ 那条链。
2. **"发表版与自己的 BPTT 不一致"写成了一条会失败的测试**——checkout 哪天修好了它会失败，
   并提示"这个文件对它的记述过期了"。这条发现因此不会悄悄变成谎话。

### 过期日不是问题

`test_layered_parity.py` 的真值是 `stream_ac.py`，而 ⑦ 要替换它。仓库自己已经走过这条路
并给了答案：金标准快照被删过，取而代之的是留一份**活的参考实现**。所以 ⑦ 的时候把
`stream_ac.py` 改名迁进 `reference/`，跟 `upstream_stream_ac.py` 一个待遇，差分不过期。

### 顺带

RTRRL 的 21 条行为检查转成 `test_rtrrl.py` 时并成 20 项（三条分组隔离的检查参数化成一条）。
③ 用的自差分快照是脚手架，用完丢弃——它证明的是"重构没改行为"，而那件事已经发生过了。

两个 RTRRL 测试文件共用一个 `build`：**两份构造器就是两个 kernel**，一旦任一份漂移，
比对就悄悄在比错的东西。

## R4m ⑥ 归一化拆件（已完成）

`Normalizer` 是两样东西焊在一起：把值变成它所属折扣累积的**累加器**，和跑动的**均值与
散度**。观测那路只用后者，回报那路两者都用。

```
Accumulation   持 Statistics.trace
Spread         持 mean / M2 / count，以及用它们缩放
Normalizer     装配：accumulate → advance → read
```

状态仍是一棵树（`test_blocks.py` 和 `test_normalization.py` 直接读它的字段），但每个组件
只点自己的字段。差分仍精确——纯重排。

**拆完暴露的一件事**：`read` 拿的是 `sample`，`advance` 拿的是 `counted`——**两者读的不是
同一个量**。这在原来的单体 `observe` 里是真的，但没有地方能看出来。

这也修正了先前的判断：我说过"前向/更新拆不了，因为 read 用刚写完的统计量"。正确版本是
**顺序确实是先 advance 后 read，但那是接线不是约束**——粒度不够让它看起来像约束。

## R4n ⑧a 评估预算（已完成）

`stream_ac.evaluate` 扫 `num_steps` 轮，而它自己的 `train` 扫 `num_steps // num_envs`。
driver 两边都传 `config.evaluation.steps`，所以**每次 shipped 评估都跑了 `num_envs` 倍于
要求的步数**——按模板的批大小是 16 倍。两边现在都是环境步数，差分里的特例删除。

## R4o ⑧b 评估独立流数（推迟，需单独讨论）

不是"加个参数"。查出来两处耦合：

**一、流数焊在 12 个位置**，且 `Environment` / `Normalization` 的 `num_envs` 是**构造器
字段**不是调用参数：`blank_timestep` 的三个 zeros、两处 `split(key, num_envs)`、回报统计量的
`initial`、`Network.carry_shape`、迹的分配、两个组件的构造。

要么这些方法全部加参数（签名一路改上去），要么**给评估建第二组组件实例**。后者是对的，
而且和"不要两个组件、传状态进去"不冲突——**形状是无状态图的构建期属性，不是状态**。
值不同就传状态，形状不同就是两个图。这个区分是本轮新得到的。

**二、`reset_on_start=False` 与不同的评估流数不兼容。** 统计量是逐流的
（`mean = zeros_like(sample)`），继承训练的统计量意味着拿 `(train_envs, obs_dim)` 的 mean
去减 `(eval_envs, obs_dim)` 的样本。而 `RunningNormalization.reset_on_start` 的默认搜索值
就是 `False`。要么拒绝这个组合，要么把逐流统计量归约后广播——后者改的是统计量的语义，
不该由这个功能顺带决定。

---

# R5 — 形式化协议（issue 46，已完成）

主文结果要能被引用，需要三件今天没有的东西：checkpoint 的 episode 数是**确切**的、
主分是**固定评估曲线的 AUC**、种子的新鲜性是**被校验并归档的身份**。契约 9→10。

## R5a 评估按 episode 数，而不是按步数

**这是本轮唯一一处非改不可的结构变动。** 一个 checkpoint 按 N 条完整 episode 计分，
而"N 条 episode"不是任何一个步数——跑多久由策略决定。于是 `evaluate` 一次 scan 装不下
它，程序契约从四支箭变成五支：

```
open_evaluation(key, state) -> eval_state          # 原来的私有 _evaluation_state
evaluate(key, eval_state, n) -> (eval_state, m)    # 原来只返回 m
```

driver 拿着 `eval_state` 反复调用，直到点名的槽位都填满。**交回状态并没有打开"评估污染
训练"这个洞**：那个状态是 `open_evaluation` 在全新环境上建的，调用方拿到的从来不是训练
状态，而 driver 也不写回。R4e-3 记的"`evaluate` 只返回 metrics，所以写不回去"这条论据
到此换了形式，结论没变。

### 哪 N 条算数：先点名，不看谁先跑完

按完成顺序取前 N 条会**系统性偏向短 episode**——短的先结束。所以槽位在 rollout 跑之前
就定好：第 i 流的第 j 条填 `j * num_envs + i`，槽号 < N 的才计分。这条规则不要求 N 被
流数整除（低位流多担一条），也不受完成时间影响；跑超了的额外 episode 同样不计分——**多
一条和少一条一样是改了那个数**。

### episode 切割只剩一份实现

原来训练走 `EpisodeTracker`（跨 chunk 续接），评估走 `rollout.complete_episodes`（单块内
切）。要跨调用凑够 N 条就必须续接，于是评估也走 tracker，`rollout.py` 整个删除，它的
`read` 搬进 `tracker.py`。tracker 多两个参数：`phase` 和 `stride`（0 = 钉在 boundary 上，
评估不推进横轴）。

第三个参数 `require_series` 是删这份重复时暴露的**真实语义差**：schema 的 series 是算法的
**更新读数**，训练缺一个是接线错误，而评估不做更新，那些读数本来就不存在。旧的
`complete_episodes` 静默跳过，tracker 严格报错——两边都对，只是对着不同的相位。

## R5b 评估有自己的 key 流

原来 driver 是 `key, eval_key = jax.random.split(key)`，也就是**跑没跑评估会改变后续训练
的 key**。同一份配置每 10k 测一次和每 100k 测一次，训练轨迹不是同一条。这不是本轮引入
的，是本轮才被写下来的。

现在 `evaluation.seed` 独立开一条流，每个 checkpoint 的 key 是 `fold_in(eval_key,
boundary)`。顺带得到两件事：评估自身可复现，且两个方法声明同一个 seed 就是在**配对的**
评估 episode 上被比较。

## R5c 打分器：两级归约

**issue 没写这一层，但它是被数据形状逼出来的**：eval 的每条 episode 在 metrics.jsonl 里
是**一行**，同一个 checkpoint 的 10 条是 10 行同 step。所以 AUC 不能对行做——那等于按
"这个 checkpoint 碰巧记了几条"给它加权，而固定 episode 数存在的意义正是让这个不变。

`auc` 和 `last_checkpoints` 先把同一 step 的行归约成该 checkpoint 的均值，再对均值序列
做归约。五个老的点归约照旧读行，未动。

AUC 取**归一化**形式（除以步跨度），量纲同 return，可与 last-five 并排读、可跨预算比；
对固定预算它是原始积分乘常数，HPO 排序完全一致。端点是**窗口内实际到达的**首末
checkpoint 而不是窗口边界——延伸到没测过的步等于外推。缺测的区间由梯形跨过（即两侧连
线），这与画曲线时看到的是同一条线。

`episodes_per_checkpoint` 是可选的一项，它把 runtime 的"确切"从一句声明变成**被检查的
断言**：某个 checkpoint 没报够条数就拒绝这份文件。

内存约束照旧（#33 的教训）：折叠只持一个未闭合 checkpoint、前一个 checkpoint、最后 k 个
均值，不随文件长度增长。

## R5d 种子是被声明的维度，不是被搜索的参数

`environment.seed`（标量）→ `environment.seeds`（列表）。每个配置在列表里**每个**种子上
各跑一次，种子**不进搜索空间**——两次只差种子的运行是同一个配置量了两遍，让采样器抽它
等于让 study 把预算花在建噪声的模型上，然后把手气最好的那次报成最优配置。

调参列一个，此时"均值"就是那一次的分数本身，与 issue 46 说的"HPO 内不做多种子聚合"一致；
正式评估列 10 个（离散）或 5 个（Brax）。#38 想要的分组种子调参因此不需要单独的调度器，
它是这个列表长度大于一时的自然行为。

run 的身份随之变成 (trial, seed)：`run_id` 加 `-s{seed}`，`identity` 加 `seed` 和 `role`，
交换区的文件名加 `-seed-NNNNNN`。**没有新增子命令**——正式评估是一份手写的实验文件，
`selection` 块声明它从哪个 study 的哪个 trial 冻结而来、调参用过哪些种子，三条拒绝规则
（种子重叠、space 还在搜、主分是 `train/` 指标）在起容器之前失败。

## R5e 没做的

- **"Original RTRRL 和 PPO 镜像"在本仓库无对象**：`entries/` 只有 rtrrl / r2d2 /
  stream_ac，`algorithms/ppo.py` 存在但没有 entry 也没有 catalog 条目。issue 里"先检查
  现有镜像是否支持精确 episode 数"这条按不适用处理。
- **契约 bump 无法回避**：`evaluation` 块换了字段而 `RunSpec` 是 `extra="forbid"`，钉住
  的镜像必须重建。issue 说的"prefer compatible adapters around pinned images"对 memo 的
  三个 entry 不成立——旧镜像本来就做不到精确 episode 数。两个 smoke 配置的 `image` 因此
  留 `TBD`，与 `rtrrl repeatprevious memo.yaml` 同样的状态。
- **`trainerctl-manual.md` 的陈旧不止这一处**：它仍写着 `epoch_steps`、
  `evaluation.num_envs`、`logging.aim` 是字符串、`enable_rerun`——都是 R1a 删掉的字段。
  本轮只改了 issue 46 触及的那几节，其余未动。
