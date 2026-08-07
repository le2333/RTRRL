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
