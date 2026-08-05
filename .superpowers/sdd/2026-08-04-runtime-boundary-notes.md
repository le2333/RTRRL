# 运行时边界讨论记录(临时,未提交)

写于 2026-08-04,供压缩上下文后核对。内容是围绕
`docs/superpowers/specs/2026-08-04-streamac-component-factory-design.md` 的讨论、查证到的
事实、以及我在过程中做错的判断。**这份文件不是结论,是待你确认的材料。**

---

## 0. 分支状态

- 我这一段的提交都在 `feature/rtrrl-lru-paper-parity`,当前 HEAD `3f80571`
  (我最后一个提交 `b644a5b` 之后,又有 7 个提交叠上来:factory boundary 的设计文档、
  清理旧配置等)。
- `refactor/streamac-factory-boundary` 从 `3f80571` 分出,13 个提交,worktree 在
  `.worktrees/streamac-factory-boundary`。
- **两条分支没有分叉**,factory 分支包含我提交的全部内容。不会冲突。
- `docs/superpowers/specs/2026-07-30-configuration-surface-design.md` **已被删除**,
  现在只剩 `2026-07-29-numerical-testbench-design.md` 和 08-04 这份。

---

## 1. 已经定下来的

### 三层结构

```
memorax ──(runtime、算法入口、配置树)──► adapter ──(扫描域、配置文件、运行指令)──► infra
```

- **memorax** 纯计算 + 算法 + 组件 + 辅助工具。
- **adapter** 抽象层,**一部分在 docker 里一部分在 infra 里**,可拆成**配置**和**入口**两半。
- **infra** 只有 HPO 和实验调度器。

### 主流程

| 阶段 | 归属 |
|---|---|
| 组件参数表 | memorax |
| 注册表 | adapter · docker 侧 |
| 扫描域 | adapter · infra 侧 |
| 采样值 | infra |
| 运行配置 | adapter · infra 侧 |
| 闭包 | memorax(本期不定) |

即 `memorax → adapter → infra → adapter → memorax`,adapter 出现两次 = 配置半 + 入口半。

### 已确认的边界

1. **adapter 两半的缝 = 序列化的注册表(JSON)**。docker 侧只做发现 + 序列化;
   infra 侧拿这份 JSON 做 resolve、扫描域、组装运行配置。
2. **runtime 直接调度 train / eval 及其他组件;算法自己调度 step / update 并组装并提供
   train / eval。** 与原计划和 runtime 目前的功能都不冲突 —— 也就是现状不变。
3. **env 在 memorax 内调度,logger 在 memorax 外调度。** 上游是 logger 在内,所以新计划
   最初也想放在内。
4. **`environment` / `training` / `evaluation` / `logging` 不属于计算图的配置树**,
   跟着 runtime 走;具体放哪取决于 runtime 放哪。
5. **入口与 worker 合并**,作为适配器 + 调度器 + 运行入口;**只有 worker 需要掌握 AWS**。
6. **图 resolve 必须做**,理论上可由 memorax、runtime 的初始化或独立初始化阶段做;
   入口本身可以仅作为总体初始化与调度。
7. 一次只处理一个 spec。本期目标:打通**组件参数表 → 注册表 → 扫描域 → 采样值 → 运行配置**,
   产物是 HPO 生成的完整配置文件。adapter 等问题后续再处理。
8. 实验模板 = `experiments/streamac template.yaml`;算法模板 = `stream_ac_factory`;
   入口应该还没改。`build` 等契约再议。
9. 命名大小写不统一(spec 写 `RTU`,模板写 `rtu`)**现在允许**。
10. **目录即声明**(2026-08-05 定)。一个文件在 `algorithm/` 下,本身就是"它应该能跑"
    的声明;import 不了就是构建失败,**不做额外处理** —— 不跳过、不记录、不加豁免名单。
    因此 `entries/rtrrl.py` 现在的 `ImportError` 是预期状态,不是待修的 bug:它按三槽
    `Network`/`FeatureExtractor` 写的,而那个形状在 `8355e6e` 就删了。catalog 建不出来
    是这条规则在起作用。迁 rtrrl 时一起解决。

---

## 2. 还没定的

### A. runtime 放 memorax 还是 adapter

讨论到最后仍未定。关键事实见第 3 节。**注意 logger 调度跟着 runtime 走**,不能拆开。

### B. logger 的分发方式(取决于 A)

- **方案 A**:memorax 的 runtime 同时挂三个 sink(写文件 + aim + rerun),
  worker 直接从本地文件生成指标。
- **方案 B**:全部只写本地,由 worker 统一二次分发(推 aim / rerun / S3)。

我当时倾向 B,理由是"aim 会污染 memorax 依赖"——**但这个理由后来被推翻**(见 3.5),
因为那是 aim 特有的问题,不是 logger 内置的问题。**B 剩下的理由**:失败隔离
(aim 端点挂掉不会弄挂一次付费运行)、memorax 对外产物是文件(最好测)。

commit：aim和rerun不影响写本地副本，而且算hui'bahpo的matrices

### ~~C. 是否退回 lox~~ —— **已定:不用。内部诊断量也不用**(2026-08-04)

验证过程和证据见 §6,保留下来是为了不再重问一遍。结论:技术上完全可用、
且不改数,但 `train` 已经把 `StepMetrics` 堆叠返回、与 spool 出来的东西逐位
相同,增量只在"通不到返回值的量";而那类量在组件层还撞 §6.3 的语境歧义。
**指标机制维持现状:`StepMetrics` 契约不动,不引入 `lox.log`。**

连带确定:§5 第 10 条(memorax 内不得 logging)**继续成立**,不作废。

不动 `pyproject.toml` 里的 lox 依赖 —— 仓库里 15 个继承来的算法文件
(`ppo.py`/`sac.py`/`dqn.py` 等)还在 import 它,那是另一笔账。

---

## 3. 查证到的事实

### 3.1 spec 与实现的差异

- **spec 没有"组节点"这个概念,契约和模板都有。**
  `training_sdk/contract.py:169` 是三选一:
  `ModuleNode = ModuleScalarSpec | ModuleStructureSpec | ModuleGroupSpec`;
  memorax 中立侧是 `type ParameterTree = dict[str, ParameterLeaf | ParameterTree]`
  —— **嵌套 dict 就是组**。模板里 `actor:` / `normalization:` 没有 `kind`,是纯命名空间。
  组是"一个模块在自己参数树内部做的命名嵌套,背后没有模块",**替代了旧设计里
  `actor_optimizer_bound` 这种前缀**。spec 的模型只有"结构参数 → 一个 kind → 递归",
  遇到没有 `kind` 的节点该怎么遍历、`valid` 怎么查、错误路径怎么拼,都没写。
- **命名规则矛盾**:spec 写「Names are never normalised or inferred」,例子是 `RTU`;
  模板写 `kind: [rtu]` / `[global_std]` / `[sgd]` / `[tbptt]`。(已允许暂不统一)
- `memo/runner/module_catalog.py` **import `training_sdk.contract`**,与"翻译由 adapter 做"
  不一致。
- **两套声明系统并存**:
  - `memorax/factory.py` — `ComponentFactory` Protocol,**实例方法** `param(self)`,
    参数类**带 `placeholder`**,有 `build_tree`,`runner/entry.py` 在用。
  - `memorax/modules.py` — `RegisteredModule` Protocol,**classmethod** `param(cls)`,
    参数类**无 `placeholder`**,多一个 `IntParameter`。
  两边都定义 `ChoiceParameter` / `FloatParameter` / `StructureParameter` /
  `ParameterTree` / `Scalar`,字段不一样。
- **两套 catalog 并存**:`runner/catalog.py:52` 仍按入口的扁平 `PARAMETERS` / `METRICS`
  建 `EntryDescriptor`;`runner/module_catalog.py` 建 `ModuleDescriptor`。
- `memorax/discovery.py` 在 memorax 里,按模型该在 adapter。
- `trainer_infra/space.py` 的 `resolve_parameters` / `sample_parameters` 在 infra 里,
  按模型该搬到 adapter。
- **全仓 `grep '^MODULES'` 零命中** —— 发现机制有了,被发现的东西还没有。
- **memorax 仍有 4 处 import SDK**(都是我写的旧 `param()`):
  `networks/backbones.py`、`networks/initialization.py`、`rl/normalization.py`、
  `rl/updates.py`。

### 3.2 training_sdk 现状:不满足新计划

| 文件 | 行数 | 处境 |
|---|---|---|
| `parameters.py` | 270 | **全删**(旧的 param/structure/placeholder/read_branch/expand) |
| `episode.py` + `rollout.py` + `reporter.py` + `sinks/*` | ≈500 | 跟 runtime 走 |
| `contract.py` | 323 | **一半是旧的**:`ParameterSpec`/`StructureSpec`/`EntryDescriptor`/`FloatSpec`/`IntSpec`/`ChoiceSpec`/`*ValidSpec`;新的只有 `ModuleScalarSpec`/`ModuleStructureSpec`/`ModuleGroupSpec`/`ModuleDescriptor` 四个,**而且新的复用了旧的标量类型**(`ModuleScalarSpec.valid: ValidSpec`、`search: SpaceEntry`) |
| `worker.py` `score.py` `objects.py` `testing.py` | — | infra 侧 |

TODO：直接把功能分散到对应层，不再保留sdk

### 3.3 分数路径:已经是分开的

```
runtime → metrics.jsonl(每行 {"step": …, "metrics": {name: float}})
          ↓                                    ← 缝在这里
worker  → compute_score(jsonl, ScoreConfig) → float → S3
infra   → _read_score → tell_value(study, trial, value)
```

`compute_score`(`training_sdk/score.py`,46 行)是纯函数,只依赖 `ScoreConfig`
(metric / window_steps / reduce / non_finite / direction)。**runtime 从来不知道 HPO 存在。**

→ **"指标怎么写入 HPO"不受 runtime 位置影响。** 唯一跟着 runtime 走的是:
指标序列的产生、和**指标名字的定义权**(`<phase>/<window>/<quantity>` 那套现在在
`episode.py`)。

### 3.4 sink 的真实依赖(很薄)

```
sinks/aim.py      from aim import Run          + RunConfig
sinks/rerun.py    rerun as rr, numpy           + objects, RunConfig, Episode
sinks/metrics.py  json, pathlib                (只有这两个标准库)
```

需要 infra 知识的是 **`Reporter.from_env()`**(读环境变量 / manifest)、
**`build_default_sinks`**、和 **worker/score** 那条 S3 路径 —— **不是 sink 本身**。

### 3.5 上游 memorax(noahfarr/memorax)

- **依赖**:`wandb>=0.20.0`、`tensorboardx>=2.6.1`、`hydra-core`、`lox` 都是**基础依赖**;
  二十多个 optional extras **全是环境**(brax / jumanji / navix / craftax / …)。
- **上游没有 aim**。aim 是我们加的,`aimrocks` 无 win_amd64 wheel,
  **这就是本仓库测试只能在 WSL 跑、仓内 `.venv` 是死路的原因**。
- `loggers/__init__.py` 把六个 logger **全部 eager import**(checkpoint / dashboard /
  file / logger / tensorboard / wandb),`wandb.py` 顶上就是 `import wandb`
  → 所以 wandb 必须是基础依赖。**这是打包问题,不是设计问题。**
- 接口本身可插拔:
  ```python
  @runtime_checkable
  class Logger(Protocol):
      def log(self, data: PyTree, step: int, **kwargs) -> None: ...
      def finish(self) -> None: ...
  class MultiLogger:  # fan-out + atexit.register(self.finish)
  ```
  `WandbLogger.__init__` 默认 `mode="disabled"`。
- **环境也不懒加载**:`environments/environment.py` 在建 `register` 表之前把 17 个环境
  模块全部 eager import。所以 optional extras 名不副实 —— 只装一部分,
  `import memorax.environments` 就会 ImportError。
- `lox.log` **28 处**:algorithms 9 个文件(stream_ac / sac / ppo / dqn / r2d2 / pqn /
  qrc / mappo / gradient_ppo)+ environments/wrappers 8 个 + `networks/sequence_models/rtu.py`
  (在 `nn.compact __call__` 里)。
- 上游 **没有任何地方把 RTUCell 组进网络**。两个 StreamAC example:gymnasium 用
  `GRUCell` + `Stack(Projection/Residual)`;minatar 用 `Conv+LayerNorm+leaky_relu`、
  无 torso、`sparse(sparsity=0.9)`。

### 3.6 lox 是什么 —— 我们已经在用

「Logging library for JAX that is compatible with transformations and primitives
such as vmap and scan」。改写 jaxpr,两种用法:

```python
lox.tap(f, callback=cb)(xs)    # 实时回调,边跑边流
y, logs = lox.spool(f)(xs)     # 收集 scan/vmap 内所有 log,随输出返回
```

**我们仓库里 15 个文件仍在用 lox**:
`algorithms/{dqn,gradient_ppo,mappo,ppo,pqn,r2d2,sac}.py` +
`environments/wrappers/{clip_action,flickering_observation,noisy_observation,
normalize_observation,normalize_reward,periodic_observation,scale_reward,sticky_action}.py`。

只有 `stream_ac` / `rtrrl` / `upstream_stream_ac` + `runner/loop.py` + `training_sdk`
改用了容器。

**关键结论**:`StepMetrics` 容器 + `record` 门控 + `TRAINING_METRICS` 逐字段声明,
**约等于手工重做 `lox.spool`**,而 lox 已经在依赖里。

---

## 4. 我在讨论中做错的判断(逐条)

1. **把"catalog schema 在 `training_sdk.contract`"当成问题已解** —— 实际那是**违反
   约束的实现**(memo 侧不该出现这个 import),该改的是实现不是 spec。
2. **说 sink 需要 infra 知识** —— 错。sink 很薄;需要 infra 知识的是
   `Reporter.from_env` / `build_default_sinks` / worker。
3. **说"runtime 放 memorax 会绑上装不了的依赖"** —— 那是 **aim 特有的问题**,
   不是 logger 内置的问题。上游内置 logger 没这毛病(wandb/tensorboardx 有 Windows wheel)。
4. **提了"第三种切法"**(`complete_episodes` 归 memorax、`drive`/sinks 归 adapter)
   —— 你指出 logger 调度跟着 runtime 走,不成立,已撤回。
5. **担心"env 要搬出算法"、以为推翻了 07-30 的论证** —— 你澄清 runtime 只调 train/eval、
   算法自己调 step/update,与现状不冲突。已撤回。
6. **最实质的一条:phase 4 建 `StepMetrics` / `episode.py` / `rollout.py` 那一套时,
   我没有查过 lox 是干什么的**,直接把上游"在 traced 区域内记日志"当缺陷处理并剥掉。
   实际 lox 正是为此设计的。这个错判影响的不只是措辞,是那一整套容器机制存在的理由。

---

## 5. spec 要补 / 要改的清单

按这条路(到"运行配置"为止)排:

1. **组节点进模型** —— 遍历规则、`valid` 检查、错误路径怎么处理没有 `kind` 的节点。
2. **覆盖语法** —— 现在只有"钉成一个值"(`hidden_dim: [32]`)的例子,
   **缩小成一个区间怎么写没有**,而那是"覆盖"这一步的主要用法。
3. **运行配置的形状** —— 本期目标产物,spec 里只有一句
   "reassemble the canonical nested runtime configuration",没有例子。
   组成应为:已解析的计算图(结构已定 + 标量已抽样 + canonical name)
   + 运行期段落(environment / training / evaluation / logging / score,原样透传)。
   **待确认**:运行期段落是否不经任何声明/校验机制原样搬入
   (现在 `TrainingConfig` 的整除性校验在 `training_sdk.contract` 里)。
4. **两个 resolve 要命名区分**:
   - **域 resolve**(adapter·infra 侧):序列化注册表 + 实验图 → 扫描域
   - **图 resolve**(adapter·docker 侧):固定运行配置 + 注册模块 → 构建好的图
5. **声明取代关系** —— 07-30 已删,spec 没说取代了什么、哪些结论仍有效
   (「结构不参与搜索」仍有效;「未激活分支取 placeholder」已作废)。
   不太重要，反正是推翻重写
6. **`param()` 名字冲突** —— 现有 `training_sdk.parameters.param()` 是"声明一个字段",
   spec 里是"返回整个模块的参数树"。迁移期会并存。
7. **descriptor 缺 metrics** —— `EntryDescriptor` 有 `metrics`(catalog 导出、
   preflight 校验 `score.metric` 用),`ModuleDescriptor` 只有
   directory/file/name/parameters。
8. **标题与文件名不一致** —— 文件 `streamac-component-factory-design`,
   H1 "Memorax Module Catalog and Resolver Contract"。
9. **memorax 与 SDK 的边界** —— 实现现在违反(4 处 `param` import +
   `module_catalog.py` import contract)。
10. **memorax 内不得 logging**(若最终 runtime 在外)—— 上游 28 处会一路带回来,
    需要写成硬约束。(曾经有个"若决定用 lox 则作废"的口子,已经关掉:见 §2C,
    不用 lox。)

---

## 6. lox 兼容性:已验证,全部可用 —— 但决定不用(见 §2C)

以下是证据,不是提案。留着是为了这个问题不用再问第二遍。

在 `/tmp/streaming-rtrrl-memo-venv`(WSL)里跑了两组探针,一组是玩具函数,
一组是真实的 `StreamAC.train`(TinyDiscreteEnv,`num_envs=3`,`num_steps=24`,
把 `_step` 和模块级 `_per_stream` 包一层加 `lox.log`)。

### 6.1 结论:能用,而且不改数

| 问题 | 结果 |
|---|---|
| `lox.log` 穿过 `jax.grad` | 可以。JVP 规则把 `lox_p` 留在 primal 侧,transpose 规则(会抛 `ValueError`)不会被触发 |
| `lox.log` 穿过 `vmap(grad(one))` | 可以。真实 `_per_stream` 里 log,出来 `(48, 1)` = 8 步 × 3 流 × 2 角色 |
| `lox.spool` 穿过 `lax.scan` | 可以。scan 内的 log 按 `(-1,)+shape[2:]` 展平,出来就是 `(步, 环境)` |
| 嵌套 scan | 可以,内层展平成 `外×内` |
| `jax.jit(spool(f))` 与 `spool(jax.jit(f))` | 都可以,形状一致 |
| spool 后数值是否变 | **不变**。整个 `trained` 状态 68 个 leaf,最大偏差 `0.0` |
| 留着 `lox.log` 但不 spool | 照跑,偏差 `0.0`(所以 log 语句可以常驻代码) |
| 按 episode 切 | 可以,而且是普通的 numpy 后处理:一起 log `done`,再按流分段 |
| `argnames=` / `tags=` 过滤 | 都生效 |
| 传 traced 的 step 下标(`lox.log({...}, step=t)`) | 可以,`logs.step["reward"]` 拿得到 |

复现脚本:`scratchpad/lox_probe.py`(11 个玩具用例)、`lox_real_probe.py`
(真实 train 的 8 个用例)、`lox_inside_probe.py`(grad 内部 log)。

### 6.2 但这**不是**支持退回 lox 的论据

决定性的一条:`train` **已经**把 `StepMetrics` 沿 scan 堆叠返回,形状与 spool
出来的 logdict 完全一样,数值逐位相同(探针 3:`logs["td_error"]` vs
`stacked.update.td_error`,偏差 `0.0`)。也就是说在**已经接到返回值里**的量上,
lox 什么都不多给。

lox 真正多给的只有一种东西:**根本没有路径通到返回值的量** —— 例如
`_per_stream` 里被微分的那个 objective(探针验证过可以拿到)。要不要为这类量
引入 lox,是一个比"`StepMetrics` 该不该退给 lox"小得多的问题。

### 6.3 同名 key:算法层可以闭包解决,组件层不行

**算法层(actor / critic)—— 闭包解决,已验证。** key 是 trace 时的普通
Python 字符串,把角色传进去即可。真实 `_per_stream` 按角色闭包后拿到
`actor/objective (24,1)` 和 `critic/objective (24,1)` 两个干净的 key,
数值偏差 `0.0`(`scratchpad/lox_collision_probe.py` 用例 A)。

**组件层 —— 名字能给回去,但只解一半。**

第一条(组件叫不出自己的名字)**是实现问题,已修**:`Sequence.__call__`
走 flax 组合,组件被绑在自己名字下,`self.name == "components_0"`;
`Sequence.walk` 是 `component.apply({"params": tree[name]}, ...)`,把组件
重新挂成 root,`self.name` 就成了 `None`。名字本来就是 sequence 的、而且
`walk` 已经拿在手里(`zip(self.names, self.components)`),所以改成
`component.clone(name=name).apply(...)` 即可。

- 测试 `test_a_component_knows_its_own_name_on_both_ways_through`
  (`tests/test_sequence.py`):先红(`[None, None]` vs
  `['components_0','components_2']`),改后绿。
- **数值完全不动**:同一个 agent 跑 train+evaluate,改前改后
  **164/164 个 leaf 逐位相同**(`scratchpad/ab_compare.sh`)。
- `tests/test_sequence.py` 全绿,套件其余部分红的还是原来那 7 个。

第二条**不是实现问题,没解**:同一个组件实例本来就从三种调用语境 log。
名字修好之后重测,两个实例确实分开了,但每个仍然:

```
'components_0/activation'  DynamicJaxprTracer  x2   ← acting forward + bootstrap forward
'components_0/activation'  LinearizeTracer     x1   ← 梯度闭包内,被 vmap 过
```

两个 dynamic 拼成 `(2,)`,linearize 那个是 `(3,1)`,于是 spool 仍然在
`logdict.__add__` → `jnp.concatenate` 处**直接报错**:
`Cannot concatenate arrays with different numbers of dimensions: got (2,), (3, 1)`。
报错在 lox 内部,消息里不含任何 key 名。

**结论**:名字区分的是"哪个组件",区分不了"哪一趟 forward"。后者只有调用方
(StreamAC:acting / bootstrap / gradient)知道,要用就得由调用方穿下去 ——
这正是 lox 本来要省掉的那种穿线。所以组件层用 lox 的门槛,比算法层高一档。

### 6.4 另外两个坑

1. **同名且形状一致时是静默拼接,不报错。** 玩具用例 9:一个 step body 里
   log 两次 `v`,出来是交错的 `[0,0,1,2,2,4,3,6]`。形状不一致才会像 6.3 那样
   炸;形状一致就悄悄合并。
2. **`vmap` 下多一个尾部 `1` 轴。** `lox.log` 先 `expand_dims(x,0)`,
   `lox_batch` 又原样返回 `batch_axes`,所以 vmap 内的 log 是 `(N, 1)` 而不是
   `(N,)`。
3. `lax.while_loop` 内的 log 会被丢掉,只打印一行 warning(不报错)。我们不在
   环境内 log,暂时不影响,但换成带 while 的环境时这是静默失败。

### 6.5 探针跑的是 worktree 那份代码

`/tmp/streaming-rtrrl-memo-venv` 里 `memorax` 的 editable 安装指向
`.worktrees/streamac-factory-boundary/memo/`,不是主 checkout。对 lox 的结论
无影响(两边 scan/vmap/grad 结构一致),但记下来。

---

## 7. 主线上还欠的两件(与本 spec 无关)

- **金快照重录** —— `test_stream_ac_golden.py` 5 条红。既是名字上的红
  (快照记的是 `actor_params/params/feature_extractor/...` 这些已不存在的叶子路径),
  也是数值上的红。重录前 stream_ac 没有对外的数值回归网。
- **rtrrl / upstream_stream_ac 迁移** —— 现在连 import 都不行,
  四个测试文件因此排除在外:`test_entries`、`test_experiments`、
  `test_hopper_reproduction`、`test_upstream_stream_ac`。

memo 当前红的七条(与阶段 3 开工前同一批):
`test_loop::test_the_catalog_is_the_entries_directory_and_nothing_written_down`、
`test_paper_parity::test_our_normalisation_is_not_the_published_normalisation`、
`test_stream_ac_golden` 5 条。
