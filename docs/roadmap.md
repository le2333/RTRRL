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

### R1 的动作

1. 定 run config 的字段（起点是现有 `RunConfig` 减掉没人读的：`training.chunk_steps`、`training.early_stop_patience`、`evaluation.num_envs` 是空声明）
2. infra 的实验 YAML schema 改成这个形状，`_configurations` 产出它
3. **补上现在缺的协调字段**：infra 有 `objective`（优化什么）但**没说结果写到哪**。这不是偏好是协调，提供方必须说
4. `RunConfig` 从跨侧契约降级成 worker 的内部类型
5. `test_template.py` 改成真往返
6. 更新 `docs/contract.md`

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

`rewrite/temp` 的 `algorithms/StreamAC/parameters.py`：

```python
@dataclass(frozen=True)
class Parameter:
    valid: Range
    search: Range

ParameterNode: TypeAlias = Parameter | Mapping[str, "ParameterNode"]
```

只有 valid + search，**没有 placeholder**，没有 `StructureSpec`；节点要么是参数要么是一组节点，嵌套即分组。但注意：那边的 StreamAC 参数树是六个平铺标量，**没有可选分支**，所以它不是"structure 不需要"的证据，只是没遇到。

另有本分支归档时丢失的两个未提交模块（只剩 `infra/src/trainer_infra/__pycache__/` 里的字节码）：`parameters.py` 声明 infra **自己的** `Choice`/`FloatRange`/`IntRange`，`parameter_adapter.py` 的 `resolve_parameters` 对着它们解析。那份工作已经在做"两侧各自声明、不共享包"。

### R2 落地后 R1 的 schema 会消失

外围配置也建模成组件之后，条件性要求（`enable_rerun` 为真才要 `rerun_s3`）就是**沿着选中的分支走一遍树**——因为结构永远被钉死，不需要条件采样，只需要条件要求。infra 的校验从"对着一份 schema"变成"走一遍树"，R1 那份 schema 自己消失。

**所以 R1 不是 R2 的临时铺垫，两步都在删东西。**

### R2 的动作

1. `search` 保持必填，`placeholder` 弃用（字段默认值就是默认值）
2. `describe_parameters` 改成走嵌套树；`ParameterSpec`/`StructureSpec` 合成 `Parameter(valid, search)`
3. 一个组件（建议 `Rtu`——它最复杂：有 carry、有 sensitivity、有 `local_jacobian`）先合并成声明 + `build()`，`backbone()` 工厂留作兼容层
4. 其余组件跟进，`read_branch` 保留（它是消费侧查表，与参数表无关）
5. 外围配置并入组件树，R1 的 schema 删除

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
