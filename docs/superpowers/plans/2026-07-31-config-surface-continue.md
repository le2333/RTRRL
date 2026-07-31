# 配置面重构 — 开发机接续说明

**分支:** `feature/rtrrl-lru-paper-parity`  
**HEAD:** `48635a4`（已推远程）  
**规格:** `docs/superpowers/specs/2026-07-30-configuration-surface-design.md`  
**仓库:** `streaming-rtrrl/`（相对 `/home/ubuntu/trainer`）

本文件只描述**配置面**这条线。同分支上还有**数值测试台**（`2026-07-29-numerical-testbench-*`），账本在 `.superpowers/sdd/progress.md`；两套计划曾共用 `task-N-report.md` 文件名，本计划报告另存为 `config-surface-task-*-report.md`。

---

## 一句话进度

规格四个阶段里：**阶段 1 完成；阶段 2 的「删 source_hash」完成；阶段 2 的 `param()` / 结构树未开工；阶段 3、4 未开工。**  
中断点：准备写阶段 2 剩余两份计划前，有两个范围问题尚未拍板。

---

## 已完成

### 阶段 1：环境 / 预算出 space + 观测删维

- `CONTRACT_VERSION` 曾升到 **3**（现已因 source_hash 再升到 **4**）。
- 实验 YAML 顶层强制 `environment` + `budget`；`space` 禁止 8 个保留名：`environment`, `env_mode`, `env_backend`, `observed`, `num_envs`, `total_steps`, `epoch_steps`, `eval_steps`。
- 观测：`SelectObservationWrapper` 按下标**删维**（非置零）；Hopper `P` → `observed: [0,1,2,3,4]`。
- memo 三入口 + `rtrrl_aaai` 从 `RunConfig` 读环境/预算。
- `experiments/` 全部迁移（含 AAAI 探针用的 `rtrrl-hopper-aaai.yaml`）。
- 计划：`docs/superpowers/plans/2026-07-30-environment-and-budget-out-of-space.md`。

### 阶段 2 之一：删除 `source_hash`

- 计划：`docs/superpowers/plans/2026-07-31-remove-source-hash.md`（已执行）。
- `EntryDescriptor` / `RunConfig` 无该字段；三处 catalog 构建器不再计算；preflight / launch / loop / Aim 不再携带或比对。
- **`CONTRACT_VERSION = 4`**。
- **保留** `images.py` 对 ECR config blob 的 `hashlib.sha256`（内容寻址，不是 source_hash）。
- 报告：`.superpowers/sdd/source-hash-task-1-report.md`（本地 gitignore，不同步）。

### 阶段 1 终审顺带修掉的活问题

1. 根目录 `/archive/` 误提交产物 → 取消跟踪 + `.gitignore` 锚定 `/archive/`；控制面下 `rtrrl/infra/control-plane/archive/` **不动**。
2. `memo/docs/gpu-abort-probe.py` 改到 `build(params, environment)` 新签名。
3. **`docs/trainerctl-manual.md`** 按新契约重写（旧示例仍教人把 `total_steps` 放进 space）。
4. **活缺陷已修：** `mock-trainer` 的 `brax_ppo_acceptance` 阶段 1 漏迁——仍从 `params["total_steps"]` 读预算，acceptance 会按随机步数训练。现已改读 `config.environment` / `config.budget`；示例改为 `brax::inverted_pendulum`（与归档一致）；守卫测试把本地别名 `env`/`backend` 也算进保留集。报告：`.superpowers/sdd/mock-trainer-migration-report.md`。

### 规格用语（已改）

不要再用「一个 default」：

| 字段 | 含义 |
|------|------|
| `search` | 搜索范围的默认（今天 `SPACE` 里的区间/列表） |
| `placeholder` | **不进搜索**时的单值占位（未激活分支塌缩、或未声明 search） |
| `valid` | 硬边界，仅校验 |

`placeholder` 不考据「推荐值」：有出处照抄，没有则取落在 `valid` 内且不引爆下游的安全值。

---

## 未完成（按依赖顺序）

### 阶段 2 之二：`param()` 三件套 + valid 校验 — **未写计划**

目标（规格 §3）：算法侧 dataclass 声明 `valid` / `search` / `placeholder`；catalog 从声明导出；实验越界 preflight 拒绝；manifest 全参数；禁止 `params.get("x", v)`。

### 阶段 2 之三：结构树 + 条件采样 — **未写计划**

目标（规格 §4）：结构选择 + 分支子参数；未激活分支参数塌缩为 `placeholder`；`ask_round` 改为 `study.ask()` 后按树 `suggest_*`；含未钉死结构时拒绝 grid。

依赖调查（已完成；调查里关于 `source_hash` 的段落作废，该字段已删）：

**声明与导出现状**

- 四入口 `SPACE: dict[str, Any]`，三种形态：`list` → `ChoiceSpec`；`{type:float/...}` → `FloatSpec`/`IntSpec`。校验只在 `EntryDescriptor.model_validate`。
- Catalog：`discover()` → validate → `model_dump`；签入的只有 `rtrrl/catalog.json` 与 mock-trainer 的；memo 的 catalog 不签入。
- `_UNIT` / `_RATE` 是**同一 dict 对象多处引用**（不是拷贝），四入口各自复制了一份。

**采样现状（无条件）**

- 链路：YAML → `Experiment.space` → `resolve_space` → `distributions` → `ask_round` → `trial.params`。
- `ask_round`：`study.ask(dict(distributions))`，每个 trial **拿到 space 全部键**；未用 `trial.suggest_*`。
- `resolve_space`：**只查键名**，override 可无阻碍越过 catalog 区间（如 `gamma` 盖成 `[2,3]`）。今日「数学有效」与「搜索区间」是同一件事。
- `check_sampler` 只拦非法采样器名，以及 grid + 非 categorical。

**条件参数（结构树要挂的）**

| 入口 | 条件 | 参数 |
|------|------|------|
| rtrrl | `update_rule==adam` | `b1`, `b2`, `rnn_grad_clip`, `freeze_gamma` |
| rtrrl | `update_rule==obgd` | `kappa`, `obgd_beta2`, `obgd_rule` |
| rtrrl | `obgd` 且 `obgd_rule==obgd` | `obgd_beta2`/`eps` 二次失效（无二阶矩） |
| rtrrl | `freeze_gamma`+obgd | **容器内硬报错**（`rtrrl.py:200-205`），会烧钱 |
| stream_ac | `bounded_rule!=obgd` | `beta2`, `eps` |
| stream_ac | 任一归一化开 | `normalization_statistics` |
| upstream | `bounded_rule!=obgd` | `beta2`；`eps` 另兼 reward 归一化 |
| aaai | LRU 拒 `rflo` | `gradient_mode` 现靠砍掉候选表达 |

无分支参数的结构（归一化开关、梯度门控）规格说与普通离散参数一致，不特判。

**结构树解决不了 / 需另议**

- `rtrrl_aaai`：`total_steps % (scan_steps × num_envs) == 0`（跨节算术，采样可抽出必崩配置）。
- `upstream_stream_ac`：`eps` 一名两义（自适应规则 eps 兼 `NormalizeRewardWrapper` eps）。

### 待用户拍板（上次问卷被中断）

1. **跨参数约束**（如 scan_steps 整除）：本轮不做 / 一起做通用机制 / 只把这一条提到 preflight。
2. **eps 一名两义：** 拆成两参数 / 保持共用挂外层 / 先查上游原实现再定。

### 阶段 3：OBGD → `bound` × `base` 两轴 — 未开工

规格 §5。依赖结构树。接口：`optax.GradientTransformationExtraArgs`。不要求与现状逐位一致。

### 阶段 4：指标 — 未开工

规格 §6。Aim `context={"scope":"step"|"episode"}`；补 `train/reward`、`train/episode_return`。按环境拆奖励分量是后话。与前三阶段无代码依赖（原与删 source_hash 同改 aim.py，现 source_hash 已删，可独立做）。

---

## CI / 约束（开发机也建议遵守）

- **本云主机禁止本地 pytest/docker**（内存会杀会话）；开发机若内存充足可本地测，但仍以 CI 为准。
- 工作流：`Memo CI`（push 自动）、`Tests`（`gh workflow run tests.yml --ref <branch>`）、AAAI 镜像（`build-aaai-image.yml`）。
- Memo CI 允许**仅** 5 个既有 `test_stream_ac_golden.py` 失败（主分支/账本已批准放行）；其它失败算回归。
- 契约 `extra="forbid"`：删字段或改 catalog 形状往往要跨项目一次绿。
- 改入口 API 后检查工作流探针（曾漏改 `parameters(chosen)`）。
- 不要 `git add -A`（会把 archive / 运行产物卷进去）。
- 代码与配置里**不要写决策理由注释**（用户明确要求）。

---

## 开发机开工清单

```bash
cd <repo>/streaming-rtrrl
git fetch origin
git checkout feature/rtrrl-lru-paper-parity
git pull
# 读规格全文
# 读 docs/superpowers/plans/2026-07-31-remove-source-hash.md（阶段2-一已完成，仅作模板）
# 先问清上面两个拍板问题，再写阶段2-二 / 2-三计划
```

建议执行方式：子代理驱动 + 红绿 CI（与阶段 1 相同）。计划落盘：`docs/superpowers/plans/YYYY-MM-DD-*.md`。

---

## 关键路径速查

| 用途 | 路径 |
|------|------|
| 规格 | `docs/superpowers/specs/2026-07-30-configuration-surface-design.md` |
| 契约 | `training-sdk/src/training_sdk/contract.py`（`CONTRACT_VERSION=4`） |
| 实验模型 / RESERVED | `rtrrl/infra/control-plane/src/trainer_infra/experiment.py` |
| 采样 | `.../space.py`, `.../study.py`, `.../loop.py` |
| memo SPACE 现状 | `memo/entries/rtrrl.py`（及 stream_ac / upstream） |
| AAAI 入口 | `rtrrl/entries/rtrrl_aaai.py` |
| mock acceptance | `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py` |
| 手册 | `docs/trainerctl-manual.md` |
| 数值测试台（并行线） | `docs/superpowers/specs/2026-07-29-numerical-testbench-design.md` |

---

## 同分支上的另一条线（勿混）

**数值测试台**与配置面共用本分支，但任务编号/报告文件名曾冲突。`.superpowers/sdd/progress.md` 是测试台账本；`task-1/5/6/7-report.md` 在 HEAD 上仍是测试台标题，本计划副本为 `config-surface-task-*-report.md`。改测试台时不要覆盖配置面报告，反之亦然。
