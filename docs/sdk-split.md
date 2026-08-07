# 拆掉 training-sdk

## 为什么

`training_sdk` 里装着三种归属不同的东西，而 Python 的依赖粒度是包，所以控制面为了两个 import 装了一整套 worker 的依赖。

infra 侧对 SDK 的全部依赖：

| import | 用途 |
| --- | --- |
| `training_sdk.contract` | Catalog / RunConfig / SpaceEntry / CONTRACT_VERSION，纯类型定义 |
| `training_sdk.objects` | S3 读写，三十行 boto3 包装 |

而 `training-sdk` 的依赖里有 `aim==3.28.*` 和 `rerun-sdk`。控制面那台 911 MiB 的机器为了读几个 pydantic 模型和调 S3，装下了 aim + aimrocks + rerun-sdk + numpy。aimrocks 没有 Windows wheel，这也是本地 `uv sync` 直接失败的原因。

更要紧的是：控制面**托管** aim server，worker **写入**它。今天把 aim 钉进控制面的依赖树，等于把服务端版本和 CLI 的安装绑死了——升级 aim server 要求重装 trainerctl，反过来也一样。

## 目标形状

| | ctrler | worker |
| --- | --- | --- |
| 职责 | HPO、task 分发、托管 aim server、aim/rerun 可视化 | 跑若干指定配置、回报结果 |
| Python 依赖 | optuna, boto3, pyyaml | 训练框架, aim(客户端), rerun-sdk, numpy, boto3 |
| 同机进程 | aim server, rerun viewer，各自安装 | — |
| 两侧之间 | **只有 S3 上的 JSON**，没有 API | |

共享包的判据只有一条：**共享的理由必须是"外部服务的 API 在这里"**。一个不 import boto3 的模块不许进 AWS 操作包。契约是**数据格式**，不是共享包——两侧各自校验，`CONTRACT_VERSION` 负责拒绝不匹配。

## 模块归属

| 模块 | memo/worker 用 | infra 用 | 依赖 | 去向 |
| --- | --- | --- | --- | --- |
| `contract.py` | 6 处 | 7 处 | pydantic | 降级成格式文档（T5） |
| `objects.py` | worker.py, sinks/rerun | 4 处 | **boto3** | AWS 操作包（T4） |
| `parameters.py` | 11 处 | — | — | `memorax/`（T1，见决策 D2） |
| `episode.py` | 6 处 | — | numpy | worker 侧（T1） |
| `rollout.py` | 2 处 | — | — | worker 侧（T1） |
| `score.py` | — | — | — | worker 侧（T1） |
| `reporter.py` | 3 处 | — | — | worker 侧（T2） |
| `sinks/*.py` | 经 reporter | — | **aim, rerun-sdk** | worker 侧（T2） |
| `worker.py` `__main__.py` | — | — | — | worker 侧（T3） |
| `testing.py` | 测试 | — | boto3, moto | 各自持有（T4） |

## 待定的决策

评估第一步时一并定。

**D1 — worker 包的名字和落点。** 建议 `memo/worker/`，与 `memorax`（库）、`entries`（入口）、`runner`（catalog）并列。执行 T1 时收窄了它的内容：只装 `reporter` `sinks` `score` `worker`，正好是"跑若干指定配置、回报结果"，不多不少。

**D2 — 哪些模块其实属于库。** 判据是**依赖方向**：`memorax` 用到的东西不能放进 `worker`，否则库依赖 worker。按这条：

- `parameters.py` → `memorax/parameters.py`。它是声明词汇，`memorax/rl/updates.py` 用 `param()` 声明 `ObBound.kappa` 的范围。
- `rollout.py` → `memorax/runtime/rollout.py`。`memorax/runtime/driver.py` 用 `complete_episodes` 把 chunk 切成 episode，这是 runtime 的活。
- `episode.py` → `memorax/runtime/episode.py`（T2）。`Episode` 是 runtime 与 reporter 之间的接口类型，由 runtime 产出，worker 消费——方向是 worker → memorax，对的。

**D3 — AWS 操作包的名字。** 候选 `cloud` / `aws_ops` / `objects`。它现在只有 S3，以后会收进 main 的 batch / ecr / cloudwatch。

## 步骤

每步做完回报并等评估。

**在 WSL 里验证**，不要只在 Windows 上跑：`aim` 的 `aimrocks` 没有 Windows wheel，`moto` 的 server 绑 `0.0.0.0` 而 Windows 连不上，`uv lock` 要构建 `mettagrid`（`cogames` 的传递依赖）而它 import `fcntl`。这三样都只在 Linux 上成立，而它们恰好盖住了 worker 侧最要紧的部分。

```bash
wsl -e bash -lc 'cd /mnt/c/.../memo && export PATH=$HOME/.local/bin:$PATH \
  && export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/memo-linux \
  && uv lock && uv sync --frozen --group development'

wsl -e bash -lc 'cd /mnt/c/.../memo && $HOME/.venvs/memo-linux/bin/python -m pytest tests \
  -p no:warnings --ignore=tests/test_entries.py --ignore=tests/test_experiments.py \
  --ignore=tests/test_module_contract.py \
  --deselect "tests/test_loop.py::test_the_catalog_is_the_entries_directory_and_nothing_written_down" \
  --deselect "tests/test_loop.py::test_the_names_an_entry_declares_are_names_a_sink_will_accept"'
```

排除的三处在 Linux 上同样坏，与本次迁移无关：`entries/rtrrl.py` 导入不存在的 `FeatureExtractor`（连带 test_entries / test_experiments / test_loop 两例）、`test_module_contract.py` 导入不存在的 `memorax.algorithm`。

基线：T1 前 **237**（Windows，排除项更多），T2 后 **337 passed, 12 skipped**（Linux）。

### T1 — 搬两个依赖叶子 ✅

原计划要一次搬四个，执行时收窄了：`episode.py` 还被留在原地的 `reporter` 和 `sinks/rerun` 用着，`score.py` 还被 `worker.py` 用着，先搬会造出一条 `training_sdk → memo` 的**反向依赖**。只有 `rollout.py` 和 `parameters.py` 在 training_sdk 内部无人引用，是干净的叶子。

**一个模块跟着它的消费者走，不跟着它的主题走**——这条在 T2/T3 继续适用。

- `parameters.py` → `memorax/parameters.py`；`rollout.py` → `memorax/runtime/rollout.py`
- 三个测试文件 `test_parameters` `test_rollout` `test_declaration_contract` → `memo/tests/`
- 13 处 import 改写，isort + black 重排
- `memo-ci.yml` 的 CHECKED 加上 `memorax/parameters.py`
- 两个模块仍 `from training_sdk.contract import ...`，那是 T5 的事

结果：memo **237 → 278 passed**（+41 是搬进来的测试），静态检查零新增失败。**新基线 278。**

### T2 — 建 `memo/worker/`，搬整个 worker 闭包 ✅

再次按依赖闭包重排：`worker.py` 引用 `reporter` 和 `score`，`reporter` 引用 `sinks`，这四个加 `__main__` 是一个连通块，一起搬才没有反向边。`episode.py` 反过来被 `reporter` 和 `sinks/rerun` 引用，先搬它会造出反向边，所以它留到 T3——**原计划的 T2/T3 分界正好画反了**。

- `reporter.py` `score.py` `worker.py` `__main__.py` `sinks/` → `memo/worker/`
- `memo/pyproject.toml`：packages include 加 `worker*`，依赖加 `aim` `rerun-sdk`，dev 组加 `moto[server]`
- `training-sdk/pyproject.toml`：去掉 `aim` `rerun-sdk`
- `Dockerfile.{cpu,gpu}` 的 `CMD` 从 `python -m training_sdk.worker` 改成 `python -m worker`
- 搬 `test_reporter` `test_score` `test_worker` `test_aim_sink` `test_rerun_sink` `test_metrics_sink`
- `memo/tests/conftest.py` 导入 `s3_base` / `s3_endpoint` 两个 fixture（`training_sdk.testing` 到 T4 才动）
- `memo-ci.yml` 的 CHECKED 加 `worker`

结果：Linux 上 **337 passed, 12 skipped**（跳过的是 `test_paper_parity` 里需要 torch 的部分，torch 在 `paper` 组，没装）。此后 `training_sdk` 只剩 `contract` `objects` `episode` `testing`，`training-sdk/uv.lock` 少了 365 行——tqdm / uvicorn / websockets 正是 aim 拖进来的。

### T3 — 搬 episode ✅

T2 之后 `episode.py` 的消费者全在 memo，且 training_sdk 内部无人再引用它。

- `episode.py` → `memorax/runtime/episode.py`（见 D2：`Episode` 是 runtime 产出、worker 消费的接口类型），12 处引用改写
- 搬 `test_episode`
- **numpy 的归属查出来是错的两次**：training-sdk 声明了 `numpy>=2` 而它**没有任何一个文件用**（`episode.py` 只 import 标准库）；memo 有九个文件直接 import numpy 却**从未声明**，一直靠 jax 的传递依赖。删掉前者，补上后者
- 此后 `training-sdk` 只剩 `contract.py` `objects.py` `testing.py`，依赖只有 boto3 / pydantic / pyyaml

结果：memo **337 → 355 passed, 12 skipped**；training-sdk 自己的 **28 passed**（它的测试这时也终于能跑了）。

### T4 — 抽出 AWS 操作包

- `objects.py` + `testing.py` → 新包（D3），依赖只有 boto3 / moto
- **infra 侧的 4 处 import 改不了：infra 源码不在本分支**（`main` 在 `rtrrl/infra/control-plane/`，`rewrite/torchrl` 在 `infra/`）。本分支只能改 worker 侧，infra 侧留给合并时处理
- 验证：`test_objects`

### T5 — contract 降级成格式

- 保留 `CONTRACT_VERSION` 与接收方校验（`worker.py` 已经在做）
- 写 `docs/contract.md`：catalog / config / manifest / score 四种 JSON 形状
- 补一条跨侧约束：**worker 的 aim 客户端与 ctrler 的 aim 服务端主版本必须一致**——今天靠"两边装同一个包"隐式保证，拆开后没人保证
- 同样受 infra 不在本分支的限制

## 本分支能做到哪

T1–T3 完整可做，T4 只能做 worker 半边，T5 只能写文档。T4/T5 的 infra 半边要等分支合并。
