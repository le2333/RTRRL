# 控制面与 worker 之间的契约

两侧不共享任何代码。它们之间只有 **S3 上的 JSON**，和这份文档。

| | ctrler | worker |
| --- | --- | --- |
| 职责 | HPO、task 分发、托管 aim server、aim/rerun 可视化 | 跑若干指定配置、回报结果 |
| Python 依赖 | optuna, pyyaml | 训练框架, aim(客户端), rerun-sdk, numpy, boto3, pydantic |
| 同机进程 | aim server、rerun viewer，各自安装 | — |

发送方负责构造，**接收方负责校验**。共享一个 pydantic 类只在两侧同时发布时才安全，而 worker 是镜像、ctrler 是控制面，本来就不同时发布。

## `CONTRACT_VERSION`

一个整数，当前 **7**，写在 `memo/worker/contract.py`。catalog 和 run configuration 各带一份。worker 收到不等于自己实现的版本就**拒绝执行**，不做兼容猜测。

改动这四种形状里的任何一种——加字段、改语义、换类型——都要同时改这个数，否则一个旧 worker 会把新配置读成它以为的样子。

## 四种形状

### catalog.json

镜像构建时由 `memo/runner/catalog.py` 扫描 `entries/` 生成，写进镜像标签。控制面从标签读它，**不导入任何 Python**——那台机器装不了 jax。

```json
{
  "contract": 6,
  "entries": {
    "stream_ac": {
      "command": ["python", "-m", "entries.stream_ac"],
      "metrics": ["train/...", "eval/..."],
      "parameters": { "<名字>": { "kind": "param"|"structure", ... } }
    }
  }
}
```

`parameters` 的节点形状由 `memo/memorax/parameters.py` 的 `ParameterSpec` / `StructureSpec` 定义。它绑定在镜像 digest 上：**改了文件不重建镜像，就不可能悄悄扩大一个实验的搜索空间**。

### manifest

控制面写，worker 从 `TRAINER_MANIFEST` 指向的位置读。

```json
{ "runs": ["s3://.../config-0.json", "s3://.../config-1.json"] }
```

worker **串行**执行其中每一个 run。

### run configuration

控制面写一份，worker 通过 `TRAINER_RUN_CONFIG` 读一份，由 `memo/worker/contract.py:RunConfig` 校验。

```json
{
  "contract": 7,
  "run_id": "...", "experiment": "...", "launch_id": "...",
  "trial": 0, "entry": "stream_ac", "digest": "sha256:...",
  "environment": { "id": "brax::hopper", "backend": "spring", "seed": 0,
                   "episode_length": 1000, "observed": [0, 2, 4] },
  "training":   { "num_envs": 16, "total_steps": 2000000, "epoch_steps": 10000 },
  "evaluation": { "steps": 1000 },
  "params":     { "<扁平名>": <标量> },
  "logging":    { "aim": "...", "enable_rerun": false,
                  "rerun_s3": null, "rerun_every_steps": null },
  "score":      { "metric": "...", "window_steps": [0, 0], "reduce": "mean",
                  "direction": "maximize", "non_finite": "worst", "s3": "s3://..." }
}
```

#### 三组字段，按读者分

| 组 | 字段 | 读者 |
| --- | --- | --- |
| **图** | `params`、`environment.{id,backend,observed,episode_length}`、`training.num_envs` | 入口的 `build` |
| **预算** | `environment.seed`、`training.{total_steps,epoch_steps}`、`evaluation.steps` | `memorax.runtime.Runtime` |
| **协调** | `contract` `run_id` `experiment` `launch_id` `trial` `entry` `digest`、`logging.*`、`score.*` | worker / reporter / sinks |

`training.num_envs` 是唯一一个从预算块穿进组装器的字段：每个 carry、trace、sensitivity 都在这个宽度上开，建图时钉死。

#### contract 6 → 7 删掉的四个字段

`training.chunk_steps`、`training.early_stop_patience`、`evaluation.num_envs`、顶层 `name`——四个都**没有任何实现读**（`name` 尤其误导：aim 的 run 名用的是 `run_id`；实验文件里的 `name` 是 optuna study 名，属于 infra，不过河）。

`evaluation.num_envs` 删得比另外三个硬：评估复用 `cfg.num_envs`，也就是训练那张图的宽度。按它的字面意思实现需要第二张图，所以它不只是没实现，是**与单图假设矛盾**。

### score

worker 写到 `score.s3`，控制面读回来喂给 Optuna。

```json
{ "run_id": "...", "trial": 0, "value": 123.4 }
```

## 一条跨侧约束，不在 JSON 里

**worker 的 aim 客户端与 ctrler 的 aim 服务端主版本必须一致。** worker 现在钉在 `aim==3.28.*`。

以前这条靠"两边装同一个包"隐式保证;拆开之后没有任何机制保证它，只有这一行。升级任何一侧都要同时升另一侧。

## 实验文件：运行配置是从哪来的

infra 的输入，样板见 `experiments/streamac template.yaml`，由 `trainer_infra.experiment` 读。它与运行配置**共用一套字段名**——不是两个 schema 加一个翻译层，是一个形状填两次。

```
[透传] environment  training  evaluation  logging  score(除 s3)
[消费] image → digest        storage → score.s3、logging.rerun_s3
[生成] contract(抄 catalog 的)  launch_id  run_id  trial
[采样] space ⊕ catalog 的搜索域 → params
[自用] name(optuna study)  description  compute  hpo
```

**运行配置 = 实验配置的外围参数（透传）+ 采样(catalog 的搜索域 ⊕ 实验配置的覆盖)**，加上只有提供方知道的协调字段。外围参数只能透传：catalog 能声明的是空间，而空间需要采样器，一个值没有空间。

infra 在**产出任何配置之前**校验自己的输入完整（`trainer_infra.experiment.REQUIRED`）。少一个 `epoch_steps` 是零个容器起来，不是一轮 HPO 起 N 个各自读到同一个空字段再死。它校验的是"我的输入格式完整吗"，不是"worker 会不会喜欢这个值"——它不理解任何名字的语义。

三个 fail-fast，都在起容器之前：

| 拒绝 | 为什么 |
| --- | --- |
| 实验文件缺字段 | 见上 |
| `space` 里有 catalog 没声明的名字 | 那是个不接线的旋钮，采样器永远不会填，运行会带着作者以为设过的值开始 |
| `image` 没钉到 digest | tag 可以移动，而搜索空间绑在镜像上；浮动 tag 会让空间在一个已记录试验的 study 底下改变 |

### 保持两份形状相等的机制

只有往返测试：catalog → `trainer_infra` 真实解析与采样 → `RunConfig` 校验 → 入口 `build` 并跑一步。链绿 = 两份拷贝仍然相等；红 = 有人只改了一边。

上半（协调与外围）在 `infra/tests/test_experiment_hpo.py`。下半（参数）等 catalog 的参数树重构后接上，见 `docs/roadmap.md` R1d。
