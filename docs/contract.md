# 控制面与 worker 之间的契约

两侧不共享任何代码。它们之间只有 **S3 上的 JSON**，和这份文档。

| | ctrler | worker |
| --- | --- | --- |
| 职责 | HPO、task 分发、托管 aim server、aim/rerun 可视化 | 跑若干指定配置、回报结果 |
| Python 依赖 | optuna, pyyaml | 训练框架, aim(客户端), rerun-sdk, numpy, boto3, pydantic |
| 同机进程 | aim server、rerun viewer，各自安装 | — |

发送方负责构造，**接收方负责校验**。共享一个 pydantic 类只在两侧同时发布时才安全，而 worker 是镜像、ctrler 是控制面，本来就不同时发布。

## `CONTRACT_VERSION`

一个整数，当前 **6**，写在 `memo/worker/contract.py`。catalog 和 run configuration 各带一份。worker 收到不等于自己实现的版本就**拒绝执行**，不做兼容猜测。

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
  "contract": 6,
  "run_id": "...", "experiment": "...", "name": "...", "launch_id": "...",
  "trial": 0, "entry": "stream_ac", "digest": "sha256:...",
  "environment": { "id": "brax::hopper", "backend": "spring", "seed": 0,
                   "episode_length": 1000, "observed": [0, 2, 4] },
  "training":   { "num_envs": 16, "total_steps": 2000000, "epoch_steps": 10000,
                  "chunk_steps": null, "early_stop_patience": null },
  "evaluation": { "steps": 1000, "num_envs": 1 },
  "params":     { "<扁平名>": <标量> },
  "logging":    { "aim": "...", "enable_rerun": false,
                  "rerun_s3": null, "rerun_every_steps": null },
  "score":      { "metric": "...", "window_steps": [0, 0], "reduce": "mean",
                  "direction": "maximize", "non_finite": "worst", "s3": "s3://..." }
}
```

`training.chunk_steps` 和 `training.early_stop_patience` **声明了但没有任何实现读它们**，`evaluation.num_envs` 同样。它们是空声明，改动或删除不影响任何运行中的东西。

### score

worker 写到 `score.s3`，控制面读回来喂给 Optuna。

```json
{ "run_id": "...", "trial": 0, "value": 123.4 }
```

## 一条跨侧约束，不在 JSON 里

**worker 的 aim 客户端与 ctrler 的 aim 服务端主版本必须一致。** worker 现在钉在 `aim==3.28.*`。

以前这条靠"两边装同一个包"隐式保证;拆开之后没有任何机制保证它，只有这一行。升级任何一侧都要同时升另一侧。

## 已知的不一致：两侧现在说的不是同一种语言

`infra/src/trainer_infra/experiment.py` 的 `_configurations` 产出的是：

```json
{ "experiment": "...", "trial": 0, "algorithm": "...", "run": {...},
  "objective": {...}, "loggers": {...}, "parameters": {...} }
```

而 worker 的 `RunConfig` 要的是上面那份。**两者字段几乎没有交集**——infra 说 `algorithm`，worker 说 `entry`；infra 说 `parameters`，worker 说 `params`；worker 要 `contract` `run_id` `digest` `environment` `training` `evaluation` `score`，infra 一个都不产出。

这不是回归：这份 infra 是从 `rewrite/temp` 拉过来的，它本来就与另一套 worker 配对。两侧现在同处一个分支，差异才第一次可见——这份文档的第一个用处就是让它可见。

对齐时要定的是**哪一侧动**：让 infra 产出 `RunConfig` 的形状，还是把 `RunConfig` 收缩到 infra 已经在产的那些字段（`run` / `objective` / `loggers` 这些嵌套块，比 worker 现在扁平的 `logging` + `score` + `training` 更接近实验 YAML 的结构）。
