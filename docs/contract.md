# 训练部署契约 v11

Infra、训练镜像中的 Worker 和 Entry 不共享 Python 类型。跨环境接口是版本化 JSON。各接收方只解析自己消费的投影。

> **本文档落后于代码。** 权威版本号是 `memo/deployment/contract.py` 里的 `ContractVersion`，当前为 `14`，对应 fixture 在 `tests/contracts/v14/`。下文的示例仍是 v11 的形状：v12 给 Catalog 条目加的 `grouped`、v13 给 manifest 加的 `groups` 都没有写进来。v14 新增的 `algorithm.environment.kwargs` 记在下面的环境一节。

## Catalog

镜像构建时运行：

```bash
python -m deployment.catalog --print-label
```

`memo/deployment/catalog.py` 只发现 `entries/` 中名称不以下划线开头、且同时导出 `PARAMETERS`、`METRICS` 和 `main` 的模块。Catalog 结构为：

```json
{
  "contract": 11,
  "entries": {
    "stream_ac": {
      "command": ["python", "-m", "entries.stream_ac"],
      "parameters": {},
      "metrics": []
    },
    "rtrrl": {
      "command": ["python", "-m", "entries.rtrrl"],
      "parameters": {},
      "metrics": []
    },
    "r2d2": {
      "command": ["python", "-m", "entries.r2d2"],
      "parameters": {},
      "metrics": []
    }
  }
}
```

Infra 从镜像产物读取 Catalog，不导入训练代码。

## 实验配置与运行配置

实验 YAML 属于 Infra，包含镜像、计算资源、HPO、搜索空间和评分策略。Infra 根据 Catalog 验证并采样，然后为每个 (trial, seed) 生成嵌套运行配置——实验 YAML 的 `environment.seeds` 是一个列表，**不参与搜索**，每个配置在其中每个种子上各跑一次：

```yaml
contract: 11
identity:
  run_id: stream-ac-launch-t0-s0
  experiment: stream-ac
  launch_id: launch
  trial: 0
  seed: 0
  role: tuning
  digest: sha256:...
entry: stream_ac
artifacts:
  root: s3://bucket/experiment/launch/run
algorithm:
  environment:
    id: brax::hopper
    backend: spring
    observed: [0, 2, 4]
    episode_length: 1000
    kwargs: {}
  num_envs: 16
  parameters: {}
training:
  seed: 0
  total_steps: 2000000
  chunk_steps: 10000
evaluation:
  every_steps: 10000
  episodes: 5
  chunk_steps: 16000
  seed: 1000
logging:
  aim:
    url: aim://host:53800
    training:
      window: { every_steps: 100000 }
  rerun:
    log_every_steps: 200000
```

运行配置不包含 `score`。评分策略由 Infra 持有，也不包含 `score.s3` 或 `logging.rerun_s3`。Worker 只需要一个 `artifacts.root`。

`environment.backend` 在该命名空间只有一种实现可选时为 `null`：brax 要选物理后端，gymnax 没有可选的。`observed` 同理，`null` 表示不裁剪观测。两者表达的都是"不适用"，与字段缺失不是一回事。

`environment.kwargs`（v14 起）是**构造环境所用的参数**，与前三个字段的区别是：前三个说的是"对造好的环境做什么"，它说的是"造出来的是哪一个"。有些任务的身份就是一个构造参数——长度 10 的 UmbrellaChain 和长度 40 的是两个任务，bsuite 自己的 sweep 扫的正是这个数——没有它，运行配置只能点出任务族，点不出其中的成员。内容原样透传给命名空间适配器的 `make`：各适配器本来就把 `**kwargs` 转发给它包的库，所以这里合法的键就是那个库的构造函数接受的键，写错的键由那个库报错。默认 `{}`，即"这个环境由名字本身唯一确定"。

gymnax 一侧还要再分一次：有些参数属于环境构造函数（UmbrellaChain 的 `n_distractor`、DiscountingChain 的 `mapping_seed`），有些属于 `EnvParams`（UmbrellaChain 的 `chain_length`）。运行配置不必知道哪个是哪个，`memorax/environments/gymnax.py` 按 `EnvParams` 自己声明的字段名来分。`max_steps_in_episode` 被明确拒绝：那是 `episode_length` 的另一种写法，一个数只在一处声明。

`identity.trial` 命名配置，`identity.seed` 命名它的这一次重复，两者合起来才唯一；`identity.role` 说明这次运行属于哪个协议（`tuning` 选配置，`formal` 量已选定的配置，只有后者可被报告）。

`evaluation.episodes` 是**恰好**多少条完整 episode，不是步数预算——跑多久由策略决定，按步数给会让 episode 数随任务和策略变。`evaluation.chunk_steps` 只是一次评估调用的内存上界，`evaluation.seed` 独立于训练种子，使"测没测"不改变训练的 key 流。

## 指标名与归约范围

指标名是 `{phase}/{scope}/{quantity}`，中段是这个数被归约的范围，也是唯一说明它怎么来的地方：

```
train/step/td_error       某一刻的读数
train/episode/return      一个 episode 的统计量
train/window/return       一段区间内每个 episode 的平均
eval/episode/return       评估，评分读的就是它
```

`metrics.jsonl` 是完整记录，永远按 episode 归约、每个 episode 一条，不可配置。Aim 是仪表盘：评估始终送达，训练只送 `logging.aim.training` 点名的范围，每个范围的间隔用它自己的单位表达：

| scope | 间隔 | 回答 |
|---|---|---|
| `step` | `every_steps` | 某一刻的读数是什么样 |
| `episode` | `every_episodes` | 典型 episode 的统计量是多少 |
| `window` | `every_steps` + `length_steps` | 一段区间内所有 episode 平均下来是多少 |

三者都不偏：`step` 选的是步，不在大小不等的对象之间做选择；`episode` 在 episode 空间里均匀；`window` 按 episode *结束* 在哪个窗口来归属，因而是对 episode 的一个划分。`window.length_steps` 默认等于 `every_steps`，即铺满整条轴、用上每个 episode；更短则是抽样若干区间——区间大小固定，同样不偏——并让累加器存活更短的时间。

`training` 块整个省略表示 Aim 只记录评估；写了 `training` 却一个范围都不点名会被拒绝。窗口内的 series 按 transition 汇总，而不是对各 episode 的均值再求均值：episode 长度不等，均值的均值不是均值，方差的均值更不是方差。`return` 和 `length` 本身就是 per-episode 的量，窗口值对 episode 取平均。

## 接收方边界

- Worker 的 v10 投影定义在 `memo/worker/envelope.py`，只解释 `contract`、`identity`、`entry` 和 `artifacts`；`algorithm`、`runtime`、`logging` 保持为交给子进程的 JSON。
- Entry 使用 `memo/entries/_contract.py` 验证完整运行配置，再分别投影到算法 assembly、Runtime 和 observability。
- Catalog 类型及版本位于 `memo/deployment/contract.py`，不属于 Worker。

## Infra 前置验证

启动 trial 之前必须满足：

- `score.metric` 是所选 Catalog Entry 声明的指标；
- 实验覆盖值或范围位于参数声明的 `valid` 域；
- 每个可达的结构参数 `kind` 在同一实验内只有一个选项；
- `space` 不包含 Catalog 未声明的参数；
- 镜像引用固定到 digest。

数值参数可以搜索；结构扫描要等基础设施显式支持后再开放。

## Manifest

Manifest 仍只按顺序列出运行配置位置：

```json
{"runs": ["s3://bucket/configs/run-t0.json"]}
```

Worker 按 manifest 顺序在隔离的 scratch 中启动各 Entry。Entry 和 logger 只在 `scratch/artifacts/` 下生成本地产物；子进程成功后，Worker 将该目录递归上传到 `artifacts.root` 并保留相对路径，最后写入 `result.json`。只有完成上传和结果写入后才清理 scratch；子进程或上传失败会立即停止 manifest，并保留失败运行的本地目录供诊断。Worker 不读取指标，也不计算 HPO 分数。

## 评分与 HPO 反馈

`ScoreSpec` 和指标文件的数值归约属于 Infra。`ExperimentRunner.run(round_executor)` 将本轮运行配置和实验的 `ScoreSpec` 一起交给 executor；executor 收集各运行的 `metrics.jsonl`，调用 Infra Scorer，并按 trial 返回数值。Infra 在整轮结果齐全后才调用 `HPO.tell()`。executor、评分或结果关联失败时，本轮已 ask 的 trial 全部标记为 `FAIL`，异常继续向上抛出，且不启动下一轮。

## 本地执行

`trainerctl run --backend local` 使用 `LocalRoundExecutor` 完整运行实验，而不是只打印首轮配置。实验的 `storage` 必须是本地 `file://` URI；executor 为每轮写入真实 config 和 manifest，启动独立 Worker 进程，读取 Worker 发布的 `result.json` 与 `metrics.jsonl`，再交给 Infra Scorer 和 HPO。Worker 命令可通过最后一个参数 `--worker-command ...` 注入；未指定时使用当前 Python 的 `python -m worker`。本地与 S3 使用同一 Worker 监督和 artifact 相对路径契约，仅对象传输 scheme 不同。

## Batch 执行

`trainerctl run --backend batch` 使用 `BatchRoundExecutor`。它从固定镜像 digest 和 `compute.instance_type` 得到 job definition，从 `--queues run|dev` 得到队列；`hpo.parallel_jobs` 决定每轮拆成几个 manifest，同一 manifest 内的 trial 仍由 Worker 串行执行。Executor 先将配置与 manifest 写入实验的 S3 前缀，再一次性提交本轮所有 job。任一 job 失败时，它终止仍未结束的同轮 sibling，读取 CloudWatch 日志尾部并令整轮失败；全部成功后才读取各 run 的 `result.json/metrics.jsonl`、评分并反馈 HPO。当前固定 AWS 区域为 `eu-north-1`。
