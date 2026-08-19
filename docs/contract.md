# 训练部署契约 v12

Infra、训练镜像中的 Worker 和 Entry 不共享 Python 类型。跨环境接口是版本化 JSON，当前版本为 `12`。同一份序列化 fixture 位于 `tests/contracts/v12/`，各接收方只解析自己消费的投影。

v12 相对 v11 新增两个可选块：`checkpoint`（这次运行多久归档一次完整状态）和 `fork`（这次运行从哪份归档状态继续）。两者都可省略，省略时文档语义与 v11 完全相同；升版的理由是反向不兼容——v11 的镜像会拒绝带这两个块的文档，于是在旧镜像上启动的分支会静默变成一次全新运行。

## Catalog

镜像构建时运行：

```bash
python -m deployment.catalog --print-label
```

`memo/deployment/catalog.py` 只发现 `entries/` 中名称不以下划线开头、且同时导出 `PARAMETERS`、`METRICS` 和 `main` 的模块。Catalog 结构为：

```json
{
  "contract": 12,
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
contract: 12
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
checkpoint:
  every_steps: 10000
  keep: null
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

`identity.trial` 命名配置，`identity.seed` 命名它的这一次重复，两者合起来才唯一；`identity.role` 说明这次运行属于哪个协议（`tuning` 选配置，`formal` 量已选定的配置，只有后者可被报告）。

`evaluation.episodes` 是**恰好**多少条完整 episode，不是步数预算——跑多久由策略决定，按步数给会让 episode 数随任务和策略变。`evaluation.chunk_steps` 只是一次评估调用的内存上界，`evaluation.seed` 独立于训练种子，使"测没测"不改变训练的 key 流。

## Checkpoint 与 fork

`checkpoint` 块在时就归档，不在就不归档；没有第二个开关。

```yaml
checkpoint:
  every_steps: 10000   # 必须是 evaluation.every_steps 的整数倍
  keep: null           # 保留几份，null 是全留
```

间隔按评估间隔计而不是按预算计：一份不落在评估边界上的 checkpoint，是运行没有测量过的时刻，从它分叉出来的结果无法放回决定分叉的那条曲线上。这条在 Entry 的 `RunSpec` 里校验，因此一整轮 job 不会先排队再逐个发现。

归档的是**整个状态**，不是权重：learner 参数、rule 自己的状态（Adam 的一二阶矩）、eligibility trace、循环 carry 及其求导状态、环境状态、归一化统计量、两个步数计数器，以及调度用的 PRNG。只存权重的 checkpoint 回答的是另一个问题——trace 和 carry 单独就决定了后面几百次更新。

产物落在 `scratch/artifacts/checkpoints/step-<12 位环境步数>.msgpack`，随 artifacts 目录整体上传，用的是既有的产物契约，没有 fork 专用的 worker、传输或目录布局。

分支就是一份带 `fork` 块的普通运行配置：

```yaml
fork:
  parent: s3://.../run-t0/checkpoints/step-000000700000.msgpack
  from_steps: 700000
  replacing: [core.rule]
```

- `from_steps` 声明 `parent` 是哪个边界，Entry 会拿对象本身核对；说的和指的不一致会让分支被标注到一个它并非来自的时刻，而下游没有任何东西会重新推导这件事。
- `training.total_steps` 是**父运行的边界加上分支自己的预算**，因为分支延续父运行的步数轴：分支在 750k 的评估就是父运行的 750k，两条曲线才读得到一起。
- `replacing` 按路径声明分支不接收的状态。合法的只有一项：`core.rule`——Adam 每个参数带矩，D-RTRRL 的两个 arm 什么都不带，所以换 rule 的分支没地方放父运行的矩。它是每次 fork 显式声明的，不从结构不匹配推断；推断会让一份来自无关运行的 checkpoint 冒充一次刻意的分支。
- 重启的运行不继承 episode 记录：tracker 从空开始，跨越 checkpoint 的那个 episode 两边都不报告，而不是各报告一半。
- 但它继承评估，且不需要携带任何东西：每个 checkpoint 的 key 由 `evaluation.seed` 与边界折出，所以分支和父运行在共享的每个边界上量的是**同一批** episode。一个时刻的两个分支之间的差别因而是它们学到的不同，不是它们被问的不同。

## 崩溃判定与分叉

读完的运行由控制面分析，不需要镜像也不启动任何东西：

```bash
trainerctl collapse --spec experiments/collapse/halfcheetah.yaml \
    --run <run-id>=<metrics.jsonl> --window-steps 50000 --decisions decisions.json
trainerctl fork --parent <父运行配置> --decision <该 seed 的判定> \
    --into <目录> --steps 50000
```

判定按 seed 逐个给出，不做跨 seed 聚合：崩溃是某一次运行里带步数的事件，五条曲线的均值既没有第一次合格崩溃也没有可分叉的 checkpoint。判定文档里带着做出它的那份 spec，所以引用一个崩溃就必然引用它的定义。`fork` 取**严格早于**崩溃步的最后一个 checkpoint，写出三份分支配置和一份 manifest，manifest 就是 Worker 一直在读的那个格式。

## 指标名与归约范围

指标名是 `{phase}/{scope}/{quantity}`，中段是这个数被归约的范围，也是唯一说明它怎么来的地方：

```
train/step/td_error       某一刻的读数
train/episode/return      一个 episode 的统计量
train/window/return       一段区间内每个 episode 的平均
eval/episode/return       评估，评分读的就是它
```

RTRRL 另外按参数组（`torso`、`actor`、`critic`）报六个描述更新尺度的量，名字里的 `update.<组>.` 段就是组：

| 名字 | 是什么 | 在哪测的 |
|---|---|---|
| `abs_td_error` | `\|delta\|`，该组拿到的那个（torso 是 `eta_f * delta`） | 每 stream |
| `used_trace_norm` | `\|\|z\|\|`，这一步乘上去的那条 trace | 每 stream |
| `raw_update_norm` | `m_raw = \|\|delta * z\|\|`，规则处理前要求的步长 | 每 stream |
| `clip_multiplier` | 规则的尺度处理实际给该组乘上的因子 | 见下 |
| `clip_fraction` | 该因子是否缩短了这一步，0/1；名字承诺的比例是某个 scope 对它取的均值 | 同上 |
| `realized_update_norm` | `\|\|dtheta\|\|`，参数真正移动的距离，量在更新上而不是从前两个推出来 | 组，广播到各 stream |

`used_trace_norm` 与既有的 `trace_norm` 差一次累加：后者是这一步累加**之后**的 trace，即下一步要读的那条。原始 clip 是 `clip_by_global_norm`，作用在流平均之后的整组上，所以同组各 block 读到同一个 multiplier；D-RTRRL 的 arm 因子是每 stream、每归一化单位的，外层 clip 的因子乘在其上，`realized_update_norm` 是独立量出来的那个校验值。

`metrics.jsonl` 是完整记录，永远按 episode 归约、每个 episode 一条，不可配置。Aim 是仪表盘：评估始终送达，训练只送 `logging.aim.training` 点名的范围，每个范围的间隔用它自己的单位表达：

| scope | 间隔 | 回答 |
|---|---|---|
| `step` | `every_steps` | 某一刻的读数是什么样 |
| `episode` | `every_episodes` | 典型 episode 的统计量是多少 |
| `window` | `every_steps` + `length_steps` | 一段区间内所有 episode 平均下来是多少 |

三者都不偏：`step` 选的是步，不在大小不等的对象之间做选择；`episode` 在 episode 空间里均匀；`window` 按 episode *结束* 在哪个窗口来归属，因而是对 episode 的一个划分。`window.length_steps` 默认等于 `every_steps`，即铺满整条轴、用上每个 episode；更短则是抽样若干区间——区间大小固定，同样不偏——并让累加器存活更短的时间。

`training` 块整个省略表示 Aim 只记录评估；写了 `training` 却一个范围都不点名会被拒绝。窗口内的 series 按 transition 汇总，而不是对各 episode 的均值再求均值：episode 长度不等，均值的均值不是均值，方差的均值更不是方差。`return` 和 `length` 本身就是 per-episode 的量，窗口值对 episode 取平均。

## 接收方边界

- Worker 的 v12 投影定义在 `memo/worker/envelope.py`，只解释 `contract`、`identity`、`entry` 和 `artifacts`；`algorithm`、`runtime`、`logging`、`checkpoint`、`fork` 保持为交给子进程的 JSON。后两个块被声明是为了让带它们的文档能通过校验，而不是为了让 Worker 对它们有看法：分支从哪份对象恢复是 Entry 的事，Worker 负责的还是那个它一直在填的 artifact 目录。
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
