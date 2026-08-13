# 训练部署契约 v8

Infra、训练镜像中的 Worker 和 Entry 不共享 Python 类型。跨环境接口是版本化 JSON，当前版本为 `8`。同一份序列化 fixture 位于 `tests/contracts/v8/`，各接收方只解析自己消费的投影。

## Catalog

镜像构建时运行：

```bash
python -m deployment.catalog --print-label
```

`memo/deployment/catalog.py` 只发现 `entries/` 中名称不以下划线开头、且同时导出 `PARAMETERS`、`METRICS` 和 `main` 的模块。Catalog 结构为：

```json
{
  "contract": 8,
  "entries": {
    "stream_ac": {
      "command": ["python", "-m", "entries.stream_ac"],
      "parameters": {},
      "metrics": []
    }
  }
}
```

Infra 从镜像产物读取 Catalog，不导入训练代码。

## 实验配置与运行配置

实验 YAML 属于 Infra，包含镜像、计算资源、HPO、搜索空间和评分策略。Infra 根据 Catalog 验证并采样，然后为每个 trial 生成嵌套运行配置：

```yaml
contract: 8
identity:
  run_id: stream-ac-launch-t0
  experiment: stream-ac
  launch_id: launch
  trial: 0
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
runtime:
  seed: 0
  total_steps: 2000000
  epoch_steps: 10000
  evaluation_steps: 1000
logging:
  aim:
    url: aim://host:53800
  rerun:
    every_steps: 20000
```

运行配置不包含 `score`。评分策略由 Infra 持有，也不包含 `score.s3` 或 `logging.rerun_s3`。Worker 只需要一个 `artifacts.root`。

## 接收方边界

- Worker 的 v8 投影定义在 `memo/worker/envelope.py`，只解释 `contract`、`identity`、`entry` 和 `artifacts`；`algorithm`、`runtime`、`logging` 保持为交给子进程的 JSON。
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
