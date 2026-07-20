# Observability SDK Task 2 Report

## RED

- 首次运行 Task 1+2 测试：收集阶段 3 个预期导入错误，缺少
  `set_current_run`、`AimAdapter`、`AimUnavailable` 等 Task 2 API。
- 自审补充边界测试：2 个预期失败，分别证明 throttle 会隐藏无效 metric，
  以及损坏 UTF-8 spool 未转换为明确的 `SpoolCorruptionError`。

## GREEN

- `52 passed`：Task 1+2 SDK 测试全部通过。
- Ruff：`All checks passed!`
- `git diff --check`：通过。

## Files

- `rtrrl/training_sdk/context.py`
- `rtrrl/training_sdk/__init__.py`
- `rtrrl/training_sdk/aim_adapter.py`
- `rtrrl/training_sdk/spool.py`
- `rtrrl/training_sdk/run.py`
- `rtrrl/tests/training_sdk/test_context.py`
- `rtrrl/tests/training_sdk/test_aim_adapter.py`
- `rtrrl/tests/training_sdk/test_spool.py`
- `.superpowers/sdd/sdk-task-2-report.md`

## Commit

- 本报告所在的 Task 2 提交（最终回复提供提交哈希）。
- Subject: `feat(sdk): add durable Aim observability`

## Self-review

- `RunContext` 强制一基 run number，并以默认空、深冻结 mappings 暴露
  `logging`/`objective`；`set_current_run` 提供明确初始化入口。
- 所有 event 均先持久 append 后发送；JSONL 使用 append-only sent marker，
  重启后仅 replay 未发送事件，损坏内容明确失败。
- `MetricEvent` 拥有唯一 event ID，数据可 JSON 序列化，metrics 拒绝 bool
  和非有限值；env steps 非负且按非递减策略验证，重复 step 合法。
- 一般 metrics 按原生 env step throttle；episode summary 永不 throttle，且
  始终包含 return、length、env_steps。
- Aim adapter 精确映射 experiment、run name、nested hparams，并以持久 event
  marker 去重；只将明确的 `AimUnavailable` 视为可恢复故障。
- `finish()` 仅在 descriptor objective 有效、目标 metric 存在且全部 final
  metrics 有限时产生 final event；Aim 先记录 metrics，再写 objective 和
  finalized marker。
- 未实现 Rerun 或脚本迁移；仅提供 `NullRerun`，留待 Task 3 替换。
