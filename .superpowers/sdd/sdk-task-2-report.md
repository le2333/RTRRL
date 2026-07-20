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

## Review Fix RED

- 新增评审回归测试后定向运行：`14 failed, 25 passed`。
- 失败分别覆盖稳定 Aim run identity、跨 adapter replay overwrite、单 event
  单 track、非临时 OSError 传播、backend-neutral exports、失败即停止 replay
  和首次目录项 fsync。

## Review Fix GREEN

- Task 1+2 SDK 测试：`62 passed`。
- Ruff：`All checks passed!`
- `git diff --check`：通过。

## Review Fix Files

- `rtrrl/training_sdk/__init__.py`
- `rtrrl/training_sdk/aim_adapter.py`
- `rtrrl/training_sdk/run.py`
- `rtrrl/training_sdk/spool.py`
- `rtrrl/tests/training_sdk/test_aim_adapter.py`
- `rtrrl/tests/training_sdk/test_spool.py`
- `.superpowers/sdd/sdk-task-2-report.md`

## Review Fix Commit

- 本节所在的 Task 2 review fix 提交（最终回复提供提交哈希）。
- Subject: `fix(sdk): make Aim replay process-safe`

## Review Fix Self-review

- `run_id` 经 SHA-256 稳定派生为 24 字符小写十六进制 Aim `run_hash`，并始终
  以 `force_resume=True` 恢复同一 run。
- 每个 `MetricEvent` 强制只含一个 metric；一般 metrics、episode summary
  和 final metrics 在 SDK 边界拆分。Aim 仅以稳定 name 和显式 env step
  track，不使用 event-specific context。
- event ID marker 继续用于审计和快速跳过；模拟 track 后 marker 前故障时，
  新 adapter 恢复同一 backend，replay overwrite 同一 sequence point。
- 默认仅 `ConnectionError`/`TimeoutError` 转换为 `AimUnavailable`；权限、
  普通 `OSError` 和其他异常保持原样，仍支持显式注入临时异常类型。
- replay 遇到首个 `AimUnavailable` 立即停止，保留后续事件和 final 顺序。
- 首次创建父目录时 fsync 其目录项，首次创建 spool 时 fsync spool 父目录；
  helper 通过 monkeypatch 测试，无平台相关 mock 细节。
- 公共包不再导出 Aim backend 类型；内部 adapter/spool 继续共享异常类型。

## Summary Sequence Fix RED

- 新增同 native env step 双 summary、summary replay 和 spool 重开恢复测试后：
  `6 failed, 21 passed`。
- 新增真实 Aim 3.28 聚焦测试后：`1 failed, 65 passed`；该失败暴露 Aim 3.28
  不会直接用指定 hash 创建首次 run，需要 adapter 先建立稳定 hash 的 metadata
  entry，再以 `force_resume=True` 打开。

## Summary Sequence Fix GREEN

- Task 1+2 SDK 测试：`66 passed`，其中包含真实 Aim 3.28 tmp repo 集成测试。
- Ruff：`All checks passed!`
- `git diff --check`：通过。

## Summary Sequence Fix Files

- `rtrrl/training_sdk/aim_adapter.py`
- `rtrrl/training_sdk/run.py`
- `rtrrl/training_sdk/spool.py`
- `rtrrl/tests/training_sdk/test_aim_adapter.py`
- `rtrrl/tests/training_sdk/test_spool.py`
- `.superpowers/sdd/sdk-task-2-report.md`

## Summary Sequence Fix Commit

- 本节所在的 Task 2 summary sequence 修复提交（最终回复提供提交哈希）。
- Subject: `fix(sdk): preserve same-step summaries`

## Summary Sequence Fix Self-review

- `MetricEvent` 现在分别持久化 native `env_steps`、Aim `aim_step` 与稳定
  `stream`；一般 metrics/final 使用 native env step，summary 使用独立 sequence。
- 每次 summary 分配单调一基 sequence，三个 mandatory events 共享同一
  `aim_step`；`train/env_steps` 的 value 和 Aim epoch 均保留真实 native step。
- `TrainingRun` 从 spool 全部历史 summary events 恢复最大 sequence，进程重启
  后继续递增；同一 event replay 保持相同 stream/step 并 overwrite。
- Aim context 仅使用固定低基数 `sdk_stream`，不包含 event ID；一般 metrics、
  episode summary 与 final 使用不同 stream。
- fake 边界验证同 native step 的两个 summary 在三个 series 中各保留两个点，
  并验证 marker 前故障后的 replay 不增加点。
- 真实 Aim 3.28 tmp repo 测试验证同 name/context/summary step overwrite、不同
  summary step 均保留，并验证 `epoch` 参数合法。

## Atomic Summary Batch Fix RED

- 新增 batch 持久化、append failure 零发送、完整重开与 torn tail 测试后：
  `4 failed, 1 passed`。
- 补充失败 batch 不消耗 summary sequence 测试：预期失败，实际 sequence
  从 2 开始。
- 补充 torn tail 恢复后继续 append/reopen 测试：预期失败，证明仅逻辑忽略
  而不截断尾部会把下一条合法记录粘到损坏数据后。

## Atomic Summary Batch Fix GREEN

- Task 1+2 SDK 测试：`70 passed`，真实 Aim 3.28 集成测试继续通过。
- Ruff：`All checks passed!`
- `git diff --check`：通过。

## Atomic Summary Batch Fix Files

- `rtrrl/training_sdk/run.py`
- `rtrrl/training_sdk/spool.py`
- `rtrrl/tests/training_sdk/test_aim_adapter.py`
- `rtrrl/tests/training_sdk/test_spool.py`
- `.superpowers/sdd/sdk-task-2-report.md`

## Atomic Summary Batch Fix Commit

- 本节所在的 Task 2 atomic summary batch 修复提交（最终回复提供提交哈希）。
- Subject: `fix(sdk): persist summaries atomically`

## Atomic Summary Batch Fix Self-review

- `append_many()` 将一个逻辑 batch 编码为单条 JSONL record，并在唯一一次
  durable append/fsync 完成后才更新内存 events 视图和允许 Aim send。
- `_emit_many()` 对 batch 只调用一次 append，随后继续逐 event send、sent
  marker 和有序 replay；batch-of-one 统一一般 metrics 路径。
- loader 原子验证完整 batch 后再展平到现有 events 视图，保留旧单-event
  record 读取兼容性；batch 内重复/非法 event 不会部分进入内存。
- append 失败不会发送 Aim event、不会留下内存事件，也不会消耗 summary
  sequence；durable batch 重开后完整恢复并可 replay 三条 mandatory events。
- 仅最终未换行的 UTF-8 torn record 会被截断、fsync 并丢弃；完整非法 JSON、
  中间损坏与非法编码仍明确抛出 `SpoolCorruptionError`。
