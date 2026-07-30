# Task 4 报告：RunConfig 环境与预算贯通

## 状态

`DONE_WITH_CONCERNS`

`build_run_config` 现在逐字段把实验的 `environment` 与 `budget` 映射成
contract v3 的 `EnvironmentConfig` 和 `BudgetConfig`，`launch.json` 同时归档这两个
section。control-plane 与 mock-trainer 的 contract 夹具已经迁移到 v3，mock-trainer
catalog 已由生成器重建。最终 CI 的 training-sdk、control-plane、mock-trainer
三个 job 全部通过。

## 提交

- `0e01f76 test(launch): require environment and budget propagation`
- `cc00861 test(launch): use the suite object store`
- `0e3b8a9 feat(launch): hand the worker its environment and its budget`
- `d446f55 test(control-plane): align fixtures with contract three`

## CI 运行（按时间顺序）

1. [30582383465](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30582383465)
   - RED 尝试。
   - training-sdk 通过；control-plane 与 mock-trainer 失败。
   - 两个新增测试在 `_launch` 设置阶段因 `NoSuchBucket` 失败，未到达目标断言；
     因此该运行不作为有效 RED，随后只修正测试对象存储设置。
2. [30582481892](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30582481892)
   - 有效 RED。
   - training-sdk 通过；control-plane 为 54 failed、94 passed、10 errors；
     mock-trainer 为 11 failed、90 passed。
   - 新增的 RunConfig 测试按要求在 `build_run_config` 内失败，错误明确列出
     `environment` 与 `budget` 缺失；归档测试以 `KeyError: 'environment'` 失败。
3. [30582797715](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30582797715)
   - 第一轮实现验证。
   - training-sdk 76 passed；mock-trainer 101 passed；control-plane
     154 passed、4 failed。
   - 剩余四项都是 contract 3 与 catalog 默认 `total_steps` spec 的旧断言：
     `tests/test_cli.py::test_validate_catalog_exits_zero_and_prints_resolved_space`、
     `tests/test_launch.py::test_launch_metadata_is_written_to_archive_and_s3`、
     `tests/test_preflight_aws.py::test_image_catalog_disagreeing_with_offline_catalog_is_rejected`、
     `tests/test_preflight_offline.py::test_example_passes_offline_checks`。
4. [30583741280](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30583741280)
   - 最终 GREEN。
   - training-sdk：76 passed。
   - control-plane：158 passed。
   - mock-trainer：101 passed。
   - 三个 job 全部通过。

## 有效 RED 的精确失败消息

```text
pydantic_core._pydantic_core.ValidationError: 2 validation errors for RunConfig
environment
  Field required [type=missing, input_value={'contract': 3, 'run_id':.../trials/t0/score.json')}, input_type=dict]
  For further information visit https://errors.pydantic.dev/2.13/v/missing
budget
  Field required [type=missing, input_value={'contract': 3, 'run_id':.../trials/t0/score.json')}, input_type=dict]
  For further information visit https://errors.pydantic.dev/2.13/v/missing
```

## 开始前的失败集合

来源：Task 4 开始前最近一次 CI
[30581864897](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30581864897)。
共 73 个失败/错误 node ID。

### control-plane（62）

- `tests/test_cli.py::test_errors_use_stderr_and_nonzero_exit`
- `tests/test_cli.py::test_validate_catalog_exits_zero_and_prints_resolved_space`
- `tests/test_cli.py::test_validate_catalog_rejects_unsupported_contract`
- `tests/test_cli.py::test_validate_catalog_rejects_unknown_score_metric`
- `tests/test_cli.py::test_validate_catalog_rejects_score_window_beyond_budget`
- `tests/test_cli.py::test_validate_catalog_rejects_unknown_space_override`
- `tests/test_cli.py::test_validate_catalog_rejects_grid_sampler_with_continuous_space`
- `tests/test_cli.py::test_validate_batch_backend_never_submits`
- `tests/test_cli.py::test_validate_batch_backend_warns_for_dev_queues`
- `tests/test_cli.py::test_run_batch_backend_exits_zero_on_success`
- `tests/test_cli.py::test_run_local_backend_exits_zero_on_success`
- `tests/test_cli.py::test_run_local_backend_exits_nonzero_on_failure`
- `tests/test_end_to_end_local.py::test_two_round_study_completes_and_reports`
- `tests/test_end_to_end_local.py::test_failing_run_stops_the_launch_and_prints_the_log`
- `tests/test_end_to_end_local.py::test_missing_score_names_the_object_and_writes_failed_report`
- `tests/test_examples.py::test_example_loads_and_passes_offline_checks[experiment-acceptance-gpu.yaml]`
- `tests/test_examples.py::test_example_loads_and_passes_offline_checks[experiment-acceptance.yaml]`
- `tests/test_examples.py::test_example_loads_and_passes_offline_checks[experiment-dev-smoke.yaml]`
- `tests/test_launch.py::test_launch_id_is_a_utc_timestamp`
- `tests/test_launch.py::test_launch_metadata_is_written_to_archive_and_s3`
- `tests/test_launch.py::test_run_config_uses_trial_params_verbatim[3]`
- `tests/test_launch.py::test_run_config_uses_trial_params_verbatim[7]`
- `tests/test_launch.py::test_trial_s3_subtrees_are_disjoint`
- `tests/test_launch.py::test_run_config_disables_rerun_when_not_configured`
- `tests/test_local_backend.py::test_successful_worker_is_reported`
- `tests/test_local_backend.py::test_terminate_stops_a_running_job`
- `tests/test_local_backend.py::test_wait_returns_early_when_a_sibling_fails`
- `tests/test_packing.py::test_configs_and_manifests_are_uploaded`
- `tests/test_packing.py::test_every_trial_appears_exactly_once_in_manifests`
- `tests/test_preflight_aws.py::test_plan_carries_digest_queue_and_job_definition`
- `tests/test_preflight_aws.py::test_dev_tier_selects_the_dev_queue`
- `tests/test_preflight_aws.py::test_loopback_aim_endpoint_is_rejected[127.0.0.1]`
- `tests/test_preflight_aws.py::test_loopback_aim_endpoint_is_rejected[127.0.1.5]`
- `tests/test_preflight_aws.py::test_loopback_aim_endpoint_is_rejected[localhost]`
- `tests/test_preflight_aws.py::test_loopback_aim_endpoint_is_rejected[::1]`
- `tests/test_preflight_aws.py::test_a_routable_aim_endpoint_is_accepted`
- `tests/test_preflight_aws.py::test_unreachable_aim_endpoint_is_rejected`
- `tests/test_preflight_aws.py::test_missing_queue_is_rejected`
- `tests/test_preflight_aws.py::test_disabled_queue_is_rejected`
- `tests/test_preflight_aws.py::test_invalid_queue_is_rejected`
- `tests/test_preflight_aws.py::test_missing_s3_bucket_is_rejected`
- `tests/test_preflight_aws.py::test_forbidden_s3_bucket_is_rejected`
- `tests/test_preflight_aws.py::test_image_without_a_registered_job_definition_is_rejected`
- `tests/test_preflight_aws.py::test_image_catalog_disagreeing_with_offline_catalog_is_rejected`
- `tests/test_preflight_aws.py::test_image_source_hash_drift_is_rejected`
- `tests/test_preflight_aws.py::test_image_parameter_space_drift_is_rejected`
- `tests/test_preflight_aws.py::test_non_ecr_image_reference_is_rejected`
- `tests/test_preflight_offline.py::test_example_passes_offline_checks`
- `tests/test_preflight_offline.py::test_unknown_entry_is_rejected`
- `tests/test_preflight_offline.py::test_metric_not_reported_by_entry_is_rejected`
- `tests/test_preflight_offline.py::test_window_beyond_smallest_total_steps_is_rejected`
- `tests/test_preflight_offline.py::test_format_space_lists_every_key`
- `tests/test_batch_backend.py::test_submit_passes_manifest_and_timeout`
- `tests/test_batch_backend.py::test_submit_tells_the_container_which_region_it_is_in`
- `tests/test_batch_backend.py::test_wait_polls_until_every_job_is_terminal`
- `tests/test_batch_backend.py::test_failed_job_exposes_its_log_tail`
- `tests/test_batch_backend.py::test_terminate_calls_batch_for_every_job`
- `tests/test_batch_backend.py::test_terminate_tolerates_already_finished_jobs`
- `tests/test_batch_backend.py::test_wait_returns_early_when_a_sibling_is_still_running`
- `tests/test_batch_backend.py::test_log_tail_without_stream_returns_empty_string`
- `tests/test_batch_backend.py::test_successful_job_has_no_reason`
- `tests/test_loop.py::test_partial_submit_failure_terminates_already_submitted_jobs`

### mock-trainer（11）

- `tests/test_catalog.py::test_catalog_declares_contract_two_and_the_reserved_parameter`
- `tests/test_runtime_cpu.py::test_installed_module_runs_real_cpu_ppo_in_subprocess`
- `tests/test_train.py::test_fast_train_retains_rollout_sdk_and_checkpoint_lifecycle`
- `tests/test_train.py::test_train_rejects_checkpoint_mismatch_before_registration`
- `tests/test_train.py::test_normal_path_calls_real_brax_entry_point_with_fixed_micro_parameters`
- `tests/test_train.py::test_launcher_fails_once_and_preserves_pre_failure_artifacts[before_training-False-False]`
- `tests/test_train.py::test_launcher_fails_once_and_preserves_pre_failure_artifacts[after_training-False-False]`
- `tests/test_train.py::test_launcher_fails_once_and_preserves_pre_failure_artifacts[after_checkpoint-True-True]`
- `tests/test_train.py::test_finalization_failure_is_rethrown_without_double_terminal`
- `tests/test_train.py::test_successful_launcher_uses_reporter_from_env_and_prints_runtime_summary`
- `tests/test_train.py::test_reported_step_matches_the_environment_step_budget`

## 最终失败集合与差异

最终运行 30583741280 的失败 node ID：无。

差异：开始前 73 个失败/错误 node ID 全部消除；新增的两个 Task 4 测试也通过，
最终三个项目共 335 个测试通过（76 + 158 + 101），失败数从 73 降为 0。

## mock-trainer catalog 生成证据

执行了要求的生成命令。由于本机新建的 uv 环境首先复用了同版本
`training-sdk==0.1.0` 的旧 wheel，第一次生成没有 diff；随后以
`uv sync --reinstall-package training-sdk` 刷新本地 path dependency，再次执行同一
生成命令。最终 diff：

```diff
 {
-  "contract": 2,
+  "contract": 3,
   "entries": {
```

只有 contract 数字改变，source hash 与 catalog 其余内容未变。

## mock-trainer 脚本 YAML 的前后 section

`rtrrl/infra/mock-trainer/scripts/brax_ppo_acceptance.yaml` 的实际内容不是实验配置，
而是旧版 trainer script descriptor；它没有顶层 `space`，也没有实验模型所需的
顶层 `environment`、`budget`。前后均为：

```yaml
defaults:
  environment:
    name: inverted_pendulum
    options: {backend: generalized}
  training_budget: {env_steps: 128}
fields:
  num_envs:
    path: algorithm.num_envs
    type: int
    default: 4
    choices: [4]
```

因此没有可从 `space` 搬出的七/八个实验键；文件中也没有 `env_mode`，无需判断
F/P/V。为避免把 descriptor 改成一个无法由其现有消费者解析的混合格式，本任务
没有修改该文件。

## 关注事项

- brief 把 `scripts/brax_ppo_acceptance.yaml` 称为实验文件，但仓库中的该路径是
  script descriptor，且不存在 `space`。这项迁移要求与实际文件结构不一致；
  需要计划维护者确认目标路径或另行定义 descriptor 到 contract v3 的迁移。
- CI 只有 GitHub Actions 执行 pytest；本机未运行 pytest 或 docker。
- 最终 CI 的唯一提示是 GitHub Actions 对 Node.js 20 action runtime 的弃用提示，
  与本任务代码无关。
