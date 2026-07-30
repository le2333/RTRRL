# Task 2 Report: Experiment Environment and Budget Sections

## Status

DONE

## Commits

- `cb4ce92` `test(experiment): define environment and budget sections`
- `7575988` `feat(experiment): give the environment and the budget their own sections`
- `d53d9e9` `fix(experiment): migrate checked-in examples`

## CI Runs

### Baseline

- Run: `30579568862`
- URL: https://github.com/le2333/RTRRL-AAAI25/actions/runs/30579568862
- Head: `dd0e9e2ce0b87216aff70869551e3150380a4555`
- Result:
  - `training-sdk`: passed
  - `mock-trainer`: 11 failed, 90 passed
  - `control-plane`: 52 failed, 78 passed, 10 errors

### RED

- Run: `30580067738`
- URL: https://github.com/le2333/RTRRL-AAAI25/actions/runs/30580067738
- Head: `cb4ce92d8c549f068c7722ebaf7a269ea3c5959d`
- Result:
  - `training-sdk`: passed
  - `mock-trainer`: unchanged at 11 failed, 90 passed
  - `control-plane`: 53 failed, 79 passed, 10 errors
  - `test_an_experiment_carries_its_environment_and_budget` failed for the predicted reason.
  - `test_a_space_may_not_name_the_environment_or_the_budget` passed.
- Exact new failure:

```text
pydantic_core._pydantic_core.ValidationError: 2 validation errors for Experiment
budget
  Extra inputs are not permitted [type=extra_forbidden, input_value={'epoch_steps': 1000, 'ev...00, 'total_steps': 2000}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/extra_forbidden
environment
  Extra inputs are not permitted [type=extra_forbidden, input_value={'backend': 'spring', 'id...erved': [0, 1, 2, 3, 4]}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/extra_forbidden
```

### First GREEN attempt

- Run: `30580245571`
- URL: https://github.com/le2333/RTRRL-AAAI25/actions/runs/30580245571
- Head: `7575988a57338638910d6d4d41645ea1dadb411d`
- Result:
  - Both new tests passed.
  - `control-plane` had six new failures because the three checked-in example YAML files did not yet carry the required sections; each failed both its image-digest and Aim-endpoint load check.
  - The examples were migrated in `d53d9e9` before final verification.

### Final GREEN

- Run: `30580387711`
- URL: https://github.com/le2333/RTRRL-AAAI25/actions/runs/30580387711
- Head: `d53d9e94e473977e758b01bdb098cfe427fab23a`
- Result:
  - `training-sdk`: passed
  - `mock-trainer`: 11 failed, 90 passed
  - `control-plane`: 52 failed, 80 passed, 10 errors
  - Both new tests passed.
  - Ruff passed in all three jobs.
  - The overall workflow remained failed only because of the pre-existing Task 1 fallout listed below.

## Pre-existing Failure Set

Baseline run `30579568862` contained the following named failures and errors.

### mock-trainer failures

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

### control-plane failures

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

### control-plane errors

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

## Failure Set After the Change

Final run `30580387711` contained the following named failures and errors.

### mock-trainer failures

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

### control-plane failures

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

### control-plane errors

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

Comparing the node IDs test by test, both set differences are empty:

- baseline minus final: empty;
- final minus baseline: empty.

No failure or error was added, removed, or renamed. The passing control-plane count increased from 78 to 80 because the two new Task 2 tests pass.

## Concerns

None for Task 2. The final workflow conclusion remains `failure` because the two stale contract-2 catalogs and existing `RunConfig` fixtures are intentionally deferred to Tasks 4 and 7.

## Fix round 1

### Commit

- `9713778` `test(experiment): cover environment and budget validation`

### Test coverage

The experiment tests now add 18 passing cases:

- five invalid environment cases;
- one fully-observed omission case;
- four invalid budget cases;
- seven additional reserved-name cases, completing all eight names; and
- one invalid epoch-stream divisibility case.

The existing valid-document test continues to assert that a YAML observation list is parsed as a tuple.

### CI verification

- Run: `30580862844`
- URL: https://github.com/le2333/RTRRL-AAAI25/actions/runs/30580862844
- Head: `9713778`
- Commands used to read the run:

```text
gh run view 30580862844 --log-failed
gh run view 30580862844 --job 91000462166 --log
```

The control-plane job shows that Ruff passed and that all 18 added cases increased the passing count from 80 to 98 without changing the existing failure or error counts:

```text
All checks passed!
52 failed, 98 passed, 1 warning, 10 errors in 29.16s
```

The failure and error node IDs in run `30580862844` are identical to the baseline and final Task 2 lists above, so the new tests introduced no additional failure or error.

## Fix round 2

### Commit

- `8ed25fd` `test(experiment): accept eval_steps of zero`

### CI verification

- Run: `30581334973`
- URL: https://github.com/le2333/RTRRL-AAAI25/actions/runs/30581334973
- Head: `8ed25fd`
- Commands used to read the run:

```text
gh run list --workflow=tests.yml --branch feature/rtrrl-lru-paper-parity --limit 3
gh run view 30581334973 --log-failed
gh run view 30581334973 --log | rg "test_eval_steps_may_be_zero|52 failed|99 passed"
```

The control-plane job shows Ruff passed and the passing count rose from 98 to 99 with the failure and error counts unchanged:

```text
52 failed, 99 passed, 1 warning, 10 errors in 21.34s
```

Pytest ran with `-q`, so individual test names are not printed; the unchanged 52 failures and 10 errors confirm `test_eval_steps_may_be_zero` is the sole new passing case.
