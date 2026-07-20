# Task 11 Report: Legacy Entry Point and Configuration Migration

## Status

The historical `rtrrl/rtrrl.py` is now a compatibility-only module. It keeps
`RTRRLParams`, `train_rtrrl`, `--config_path`, legacy YAML fields, and CLI
overrides, while Memorax owns normalization, component/program selection,
training, evaluation, historical metric translation, and logger lifecycle.

## Preserved Work Audit and TDD

Two interrupted agents left valid uncommitted implementation and tests but no
report, commit, terminal record, or recoverable behavior-level RED output. That
work was audited rather than discarded. The retained compatibility test file
was green locally before further implementation (7 tests, 44,748 KiB peak
RSS), so no earlier RED claim is made.

The audit found one remaining behavior gap: discovery used `rtrrl*.yml` in
`memo/config` and silently omitted
`memo/config/independent_rtrrl_hopper_maskP_lru.yml`.

- RED: the repository audit regression expected 697 runtime RTRRL YAMLs but
  received 696.
- GREEN: discovery now accepts both `.yml` and `.yaml`, and any filename
  containing `rtrrl`, in both runtime config directories. The full entrypoint
  test file then passed 7/7.

No production change made during this continuation preceded its failing test.

## Delegation and Old-Core Removal

- `rtrrl/rtrrl.py` contains CLI/path compatibility and delegation only.
- Parse/build imports stop at the lightweight Memorax compatibility and
  entrypoint modules. Package exports were made lazy so this path does not
  import JAX, Brax, Optax, Distrax, environments, or the AAAI oracle.
- `normalize_legacy_invocation()` maps old budget names and delegates to
  `normalize_legacy_config()`.
- `describe_legacy_build()` delegates effective recipe construction to
  `to_component_config()` without creating an environment.
- Training delegates dynamically to `experiments/rtrrl_hopper/run.py`;
  `train_legacy()` selects the Memorax shared or independent builder and calls
  the common `train_loop()`.
- Evaluation remains in that common program lifecycle. `run_legacy_experiment`
  delegates logger creation/finalization to `with_logger` while preserving
  project `"RTRRL"`, legacy run names, logger backend, and repository.
- Historical strict logging now calls the shared
  `historical_rtrrl_metrics()` translator.

Against base `87fbcf5`, the external script changed from 983 lines to 72
lines (965 deletions, 54 compatibility additions). Its former TD, recurrent,
trace, optimizer, train-step, evaluation, and logging mathematics are absent.
The AST contract rejects JAX/Optax/Distrax/oracle imports and the former core
definitions.

## Subprocess Parse/Build Evidence

Representative legacy and Memorax YAMLs are parsed in fresh subprocesses:

- `rtrrl/config/rtrrl_hop_533.yml`: 10,000,000 steps, 10,000 epochs,
  `memo_experimental`, Aim, old run name, TD LR `3e-5`, RNN LR `2e-6`, and
  RNN clip `1.0`.
- `memo/config/rtrrl_hopper_533.yml`: 1,000,000 steps, 20 epochs,
  `memo_experimental`, no logger, Memorax run name, and the same optimizer
  values.

Both report `environment_started: false` and `jax_imported: false`. These
actions resolve only the static AgentProgram recipe; no Brax environment,
JAX trace, or compilation occurs.

## Exact Historical Mock Epoch

`--compat-action mock-epoch` emits the frozen 16-key historical dictionary
exactly, including `steps=30`, `mean_reward=4.0`,
`mean_delta=0.8333333134651184`,
`total_td_loss=19.399999618530273`,
`v_targ=20.66666603088379`, both optimizer learning rates, and all three
historical norm paths. The subprocess compares the complete decoded dictionary,
not a subset or tolerance. A separate test replaces the shared translator and
proves the mock epoch goes through that production logging boundary.

## Repository YAML Discovery and Classification

The plan expected 686 runtime RTRRL YAMLs. Current discovery found 697:

- `rtrrl/config`: 684
- `memo/config`: 13
- delta from plan: +11

Classification:

- accepted: 697
- unsupported explicit branch: 0
- unknown fields: 0
- deprecated no-op: 0

All supported runtime files parsed without edits; no YAML file changed.
Synthetic fixtures additionally prove explicit CTRNN and no-RNN branches are
classified as unsupported, unknown fields separately, and deprecated
`save_model` as a no-op warning. HPO study specifications and the GitHub
workflow whose filenames contain `rtrrl` are not runtime CLI configuration
files and are intentionally outside these two config-directory counts.

## Resource Incident Handling

Before testing, the process table showed no stale pytest, JAX, RTRRL training,
or environment process. No complete RL environment ran locally.

The earlier server-exhaustion run left no durable test output, so no result is
attributed to it. Local work was limited to config/static/focused checks:

- entrypoint-only: 7 passed, 44,440 KiB peak RSS;
- config plus entrypoint: 79 passed, 465,544 KiB peak RSS;
- ruff, compileall, `git diff --check`, and IDE diagnostics passed.

JAX/mock/full relevant parity ran on authorized AWS Batch queue
`rtrrl-cpu2-queue`, job definition `rtrrl-cpu-job:14`, with an explicit
8,192 MiB/4-vCPU override on a `c7a.2xlarge` compute environment.

## Batch and Static Evidence

- `843beb65-2b6b-4f68-b829-63adca40946f`: infrastructure-only failure before
  tests because the image lacked `/usr/bin/time` (exit 127). The retry installed
  that utility.
- `acd414ac-8b29-40c5-9309-b3e3a2f5e557`: exercised the complete selected
  set and measured 4,112,708 KiB peak RSS. It found only the two already
  documented Task-10 StreamAC x64-disabled int64/int32 baseline mismatches.
- `7e4fdedd-e0d9-43b7-8889-5e644f6830c1`: after excluding exactly those two
  known StreamAC cases and restoring the passing RTRRL parameterization,
  produced 207 passed and five authorized directional finite-difference skips;
  peak RSS was 3,944,868 KiB. Ruff passed. Its tests were green, but the job
  exited on pyright environment discovery because the temporary venv was not
  at the repository-configured `.venv` path.
- `95a29ae6-119c-48c2-a00d-8ff5ecf27010`: final Batch static retry binds that
  temporary environment at `.venv`, but failed before checks while installing
  the editable package because `git` was absent from the image.
- `e6e70a93-0d1b-40b2-8c32-66733a56c16b`: installed the missing prerequisite
  and succeeded. Pyright reported 0 errors/0 warnings, ruff and compileall
  passed, and all seven entrypoint contracts passed.

## Concerns

The two full selected-suite failures are unchanged pre-existing StreamAC
counter dtype differences under JAX x64-disabled operation, previously
recorded in Task 10; Task 11 does not modify StreamAC. HPO study YAMLs are
configuration generators rather than direct legacy CLI inputs and therefore
are reported as explicitly out of runtime-audit scope.

## Review-Blocker Corrections (Authoritative)

This section supersedes the earlier build-description, parameter-API, CLI,
classification, mock-provenance, local-count, and final-Batch statements.

### TDD Evidence

The review regressions were added before their production fixes:

- build subprocesses returned a fixed construction label and omitted the
  effective environment and selected builder;
- top-level Memorax `env_name: halfcheetah` left
  `env_params.env_name` at `StatelessCartPoleEasy`, and conflicting top-level
  and nested names did not fail;
- static build description reported top-level backend even when the runner
  would use nested `env_params.init_kwargs.backend`;
- `RTRRLParams` was the frozen Memorax schema rather than the historical
  mutable dataclass, so mutation and `rtrrl_fixed.py` parse-then-assignment
  failed;
- `--field=value` and boolean disable forms were not parsed;
- invalid profile/value files were reported as accepted and every other
  `ValueError` was treated as unknown; malformed YAML aborted the audit;
- the mock expected dictionary was duplicated in production/test and its
  `num_episodes=1` did not come from the approved Task-9 synthetic
  characterization.

Each RED failed for the named missing behavior. The corrected entrypoint suite
passes all 17 contracts.

### Effective Environment and Builder Contract

Top-level Memorax environment names now populate the exact nested environment
configuration consumed by `make_legacy_env`; bare names are canonicalized to
`brax-*`. If top-level and nested names are both explicit, canonical equality
is accepted and a conflict raises `InvalidRTRRLConfig`. Nested
`env_params.init_kwargs.backend` retains the runner's legacy precedence over
the top-level backend.

Static build output no longer claims a fixed construction string. It reports:

- final environment name, mode, and effective backend;
- shared versus independent topology and exact builder function;
- selected recurrent, feature, and actor components;
- meta-RL, observation/reward normalization, and pass-observation inputs.

Subprocess contracts cover the historical shared YAML, the top-level Memorax
shared YAML, and
`memo/config/independent_rtrrl_hopper_maskP_lru.yml`. Every parse/build
subprocess still reports no environment startup and no JAX import.

### Historical Mutable Parameter API

`rtrrl/rtrrl.py` contains compatibility-only copies of historical
`EnvironmentParams`, `OptimizerConfig`, and
`@dataclass(unsafe_hash=True) RTRRLParams`. The RTRRL class preserves base
`87fbcf5` field order, defaults, constructor surface, class name, and mutable
assignment behavior. Tests cover representative construction/mutation,
all historical field names, nested defaults, `rtrrl_fixed.py` parsing followed
by both assignments, and `train_rtrrl` normalization/delegation. No training
math was restored.

### CLI and Classification

Overrides support spaced and equals forms, dotted optimizer paths, boolean
enable forms, and `--no-*`/`--no_*` disable forms. Tests run the external
script from both repository root and `rtrrl/`; CLI values deterministically
override YAML budgets, optimizer values, and booleans.

Audit categories are now:

- accepted;
- unsupported explicit branch;
- unknown field (`UnknownRTRRLField`);
- deprecated no-op;
- invalid config/profile/value.

Synthetic fixtures cover all categories, invalid root mappings, and malformed
YAML. The actual repository result remains 697 accepted, 0 unsupported,
0 unknown, 0 deprecated no-op, and 0 invalid, with no YAML edits.

### Versioned Historical Mock Fixture

The exact expected dictionary now lives only in
`tests/rtrrl_parity/golden/historical_mock_epoch_v1.json`. Its metadata names
the approved Task-9 `_synthetic_summary` characterization and base
`87fbcf5`. Production independently constructs that synthetic epoch from step
rewards, dones, TD errors, values, targets, entropy/loss observations, state
norm vectors, and float32 learning rates, then calls the production translator.
The exact fixture has `steps=30` and `num_episodes=3`.

### Corrected Verification

- Local config/entrypoint: 89 passed, peak RSS 468,268 KiB.
- Local ruff, compileall, `git diff --check`, and IDE diagnostics passed.
- Authorized 8,192-MiB Batch
  `971dfb2f-f5ac-443a-989b-7c43ab8b4a5b` succeeded.
- Batch relevant parity/builders: 217 passed and five authorized directional
  finite-difference skips, with the same two documented StreamAC dtype cases
  excluded; peak RSS 3,892,576 KiB.
- Batch ruff and compileall passed; pyright reported 0 errors and 0 warnings.

No complete RL environment ran locally.

## Final Review Findings: Budget and Optimizer Sources

### Honest RED Evidence

The focused pre-implementation command selected nine parametrized outcomes.
Six failed for the requested missing behavior and three nested-only/override
controls passed:

- direct mutable `RTRRLParams` produced `total_timesteps=15` instead of 960
  when `episodes=3`, `steps=5`, and `env_params.batch_size=64`;
- strict and experimental top-level `td_lr`/`rnn_lr` remained metadata while
  final component optimizer learning rates stayed at `1e-4`;
- conflicting top-level and dotted nested optimizer rates did not fail;
- a direct `LegacyRTRRLConfig(td_lr=...)` did not canonicalize its nested
  optimizer rate.

This RED run used only config/entrypoint tests, exited 1, and peaked at
46,780 KiB RSS.

### Budget Resolution

`normalize_legacy_invocation` now reads `batch_size` from either a mapping
`env_params` or the mutable compatibility `EnvironmentParams` object. Budget
precedence is:

1. explicit `total_timesteps` is preserved;
2. otherwise `total_timesteps = episodes * steps * num_envs`;
3. explicit top-level `num_envs` overrides `env_params.batch_size`;
4. otherwise `env_params.batch_size` supplies `num_envs`.

Regression coverage proves direct-object/YAML equivalence at 64 environments
(`3 * 5 * 64 = 960`) and explicit `num_envs`/`total_timesteps` precedence.

### Canonical Optimizer Resolution

Top-level and nested learning-rate aliases now resolve before construction:

- only `td_lr`/`rnn_lr` explicit: propagate into
  `optimizer_params_td.learning_rate` /
  `optimizer_params_rnn.learning_rate`;
- only the dotted nested field explicit: copy it into the top-level
  compatibility alias;
- both explicit and equal: accept;
- both explicit and unequal: raise `InvalidRTRRLConfig` naming both the
  top-level and dotted nested paths.

Direct `LegacyRTRRLConfig` instances use non-default values to identify the
source when `explicit_fields` metadata is unavailable. Direct mutable
`RTRRLParams` nested optimizer objects normalize to the same canonical result.
Strict and experimental tests prove that non-default values agree across the
canonical config, final `RTRRLComponentConfig.optimizer_params_td/rnn`, and
`describe_legacy_build`. The description now reads only from that final
component config.

### Final Verification

- Local config/entrypoint: 100 passed, peak RSS 468,052 KiB.
- Focused GREEN for the original RED selection: nine passed, peak RSS
  45,092 KiB.
- Local ruff, pyright (0 errors/0 warnings), compileall, `git diff --check`,
  and IDE diagnostics passed.
- Repository audit remains 697 discovered and accepted; all unsupported,
  unknown, deprecated no-op, and invalid counts remain zero.
- Authorized 8,192-MiB Batch
  `5522e151-8cbe-472c-b6c8-cb20446a5a92` succeeded with exit 0.
- Batch relevant parity/builders: 225 passed and five intentional authorized
  directional finite-difference skips, plus the separately selected RTRRL
  combined-normalization builder case passed; peak RSS 3,909,088 KiB.
- Batch ruff and compileall passed; pyright reported 0 errors and 0 warnings.

No complete RL environment ran locally. The only warning was the pre-existing
TensorFlow Probability/JAX deprecation warning.

## Final Gradient-Clip Alias Correction

### Honest RED Evidence

The focused pre-implementation command selected nine outcomes. Six failed for
the missing behavior and three nested-only/equal controls passed:

- strict and experimental top-level `rnn_grad_clip` changed compatibility
  metadata while final `optimizer_params_rnn.gradient_clip` remained `1.0`;
- conflicting mapping sources did not fail;
- direct `LegacyRTRRLConfig` top-only input did not update the nested
  optimizer;
- mutable `RTRRLParams` top-only input did not update its nested optimizer;
- conflicting mutable inputs did not fail.

The RED command exited 1 and peaked at 47,996 KiB RSS.

### Canonical Resolution

`rnn_grad_clip` and `optimizer_params_rnn.gradient_clip` now use the same
source-resolution contract as learning rates:

- only the top-level alias explicit: propagate it into the canonical nested
  optimizer;
- only the dotted nested value explicit: copy it into the top-level
  compatibility alias;
- both explicit and equal: accept;
- both explicit and unequal: raise `InvalidRTRRLConfig` naming
  `rnn_grad_clip` and `optimizer_params_rnn.gradient_clip`.

The mapping path tracks whether the dotted `gradient_clip` key itself was
present. Direct `LegacyRTRRLConfig` objects infer explicitness from
`explicit_fields` when available and otherwise from non-default values.
Mutable historical `RTRRLParams` receives equivalent nested-only, top-only,
equal, and conflict behavior; its implicit nested default does not masquerade
as an explicit value when a dynamically supplied top-level alias is present.

`describe_legacy_build` now reports
`effective.optimizer_params_rnn.gradient_clip`, which is the final value used
by state-machine optimizer construction.

### Verification

- Focused gradient-clip GREEN: nine passed, peak RSS 45,128 KiB.
- Local complete config/entrypoint suite: 110 passed, peak RSS 463,240 KiB.
- Repository audit: 697 discovered and accepted; all unsupported, unknown,
  deprecated no-op, and invalid counts are zero.
- Local ruff, pyright (0 errors/0 warnings), compileall, `git diff --check`,
  and IDE diagnostics passed.
- Authorized 8,192-MiB Batch
  `b17f36c5-f316-427e-b74a-a1a225e07339` succeeded with exit 0.
- Batch relevant parity/builders: 234 passed and five intentional authorized
  directional finite-difference skips, plus the separately selected RTRRL
  combined-normalization builder case passed; peak RSS 3,890,084 KiB.
- Batch ruff and compileall passed; pyright reported 0 errors and 0 warnings.

No complete RL environment ran locally. The only test warning remains the
pre-existing TensorFlow Probability/JAX deprecation warning.
