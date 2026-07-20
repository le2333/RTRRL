# RTRRL Numerical Parity and Resource Report

## Scope and conclusion

This report records fresh Task 12 evidence for commit
`33448c2a12ef93edf4389b9286b1da60d8a8a17f`.  All JAX parity,
finite-difference, full `online_ac`, independent-RTRRL, and Brax work ran on the
authorized AWS Batch queue.  The local host ran only static/configuration
checks; it did not construct or execute a complete RL environment.

The strict Memorax LRU path passes its complete parity suite and a real
legacy-entrypoint Brax smoke.  The full `online_ac` suite is **not green**:
head has 12 failures and base `5a89953b5d09909b35c5016118dc11a1adb0dec2`
has the same 12 failures plus six old meta failures.  There are no head-only
failures.  Those baseline exclusions are listed below rather than hidden.

## Fixture provenance

The versioned fixture is `tests/rtrrl_parity/golden/manifest.json` plus its
NPZ payload.  The manifest pins:

- source: `RTRRL-AAAI25`;
- source commit: `4301943c349171d828d0fcf3e40944c286451415`;
- algorithm and seed: LRU, seed 7;
- dimensions: hidden 2, input 4, action 2, batch 1;
- dtype policy: float32/complex64;
- capture runtime: Python 3.12.13, JAX/JAXLIB 0.4.38, CPU;
- state-machine protocol: one deterministic mock environment, terminal on the
  second transition, three eager transitions;
- instrumentation: only the nested step JIT was removed, observation hooks
  were inserted without changing statements/returns, and `lax.scan` was
  replaced by an eager test driver.

The fixture was not regenerated or recalibrated in Task 12.

## Batch runtime and resources

All successful final jobs used queue `rtrrl-cpu2-queue`, job definition
`rtrrl-cpu-job:14`, 4 vCPUs, and 8,192 MiB.  The EC2 compute environment is
`rtrrl-cpu2-ce` on `c7a.2xlarge`.  Test and numerical-comparison runtime was
Python 3.12.13, JAX/JAXLIB 0.10.0, Flax 0.12.7, backend `cpu`, device
`cpu:0`.  The Brax smoke additionally used Brax 0.14.2 with the `spring`
physics backend.

Fresh measurements:

| Workload | Result | Duration | Peak RSS |
| --- | --- | ---: | ---: |
| Complete `tests/rtrrl_parity` | 200 passed, 5 skipped | 44.83 s | 2,145,140 KiB |
| RTRRL/meta/independent-selected `online_ac` | 34 passed, 2 baseline failures | 82.632 s | 3,126,540 KiB |
| `tests/test_independent_rtrrl.py` | 11 passed | 25.545 s | 1,484,776 KiB |
| Full head `tests/online_ac` | 101 passed, 12 baseline failures | 210.409 s | 4,880,076 KiB |
| Full base `tests/online_ac` | 90 passed, 18 failures | 191.380 s | 4,679,072 KiB |
| Strict real Brax smoke | succeeded | 22.34 s | 1,392,252 KiB |
| Complete-step eager/JIT comparison | succeeded | 13.42 s | 1,079,284 KiB |
| Local 697-file config audit | 697 accepted | 1.64 s | 18,412 KiB |

The five parity skips are the deliberately opt-in directional finite-difference
parameterizations.  They are skipped unless their separate authorization flag
is enabled; all non-skipped credit, eager/JIT, and fixture comparisons passed.

An 8-GiB, 4-vCPU EC2/Batch allocation remains the recommendation.  The largest
measured resident set was 4.66 GiB for full `online_ac`; 8 GiB leaves practical
headroom for installation, compiler, and teardown.  The shortest Brax
integration needed only 1.33 GiB RSS, so no memory increase was justified.

## Numerical differences by module

The final parity job instrumented floating assertions without changing their
pass/fail policies.  “Exact” below counts observed floating comparison events,
not unique fixture paths.  No broad ULP tolerance is inferred from these
measurements.

| Module | Exact / compared | Maximum absolute | Maximum relative | Measured named ULP |
| --- | ---: | ---: | ---: | --- |
| strict heads | 20 / 21 | `2.9802322e-08` | `1.0697146e-07` | one ULP observed, but the NumPy assertion supplied no leaf label; no ULP bound adopted |
| strict LRU forward | 35 / 35 | `0` | `0` | 0 |
| strict LRU credit | 40 / 61 | `2.3841858e-07` | `2.4061205e-07` at `1/B_img` | 3 at `B_img` |
| historical initialization | 128 / 128 | `0` | `0` | 0 |
| pure update rules | 28 / 37 | `1.1920929e-07` | `2.2351742e-07` | no bound: the largest bit distance crossed zero and had no named leaf |
| complete/optimizer step assertions | 1004 / 1270 | `1.1920929e-07` at `action` | `6.7169371e-05` at `params/rnn/OnlineLRUCell_0/LRUCell_0/gamma_log` | 766 at that exact `gamma_log` leaf; measured only, not generalized |
| historical logging reductions | 10 / 17 | `1.2715658e-06` | `8.9406967e-08` | no bound: zero/sign-crossing synthetic reductions had no named leaf |

The helper’s intentional one-ULP self-test in `test_public_api.py` is excluded
from algorithm-module maxima.

## CPU eager, CPU JIT, and complete scan

The selected Batch backend is CPU, so eager and JIT were compared in the same
fresh Batch runtime and against the CPU oracle fixture.  Canonical state
structure, shape, dtype, and PRNG keys were checked before values.

| Comparison | Exact leaves | Max abs leaf/value | Max rel leaf/value | Max ULP leaf/value |
| --- | ---: | --- | --- | --- |
| one-step eager vs oracle | 76 / 82 | `traces/rnn/D`, `1.4901161e-08` | `traces/rnn/OnlineLRUCell_0/LRUCell_0/nu_log`, `2.2639420e-06` | same `nu_log`, 26 |
| one-step JIT vs oracle | 70 / 82 | `traces/rnn/D`, `1.4901161e-08` | same `nu_log`, `2.2639420e-06` | same `nu_log`, 26 |
| one-step JIT vs eager | 76 / 82 | `traces/td/critic/kernel`, `3.7252903e-09` | TD actor Adam `nu` kernel, `2.5525941e-07` | that exact kernel, 4 |
| three-step eager vs oracle | 54 / 82 | `action`, `1.1920929e-07` | RNN Adam `nu/.../nu_log`, `4.6432751e-06` | that exact leaf, 55 |
| three-step JIT vs oracle | 46 / 82 | `action`, `1.1920929e-07` | same RNN Adam leaf, `4.6432751e-06` | same leaf, 55 |
| three-step JIT vs eager | 61 / 82 | `action`, `5.9604645e-08` | TD actor Adam `nu` kernel, `4.3495504e-07` | that exact kernel, 6 |

The one-step and three-step output keys were bitwise exact between eager and
JIT.  The three-step comparison includes the terminal transition and every
persisted final-state leaf.  Existing tests separately compare each of the
three eager intermediate states, complete debug observables, gradients,
traces, optimizer updates, and slow parameters to the fixture.  Production
scan tests confirm fixed pytree schemas and scalar/event-only epoch summaries.

The ULP values in this section are observations for the exact named leaves and
runtime only.  They are not tree-wide acceptance limits.

## Configuration and unsupported branches

The fresh lightweight repository audit discovered 697 runtime RTRRL YAML files,
11 more than the plan’s original 686:

- `rtrrl/config`: 684;
- `memo/config`: 13;
- accepted: 697;
- unsupported explicit branch: 0;
- unknown fields: 0;
- deprecated no-op: 0;
- invalid config/profile/value: 0.

No YAML was edited.  Synthetic contracts still prove the classification paths:
explicit `rnn_model: ctrnn` and explicit no-RNN/`None` are unsupported in the
strict profile and fail during construction.  CTRNN wiring, CTRNN RFLO/RTRL,
and no-RNN training remain intentionally unsupported.  RTU, encoding,
bounded actor, clipping, frozen gain, prediction, fresh traces, sum reduction,
normalization, and independent topology remain explicit
`memo_experimental` components rather than strict-baseline options.

## Logging compatibility

The complete parity suite verifies the historical training keys, optional
learning-rate/magnitude keys, `norms/*`, logger scan-step semantics,
`eval/rewards`, `eval/best_eval_reward`, and `env/video`.  The pinned mock epoch
remains an exact versioned dictionary comparison.

The real Brax smoke emitted, at historical logger step 1:

`steps`, `mean_reward`, `num_episodes`, `mean_delta`, `mean_r_bar`, `mean_v`,
`total_td_loss`, `actor_loss`, `critic_loss`, `entropy`, `v_targ`,
`eval/rewards`, and `eval/best_eval_reward`.

Its evaluation reward was `49.92558288574219`.  Rendering was disabled, so
`env/video` was correctly not requested in this shortest integration run.

## Real legacy/Memory Brax integration

Final job `64b129a9-964e-4293-a724-c1cd25bdd839` called the public historical
`rtrrl.train_rtrrl` entrypoint with an explicit `aaai25_strict_lru` mapping.
That delegated to Memorax, constructed real `brax-hopper` with the `spring`
backend, compiled the closed strict program, executed one real training
environment transition/update, executed a 1,000-transition evaluation scan,
and finalized the recording logger.  One training transition is the minimum
valid strict epoch budget.  A full training run was neither needed nor run.

Two earlier smoke calibrations are not acceptance evidence: the first selected
the experimental profile through an over-complete mutable compatibility
dataclass and used a falsey empty recorder; the second proved strict training
but only two evaluation transitions, too short to complete a Hopper episode
and therefore too short to emit `eval/rewards`.  Neither revealed a production
defect.

## Full `online_ac` baseline classification

Head has 12 failures; base has those same 12 plus six.  Set comparison is
exact: `head_only = []`.  The shared failures are:

1. `test_evaluation_parity.py::test_standard_legacy_evaluate_matches_leaf_for_leaf`
2. `test_legacy_builders.py::test_legacy_builder_combined_normalization_train_and_eval_parity[stream-ac-rtrl]`
3. `test_legacy_builders.py::test_stream_ac_legacy_builder_translates_exact_rtrl_and_runs_one_step`
4. `test_legacy_characterization.py::test_rtrrl_lru_matches_versioned_legacy_oracle[fresh]`
5. `test_legacy_characterization.py::test_rtrrl_lru_matches_versioned_legacy_oracle[incoming]`
6. `test_legacy_characterization.py::test_stream_ac_rtu_matches_versioned_legacy_oracle[adaptive]`
7. `test_legacy_characterization.py::test_stream_ac_rtu_matches_versioned_legacy_oracle[non_adaptive]`
8. `test_standard_parity.py::test_standard_continuous_wrapper_clip_does_not_replace_feedback`
9. `test_standard_parity.py::test_standard_fresh_trace_resets_nonzero_incoming_at_initial_boundary`
10. `test_standard_parity.py::test_standard_init_matches_legacy_and_golden_leaf_for_leaf`
11. `test_standard_parity.py::test_standard_one_step_matches_legacy_every_exposed_leaf`
12. `test_standard_parity.py::test_standard_three_steps_match_each_state_and_terminal_zeroing`

Base alone also fails six pre-modularization `test_meta_parity.py` cases:
initialization, captured-parts immutability, incoming/fresh one-step views,
three-step intermediate/final state, and prediction direct gradient.  Head
makes all six green.  The 12 shared failures are the known versioned
legacy-golden/path and JAX x64-disabled int64/int32 StreamAC/standard
characterization differences.  The two selected tests whose names include
`rtrrl_lru` exercise that old generic meta characterization, not the new strict
AAAI25 fixture suite; they fail identically at base and head.  They are reported
as exclusions, not described as passing.

## Repository-state verification

- The Task 12 worktree was clean before this report and only this worktree was
  modified by Task 12.
- Other pre-existing worktrees are not clean.  Their changes predate this task
  and were not touched, so the stronger literal statement “only one worktree
  in the repository is dirty” cannot be made.
- A tracked-text credential scan found no AWS access-key IDs, private-key
  blocks, or assigned API key/password/secret literals.
- `rtrrl/rtrrl.py` is a 164-line compatibility/delegation entrypoint.  Its AST
  contract and source audit find no JAX/Optax/Distrax imports or TD, recurrent,
  trace, optimizer, train-step, or evaluation mathematics.
- The remaining `rtrrl/` package still contains other historical algorithms
  and utilities (`rtrrl_lru.py`, generic optimizers, environment wrappers).
  The old AAAI25 strict mathematical core removed from `rtrrl/rtrrl.py` is not
  duplicated there; strict lifecycle construction delegates to Memorax.
- No full RL environment ran locally.  Local activity was configuration audit,
  source/state inspection, documentation, and static verification only.
- Task 12 changed no Python source.  Expanded branch-wide ruff and compileall
  passed.  Expanded branch-wide pyright is not green: it reports three existing
  `with_logger` annotation errors in `experiments/base/experiment.py`, one
  existing `base.experiment` test-import error in
  `tests/test_independent_rtrrl.py`, and 12 dynamic-export warnings.  These are
  recorded as baseline static-analysis debt rather than silently excluded or
  changed by this documentation-only task.

## Acceptance boundary

Fresh evidence supports strict fixture parity, complete eager and JIT steps,
the terminal short scan, configuration compatibility, construction-time
unsupported-branch rejection, historical logging, legacy delegation,
scalar-only production scans, retained extension/independent contracts, and a
real Brax integration.

This report does **not** claim that full `online_ac` is green, does not claim
bitwise equality across arbitrary backends, does not turn measured ULP values
into broad bounds, and does not claim that unrelated worktrees are clean.
