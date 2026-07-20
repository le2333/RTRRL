# RTRRL Numerical Parity and Resource Report

> **SUPERSEDING DECISION — 2026-07-20:** The strict-parity result in this
> report applies to Memo's modular implementation. `rtrrl/rtrrl.py` has been
> restored byte-for-byte to functional base `5f7ff4e` as a non-runtime
> backup/reference; it does not delegate to Memo and retains its original
> mathematics. Its separate, non-parity verdict is documented in
> [`rtrrl-preserved-original-comparison.md`](rtrrl-preserved-original-comparison.md).

## Scope and evidence identity

This report closes Task 12 against the true feature base and records the
review fixes found by that comparison. The complete machine-readable record is
[`rtrrl-task12-evidence.json`](rtrrl-task12-evidence.json). Exact commands,
working directories, relevant environment variables, exits, pytest counts,
nodeids, failures, timing, RSS, numerical leaves, smoke payload, and source
hashes are recorded there. The committed command source is
`tests/rtrrl_parity/task12_batch_verification.sh`.

Commit identities are intentionally distinct:

- true feature functional base: `5f7ff4e40e66da0b7df4f3edc9a928185ad73ae6`;
- initial docs-only feature root: `ffa90b5ae50b67bae1cbfe84c85eb0e21325eac3`;
- Task 10-local comparison base: `5a89953b5d09909b35c5016118dc11a1adb0dec2`;
- pre-report functional head: `33448c2a12ef93edf4389b9286b1da60d8a8a17f`;
- first Task 12 report commit / review parent:
  `62246110d39256dba5641293920cebbb0b626a65`;
- reviewed functional overlay applied to `33448c2`:
  SHA-256 `9452b8661b2de7ee2afb09ab80a30dc8f94ec5527021e43d46192f8e45770052`.

The final report commit is the commit containing this document. Its SHA is
reported by the task handoff rather than embedded recursively in its own
contents.

## Final Batch runtime and resources

Superseding acceptance job `a5f1b64e-3ad3-4994-a107-6ca4241b182d`
succeeded with exit 0.
It used queue `rtrrl-cpu2-queue`, definition `rtrrl-cpu-job:14`, 4 vCPUs,
8,192 MiB, and compute environment `rtrrl-cpu2-ce` (`c7a.2xlarge`). Overall
container runtime was 879.285 seconds. Runtime versions were Python 3.12.13,
JAX/JAXLIB 0.10.0, Flax 0.12.7, Brax 0.14.2, backend `cpu`, device `cpu:0`.

| Workload | Result | Wall time | Peak RSS |
| --- | --- | ---: | ---: |
| strict parity, accelerated cases enabled | 203 passed | 52.35 s | 2,158,956 KiB |
| five directional finite differences | 5 passed | 6.13 s | 468,508 KiB |
| selected RTRRL/meta/legacy-builder `online_ac` | 36 passed | 109.08 s | 3,316,524 KiB |
| independent RTRRL | 11 passed | 28.41 s | 1,494,664 KiB |
| full head `online_ac` | 112 passed, 1 failed | 267.61 s | 5,144,648 KiB |
| full true-base `online_ac` | 105 passed, 1 failed | 211.31 s | 5,079,376 KiB |
| eager/JIT/oracle harness | succeeded | 12.40 s | 1,078,424 KiB |
| preserved/oracle isolated probes + comparison | succeeded | 4.83 s | 364,700 KiB |
| strict real Brax smoke | succeeded | 22.11 s | 1,374,304 KiB |
| ruff | passed | 0.10 s | 24,048 KiB |
| compileall | passed | 0.12 s | 16,472 KiB |

The maximum measured RSS was 4.91 GiB. The existing 8-GiB allocation remains
appropriate; compilation did not demonstrate a need to increase memory.

## Fixture provenance

`tests/rtrrl_parity/golden/manifest.json` and its NPZ payload pin source
`RTRRL-AAAI25` commit `4301943c349171d828d0fcf3e40944c286451415`,
LRU seed 7, hidden/input/action dimensions 2/4/2, batch 1,
float32/complex64, and capture runtime Python 3.12.13 with JAX/JAXLIB 0.4.38
on CPU. The fixture protocol contains one deterministic mock environment,
terminal on transition two, and three eager transitions. Task 12 did not
regenerate or recalibrate the fixture.

## Fresh finite differences

The final command explicitly set `RTRRL_RUN_ACCELERATED_NUMERICS=1`; it ran all
five parameterizations with no skips:

| Parameter group | Cosine | Relative error |
| --- | ---: | ---: |
| `nu_log` | 1.00000012 | 0.000119965051 |
| `theta_log` | 0.99999994 | 0.0000730149404 |
| `gamma_log` | 1.0 | 0.000206126366 |
| `B_real` | 1.0 | 0.0000286421237 |
| `B_img` | 1.0 | 0.0000388202425 |

The five tests retain their original acceptance assertions:
cosine at least 0.999 and relative error at most 0.01.

## Exact eager/JIT/oracle leaves

`task12_numerical_evidence.py` is committed and its SHA-256 is recorded in the
evidence JSON. It runs after the unchanged acceptance suite and does not wrap,
replace, or relax any assertion. Every comparison first requires identical
canonical path sets, shapes, and dtypes. There are 82 canonical leaves,
including 73 floating/complex leaves. “Exact” means `numpy.array_equal`.

| Comparison | Exact all / float | Maximum absolute | Maximum relative | Maximum ULP |
| --- | ---: | --- | --- | --- |
| one-step eager vs oracle | 76/82; 67/73 | `traces/rnn/D`, shape `[1,2,4]`, float32, `1.4901161193847656e-08` | `traces/rnn/OnlineLRUCell_0/LRUCell_0/nu_log`, shape `[1,2]`, float32, `2.263941951241577e-06` | same `nu_log` path/shape/dtype, 26 |
| one-step JIT vs oracle | 70/82; 61/73 | `traces/rnn/D`, shape `[1,2,4]`, float32, `1.4901161193847656e-08` | `traces/rnn/OnlineLRUCell_0/LRUCell_0/nu_log`, shape `[1,2]`, float32, `2.263941951241577e-06` | same `nu_log` path/shape/dtype, 26 |
| one-step JIT vs eager | 76/82; 67/73 | `traces/td/critic/kernel`, shape `[1,2,1]`, float32, `3.725290298461914e-09` | `optimizer_state/inner_states/td/inner_state/inner_state/2/0/nu/td/actor/kernel`, shape `[2,4]`, float32, `2.552594082771975e-07` | same optimizer path/shape/dtype, 4 |
| three-step eager vs oracle | 54/82; 45/73 | `action`, shape `[1,2]`, float32, `1.1920928955078125e-07` | `optimizer_state/inner_states/rnn/inner_state/inner_state/2/0/nu/rnn/OnlineLRUCell_0/LRUCell_0/nu_log`, shape `[2]`, float32, `4.643275133275893e-06` | same optimizer path/shape/dtype, 55 |
| three-step JIT vs oracle | 46/82; 37/73 | `action`, shape `[1,2]`, float32, `1.1920928955078125e-07` | `optimizer_state/inner_states/rnn/inner_state/inner_state/2/0/nu/rnn/OnlineLRUCell_0/LRUCell_0/nu_log`, shape `[2]`, float32, `4.643275133275893e-06` | same optimizer path/shape/dtype, 55 |
| three-step JIT vs eager | 61/82; 52/73 | `action`, shape `[1,2]`, float32, `5.960464477539063e-08` | `optimizer_state/inner_states/td/inner_state/inner_state/2/0/nu/td/actor/kernel`, shape `[2,4]`, float32, `4.349550408733194e-07` | same optimizer path/shape/dtype, 6 |

One- and three-step output keys were bitwise exact between eager and JIT. The
three-step result includes the terminal transition and every persisted final
state leaf. ULP values above are observations only for the named leaf,
shape, dtype, runtime, and comparison; they are not generalized bounds.

The strict component modules all pass their unchanged tests: heads, LRU
forward, two-step credit, initialization, update rules, complete step, short
scan, and logging reductions. Those individual assertion sites do not all
provide stable canonical leaf labels, so this report does not repeat the old
unlabeled module maxima or infer leaf-specific ULP bounds from them. The
reviewable complete-state table above is the authoritative numerical maximum
record.

## True-base `online_ac` comparison

Both revisions ran in the same virtual environment and Batch container. Exact
collection contains 106 base nodeids and 113 head nodeids. Fourteen nodeids
were added and seven old `test_meta_parity.py` nodeids were replaced; all lists
are stored verbatim in the evidence JSON.

After fixing review-discovered compatibility regressions, base and head have
the exact same one-test failure set:

`tests/online_ac/test_standard_parity.py::test_standard_one_step_matches_legacy_every_exposed_leaf`

Therefore `head_only_failures = []`. The failure is a small exact-zero
floating comparison against the old standard legacy path and is not hidden or
called green. The selected RTRRL/meta/legacy-builder set is fully green:
36 passed.

The fixes restore the old `online_ac` helper’s permissive numeric/dtype
contract, preserve historical `RTRRL._update_step` logging while returning
`aux=None`, and remove the branch-introduced static diagnostics. Task 10’s
older `5a89953` comparison remains historical Task 10 evidence only; it is not
used as the final branch baseline.

## Static correctness

The original report’s “four errors and 12 warnings are baseline” statement was
false. Same-runtime evidence showed that three `with_logger` errors and 12
lazy-export warnings were branch-introduced; only
`tests/test_independent_rtrrl.py`’s dynamic `base.experiment` import was present
at true base. The branch-introduced diagnostics were fixed.

The final review-scope pyright command reports 1 error, 0 warnings at head: the
same dynamic test import. The corresponding base command reports 34 errors,
0 warnings because the base experiment annotations have additional debt.
Full-project pyright remains non-green: head 465 errors/2 warnings versus base
653 errors/2 warnings. This report does not relabel full-project debt as green.
Exact commands and exits are in the evidence JSON. Ruff and compileall pass.

## Configuration, logging, and unsupported branches

The configuration audit still discovers and accepts all 697 runtime YAMLs:
684 under `rtrrl/config`, 13 under `memo/config`, with zero unknown,
deprecated-no-op, unsupported, or invalid files. Synthetic contracts continue
to reject explicit CTRNN and no-RNN strict construction. CTRNN wiring/RFLO/RTRL
and no-RNN training remain unsupported in strict mode; retained RTU,
normalization, clipping, prediction, fresh-trace, and independent variants are
explicit experimental components.

Historical logging keys remain covered by parity tests. The review fix also
restores the old per-step lox log schema on the experimental compatibility
facade while preserving `aux=None`.

## Preserved external-script audit

The external copy is exact functional-base content (byte SHA-256
`f8aedcd9c315445af93e7f4a2475c50e9828c5188bd487ed39b85d7ec7da61cf`).
Separate JAX 0.5.0 preserved and JAX 0.4.38 AAAI25 processes measured exact
explicit-parameter LRU forward/carry, trace/update, and objective values.
Source-native PRNG and initialization differed across those runtimes. More
importantly, the fixed-noise actor-gradient maxima differed by
`1.0449076890945435` (location) and `0.5879773795604706` (raw scale), matching
the audited unconditional sampled-action `stop_gradient` difference.

Therefore Memo strict parity does not imply parity of the preserved external
script. Its detailed option mapping and unverified branches are in
`rtrrl-preserved-original-comparison.md`.

## Real Memo-runner Brax smoke

The superseding Batch job invokes committed `task12_brax_smoke.py`, which calls
`experiments.rtrrl_hopper.run.train_legacy` directly after Memo configuration
normalization, with the exact payload stored in the evidence JSON:
`aaai25_strict_lru`, real `brax-hopper`, spring backend, one training
transition/update, and 1,000 evaluation transitions. It emitted one logger
record at historical step 1, finalized the logger, and produced
`eval/rewards = eval/best_eval_reward = 49.92558288574219`. Rendering was
disabled, so `env/video` was intentionally not requested.

## Isolation and acceptance boundary

Task 12 did not run a complete RL environment locally. Local work was limited
to source/static/configuration checks and report assembly; real Brax execution
occurred on authorized Batch.

The final acceptance is scoped to this worktree and this change:

- no credential-like content or secret filename is present in the Task 12 diff;
- the external historical script exactly matches its preserved base copy and
  is not a Memo runtime path;
- its retained mathematical core is explicitly a backup/reference and has a
  separate numerical audit rather than a Memo-parity claim;
- unrelated worktrees may have pre-existing changes and are not claimed clean;
- worktree cleanliness, no untracked files, and final diff checks are recorded
  after the report commit in the task handoff.

Fresh evidence supports strict parity including accelerated finite
differences, complete eager/JIT one- and three-step state comparisons,
configuration compatibility, historical logging, true-base regression
closure, and real Brax integration. It does not claim arbitrary-backend
bitwise equality, broad ULP limits, a globally green pyright run, or a fully
green `online_ac` suite.
