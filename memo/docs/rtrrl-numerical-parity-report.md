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
- tested committed parent and evidence `functional_head_sha`:
  `b50100dc66305e4005bed93f3d1750df8b474862`;
- tested implementation manifest applied over that parent:
  SHA-256 `3ece46030ffd747a13d884273bac5b62b0e39c0c6b61f42c6697fc776625fdda`;
- exact uploaded overlay archive:
  SHA-256 `4d6a1978b4c76c1433d02ec2d949e192a417529c665b3f4392d08f7c5e3b53ad`;
- executed orchestration script:
  SHA-256 `5c392283d83789323ce626b4d94cbdfc2dde37fdfbca1e1fb59fd8067db5b709`;
- resulting evidence JSON:
  SHA-256 `eddb4a4f6206593fa39f7e240f3a8340deacfc4546fb6c9714cc9b0dac5ce054`.

The tested implementation is therefore the committed parent plus the pinned
manifest/archive, not the parent SHA alone. The final evidence/report commit
contains that tested overlay and this document; its SHA is reported by the
task handoff rather than embedded recursively in its own contents.

## Final Batch runtime and resources

Superseding acceptance job `ee678a9a-f600-403b-b4a8-c801b37abf22`
succeeded with exit 0.
It used queue `rtrrl-cpu2-queue`, definition `rtrrl-cpu-job:14`, 4 vCPUs,
8,192 MiB, and compute environment `rtrrl-cpu2-ce` (`c7a.2xlarge`). Overall
container runtime was 769.076 seconds. Runtime versions were Python 3.12.13,
JAX/JAXLIB 0.10.0, Flax 0.12.7, Brax 0.14.2, backend `cpu`, device `cpu:0`.

| Workload | Result | Wall time | Peak RSS |
| --- | --- | ---: | ---: |
| strict parity, accelerated cases enabled | 214 passed | 41.92 s | 2,148,048 KiB |
| five directional finite differences | 5 passed | 4.80 s | 466,508 KiB |
| selected RTRRL/meta/legacy-builder `online_ac` | 36 passed | 85.22 s | 3,201,080 KiB |
| independent RTRRL | 11 passed | 22.64 s | 1,480,832 KiB |
| full head `online_ac` | 112 passed, 1 failed | 217.24 s | 5,126,092 KiB |
| full true-base `online_ac` | 105 passed, 1 failed | 215.59 s | 5,041,708 KiB |
| eager/JIT/oracle harness | succeeded | 12.41 s | 1,080,152 KiB |
| preserved/oracle probes, 2×2 comparison, source audit | succeeded | 4.93 s | 367,352 KiB |
| strict real Brax smoke | succeeded | 22.34 s | 1,376,552 KiB |
| ruff | passed | 0.14 s | 24,068 KiB |
| compileall | passed | 0.12 s | 16,352 KiB |

The maximum measured RSS was 4.89 GiB. The existing 8-GiB allocation remains
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

The final review-scope Pyright command reports 1 error, 0 warnings at head: the
same dynamic test import. The corresponding base command reports 133 errors,
0 warnings after adding the true-base `memorax/algorithms/rtrrl.py` monolith to
the compared scope. Diagnostics are canonicalized by normalized relative path,
severity, rule, and whitespace-normalized message; the moved
`rtrrl.py`/`rtrrl/__init__.py` paths share one logical identity. The
machine-enforced result has zero head-only canonical diagnostics.

Full-project Pyright remains non-green: head 480 errors/2 warnings versus base
653 errors/2 warnings. These raw totals are evidence only and do not decide the
regression gate. Exact commands, diagnostics, and exits are in the evidence
JSON. Ruff and compileall pass.

## Machine-enforced final gates

The collector emits gate details and exits nonzero when any gate fails; the
orchestration propagates that failure to `TASK12_OVERALL`. Final results:

- selected `online_ac`: exit 0, 36 passed;
- complete `online_ac`: head and base each fail only
  `tests/online_ac/test_standard_parity.py::test_standard_one_step_matches_legacy_every_exposed_leaf`;
  head-only failure nodeids: none;
- review-scope Pyright: head-only canonical diagnostics: none.

All three gates passed, so `TASK12_OVERALL=0`. The complete suites and
full-project Pyright remain visible rather than being treated as raw-success
requirements.

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
explicit-parameter LRU forward/carry and trace/update values. Source-native
PRNG and initialization differed across those runtimes. Each runtime also ran
both fixed-noise detached and reparameterized objectives: changing only
semantics produced gradient maxima of `1.0449076890945435` (location) and
`0.5879773795604706` (raw scale). Same-semantics cross-runtime controls are
reported separately. Structural AST evidence identifies
`stop_gradient(action)` versus `action`; it does not claim direct execution of
the nested source objective.

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

The command ran with working directory `/tmp/head/memo`. Its runtime provenance
gate found no loaded module under `/tmp/head/rtrrl` and confirmed
`memorax`, `memorax.algorithms.rtrrl.entrypoint`, and
`experiments.rtrrl_hopper.run` all resolved under `/tmp/head/memo`.

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
