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
  `ca9f80f76c1b7df7a804eeba656fb02ca1ad11f3`;
- tested implementation manifest applied over that parent:
  SHA-256 `aeabbe826618b49b6228412a40e5b5b978cebef6ffa91b256689201913245c92`;
- exact uploaded overlay archive:
  SHA-256 `f015b6010aaeb892d09a9049857cee3017c4728dec32e2125932ec82948ac26c`;
- executed orchestration script:
  SHA-256 `41082ca2e5915ea5afe34e06d88e043ffd7ec583f4a9826793cc6b2468f7b3f6`;
- resulting evidence JSON:
  SHA-256 `e31626db4fabda003ee5fe9e7431d3683737fb2b60d7fad4937b6219a9fd0f32`.

The tested implementation is therefore the committed parent plus the pinned
manifest/archive, not the parent SHA alone. The final evidence/report commit
contains that tested overlay and this document; its SHA is reported by the
task handoff rather than embedded recursively in its own contents.

Before extraction, submitted expected and downloaded actual SHA-256 values
matched for head
`f015b6010aaeb892d09a9049857cee3017c4728dec32e2125932ec82948ac26c`,
base
`702ff4fdd485cd537ffeef599046b93fcc1510268adbaa7ec24665bb008a3a5f`,
and oracle
`f9a97fd54cb2786324ea83dc10078b4f9dbcc6e83bc2becf51c0d62d497b9740`.
The submitted Batch command first binds the downloaded orchestration script to
its committed SHA-256 with system `sha256sum -c`. That script then computes all
three archive digests with system `sha256sum`, compares them in shell before
the first `tar -xf`, and uses literal inline Python only to serialize evidence.
No mutable external verifier is downloaded or executed.

## Final Batch runtime and resources

Superseding acceptance job `83804547-56dc-4ff5-bd31-e9c11e141230`
succeeded with exit 0.
It used queue `rtrrl-cpu2-queue`, definition `rtrrl-cpu-job:14`, 4 vCPUs,
8,192 MiB, and compute environment `rtrrl-cpu2-ce` (`c7a.2xlarge`). Overall
container runtime was 753.801 seconds. Runtime versions were Python 3.12.13,
JAX/JAXLIB 0.10.0, Flax 0.12.7, Brax 0.14.2, backend `cpu`, device `cpu:0`.

| Workload | Result | Wall time | Peak RSS |
| --- | --- | ---: | ---: |
| strict parity, accelerated cases enabled | 226 passed | 40.66 s | 2,141,628 KiB |
| five directional finite differences | 5 passed | 4.65 s | 466,696 KiB |
| selected RTRRL/meta/legacy-builder `online_ac` | 36 passed | 81.58 s | 3,180,048 KiB |
| independent RTRRL | 11 passed | 21.93 s | 1,474,824 KiB |
| full head `online_ac` | 113 passed | 212.52 s | 5,114,968 KiB |
| full true-base `online_ac` | 105 passed, 1 failed | 209.70 s | 5,034,444 KiB |
| eager/JIT/oracle harness | succeeded | 11.93 s | 1,071,656 KiB |
| preserved/oracle probes, 2×2 comparison, source audit | succeeded | 5.00 s | 367,180 KiB |
| strict real Brax smoke | succeeded | 21.55 s | 1,375,656 KiB |
| ruff | passed | 0.13 s | 23,864 KiB |
| compileall | passed | 0.12 s | 16,268 KiB |

The maximum measured RSS was 4.88 GiB. The existing 8-GiB allocation remains
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

The true base retains its pre-existing failure in
`test_standard_one_step_matches_legacy_every_exposed_leaf`, while the head now
passes all 113 tests. The root cause was comparing the legacy eager step with
the composed program inside `lax.scan`; scan lowering changed critic Jacobian
evaluation order by one or a few float32 ULPs. The corrected test accesses both
direct kernels through test-only debug interfaces and requires exact equality
for gradients, traces, ObGD state, parameter application, and final state.
Production scan-to-scan parity remains covered separately. No tolerance was
relaxed, and `head_only_failures = []`.

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

The final review-scope Pyright command checks the entire new
`memorax/algorithms/rtrrl` package plus changed experiment, independent-RTRRL,
online-AC, consumer-test, and package-export paths. The true-base command checks
the old `memorax/algorithms/rtrrl.py` monolith and corresponding paths. It
reports 1 error, 0 warnings at head and 190 errors, 0 warnings at base.

Diagnostics are canonicalized by normalized relative path, severity, rule, and
whitespace-normalized message; every new `rtrrl/*.py` module maps to the old
monolith's logical identity. Comparison uses `Counter` multiplicities rather
than sets. The machine-enforced result has zero head-only diagnostics, including
duplicate occurrences.

Full-project Pyright remains non-green: head 512 errors/2 warnings versus base
653 errors/2 warnings. These raw totals are evidence only and do not decide the
regression gate. Exact commands, diagnostics, and exits are in the evidence
JSON. Ruff and compileall pass.

## Machine-enforced final gates

The collector emits gate details and exits nonzero when any gate fails; the
orchestration propagates that failure to `TASK12_OVERALL`. Final results:

- downloaded archives: head, base, and oracle expected/actual SHA-256 values
  all match before extraction;
- selected `online_ac`: valid JUnit, exit 0, 36 passed;
- complete `online_ac`: both JUnit documents are valid; head exits 0 with
  113 passes, while base exits 1 and records only
  `tests/online_ac/test_standard_parity.py::test_standard_one_step_matches_legacy_every_exposed_leaf`;
  head-only failure/error nodeids: none;
- review-scope Pyright: head-only canonical diagnostic multiplicities: none.

Pytest exit semantics are explicit: 0 is accepted only with no JUnit
failures/errors; 1 only with recorded outcomes; 2 or higher is always rejected.
Missing, malformed, or internally inconsistent JUnit is rejected.

All four gates passed, so `TASK12_OVERALL=0`. The head complete suite is green;
the true-base failure and full-project Pyright diagnostics remain visible.

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
