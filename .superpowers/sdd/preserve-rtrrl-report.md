# Preserve Original RTRRL — Implementation Report

## Status

The superseding decision is implemented. `rtrrl/rtrrl.py` is restored exactly
from functional base `5f7ff4e`; Memo owns all maintained compatibility helpers
and runtime execution. Authorized Batch job
`8c9035b9-ad43-444b-9564-614c5edf96e8` succeeded with exit 0. No complete RL
environment ran locally.

## RED / GREEN preservation contract

RED command:

```text
memo/.venv/bin/pytest -q memo/tests/rtrrl_parity/test_legacy_entrypoint.py::test_external_rtrrl_script_is_exact_preserved_original
```

Before restoration it failed as intended:

```text
actual   5738c0baa6c4988a0c54acdb38bc7b8336eb36bad47ddd825584ed83d6f0e9d0
expected f8aedcd9c315445af93e7f4a2475c50e9828c5188bd487ed39b85d7ec7da61cf
```

GREEN/restore command:

```text
git restore --source=5f7ff4e40e66da0b7df4f3edc9a928185ad73ae6 -- rtrrl/rtrrl.py
memo/.venv/bin/pytest -q memo/tests/rtrrl_parity/test_legacy_entrypoint.py::test_external_rtrrl_script_is_exact_preserved_original
git diff --exit-code 5f7ff4e40e66da0b7df4f3edc9a928185ad73ae6 -- rtrrl
```

Result: 1 passed and empty `rtrrl/` diff. Frozen identities:

- byte SHA-256:
  `f8aedcd9c315445af93e7f4a2475c50e9828c5188bd487ed39b85d7ec7da61cf`;
- canonical AST SHA-256:
  `46d3b46a45ab72c3a9550763ae6f6fb0c5bda49a103731e953183f707e388ee9`.

The full refactored helper/preservation file subsequently passed 21 tests.
Mutable historical `RTRRLParams` is tested by executing its exact preserved AST
with lightweight data-carrier dependencies, avoiding an import-time Brax
requirement. Memo normalization, build description, mock metrics, CLI override,
and repository audit tests call Memo helpers directly and make no delegation
claim.

## Reviewer-control RED / GREEN

The review contracts were added before their implementations:

- RED: `test_preserved_original_audit.py` failed collection because
  `compare_probes` did not exist; the source-audit module also did not exist.
  GREEN: two contracts pass, covering precise AST normalization, structural
  actor evidence, and synthetic 2×2 result classification.
- RED: the new provenance assertions in `test_program_contract.py` and
  `test_logging_compat.py` both resolved `logging_util` to
  `rtrrl/logging_util.py`. GREEN: external path injection was removed, Memo's
  root was added explicitly, and each isolated provenance contract resolves
  `memo/logging_util.py`.
- Batch RED `b90fbf31-ac9c-4b2f-bf29-880d0fe683ea`: 206 strict tests passed
  and the source-audit contract failed because its local absolute oracle path
  was absent. GREEN `98ce6669-9be8-49f5-a66e-d57b09985f42`: the explicit
  `RTRRL_AAAI25_ROOT=/tmp/oracle` contract produced 207/207 strict passes.
- Final-gate RED: `test_task12_evidence_gates.py` failed collection because
  `evaluate_gates` and smoke provenance APIs did not exist. Hardened-gate RED
  later failed because JUnit artifact parsing and archive verification APIs did
  not exist. GREEN: 18 unit contracts now cover selected-suite exit, valid and
  malformed/missing JUnit, exits 0/1/2+, failure/error nodeids, canonical
  monolith/package paths, diagnostic multiplicity, collector nonzero exit,
  archive mismatch, and smoke provenance.
- Pyright-parser RED: node bootstrap text before JSON caused a missing
  full-head diagnostic summary. GREEN: the parser extracts the structured
  Pyright payload after arbitrary prefix noise; definitive Batch evidence
  records head 512 errors/2 warnings and base 653/2.

The local focused set covering audit, provenance, program, logging, and
preservation produced 48 passed. `git diff --check` and focused Ruff passed.
The final direct-trust focused command produced 24 passed; no RL environment
was executed locally.

## Runtime and smoke decision

`task12_brax_smoke.py` now normalizes with
`memorax.algorithms.rtrrl.entrypoint.normalize_legacy_invocation` and invokes
`experiments.rtrrl_hopper.run.train_legacy` directly. Its Batch payload used
strict LRU, one real Hopper/spring training transition, and 1,000 evaluation
transitions. It emitted one historical metric record at step 1, finalized the
logger, and returned `49.92558288574219`.

`PYTHONPATH` for Memo parity/smoke contains `/tmp/head/memo`, not the external
`rtrrl/` project. Among the two Docker image workflows, `memo/**` triggers
`build-memo-image.yml` and does not trigger `build-rtrrl-image.yml`, whose path
filter requires `rtrrl/**`. Separately, `memo-ci.yml` also triggers on
`memo/**` for pushes and pull requests.

The final smoke runs from `/tmp/head/memo`. Runtime provenance checks every
loaded module origin, finds none under `/tmp/head/rtrrl`, and confirms the
Memo package, RTRRL entrypoint, and experiment runner all resolve under
`/tmp/head/memo`.

## Numerical audit

Oracle identity:
`4301943c349171d828d0fcf3e40944c286451415`.

The preserved process used Python 3.12.13, JAX/JAXLIB 0.5.0, Flax 0.10.2, CPU.
The oracle process separately used Python 3.12.13, JAX/JAXLIB 0.4.38,
Flax 0.10.2, CPU. Deterministic probes used explicit float32 inputs, two LRU
steps, fixed trace tensors, fixed actor noise, and a 2×2 actor control: both
detached and reparameterized semantics execute in each pinned runtime.

Measured exact (maximum absolute difference 0):

- explicit LRU parameter tree;
- initial carry;
- first/second LRU carry and output;
- accumulated eligibility trace;
- trace-derived update;
- actor objective value.

Within each runtime, changing only actor semantics measured:

- actor location gradient maximum: `1.0449076890945435`;
- actor raw-scale gradient maximum: `0.5879773795604706`.

Across the two source-native runtimes, initialization measured:

- source-native uint32 PRNG split maximum: `2793280701`;
- source-native LRU parameter maximum: `5.048454284667969`;
- source-native first/second LRU output maxima:
  `1.8039704756811261` / `1.5446054488420486`.

The native initialization differences are runtime-sensitive because the two
pinned JAX releases produce different PRNG values. They are reported rather
than attributed only to source. Explicit equal parameters isolate and confirm
the LRU/trace equations.

Cross-runtime comparisons hold actor semantics fixed; those controls determine
whether the two pinned JAX versions add a gradient difference. A separate
structural AST audit finds the preserved `log_prob` argument is
`stop_gradient(action)`, whereas AAAI25's is `action`. The AST result is source
structure, not direct execution of the nested training closure. Combined with
the 2×2 synthetic objective, it supports the conclusion that no preserved
option disables detachment and that the preserved path differs for actor
gradients and downstream trace/optimizer state.

The cross-runtime same-semantic maxima were exactly zero for actor objective,
location gradient, and raw-scale gradient under both detached and
reparameterized semantics. Both runtimes independently reproduced the same
within-runtime semantic deltas above.

AST normalization is also recorded exactly: `traces.py` becomes equal after
ordinary leading-docstring removal; `models/online_lru.py` does not. Both
become equal only after removing every standalone string expression statement,
including the descriptive class-body string following field declarations.

Closest options:

- `align_action_logprob=False` preserves AAAI25 forward sampled-action values,
  but not its gradient;
- `update_trace_before_td=False` reproduces AAAI25 incoming-trace order.

Intentional differences:

- `align_action_logprob=True` clips action before log-probability;
- `update_trace_before_td=True` uses fresh/current trace;
- `run_name` affects logging only.

Unverified complete preserved-script branches: terminal/multi-env scans,
normalization, Dutch traces, average reward, discrete actor, CTRNN/no-RNN,
dropout, MLP actor, rendering/evaluation, and a full preserved optimizer scan.
Complete-step parity is not claimed. Detailed evidence and verdict:
`memo/docs/rtrrl-preserved-original-comparison.md`.

## Batch commands and jobs

Final orchestration is committed as
`memo/tests/rtrrl_parity/task12_batch_verification.sh`. The implementation
manifest SHA-256 embedded in evidence is
`f33468f9e451a8a298716f9e896fb3e0944c70198384d30c78b99702e50acd20`;
the script SHA-256 is
`cfe60c0b8d4b4b76f9686a329a33ab849966fcc8ff10e211cac8d01e0375630b`.
The submitted command verifies that script hash with system `sha256sum -c`;
the bound script computes archive hashes with system `sha256sum` and compares
them in shell before extraction. No external verifier is downloaded.

Inputs:

- head overlay archive SHA-256:
  `04138a897ad7cdc0e037a2b3ca212c602f1611ab77ff0ced271e517ddb070317`;
- true-base archive SHA-256:
  `702ff4fdd485cd537ffeef599046b93fcc1510268adbaa7ec24665bb008a3a5f`;
- corrected AAAI25 archive SHA-256:
  `f9a97fd54cb2786324ea83dc10078b4f9dbcc6e83bc2becf51c0d62d497b9740`.

Final submission:

```text
aws batch submit-job
  --job-name rtrrl-direct-trust-anchor-final-20260721
  --job-queue rtrrl-cpu2-queue
  --job-definition rtrrl-cpu-job:14
  --container-overrides <bash orchestration; 4 vCPU; 8192 MiB>
```

Final job `8c9035b9-ad43-444b-9564-614c5edf96e8` succeeded. Earlier attempts
are retained as diagnostic history:

- `0dc0aa1b-54c1-43cc-bfeb-e5694ed5bff9`: failed because the uploaded oracle
  archive was created from the wrong repository;
- `bd1847e5-6888-4401-b2b9-0538309d73cb`: correctly exposed that native
  cross-JAX PRNG initialization cannot be asserted equal; the probe was then
  corrected to report native initialization separately and compare equations
  under explicit equal parameters.
- `b90fbf31-ac9c-4b2f-bf29-880d0fe683ea`: the new source-audit test used a
  workstation-only absolute oracle path. It failed one strict test while the
  standalone 2×2 comparison and source audit passed. The test now accepts
  `RTRRL_AAAI25_ROOT`, and the successful rerun sets it to `/tmp/oracle`.
- `99ed8c30-1503-48f8-8a4d-ec9c46bfe4a6`: all new machine gates passed, but
  Pyright's one-time node bootstrap prefix prevented the collector from
  summarizing the full-head JSON count. The parser received a failing unit
  contract and was fixed before rerun `ee678a9a`.
- `17ab0a34-baf3-40c1-8b20-ff58d009ff97`: all hardened gates and archive
  checks passed, but the evidence identity still named the prior committed
  parent. The identity was corrected to `517e073` before final job `f2a0d8ba`.
- `f2a0d8ba-bdc2-44fb-85c2-890cfc99989f`: gates and identities passed, but
  archive validation depended on a mutable downloaded verifier. The final job
  removes that dependency and anchors both orchestration and archive hashes.

## Final Batch tests

- strict parity: 226 passed, 0 failed, 41.72 s, 2,142,944 KiB peak RSS;
- finite differences: 5 passed, 0 failed, 4.81 s, 466,476 KiB;
- selected RTRRL/meta/independent online AC: 36 passed, 0 failed, 85.35 s,
  3,200,744 KiB;
- independent RTRRL: 11 passed, 0 failed, 22.61 s, 1,476,600 KiB;
- numerical complete-state harness: exit 0;
- preserved/oracle probes, 2×2 semantic comparison, and source audit: exit 0;
- direct Memo real Brax smoke: exit 0;
- Ruff and compileall: exit 0.

Machine gates all passed: all three downloaded archive hashes matched before
extraction; selected `online_ac` had valid JUnit and exited 0; full head/base
JUnit failure and error nodeid multisets have no head-only entries; canonical
review-scope Pyright `Counter` comparison has no head-only diagnostics. Full
`online_ac` remains non-green: head has
112 passed/1 failed and true base has 105 passed/1 failed; both fail only
`test_standard_one_step_matches_legacy_every_exposed_leaf`. Full-project
Pyright remains diagnostic debt (head 512 errors/2 warnings, base 653/2).
Expanded review scope is head 1 error versus base 190, with the one head diagnostic
present at base after path/message canonicalization.

Machine-readable evidence:
`memo/docs/rtrrl-task12-evidence.json`, SHA-256
`fe6fb72f000a03f0cf5b4b09f1796f052f6161a7ded4a87b84b3bb7b061c945a`.

## Concerns

- The preserved script is intentionally duplicate backup code and remains
  numerically non-equivalent to AAAI25 actor-gradient semantics.
- Cross-version native PRNG/initialization differences are measured but are not
  generalized beyond the pinned runtimes.
- Full preserved-script training, unsupported branches, full `online_ac`, and
  project-wide Pyright are not claimed green.
- No image was built locally. On integration to `main`, only the Memo Docker
  image workflow is eligible among the two image workflows; Memo CI is also
  eligible for `memo/**`.
