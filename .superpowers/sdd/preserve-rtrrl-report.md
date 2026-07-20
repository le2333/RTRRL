# Preserve Original RTRRL — Implementation Report

## Status

The superseding decision is implemented. `rtrrl/rtrrl.py` is restored exactly
from functional base `5f7ff4e`; Memo owns all maintained compatibility helpers
and runtime execution. Authorized Batch job
`98ce6669-9be8-49f5-a66e-d57b09985f42` succeeded with exit 0. No complete RL
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

The local focused set covering audit, provenance, program, logging, and
preservation produced 48 passed. `git diff --check` and focused Ruff passed.

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
- actor raw-scale gradient maximum: `0.5879773795604706`;
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
`5d9a6af579ce55215cd2f12d2e652ae7f43cc10f4d7eccf694ebac89bb64f269`;
the script SHA-256 is
`db80e0b84de1b75af1501cbcdab07a413c48889fe500231412130ab6c4a52fb9`.

Inputs:

- head overlay archive SHA-256:
  `54a4044999ff299f52ef208b2efe6eacc75de8ef6a9f9f543f0fb583f7bae0dc`;
- corrected AAAI25 archive SHA-256:
  `f9a97fd54cb2786324ea83dc10078b4f9dbcc6e83bc2becf51c0d62d497b9740`.

Final submission:

```text
aws batch submit-job
  --job-name rtrrl-preserved-review-controls-rerun-20260721
  --job-queue rtrrl-cpu2-queue
  --job-definition rtrrl-cpu-job:14
  --container-overrides <bash orchestration; 4 vCPU; 8192 MiB>
```

Final job `98ce6669-9be8-49f5-a66e-d57b09985f42` succeeded. Earlier attempts
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

## Final Batch tests

- strict parity: 207 passed, 0 failed, 42.22 s, 2,145,452 KiB peak RSS;
- finite differences: 5 passed, 0 failed, 4.85 s, 466,616 KiB;
- selected RTRRL/meta/independent online AC: 36 passed, 0 failed, 85.41 s,
  3,196,132 KiB;
- independent RTRRL: 11 passed, 0 failed, 22.51 s, 1,477,888 KiB;
- numerical complete-state harness: exit 0;
- preserved/oracle probes, 2×2 semantic comparison, and source audit: exit 0;
- direct Memo real Brax smoke: exit 0;
- Ruff and compileall: exit 0.

Full `online_ac` remains non-green but unchanged in failure set: head has
112 passed/1 failed and true base has 105 passed/1 failed; both fail only
`test_standard_one_step_matches_legacy_every_exposed_leaf`. Full-project
Pyright remains diagnostic debt (head 465 errors/2 warnings, base 653/2).
Review-scope head Pyright has one existing dynamic-import error.

Machine-readable evidence:
`memo/docs/rtrrl-task12-evidence.json`, SHA-256
`cf817f8ffaa3d3e48dd07ee4d021cd1aa69768bcbeb206da4f4810a98fc21aba`.

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
