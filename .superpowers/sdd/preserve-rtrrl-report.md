# Preserve Original RTRRL — Implementation Report

## Status

The superseding decision is implemented. `rtrrl/rtrrl.py` is restored exactly
from functional base `5f7ff4e`; Memo owns all maintained compatibility helpers
and runtime execution. Authorized Batch job
`a5f1b64e-3ad3-4994-a107-6ca4241b182d` succeeded with exit 0. No complete RL
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

## Runtime and smoke decision

`task12_brax_smoke.py` now normalizes with
`memorax.algorithms.rtrrl.entrypoint.normalize_legacy_invocation` and invokes
`experiments.rtrrl_hopper.run.train_legacy` directly. Its Batch payload used
strict LRU, one real Hopper/spring training transition, and 1,000 evaluation
transitions. It emitted one historical metric record at step 1, finalized the
logger, and returned `49.92558288574219`.

`PYTHONPATH` for Memo parity/smoke contains `/tmp/head/memo`, not the external
`rtrrl/` project. The main-branch workflow path filters remain separate:
`memo/**` triggers only `build-memo-image.yml`; `build-rtrrl-image.yml` requires
an `rtrrl/**` diff, which final verification forbids.

## Numerical audit

Oracle identity:
`4301943c349171d828d0fcf3e40944c286451415`.

The preserved process used Python 3.12.13, JAX/JAXLIB 0.5.0, Flax 0.10.2, CPU.
The oracle process separately used Python 3.12.13, JAX/JAXLIB 0.4.38,
Flax 0.10.2, CPU. Deterministic probes used explicit float32 inputs, two LRU
steps, fixed trace tensors, fixed actor noise, and source-specific actor
differentiation.

Measured exact (maximum absolute difference 0):

- explicit LRU parameter tree;
- initial carry;
- first/second LRU carry and output;
- accumulated eligibility trace;
- trace-derived update;
- actor objective value.

Measured different:

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

The decisive source-level mismatch is not runtime-sensitive:
the preserved script always applies `stop_gradient` to the sampled action
before `log_prob`; AAAI25 differentiates through the reparameterized sample.
No preserved option disables detachment. Thus the preserved path is
numerically inconsistent with AAAI25 for actor gradients and downstream
trace/optimizer state.

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
`a2c062212511c4cd59dedb3cfe355c1246521fc055cc1b2ab0f6ffb79ec030b2`;
the script SHA-256 is
`31099c6020e510f65c570d7d263ddabfcf2605494d7e68adcbf74ab5877fb300`.

Inputs:

- head overlay archive SHA-256:
  `8968497929847c9927e4828ac73611b5a0cf10e91d54d4344dfb89faf70974dd`;
- corrected AAAI25 archive SHA-256:
  `f9a97fd54cb2786324ea83dc10078b4f9dbcc6e83bc2becf51c0d62d497b9740`.

Final submission:

```text
aws batch submit-job
  --job-name rtrrl-preserved-original-final-20260720
  --job-queue rtrrl-cpu2-queue
  --job-definition rtrrl-cpu-job:14
  --container-overrides <bash orchestration; 4 vCPU; 8192 MiB>
```

Final job `a5f1b64e-3ad3-4994-a107-6ca4241b182d` succeeded. Two earlier attempts
are retained as diagnostic history:

- `0dc0aa1b-54c1-43cc-bfeb-e5694ed5bff9`: failed because the uploaded oracle
  archive was created from the wrong repository;
- `bd1847e5-6888-4401-b2b9-0538309d73cb`: correctly exposed that native
  cross-JAX PRNG initialization cannot be asserted equal; the probe was then
  corrected to report native initialization separately and compare equations
  under explicit equal parameters.

## Final Batch tests

- strict parity: 203 passed, 0 failed, 52.35 s, 2,158,956 KiB peak RSS;
- finite differences: 5 passed, 0 failed, 6.13 s, 468,508 KiB;
- selected RTRRL/meta/independent online AC: 36 passed, 0 failed, 109.08 s,
  3,316,524 KiB;
- independent RTRRL: 11 passed, 0 failed, 28.41 s, 1,494,664 KiB;
- numerical complete-state harness: exit 0;
- preserved/oracle probes and semantic comparison: exit 0;
- direct Memo real Brax smoke: exit 0;
- Ruff and compileall: exit 0.

Full `online_ac` remains non-green but unchanged in failure set: head has
112 passed/1 failed and true base has 105 passed/1 failed; both fail only
`test_standard_one_step_matches_legacy_every_exposed_leaf`. Full-project
Pyright remains diagnostic debt (head 465 errors/2 warnings, base 653/2).
Review-scope head Pyright has one existing dynamic-import error.

Machine-readable evidence:
`memo/docs/rtrrl-task12-evidence.json`, SHA-256
`8a97bfb8d9f4872c88b9bdfe40cc75f1e580dbe04d3e05e81f4e5013b36ec31e`.

## Concerns

- The preserved script is intentionally duplicate backup code and remains
  numerically non-equivalent to AAAI25 actor-gradient semantics.
- Cross-version native PRNG/initialization differences are measured but are not
  generalized beyond the pinned runtimes.
- Full preserved-script training, unsupported branches, full `online_ac`, and
  project-wide Pyright are not claimed green.
- No image was built locally. On integration to `main`, only the Memo workflow
  is eligible because final `rtrrl/**` diff is empty.
