# Task 12 Implementation Report

> **SUPERSEDED 2026-07-20:** The external `rtrrl/rtrrl.py` delegate described
> below was reverted. It is now an exact `5f7ff4e` backup/reference, while the
> fresh real Brax smoke invokes Memo's runner directly. The authoritative
> preservation implementation report is
> `.superpowers/sdd/preserve-rtrrl-report.md`.

## Status

Review blockers are resolved. Superseding acceptance job
`8c9035b9-ad43-444b-9564-614c5edf96e8` succeeded with exit 0 on authorized
Batch. Strict parity, all five accelerated finite differences, selected
RTRRL/meta/legacy-builder tests, independent RTRRL, numerical harness, Brax
smoke, ruff, and compileall passed.

Full `online_ac` is not claimed green. True base `5f7ff4e` and final head each
have the same single failure; there are no head-only failures. The final
report is `memo/docs/rtrrl-numerical-parity-report.md`; exact evidence is
`memo/docs/rtrrl-task12-evidence.json`.

## Commit and baseline identities

- feature functional base: `5f7ff4e40e66da0b7df4f3edc9a928185ad73ae6`;
- docs-only feature root: `ffa90b5ae50b67bae1cbfe84c85eb0e21325eac3`;
- Task 10 comparison base only: `5a89953b5d09909b35c5016118dc11a1adb0dec2`;
- tested committed parent: `8e9dbace42dc9e75d4fb40505a66a78382a25e30`;
- tested implementation manifest SHA-256:
  `f33468f9e451a8a298716f9e896fb3e0944c70198384d30c78b99702e50acd20`;
- uploaded overlay archive SHA-256:
  `04138a897ad7cdc0e037a2b3ca212c602f1611ab77ff0ced271e517ddb070317`;
- orchestration script SHA-256:
  `cfe60c0b8d4b4b76f9686a329a33ab849966fcc8ff10e211cac8d01e0375630b`.

The submitted command verifies the downloaded orchestration script with
system `sha256sum -c`. The bound script hashes all archives with system
`sha256sum` and aborts before `tar -xf` on any mismatch. No external verifier
is downloaded.

The evidence/report commit contains the tested overlay. Its SHA is returned by
the task handoff; embedding it in the file contained by that commit would be
recursive.

## Final commands and results

The exact command, cwd, environment, exit, counts, duration, and RSS for every
run are stored in `memo/docs/rtrrl-task12-evidence.json`. The exact orchestration
source is committed as
`memo/tests/rtrrl_parity/task12_batch_verification.sh`.

Final Batch resources: `rtrrl-cpu2-queue`, `rtrrl-cpu-job:14`, 4 vCPU,
8,192 MiB, `rtrrl-cpu2-ce` on `c7a.2xlarge`. Runtime: Python 3.12.13,
JAX/JAXLIB 0.10.0, Flax 0.12.7, CPU.

- strict parity with `RTRRL_RUN_ACCELERATED_NUMERICS=1`: 226 passed,
  41.72 s, 2,142,944 KiB;
- separate five-case finite differences: 5 passed, 4.81 s, 466,476 KiB;
- selected online_ac: 36 passed, 85.35 s, 3,200,744 KiB;
- independent RTRRL: 11 passed, 22.61 s, 1,476,600 KiB;
- full head online_ac: 112 passed, 1 failed, 216.60 s, 5,124,492 KiB;
- full base online_ac: 105 passed, 1 failed, 212.72 s, 5,038,128 KiB;
- eager/JIT/oracle harness: exit 0, 12.38 s, 1,079,596 KiB;
- isolated preserved/oracle 2×2 comparison and source audit: exit 0;
- direct Memo strict Brax smoke: exit 0, 22.25 s, 1,382,480 KiB;
- ruff and compileall: exit 0.

Finite-difference cosine/relative-error pairs:

- `nu_log`: 1.00000012 / 0.000119965051;
- `theta_log`: 0.99999994 / 0.0000730149404;
- `gamma_log`: 1.0 / 0.000206126366;
- `B_real`: 1.0 / 0.0000286421237;
- `B_img`: 1.0 / 0.0000388202425.

## Review defects fixed

The true-base comparison invalidated two claims in the first report.

1. Full base `online_ac` had one failure, not 18. Eleven head-only failures
   were real review blockers. Root causes were a test helper that replaced the
   established permissive numeric/dtype/path contract with strict RTRRL
   assertions, and the experimental compatibility facade returning debug
   metrics instead of historical `aux=None` without emitting the old lox
   schema. Existing failing tests provided RED evidence. The helper contract
   and compatibility logging were restored. Final base/head failure sets are
   identical.
2. Three `with_logger` errors and 12 lazy-export warnings were branch
   introduced, not baseline. They were fixed. Review-scope pyright now has
   one error and zero warnings, the true-base `base.experiment` test import.

Full-project pyright is still diagnostic rather than green: head
465 errors/2 warnings, base 653 errors/2 warnings. This is reported exactly.

## Numerical and Brax evidence

Committed `task12_numerical_evidence.py` checks all 82 canonical state leaves,
including path, shape, dtype, and 73 floating/complex leaves. It runs after
unchanged acceptance assertions and does not modify them. The report and JSON
contain the full canonical path, shape, dtype, max absolute, relative, and ULP
measurement for all six eager/JIT/oracle comparisons.

Committed `task12_brax_smoke.py` calls Memo's `train_legacy` runner with strict
LRU, real hopper/spring, one training update, and 1,000 evaluation transitions.
It emitted and recorded historical train/eval metrics, finalized the logger,
and returned reward `49.92558288574219`.

## Concerns and acceptance boundary

- Full `online_ac` retains one base/head shared exact-zero legacy-standard
  failure and is not reported green.
- Full-project pyright retains broad repository debt and is not reported
  green.
- ULP values are leaf/runtime observations, not broad tolerance policy.
- Isolation and secret claims apply to this Task 12 diff/worktree only, not
  unrelated worktrees or the entire host.
- No complete RL environment ran locally.
