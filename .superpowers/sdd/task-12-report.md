# Task 12 Implementation Report

## Status

Final strict parity, independent RTRRL, CPU eager/JIT comparison, configuration
audit, and real strict-LRU Brax integration succeeded with fresh evidence.
Full `memo/tests/online_ac` remains non-green: head has 12 failures, all also
present at base `5a89953`; base has six additional meta failures.  No
Task-12-owned production defect was found, so Task 12 changes documentation
only.

The local Task 12 commit SHA is reported in the final handoff because this
ignored implementation report is finalized before the commit is created.

## Commands and results

### Local, non-RL checks

- `python3 rtrrl/rtrrl.py --compat-action audit`
  - exit 0; 697 discovered/accepted; 0 unsupported, unknown, deprecated no-op,
    or invalid; 1.64 s; 18,412 KiB peak RSS.
- repository/worktree status, tracked credential regex audit, delegation/core
  source audit, and branch `git diff --check`.
  - Task worktree was clean before report creation.
  - No credential pattern was found.
  - Other worktrees contain pre-existing changes and were not modified.
  - `rtrrl/rtrrl.py` delegates and contains no old training mathematics.
- No full RL environment ran locally.

### Static gates

- Task 12 changes no Python source, so task-owned pyright input is empty.
- Expanded branch-wide ruff over Python files changed since `5a89953`: passed.
- Expanded branch-wide `compileall`: passed.
- Expanded branch-wide pyright, run from `memo` with its configured `.venv`:
  non-green with four existing errors and 12 dynamic-export warnings.  The
  errors are three `with_logger` annotations in
  `experiments/base/experiment.py` and the long-standing
  `base.experiment` test import path in `tests/test_independent_rtrrl.py`.
  They are not Task-12-owned and were not changed.
- IDE diagnostics for the new report: no errors.
- Final staged `git diff --check` is run immediately before commit.

### Authorized Batch test matrix

Queue/definition/resources for final jobs:
`rtrrl-cpu2-queue` / `rtrrl-cpu-job:14`, 4 vCPU, 8,192 MiB,
`c7a.2xlarge` compute environment.

Runtime: Python 3.12.13, JAX/JAXLIB 0.10.0, Flax 0.12.7, CPU.

- Job `c73f26ab-c908-4358-852f-de721986833c`:
  `pytest memo/tests/rtrrl_parity -q` with passive comparison measurement.
  - 200 passed, five authorized directional finite-difference skips;
    44.83 s; 2,145,140 KiB peak RSS.
- Job `acad4da8-c1ba-45bc-8de1-10444fd437fd`:
  - RTRRL/meta/independent-selected `online_ac`: 34 passed, two shared
    base/head legacy-characterization failures; 82.632 s; 3,126,540 KiB.
  - `pytest memo/tests/test_independent_rtrrl.py -q`: 11 passed; 25.545 s;
    1,484,776 KiB.
  - full head `memo/tests/online_ac`: 101 passed, 12 failed; 210.409 s;
    4,880,076 KiB.
  - full base `memo/tests/online_ac`: 90 passed, 18 failed; 191.380 s;
    4,679,072 KiB.
  - exact failure-set comparison: no head-only failures; base-only failures
    are the six old `test_meta_parity.py` cases restored by this branch.
  - Job exit 1 is expected from the explicitly retained diagnostics and an
    initial invalid pytest plugin path; the parity command was rerun
    successfully in the preceding job.
- Job `838ef324-dead-4a65-b035-269e810109a1`:
  canonical complete one-step and terminal three-step CPU eager/JIT/oracle
  comparison.
  - succeeded; exact path/shape/dtype/key checks; 13.42 s; 1,079,284 KiB.
  - one-step JIT-vs-eager max abs `3.7252903e-09`, max rel
    `2.5525941e-07`, max ULP 4 at the exact TD actor Adam `nu` kernel.
  - three-step JIT-vs-eager max abs `5.9604645e-08`, max rel
    `4.3495504e-07`, max ULP 6 at that exact kernel.

### Real Brax integration

Final job `64b129a9-964e-4293-a724-c1cd25bdd839`:

- public `rtrrl.train_rtrrl` legacy entrypoint;
- explicit `aaai25_strict_lru`;
- real `brax-hopper`, Brax 0.14.2, spring backend;
- one compiled real training transition/update (minimum valid epoch);
- 1,000 evaluation transitions, enough for a completed episode;
- historical train and eval metrics emitted at logger step 1;
- `eval/rewards = eval/best_eval_reward = 49.92558288574219`;
- 22.34 s, 1,392,252 KiB peak RSS, exit 0.

Calibration jobs not used as acceptance evidence:

- `7da88cd4-fd52-49c2-bf23-2a05142794a2`: selected the experimental profile
  through an over-complete compatibility dataclass and used a falsey recorder.
- `0cabca63-193d-4036-bd01-2452ce075ee6`: strict one-step training succeeded,
  but two evaluation transitions were insufficient to emit an episode return.
- JIT measurement jobs `81951c02-5f46-4a48-a17b-c44b0132029d`,
  `943c4ad2-cd95-4dc6-99a1-beabc7de02e6`, and
  `a9ca0974-5821-4bfb-9810-44bc366c3792` calibrated only the standalone
  measurement harness (test-package import, environment pytree registration,
  and nested-tree flattening).  They did not expose production defects.

## Numerical and acceptance evidence

The complete report at `memo/docs/rtrrl-numerical-parity-report.md` records:

- fixture source commit/runtime/protocol;
- module-by-module abs/rel/ULP observations;
- complete eager/JIT one-step and terminal three-step results;
- all 697 runtime config counts;
- historical logging and real smoke metric keys;
- peak RSS and 8-GiB/4-vCPU recommendation;
- explicit CTRNN/no-RNN exclusions;
- exact full-`online_ac` base/head failure sets;
- repository isolation, delegation, secret scan, duplicate-core audit, and
  no-local-full-RL boundary.

No tree-wide ULP policy was introduced.  Named ULP values are measurements for
the exact recorded leaf/runtime only.

## Concerns

- Full `online_ac` is not green and is not reported as green.  Its 12 head
  failures are unchanged base failures in old generic RTRRL/StreamAC/standard
  characterization under the current JAX runtime.
- Other pre-existing worktrees are dirty, so only the narrower claim that Task
  12 itself modified the isolated worktree is supported.
- The five directional finite-difference cases remain explicitly opt-in and
  skipped in the complete parity command.
- Expanded branch-wide pyright remains non-green with the four exact
  pre-existing errors recorded above; Task 12 does not claim a green result.

## Report path

`memo/docs/rtrrl-numerical-parity-report.md`
