# Training/Evaluation Task 1 Report

## Scope and files changed

- `training-sdk/src/training_sdk/contract.py`
  - Raised `CONTRACT_VERSION` from 4 to 5.
  - Replaced environment stream count with the required non-negative `seed`.
  - Replaced `BudgetConfig` with `TrainingConfig` and `EvaluationConfig`, including
    whole-epoch, stream-round, chunk, and early-stop validation.
  - Split `RunConfig.budget` into `training` and `evaluation` and removed the
    validator whose ownership moved into `TrainingConfig`.
- `training-sdk/tests/test_contract.py`
  - Added the v5 contract tests and updated the shared RunConfig fixture.
- `training-sdk/tests/test_reporter.py`
  - Updated the RunConfig fixture to v5 fields and algorithm-only params.
- `training-sdk/tests/test_worker.py`
  - Updated the catalog fixture to contract version 5.
- `training-sdk/tests/test_aim_sink.py`
  - Updated the Aim assertion for the algorithm-only parameter fixture.

## Test and check results

| Command | Result | Full summary |
| --- | --- | --- |
| `uv run pytest training-sdk/tests/test_contract.py` (repository root) | FAIL | `pytest` was not found in the root environment: `Failed to spawn: pytest; program not found`. |
| `uv run pytest tests/test_contract.py` (`training-sdk`) | FAIL | uv could not read its shared cache: `C:\Users\le233\AppData\Local\uv\cache\sdists-v9\.git: Access denied`. |
| `uv sync --all-groups` (`training-sdk`) | NOT RUN | The required sandbox escalation was rejected because the machine instructions prohibit local dependency synchronization. |
| `uv run ruff check .; uv run pytest tests/test_contract.py tests/test_reporter.py tests/test_worker.py tests/test_aim_sink.py` (`training-sdk`) | FAIL | uv downloaded CPython and created `.venv`, but dependency resolution stopped before either command ran: `aimrocks==0.5.2` has no source distribution or wheel for Windows `win_amd64`. |
| `uvx ruff check .` (`training-sdk`) | FAIL | uv again could not read its shared cache (`sdists-v9\.git: Access denied`). |
| `git diff --check` | PASS | Exit code 0; no whitespace errors reported. |

The red test phase was written before production changes. Its focused pytest
invocation could not reach collection because the local Python environment was
unavailable, so its expected missing-import failure could not be observed.

## Commits

- `2db4224` — `test(contract): require split training and evaluation sections`
- `51e21a2` — `feat(contract): split training and evaluation run sections`

## Remote CI and push

`git push origin HEAD` was attempted and failed because the sandbox could not
connect to GitHub. A retry requesting external access was rejected because the
destination and export were not explicitly authorized by the environment.
Consequently, no `gh workflow run tests.yml` invocation was attempted and no
remote CI result is available.

## Self-review notes

- `EnvironmentConfig` exactly carries `id`, `backend`, `seed`, and optional
  `observed`; `num_envs` is rejected by the model's existing `extra="forbid"`
  policy.
- `TrainingConfig` owns all training stream divisibility checks, including
  `chunk_steps * num_envs` dividing both totals.
- `EvaluationConfig` uses the requested rollout `steps` and its own `num_envs`.
- All listed test fixtures now construct v5 `RunConfig` objects with separate
  `training` and `evaluation` fields and no training budget in `params`.

## Concerns

- The required pytest and ruff checks could not execute locally because the
  locked Aim dependency does not support Windows and the shared uv cache is
  inaccessible in the sandbox.
- Remote validation remains pending because GitHub push/workflow authorization
  was denied.
