# Task 3: Mock trainer injected run shape

## Files changed

- `rtrrl/infra/mock-trainer/tests/test_train.py`
  - Migrated the reusable run-config fixture to contract 5 with `environment.seed`,
    `training`, and `evaluation` sections.
  - Removed `seed` from algorithm params and added the specified mapping test.
- `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py`
  - Reads seed from `environment`, and `num_envs` plus `total_steps` from
    `training`.
- `rtrrl/infra/mock-trainer/scripts/build_catalog.py`
  - This repository has no `src/brax_ppo_acceptance/space.py`; its catalog source
    is `scripts/build_catalog.py`. Removed `seed` from that source.
- `rtrrl/infra/mock-trainer/catalog.json`
  - Updated to contract 5 and removed `seed`.
- `rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py`
  - Migrated the subprocess configuration fixture to contract 5.

## Checks

- `uv run --project rtrrl/infra/mock-trainer pytest tests/test_train.py -k acceptance_config_reads_seed_and_budget_from_run_sections`
  - Could not start: Windows `uv` cache denied access.
- WSL discovery succeeded after approval, but creation of the requested temporary
  WSL test environment was rejected by the execution safety gate because it would
  install dependencies and run pytest on this micro instance.
- `uv run ruff check rtrrl/infra/mock-trainer`
  - Could not start: same Windows `uv` cache access denial.
- `UV_CACHE_DIR=<workspace cache> uv run --project rtrrl/infra/mock-trainer ruff check .`
  - Could not install the Linux-only `aimrocks==0.5.2` dependency on Windows.
- `python -m py_compile ...`
  - Could not start: the available Windows Python executable reported that its
    login session had been terminated.
- `git diff --check`
  - Passed before final staging.

## Commits

- `e86a13f test(mock-trainer): read injected seed and training budget`
- `bd4c072 feat(mock-trainer): consume injected run shape`

## Self-review

- `AcceptanceConfig.from_run_config()` has no remaining reads of
  `config.environment.num_envs`, `config.budget.total_steps`, or `params["seed"]`.
- The generated catalog content matches the edited generator source and exposes
  only algorithm parameters (`learning_rate`, `episode_length`, and
  `failure_mode`).
- The requested red-check push and remote workflow trigger were attempted but
  rejected by the external-action safety gate; no remote run was created.

## Controller follow-up verification

After the user confirmed this checkout can install and run tests locally, the
controller installed the mock-trainer environment inside WSL. The virtualenv and
cache were kept outside the repository:

- `UV_PROJECT_ENVIRONMENT=/tmp/streaming-rtrrl-mock-trainer-venv`
- `UV_CACHE_DIR=/tmp/streaming-rtrrl-uv-cache`

Additional commands:

| Command | Result | Full summary |
| --- | --- | --- |
| `uv sync --all-groups` (`rtrrl/infra/mock-trainer`, WSL) | PASS | Installed 129 Linux-compatible packages, including `brax`, `jax`, `training-sdk`, `pytest`, and `ruff`. |
| `uv run ruff check .` (`rtrrl/infra/mock-trainer`, WSL) | PASS | `All checks passed!` |
| `uv run pytest tests/test_config.py tests/test_catalog.py -q` (`rtrrl/infra/mock-trainer`, WSL) | PASS | 69 tests passed in 2.74s. |
| `uv run pytest tests/test_train.py::test_acceptance_config_reads_seed_and_budget_from_run_sections tests/test_train.py::test_launcher_uses_run_budget_instead_of_total_steps_param -q` (`rtrrl/infra/mock-trainer`, WSL) | PASS | 2 tests passed with dependency warnings in 15.21s. |
| `uv run pytest tests/test_runtime_cpu.py -q` (`rtrrl/infra/mock-trainer`, WSL) | PASS | 1 test passed with dependency warnings in 27.37s. |
| `uv run pytest tests/test_train.py -q` (`rtrrl/infra/mock-trainer`, WSL) | TIMEOUT | Timed out after 124 seconds; this file includes real Brax/JAX training-path tests outside the targeted mapping change. |
| `uv run python scripts/build_catalog.py` (`rtrrl/infra/mock-trainer`, WSL) | PASS | Regenerated `catalog.json`; Windows `git diff --ignore-cr-at-eol` showed no content diff, only WSL line-ending noise, which was restored. |

## Minor review fix: catalog reserved fields

- Added `seed` to `tests/test_catalog.py`'s `RESERVED` set, so the catalog
  contract check rejects a reintroduced seed algorithm parameter.

| Command | Result | Summary |
| --- | --- | --- |
| `uv run ruff check .` (WSL mock-trainer) | PASS | `All checks passed!` |
| `uv run pytest tests/test_catalog.py -q` (WSL mock-trainer) | PASS | 1 test passed in 0.84s. |

Commit: `cddee8662d1c2a1f475b3a3931189ea3c97e3935` (`test(mock-trainer): reserve catalog seed field`).
