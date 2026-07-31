# Task 4 report: entries consume injected run fields

## Changes

- Removed sampled `seed` from the three Memo entry `SPACE` mappings. Their
  builds now receive `training`, and their runs send the injected environment
  seed plus training/evaluation run shape to `drive()`.
- Removed sampled `seed`, `scan_steps`, `eval_envs`, and `patience` from the
  AAAI entry. `settings()` and `parameters()` now receive environment,
  training, and evaluation config objects; chunking and early-stop settings
  come from the training configuration.
- Updated entry tests to use contract-v5 environment/training/evaluation
  configuration and reserve the injected field names.
- Updated `rtrrl/catalog.json` to contract 5 and removed the four AAAI runtime
  fields.

## Commands and results

| Command | Result |
| --- | --- |
| `wsl.exe bash -lc '... uv run pytest tests/test_entry.py -q'` | Blocked before execution: WSL instance access was denied by the sandbox; its escalated retry was rejected because the micro-instance policy forbids pytest. |
| `uv run ruff check memo rtrrl/entries rtrrl/tests` | Could not spawn `ruff` from the repository root. |
| `uv run --project memo ruff check entries tests; uv run --project rtrrl ruff check entries tests` | Could not resolve the Windows environments: `aimrocks==0.5.2` has no Windows wheel. |
| `cd rtrrl; uv run --no-sync python scripts/build_catalog.py` | Could not run the generator locally: the incomplete Windows environment lacks `training_sdk` / `pydantic`. The catalog diff was applied to the generator's deterministic, sorted output shape and checked structurally. |
| `git diff --check` | Passed. |
| PowerShell JSON consistency check | Passed: catalog contract is 5 and `rtrrl_aaai` declares none of `seed`, `scan_steps`, `eval_envs`, or `patience`. |

## Commits

- `99290f8a88efbc841cfa3706037c22c435c10151` — `test(entries): injected run fields are not parameters`
- `2df1f3b15c70b6f329baf74dec0995df351d6409` — `feat(entries): consume injected seed and run shape`

## Self-review

- Confirmed no entry still reads `config.budget`, sampled `seed`, or
  `environment.num_envs` for agent stream count.
- Confirmed Memo drive calls use training total/epoch/env count and evaluation
  steps, while AAAI converts training chunk and early-stop settings exactly at
  the entry boundary.
- Confirmed the catalog is contract v5 with only sampler-owned AAAI fields.

## Concerns

- Remote GitHub Actions have not been triggered from this agent: sandbox review
  rejected the requested push/workflow dispatch as an unverified external
  mutation and possible billed image build. The controller should run the
  prescribed remote red/green checks.
- Local pytest/Ruff/catalog generation could not execute on this Windows micro
  instance for the reasons recorded above.

## Controller follow-up verification

After the user confirmed this checkout can install and run tests locally, the
controller ran Task 4 checks in WSL. Virtualenvs and cache were kept outside the
repository:

- `UV_PROJECT_ENVIRONMENT=/tmp/streaming-rtrrl-memo-venv`
- `UV_CACHE_DIR=/tmp/streaming-rtrrl-uv-cache`

Additional commands:

| Command | Result | Full summary |
| --- | --- | --- |
| `uv sync --group development` (`memo`, WSL) | PASS | Installed the memo development environment and local `training-sdk` dependency. |
| `uv run --group development ruff check entries tests/test_entries.py` (`memo`, WSL) | PASS | `All checks passed!` |
| `uv run --group development pytest tests/test_entries.py -q` (`memo`, WSL) | PASS | 14 tests passed with dependency warnings. |
| `uv sync --group dev` (`rtrrl`, WSL) | BLOCKED | `pytinyrenderer==0.0.14` failed to build because WSL lacks `x86_64-linux-gnu-g++`; this is a full fork dependency, not needed by the lightweight entry mapping tests. |
| `PYTHONPATH='.:../training-sdk/src' uv run --project ../memo --group development pytest tests/test_entry.py -q` (`rtrrl`, WSL) | PASS | 11 tests passed in 0.60s using the memo development environment plus rtrrl/training-sdk import paths. |
| `PYTHONPATH='.:../training-sdk/src' uv run --project ../memo --group development ruff check entries/rtrrl_aaai.py tests/test_entry.py scripts/build_catalog.py` (`rtrrl`, WSL) | PASS | `All checks passed!` |
| `PYTHONPATH='.:../training-sdk/src' uv run --project ../memo --group development python scripts/build_catalog.py` (`rtrrl`, WSL) | PASS | Regenerated `catalog.json`; Windows `git diff --ignore-cr-at-eol` showed no content diff. |
