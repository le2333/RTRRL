# Task 2: Control Plane Experiment and Launch Schema

## Commits

- `827dec1 test(control-plane): require training and evaluation sections`
- `0c7327c feat(control-plane): model training and evaluation sections`

## Files changed

- `rtrrl/infra/control-plane/src/trainer_infra/experiment.py`
- `rtrrl/infra/control-plane/src/trainer_infra/preflight.py`
- `rtrrl/infra/control-plane/src/trainer_infra/launch.py`
- `rtrrl/infra/control-plane/tests/helpers.py`
- `rtrrl/infra/control-plane/tests/test_experiment.py`
- `rtrrl/infra/control-plane/tests/test_preflight_offline.py`
- `rtrrl/infra/control-plane/tests/test_launch.py`
- `rtrrl/infra/control-plane/tests/data/experiment.yaml`

## Verification

- `git diff --check`: passed (exit 0) before the implementation commit.
- `git show --check --oneline 0c7327c`: passed (exit 0); the committed patch has no whitespace errors.
- `uv run --project rtrrl/infra/control-plane pytest rtrrl/infra/control-plane/tests/test_experiment.py`: not run. The sandboxed attempt failed before test collection because uv could not open its local cache (`sdists-v9/.git`, access denied); the approved rerun was rejected because the active `AGENTS.md` policy prohibits pytest on this micro instance.
- `uv run ruff check rtrrl/infra/control-plane`: failed (exit 1) because `ruff` is not available in the root environment.
- `uv run --project rtrrl/infra/control-plane ruff check rtrrl/infra/control-plane`: not run to completion. The sandboxed attempt hit the same uv-cache access denial; the approved retry failed during environment resolution because `aimrocks==0.5.2` has no Windows wheel. The WSL retry failed before execution with `Wsl/Service/CreateInstance/E_ACCESS_DENIED`.
- Required remote red/green checks (`git push origin HEAD` and `gh workflow run tests.yml --ref <branch>`): not run. The sandboxed attempt could neither acquire `.git/index.lock` nor reach GitHub; the escalation was rejected because the remote destination was not explicitly trusted for code egress.

No pytest or Docker command was executed.

## Self-review

- `Experiment` now carries `environment`, `training`, and `evaluation`; the old `Budget` model and cross-model budget/environment validator are removed.
- The new control-plane models mirror Task 1 validation: environment seed validation; training stream, whole-epoch, chunk, and early-stop validation; and evaluation step/environment-count validation.
- Reserved search-space names include both legacy aliases and all new configuration fields. The fixture moves seed out of the search space, and launch archive/config output uses the split sections while `LaunchPlan.space` remains unchanged.
- `helpers.CATALOG` was updated to contract version 5 so the offline fixture agrees with Task 1's contract version.

## Concerns

- The code and tests are committed but unpushed. Remote CI must be dispatched after an authorized push.
- Local pytest and Ruff verification remain unavailable due the machine policy/environment constraints above.
- Later migration tasks are still expected to update mock-trainer and entry-side budget readers; those files were intentionally outside this task's scope.

## Review fix

### Files changed

- `rtrrl/infra/control-plane/examples/experiment-dev-smoke.yaml`
- `rtrrl/infra/control-plane/examples/experiment-acceptance.yaml`
- `rtrrl/infra/control-plane/examples/experiment-acceptance-gpu.yaml`
- `rtrrl/infra/control-plane/tests/conftest.py`

All three example experiments now place the fixed seed in `environment`, split
the old environment/budget values into `training` and `evaluation`, and keep
reserved configuration fields out of `space`. The end-to-end fixture trainer
now reads `training.total_steps` from the v5 run configuration.

### Commands and results

- `wsl.exe bash -lc '... uv run pytest'`: not run. WSL instance creation was
  denied in the sandbox; the required escalated retry was rejected because the
  host policy prohibits local pytest on this micro instance.
- `uv run ruff check .` (WSL): passed (`All checks passed!`).

### Commit

- `7cc05e5 fix(control-plane): migrate example experiment configs`

### Concerns

- Full pytest could not be run locally because sandbox policy rejected WSL
  execution on this micro instance.

## Controller follow-up verification

After the user confirmed this checkout can install and run tests locally, the
controller installed the full control-plane environment inside WSL because the
control-plane dev dependencies include `training-sdk[testing]`, which pulls
`aimrocks==0.5.2`; that package has no Windows `win_amd64` distribution. The
virtualenv and cache were kept outside the repository:

- `UV_PROJECT_ENVIRONMENT=/tmp/streaming-rtrrl-control-plane-venv`
- `UV_CACHE_DIR=/tmp/streaming-rtrrl-uv-cache`

Additional commands:

| Command | Result | Full summary |
| --- | --- | --- |
| `uv sync --all-groups` (`rtrrl/infra/control-plane`, WSL) | PASS | Installed 99 Linux-compatible packages, including `trainer-infra`, `training-sdk`, `aim`, `aimrocks`, `optuna`, `pytest`, and `ruff`. |
| `uv run ruff check .` (`rtrrl/infra/control-plane`, WSL) | PASS | `All checks passed!` |
| `uv run pytest tests/test_experiment.py tests/test_preflight_offline.py tests/test_launch.py` (`rtrrl/infra/control-plane`, WSL) | PASS | 47 tests collected; 47 passed with 1 Aim/SQLAlchemy deprecation warning in 2.29s. |
