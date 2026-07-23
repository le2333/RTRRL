# Infra-only Task 3 Report

## Status

Task 3 was implemented from baseline `23f229e` as the standalone
`rtrrl/infra/mock-trainer` project and committed as:

- `1eb355e feat(infra): define isolated acceptance trainer`

No training implementation, catalog, Docker, controller, memo, AWS, or Docker
operation was added or run.

## RED / GREEN

### RED

The plan's repository-root command does not change into the `--project`
directory under local `uv 0.11.25`. Before `pyproject.toml` existed it selected
the parent `rtrrl` project; after the project contract was added it still looked
for `tests/test_config.py` relative to the repository root. The meaningful RED
run therefore used the project as its working directory:

```text
uv run --project . pytest tests/test_config.py -q
ModuleNotFoundError: No module named 'brax_ppo_acceptance'
1 error during collection
```

No package implementation existed during this run. The parent `rtrrl/uv.lock`
change produced by the first misresolved command was inspected and fully
reverted before commit.

### GREEN

```text
uv run --project rtrrl/infra/mock-trainer \
  pytest rtrrl/infra/mock-trainer/tests/test_config.py -q
50 passed in 0.22s

uv run --project rtrrl/infra/mock-trainer \
  ruff check rtrrl/infra/mock-trainer/src rtrrl/infra/mock-trainer/tests
All checks passed!
```

The tests cover exact key sets at all eight mapping levels, missing and extra
fields, supported literals, deep immutability, strict integers that reject
booleans, positive/non-negative bounds, finite positive learning rates,
environment/backend restrictions, environment-budget consistency, resolved
`num_timesteps`, all failure modes, both environment gates, and a default
environment snapshot that is not dynamically changed through `os.environ`.

## Configuration Contract

- `AcceptanceConfig` is a standard frozen, slotted dataclass; nested mappings
  are recursively frozen with mapping proxies.
- Only `inverted_pendulum` with the `generalized` backend is accepted.
- `training_budget.env_steps` is positive, resolves `num_timesteps`, and must
  equal `num_envs * episode_length`.
- Non-`none` failure injection requires
  `BRAX_ACCEPTANCE_TEST_MODE=1`.
- Fast mode requires both `BRAX_ACCEPTANCE_TEST_MODE=1` and
  `BRAX_ACCEPTANCE_E2E_FAST=1`; invalid gate values fail closed.

## Lock and Isolation Evidence

```text
uv sync --project rtrrl/infra/mock-trainer --frozen
Checked 95 packages

uv lock --project rtrrl/infra/mock-trainer --check
Resolved 113 packages
```

- The lock resolves `brax==0.14.2`, `jax==0.10.0`, and `jaxlib==0.10.0`.
- `boto3` is a runtime dependency and resolved to `1.43.55`.
- The lock records `training-sdk` as
  `directory = "../../../training-sdk"`; an isolated-environment import
  resolved to the mock-trainer virtual environment installation.
- Forbidden `memo` / `trainer_infra` import search returned no matches.
- `git diff --check` passed.
- The pre-commit diff boundary contained only
  `rtrrl/infra/mock-trainer/**`.
- IDE lint reported no findings.

## Concerns

- With this `uv` version, the plan's repository-root pytest command must use
  the repository-relative test path, or run with the mock-trainer directory as
  its working directory. `--project` selects project metadata but does not
  change the process working directory.
- No full JAX execution was performed. Verification was intentionally limited
  to lock/sync, configuration tests, dependency imports, Ruff, forbidden
  imports, diff checks, and IDE lint.
