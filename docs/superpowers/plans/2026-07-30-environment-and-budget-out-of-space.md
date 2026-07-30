# Environment and Budget Out of Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the environment and the budget out of the searched space into their own sections of the experiment file, and specify a partially observed task by listing the observation indices to keep, deleting the rest rather than zeroing them.

**Architecture:** The experiment YAML gains an `environment` section and a `budget` section beside `space`. Both travel to the worker through new fields on `RunConfig` rather than through sampled parameters, so no trial can vary them. On the algorithm side an observation wrapper indexes the observation down to the kept dimensions and reports the reduced shape, replacing the boolean-mask multiply and the named F/P/V table.

**Tech Stack:** Python 3.12, pydantic v2, Optuna, JAX/Flax, gymnax-style environment wrappers, uv, pytest, GitHub Actions.

This is phase 1 of `docs/superpowers/specs/2026-07-30-configuration-surface-design.md`. Phases 2 to 4 get their own plans.

## Global Constraints

- Never run pytest on this machine. It is a micro instance with ~250 MiB free and running the suite has killed the editor session (`AGENTS.md:16-38`). Tests are written here and executed in CI.
- Never run `docker` here in any form.
- `memo/**` changes: pushing to any branch runs the `Memo CI` workflow automatically.
- `training-sdk/**`, `rtrrl/infra/control-plane/**`, `rtrrl/infra/mock-trainer/**` changes: the `Tests` workflow only auto-runs on `main`. On a feature branch trigger it by hand with `gh workflow run tests.yml --ref $(git branch --show-current)`.
- `Memo CI` is red at a known baseline of exactly five `test_stream_ac_golden.py` failures caused by a seed-spending change on main, left in place deliberately. A memo task passes when its failure set equals exactly those five and nothing else. Task 6 is the single exception and says so in its own text: it may additionally fail `test_experiments.py`, and nothing else. No other task may exceed the five.
- Two CI round trips per task: one to watch the new test fail, one to watch it pass. Do not collapse them. A test that was never seen red is not evidence.
- Do not write rationale into code or configuration files. State what the code does, not why the decision was made or what was measured.
- `CONTRACT_VERSION` becomes `3` in Task 1. Every catalog and every image must be rebuilt before anything is launched; no launch happens inside this plan.
- Two catalogs are checked in and both currently declare `"contract": 2`: `rtrrl/infra/mock-trainer/catalog.json` and `rtrrl/catalog.json`. `check_offline` refuses a catalog whose contract is not the running one, so each must be regenerated in the task that first needs it green — the mock trainer's in Task 4, `rtrrl`'s in Task 7. Regenerate rather than hand-edit:

```bash
uv run --project rtrrl/infra/mock-trainer python rtrrl/infra/mock-trainer/scripts/build_catalog.py
cd rtrrl && uv run python scripts/build_catalog.py && cd ..
```

  Both are deliberately importable without a training framework, so they run on this machine. `memo` has no checked-in catalog; its tests discover entries live.

---

### Task 1: The contract carries an environment and a budget

**Files:**
- Modify: `training-sdk/src/training_sdk/contract.py:7` and `:109-121`
- Test: `training-sdk/tests/test_contract.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `EnvironmentConfig(id: str, backend: str, num_envs: int, observed: tuple[int, ...] | None)`, `BudgetConfig(total_steps: int, epoch_steps: int, eval_steps: int)`, both exported from `training_sdk.contract`. `RunConfig` gains required fields `environment: EnvironmentConfig` and `budget: BudgetConfig`. `CONTRACT_VERSION == 3`.

- [ ] **Step 1: Write the failing tests**

Append to `training-sdk/tests/test_contract.py`:

```python
import pytest
from pydantic import ValidationError

from training_sdk.contract import BudgetConfig, EnvironmentConfig


def test_an_environment_names_a_task_and_how_many_copies_of_it():
    environment = EnvironmentConfig(
        id="brax::hopper", backend="spring", num_envs=1, observed=(0, 1, 2, 3, 4)
    )

    assert environment.observed == (0, 1, 2, 3, 4)


def test_an_environment_without_observed_is_fully_observed():
    environment = EnvironmentConfig(id="brax::hopper", backend="spring", num_envs=1)

    assert environment.observed is None


@pytest.mark.parametrize(
    "observed", [(), (0, 0, 1), (-1, 0)], ids=["empty", "repeated", "negative"]
)
def test_an_index_list_that_selects_nothing_usable_is_refused(observed):
    with pytest.raises(ValidationError):
        EnvironmentConfig(
            id="brax::hopper", backend="spring", num_envs=1, observed=observed
        )


def test_a_budget_must_divide_into_whole_epochs():
    with pytest.raises(ValidationError):
        BudgetConfig(total_steps=1000, epoch_steps=300, eval_steps=0)


def test_a_budget_of_whole_epochs_is_accepted():
    budget = BudgetConfig(total_steps=900, epoch_steps=300, eval_steps=0)

    assert budget.total_steps == 900
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
gh workflow run tests.yml --ref $(git branch --show-current)
gh run watch
```

Expected: the `training-sdk` job fails with `ImportError: cannot import name 'BudgetConfig' from 'training_sdk.contract'`.

- [ ] **Step 3: Add the two models and wire them into `RunConfig`**

In `training-sdk/src/training_sdk/contract.py`, change line 7 to:

```python
CONTRACT_VERSION = 3
```

Insert after the `Catalog` class:

```python
class EnvironmentConfig(_Frozen):
    id: str
    backend: str
    num_envs: int
    observed: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _usable(self) -> "EnvironmentConfig":
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.observed is None:
            return self
        if not self.observed:
            raise ValueError("observed must name at least one index")
        if len(set(self.observed)) != len(self.observed):
            raise ValueError("observed must not repeat an index")
        if any(index < 0 for index in self.observed):
            raise ValueError("observed indices must not be negative")
        return self


class BudgetConfig(_Frozen):
    total_steps: int
    epoch_steps: int
    eval_steps: int

    @model_validator(mode="after")
    def _whole(self) -> "BudgetConfig":
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.epoch_steps < 1:
            raise ValueError("epoch_steps must be positive")
        if self.eval_steps < 0:
            raise ValueError("eval_steps must not be negative")
        if self.total_steps % self.epoch_steps:
            raise ValueError(
                f"total_steps {self.total_steps} is not whole epochs of "
                f"{self.epoch_steps}"
            )
        return self
```

Add two fields to `RunConfig`, after `digest`:

```python
    environment: EnvironmentConfig
    budget: BudgetConfig
```

and a validator at the end of `RunConfig`:

```python
    @model_validator(mode="after")
    def _epochs_hold_whole_rounds_of_streams(self) -> "RunConfig":
        if self.budget.epoch_steps % self.environment.num_envs:
            raise ValueError(
                f"epoch_steps {self.budget.epoch_steps} is not "
                f"{self.environment.num_envs} streams' worth"
            )
        return self
```

- [ ] **Step 4: Update every `RunConfig` construction in the training-sdk tests**

`RunConfig` gained two required fields, so its existing constructions no longer validate. In `training-sdk/tests/`, find them and add both fields:

```bash
rg -n "RunConfig\(" training-sdk/tests/
```

For each, add:

```python
        environment=EnvironmentConfig(id="brax::hopper", backend="spring", num_envs=1),
        budget=BudgetConfig(total_steps=100, epoch_steps=100, eval_steps=0),
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
gh workflow run tests.yml --ref $(git branch --show-current)
gh run watch
```

Expected: the `training-sdk` job passes. The `control-plane` and `mock-trainer` jobs fail, because they build `RunConfig` without the new fields; Task 4 fixes the control plane and Task 4's step 5 fixes the mock trainer.

- [ ] **Step 6: Commit**

```bash
git add training-sdk/src/training_sdk/contract.py training-sdk/tests/test_contract.py
git commit -m "feat(contract): carry the environment and the budget beside the params"
```

---

### Task 2: The experiment file has an environment section and a budget section

**Files:**
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/experiment.py:66-82`
- Test: `rtrrl/infra/control-plane/tests/test_experiment.py`

**Interfaces:**
- Consumes: nothing from Task 1; this model is the control plane's own.
- Produces: `Environment(id: str, backend: str, num_envs: int, observed: tuple[int, ...] | None)` and `Budget(total_steps: int, epoch_steps: int, eval_steps: int)` in `trainer_infra.experiment`. `Experiment` gains required fields `environment: Environment` and `budget: Budget`, and rejects a `space` naming any of `RESERVED`.

- [ ] **Step 1: Write the failing tests**

Append to `rtrrl/infra/control-plane/tests/test_experiment.py`:

```python
def test_an_experiment_carries_its_environment_and_budget(tmp_path):
    document = _document()
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    experiment = load_experiment(path)

    assert experiment.environment.observed == (0, 1, 2, 3, 4)
    assert experiment.budget.total_steps == 2000


def test_a_space_may_not_name_the_environment_or_the_budget(tmp_path):
    document = _document()
    document["space"]["total_steps"] = [2000]
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValidationError) as raised:
        load_experiment(path)

    assert "total_steps" in str(raised.value)
```

Add this helper to `rtrrl/infra/control-plane/tests/helpers.py` and import it here; Task 3 imports the same one:

```python
def _document() -> dict:
    return {
        "experiment": "demo",
        "name": "one",
        "image": "example.invalid/image@sha256:" + "0" * 64,
        "entry": "demo_entry",
        "storage": "s3://bucket/prefix",
        "environment": {
            "id": "brax::hopper",
            "backend": "spring",
            "num_envs": 1,
            "observed": [0, 1, 2, 3, 4],
        },
        "budget": {"total_steps": 2000, "epoch_steps": 1000, "eval_steps": 100},
        "compute": {"instance_type": "c7a.medium", "timeout_minutes": 60},
        "hpo": {
            "sampler": "tpe",
            "rounds": 1,
            "trials_per_round": 1,
            "parallel_jobs": 1,
        },
        "space": {"learning_rate": [0.001]},
        "score": {
            "metric": "eval/episode_return",
            "window_steps": [0, 2000],
            "reduce": "max",
            "direction": "maximize",
            "non_finite": "worst",
        },
        "logging": {"aim": "aim://127.0.0.1:53801", "every_steps": 1},
    }
```

Make sure the module imports `pytest`, `yaml`, `ValidationError` from `pydantic`, and `load_experiment` from `trainer_infra.experiment`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
gh workflow run tests.yml --ref $(git branch --show-current)
gh run watch
```

Expected: the `control-plane` job fails on `test_an_experiment_carries_its_environment_and_budget` with a pydantic error naming `environment` as an unexpected field, because `Experiment` sets `extra="forbid"`.

- [ ] **Step 3: Add the two sections to the model**

In `rtrrl/infra/control-plane/src/trainer_infra/experiment.py`, insert before `class Experiment`:

```python
RESERVED = frozenset(
    {
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
    }
)


class Environment(_Frozen):
    id: str
    backend: str
    num_envs: int
    observed: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _usable(self) -> "Environment":
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.observed is None:
            return self
        if not self.observed:
            raise ValueError("observed must name at least one index")
        if len(set(self.observed)) != len(self.observed):
            raise ValueError("observed must not repeat an index")
        if any(index < 0 for index in self.observed):
            raise ValueError("observed indices must not be negative")
        return self


class Budget(_Frozen):
    total_steps: int
    epoch_steps: int
    eval_steps: int

    @model_validator(mode="after")
    def _whole(self) -> "Budget":
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.epoch_steps < 1:
            raise ValueError("epoch_steps must be positive")
        if self.eval_steps < 0:
            raise ValueError("eval_steps must not be negative")
        if self.total_steps % self.epoch_steps:
            raise ValueError(
                f"total_steps {self.total_steps} is not whole epochs of "
                f"{self.epoch_steps}"
            )
        return self
```

Add the two fields to `Experiment`, after `storage`:

```python
    environment: Environment
    budget: Budget
```

and a validator at the end of `Experiment`:

```python
    @model_validator(mode="after")
    def _space_is_only_algorithm(self) -> "Experiment":
        taken = sorted(RESERVED & set(self.space))
        if taken:
            raise ValueError(
                f"space names {', '.join(taken)}, which belong to the environment "
                "and budget sections and are not searched"
            )
        if self.budget.epoch_steps % self.environment.num_envs:
            raise ValueError(
                f"epoch_steps {self.budget.epoch_steps} is not "
                f"{self.environment.num_envs} streams' worth"
            )
        return self
```

- [ ] **Step 4: Update the control-plane test fixtures**

Every fixture experiment now needs the two sections:

```bash
rg -ln "storage" rtrrl/infra/control-plane/tests/ rtrrl/infra/control-plane/tests/data/
```

For each YAML or dict fixture, add the two blocks shown in Step 1's `_document`, and remove any of the `RESERVED` names from its `space`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
gh workflow run tests.yml --ref $(git branch --show-current)
gh run watch
```

Expected: the two new tests pass. Other control-plane tests still fail on `RunConfig`; Task 4 fixes them.

- [ ] **Step 6: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/experiment.py rtrrl/infra/control-plane/tests/
git commit -m "feat(experiment): give the environment and the budget their own sections"
```

---

### Task 3: Preflight scores against the budget instead of the space

**Files:**
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/space.py:50-60`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/preflight.py:50-58`
- Test: `rtrrl/infra/control-plane/tests/test_preflight_offline.py`
- Test: `rtrrl/infra/control-plane/tests/test_space.py`

**Interfaces:**
- Consumes: `Experiment.budget.total_steps` from Task 2.
- Produces: `minimum_total_steps` no longer exists in `trainer_infra.space`. `check_offline` raises `PreflightError` when `experiment.score.window_steps[1] > experiment.budget.total_steps`.

- [ ] **Step 1: Write the failing test**

`check_offline(experiment: Experiment, catalog: Catalog) -> dict[str, SpaceEntry]` returns the resolved space, so the accepting case asserts on that rather than on the absence of a raise.

Append to `rtrrl/infra/control-plane/tests/test_preflight_offline.py`:

```python
def _written(tmp_path, *, window, total_steps):
    document = _document()
    document["score"]["window_steps"] = list(window)
    document["budget"]["total_steps"] = total_steps
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_experiment(path)


def test_a_score_window_past_the_budget_is_refused(tmp_path):
    experiment = _written(tmp_path, window=(0, 4000), total_steps=2000)

    with pytest.raises(PreflightError) as raised:
        check_offline(experiment, _catalog())

    assert "4000" in str(raised.value)


def test_a_score_window_inside_the_budget_is_accepted(tmp_path):
    experiment = _written(tmp_path, window=(0, 2000), total_steps=2000)

    assert "learning_rate" in check_offline(experiment, _catalog())
```

`_document` is the helper Task 2 Step 1 put in `rtrrl/infra/control-plane/tests/helpers.py`; import it here rather than writing it twice. `_catalog()` builds a `Catalog` at `CONTRACT_VERSION` whose entries contain `demo_entry` with `metrics=("eval/episode_return",)` and `space={"learning_rate": ChoiceSpec(choices=(0.001,))}`; that file already builds catalogs this way for its other tests, so reuse its existing constructor.

- [ ] **Step 2: Run the test to verify it fails**

```bash
gh workflow run tests.yml --ref $(git branch --show-current)
gh run watch
```

Expected: `test_a_score_window_past_the_budget_is_refused` fails, because `minimum_total_steps` reads `total_steps` out of the space and the space no longer has it.

- [ ] **Step 3: Delete `minimum_total_steps` and read the budget instead**

In `rtrrl/infra/control-plane/src/trainer_infra/space.py`, delete the whole `minimum_total_steps` function.

In `rtrrl/infra/control-plane/src/trainer_infra/preflight.py`, change the import on line 13 to drop the name:

```python
from trainer_infra.space import distributions, resolve_space
```

and replace the two statements in `check_offline` that compute and compare the budget with:

```python
    if experiment.score.window_steps[1] > experiment.budget.total_steps:
        raise PreflightError(
            f"score window upper bound {experiment.score.window_steps[1]} exceeds "
            f"the budget's total_steps ({experiment.budget.total_steps})"
        )
```

- [ ] **Step 4: Delete the tests of the deleted function**

In `rtrrl/infra/control-plane/tests/test_space.py`, delete every test that calls `minimum_total_steps` and drop it from that file's imports.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
gh workflow run tests.yml --ref $(git branch --show-current)
gh run watch
```

Expected: both new tests pass and no test references `minimum_total_steps`.

- [ ] **Step 6: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/space.py rtrrl/infra/control-plane/src/trainer_infra/preflight.py rtrrl/infra/control-plane/tests/
git commit -m "feat(preflight): hold the score window against the declared budget"
```

---

### Task 4: The run config a worker receives carries both sections

**Files:**
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/launch.py:10-16`, `:47-62`, `:88-113`
- Modify: `rtrrl/infra/mock-trainer/` wherever it builds or reads a `RunConfig`
- Test: `rtrrl/infra/control-plane/tests/test_launch.py`

**Interfaces:**
- Consumes: `EnvironmentConfig` and `BudgetConfig` from Task 1; `Experiment.environment` and `Experiment.budget` from Task 2.
- Produces: `build_run_config` returns a `RunConfig` whose `environment` and `budget` mirror the experiment's. `launch.json` gains `environment` and `budget` keys.

- [ ] **Step 1: Write the failing test**

Append to `rtrrl/infra/control-plane/tests/test_launch.py`:

```python
def test_the_run_config_carries_the_environment_and_the_budget(tmp_path):
    launch = _launch(tmp_path)

    config = build_run_config(launch, trial=0, params={"learning_rate": 0.001})

    assert config.environment.id == "brax::hopper"
    assert config.environment.observed == (0, 1, 2, 3, 4)
    assert config.budget.total_steps == 2000


def test_the_archived_launch_records_both_sections(tmp_path):
    launch = _launch(tmp_path)

    archived = json.loads((launch.archive / "launch.json").read_text(encoding="utf-8"))

    assert archived["environment"]["observed"] == [0, 1, 2, 3, 4]
    assert archived["budget"]["epoch_steps"] == 1000
```

Reuse the file's existing helper for building a `Launch`; it needs the experiment document from Task 2 Step 1.

- [ ] **Step 2: Run the test to verify it fails**

```bash
gh workflow run tests.yml --ref $(git branch --show-current)
gh run watch
```

Expected: `test_the_run_config_carries_the_environment_and_the_budget` fails with a pydantic error naming `environment` and `budget` as missing.

- [ ] **Step 3: Thread both sections through**

In `rtrrl/infra/control-plane/src/trainer_infra/launch.py`, extend the import block:

```python
from training_sdk.contract import (
    CONTRACT_VERSION,
    BudgetConfig,
    EnvironmentConfig,
    LoggingConfig,
    RunConfig,
    Scalar,
    ScoreConfig,
)
```

Add two keys to `launch_payload`, after `"entry"`:

```python
        "environment": experiment.environment.model_dump(mode="json"),
        "budget": experiment.budget.model_dump(mode="json"),
```

Add two arguments to the `RunConfig(...)` call in `build_run_config`, after `digest`:

```python
        environment=EnvironmentConfig(
            id=experiment.environment.id,
            backend=experiment.environment.backend,
            num_envs=experiment.environment.num_envs,
            observed=experiment.environment.observed,
        ),
        budget=BudgetConfig(
            total_steps=experiment.budget.total_steps,
            epoch_steps=experiment.budget.epoch_steps,
            eval_steps=experiment.budget.eval_steps,
        ),
```

- [ ] **Step 4: Fix the remaining control-plane and mock-trainer constructions**

```bash
rg -n "RunConfig\(" rtrrl/infra/control-plane/ rtrrl/infra/mock-trainer/
```

Add the same two arguments to each. Where a test asserts on a `config.json` payload, add the two keys to the expected document.

Then regenerate the mock trainer's catalog so it declares contract 3, and give its experiment the two new sections:

```bash
uv run --project rtrrl/infra/mock-trainer python rtrrl/infra/mock-trainer/scripts/build_catalog.py
git diff --stat rtrrl/infra/mock-trainer/catalog.json
```

`rtrrl/infra/mock-trainer/scripts/brax_ppo_acceptance.yaml` is that trainer's own experiment file and lives beside it rather than in `experiments/`, so Task 8 will not reach it; move its seven keys into `environment` and `budget` here.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
gh workflow run tests.yml --ref $(git branch --show-current)
gh run watch
```

Expected: all three jobs — `training-sdk`, `control-plane`, `mock-trainer` — pass.

- [ ] **Step 6: Commit**

```bash
git add rtrrl/infra/control-plane/ rtrrl/infra/mock-trainer/
git commit -m "feat(launch): hand the worker its environment and its budget"
```

---

### Task 5: An observation is selected by index and the rest are deleted

**Files:**
- Create: `memo/memorax/environments/wrappers/select_observation.py`
- Delete: `memo/memorax/environments/wrappers/mask_observation.py`
- Delete: `memo/tests/test_masking.py`
- Modify: `memo/memorax/environments/wrappers/__init__.py:9`
- Modify: `memo/memorax/environments/brax.py:1-30`, `:72-85`
- Test: `memo/tests/test_observation_selection.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SelectObservationWrapper(env, observed)` exported from `memorax.environments.wrappers`. `memorax.environments.brax.make(env_id, observed=None, backend="generalized", **kwargs)`. `memorax.environments.brax.masks` no longer exists.

- [ ] **Step 1: Write the failing tests**

Create `memo/tests/test_observation_selection.py`:

```python
"""Selecting observation dimensions by index."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from memorax.environments import make

HOPPER_WIDTH = 11
KEPT = (0, 1, 2, 3, 4)


def test_the_environment_reports_only_the_selected_dimensions():
    env, params = make("brax::hopper", observed=KEPT, backend="spring")

    observation, _ = env.reset(jax.random.key(0), params)

    assert observation.shape == (len(KEPT),)
    assert env.observation_space(params).shape == (len(KEPT),)


def test_a_full_observation_is_what_the_task_reports():
    env, params = make("brax::hopper", backend="spring")

    observation, _ = env.reset(jax.random.key(0), params)

    assert observation.shape == (HOPPER_WIDTH,)


def test_the_selected_dimensions_are_the_ones_asked_for():
    wide, params = make("brax::hopper", backend="spring")
    narrow, _ = make("brax::hopper", observed=KEPT, backend="spring")
    key = jax.random.key(0)

    whole, _ = wide.reset(key, params)
    part, _ = narrow.reset(key, params)

    assert jnp.array_equal(part, whole[jnp.asarray(KEPT)])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
git add memo/tests/test_observation_selection.py
git commit -m "test(env): select observation dimensions by index"
git push
gh run watch
```

Expected: two of the three fail with `TypeError: make() got an unexpected keyword argument 'observed'`, on top of the five known golden failures. `test_a_full_observation_is_what_the_task_reports` passes already, because it passes no `observed` and the current default mode `F` keeps all eleven dimensions; it is here to hold that behaviour through the change, not to fail first.

- [ ] **Step 3: Write the wrapper**

Create `memo/memorax/environments/wrappers/select_observation.py`:

```python
from __future__ import annotations

from typing import Union

import jax.numpy as jnp
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper

from memorax.utils.typing import Array, Key


class SelectObservationWrapper(GymnaxWrapper):
    def __init__(self, env, observed):
        super().__init__(env)
        self.observed = jnp.asarray(observed, dtype=jnp.int32)

    def observation_space(self, params=None) -> spaces.Box:
        inner = self._env.observation_space(params)
        return spaces.Box(
            low=inner.low,
            high=inner.high,
            shape=(int(self.observed.size),),
            dtype=inner.dtype,
        )

    def reset(
        self, key: Key, params: environment.EnvParams | None = None
    ) -> tuple[Array, environment.EnvState]:
        observation, state = self._env.reset(key, params)
        return observation[..., self.observed], state

    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, environment.EnvState, float, bool, dict]:
        observation, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        return observation[..., self.observed], state, reward, done, info
```

`MaskObservationWrapper` overrode no `observation_space`, so today a masked task still reports the full width; the override above is what makes the reduced width visible to the network that reads it.

- [ ] **Step 4: Swap the wrapper in and delete the mask table**

Delete `memo/memorax/environments/wrappers/mask_observation.py`.

In `memo/memorax/environments/wrappers/__init__.py`, replace line 9 with:

```python
from .select_observation import SelectObservationWrapper
```

and update `__all__` in that file if it lists `MaskObservationWrapper`.

In `memo/memorax/environments/brax.py`, change the import on line 6 to:

```python
from memorax.environments.wrappers import GymnaxWrapper, SelectObservationWrapper
```

delete the whole `masks` dict at lines 9 to 30, and replace `make` with:

```python
def make(env_id: str, observed=None, backend="generalized", **kwargs) -> tuple:
    from brax import envs
    from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper

    env = envs.get_environment(env_name=env_id, backend=backend, **kwargs)
    env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
    env = AutoResetWrapper(env)
    env = BraxGymnaxWrapper(env)
    if observed is not None:
        env = SelectObservationWrapper(env, observed)

    env_params = env.default_params
    return env, env_params
```

Delete `memo/tests/test_masking.py`.

- [ ] **Step 5: Fix the remaining references to the deleted table**

```bash
rg -n "masks|MaskObservationWrapper" memo/
```

`memo/entries/rtrrl.py`, `memo/entries/stream_ac.py` and `memo/entries/upstream_stream_ac.py` each build their `environment` choice list from `sorted(masks)`. Replace that list with the literal task names, so the entries keep importing nothing from the deleted table:

```python
BRAX_TASKS = ("ant", "halfcheetah", "hopper", "walker2d")
```

and use `[f"brax::{task}" for task in BRAX_TASKS]`. Task 6 removes the key entirely; this step only keeps the module importable.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
git add memo/
git commit -m "feat(env): select observation dimensions rather than zeroing them"
git push
gh run watch
```

Expected: the three new tests pass and the failure set is exactly the five known `test_stream_ac_golden.py` failures.

---

### Task 6: The memo entries stop declaring the environment and the budget

**Files:**
- Modify: `memo/entries/rtrrl.py:25` (drop the `masks` import), `:39-56` (`SPACE`), `:126-142` (`build`), `:203-215` (`run`), `:219` (`main`)
- Modify: `memo/entries/stream_ac.py:56-88`, `:134-136`, `:203-206`
- Modify: `memo/entries/upstream_stream_ac.py:66-78`, `:112-114`, `:179-182`
- Test: `memo/tests/test_entries.py`

**Interfaces:**
- Consumes: `make(env_id, observed=..., backend=...)` from Task 5; `RunConfig.environment` and `RunConfig.budget` from Task 1.
- Produces: each entry's `SPACE` no longer contains `environment`, `env_mode`, `env_backend`, `num_envs`, `total_steps`, `epoch_steps`, `eval_steps`. Each entry exposes `run(reporter, config)` where `config` is a `RunConfig`, and `build(params, environment)` where `environment` is a `RunConfig.environment`.

- [ ] **Step 1: Write the failing test**

Append to `memo/tests/test_entries.py`:

```python
RESERVED = frozenset(
    {
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
    }
)


def test_no_entry_declares_the_environment_or_the_budget():
    for name, module in ENTRIES.items():
        taken = RESERVED & set(module.SPACE)
        assert not taken, f"{name} still declares {sorted(taken)}"
```

`ENTRIES = discover()` is already at `memo/tests/test_entries.py:86`; use it rather than calling `discover` again.

- [ ] **Step 2: Run the test to verify it fails**

```bash
git add memo/tests/test_entries.py
git commit -m "test(entries): the environment and the budget are not parameters"
git push
gh run watch
```

Expected: fails listing all seven names for each of the three entries.

- [ ] **Step 3: Rewrite `memo/entries/rtrrl.py`**

Delete these seven keys from `SPACE`: `environment`, `env_mode`, `env_backend`, `num_envs`, `total_steps`, `epoch_steps`, `eval_steps`. Delete the `BRAX_TASKS` constant Task 5 added to this file, which had no other reader.

Change `build` to take the environment:

```python
def build(params: Mapping[str, Any], environment) -> RTRRL:
    """Assemble the agent this file is about."""

    env, env_params = make(
        environment.id,
        observed=environment.observed,
        backend=environment.backend,
    )
```

and inside it use `num_envs=environment.num_envs` in place of `int(params["num_envs"])`.

Change `run` to take the whole config:

```python
def run(reporter, config) -> None:
    agent = build(config.params, config.environment)
    drive(
        reporter,
        init_fn=agent.init,
        train_fn=agent.train,
        evaluate_fn=agent.evaluate,
        total_steps=config.budget.total_steps,
        epoch_steps=config.budget.epoch_steps,
        eval_steps=config.budget.eval_steps,
        num_envs=config.environment.num_envs,
        seed=int(config.params["seed"]),
        training_report=training_report,
    )
```

and `main`:

```python
def main(argv: list[str] | None = None) -> int:
    del argv
    with Reporter.from_env() as reporter:
        run(reporter, reporter.config)
    return 0
```

- [ ] **Step 4: Make the same three changes in the two StreamAC entries**

`memo/entries/stream_ac.py` and `memo/entries/upstream_stream_ac.py` each have the same seven `SPACE` keys, the same `make(...)` call with `mode=` and `backend=`, the same `num_envs` read inside their config construction, and the same `drive(...)` call. Apply the identical edits: delete the seven keys, give `build` an `environment` argument that supplies `id`, `observed`, `backend` and `num_envs`, give `run` the whole config, and have `main` pass `reporter.config`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
git add memo/entries/
git commit -m "feat(entries): read the environment and the budget off the run config"
git push
gh run watch
```

Expected: the new test passes; `test_experiments.py` now fails on every experiment file, because each still names the seven keys.

This is the one task in the plan that ends redder than the baseline, and the Global Constraints carve it out by name. It is unavoidable: `test_experiments.py` checks the entries and the experiment files against each other, so whichever side moves first breaks it. Task 8 is what closes it. Do not "fix" it by weakening the test, and do not report it as a defect — but do check that `test_experiments.py` is the *only* thing that went red beyond the five, because anything else here is a real regression.

---

### Task 7: The AAAI entry stops declaring them too

**Files:**
- Modify: `rtrrl/entries/rtrrl_aaai.py:47-60` (the `MASKS` table and its comment), `:70-102` (`SPACE`), `:154-228` (`settings`), `:231-244` (`parameters`), `:247` (`run`), and `main` at the end of the file
- Test: `rtrrl/tests/test_entry.py`

**Interfaces:**
- Consumes: `RunConfig.environment` and `RunConfig.budget` from Task 1.
- Produces: `rtrrl_aaai.SPACE` without the seven names. `settings(params, environment, budget)` and `parameters(params, environment, budget)` take the two new arguments; `run(reporter, config)` takes the whole config. The entry maps `environment.observed` onto the authors' `EnvironmentParams.obs_mask` and `environment.num_envs` onto their `batch_size`.

**Two facts about their side, verified in `RTRRL-AAAI25/envs/environments.py`:**
- `obs_mask` is a sequence of the indices to **keep**, and `OBS_SIZE = len(obs_mask)` (`:105-106`), so their code already deletes what it drops. `get_obs_mask` (`:54-67`) treats a falsy `obs_mask` as the full `range(base_obs_size)`, so `None` is the right spelling of fully observed.
- It must be a **tuple**, not a list. `RTRRLParams` is declared `unsafe_hash=True` and holds the environment parameters frozen, so an unhashable field makes the whole dataclass unhashable.

- [ ] **Step 1: Write the failing test**

Append to `rtrrl/tests/test_entry.py`:

```python
RESERVED = frozenset(
    {
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
    }
)


def test_the_entry_declares_neither_the_environment_nor_the_budget():
    from entries import rtrrl_aaai

    assert not RESERVED & set(rtrrl_aaai.SPACE)


def _defaults() -> dict:
    """One value per declared parameter, taken as the first of each domain."""

    from entries import rtrrl_aaai

    chosen = {}
    for name, spec in rtrrl_aaai.SPACE.items():
        chosen[name] = spec[0] if isinstance(spec, list) else spec["low"]
    return chosen


def test_the_kept_indices_reach_their_obs_mask():
    from entries import rtrrl_aaai
    from training_sdk.contract import BudgetConfig, EnvironmentConfig

    chosen = rtrrl_aaai.settings(
        _defaults(),
        EnvironmentConfig(
            id="brax::hopper", backend="spring", num_envs=1, observed=(0, 1, 2, 3, 4)
        ),
        BudgetConfig(total_steps=2000, epoch_steps=1000, eval_steps=100),
    )

    assert chosen["environment"]["obs_mask"] == (0, 1, 2, 3, 4)
    assert chosen["environment"]["batch_size"] == 1


def test_a_fully_observed_task_asks_for_no_mask():
    from entries import rtrrl_aaai
    from training_sdk.contract import BudgetConfig, EnvironmentConfig

    chosen = rtrrl_aaai.settings(
        _defaults(),
        EnvironmentConfig(id="brax::hopper", backend="spring", num_envs=1),
        BudgetConfig(total_steps=2000, epoch_steps=1000, eval_steps=100),
    )

    assert chosen["environment"]["obs_mask"] is None
```

`settings` is the existing function at `rtrrl_aaai.py:154`; it returns the plain-value dictionary that `parameters` feeds to their dataclasses, and it is deliberately importable without a training framework, which is why the test can call it at all.

- [ ] **Step 2: Run the test to verify it fails**

```bash
gh workflow run build-aaai-image.yml --ref $(git branch --show-current)
gh run watch
```

Expected: the test step fails on the seven names still being declared.

- [ ] **Step 3: Take the environment and budget from the config**

Delete the seven keys from `SPACE`. Delete the `MASKS` table at lines 47 to 60 together with its comment; `SPACE["environment"]` was its only reader, and its comment cites `memo/tests/test_masking.py`, which Task 5 deletes.

Change the signature at line 154 and the three reads under it:

```python
def settings(
    params: Mapping[str, Any], environment, budget
) -> dict[str, Any]:
    """Every field of theirs this entry sets, as plain values."""

    scan_steps = int(params["scan_steps"])
    total = iterations(
        total_steps=budget.total_steps,
        scan_steps=scan_steps,
        num_envs=environment.num_envs,
    )
    per_epoch = iterations(
        total_steps=budget.epoch_steps,
        scan_steps=scan_steps,
        num_envs=environment.num_envs,
    )
```

Change `"eval_steps": int(params["eval_steps"])` to `"eval_steps": budget.eval_steps`.

Replace the environment block at lines 205 to 218 with:

```python
        "environment": {
            "env_name": environment.id.replace("::", "-"),
            "batch_size": environment.num_envs,
            "max_ep_length": MAX_EPISODE_LENGTH,
            "render": False,
            "obs_mask": tuple(environment.observed) if environment.observed else None,
            "env_kwargs": {"backend": environment.backend},
        },
```

The `replace("::", "-")` is theirs, not ours: `make_env` splits on the first hyphen and hands the rest to `brax.envs.get_environment`. Only the source of the string changes here.

Give `parameters` the same two arguments and pass them through:

```python
def parameters(params: Mapping[str, Any], environment, budget) -> RTRRLParams:
    """Assemble the dataclass their training function takes."""

    from envs.environments import EnvironmentParams
    from optimizers import OptimizerConfig
    from rtrrl import RTRRLParams

    chosen = dict(settings(params, environment, budget))
    return RTRRLParams(
        env_params=EnvironmentParams(**chosen.pop("environment")),
        optimizer_params_td=OptimizerConfig(**chosen.pop("td")),
        optimizer_params_rnn=OptimizerConfig(**chosen.pop("rnn")),
        **chosen,
    )
```

Change `run(reporter, params)` at line 247 to `run(reporter, config)`, have it call `parameters(config.params, config.environment, config.budget)`, and have `main` pass `reporter.config`.

- [ ] **Step 4: Regenerate the checked-in catalog**

`rtrrl/catalog.json` still declares contract 2 and the old space. It is what `memo/tests/test_experiments.py` validates the AAAI experiment file against in Task 8, so it has to be current before that task can go green.

```bash
cd rtrrl && uv run python scripts/build_catalog.py && cd ..
git diff rtrrl/catalog.json
```

Expected in the diff: `"contract"` moves from 2 to 3, and the seven names disappear from the `rtrrl_aaai` entry's space.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
git add rtrrl/entries/ rtrrl/tests/ rtrrl/catalog.json
git commit -m "feat(aaai): read the environment and the budget off the run config"
git push
gh workflow run build-aaai-image.yml --ref $(git branch --show-current)
gh run watch
```

Expected: the entry tests pass.

---

### Task 8: Every experiment file moves to the three sections

**Files:**
- Modify: all 25 files under `experiments/`
- Test: `memo/tests/test_experiments.py`

**Interfaces:**
- Consumes: the `Experiment` model from Task 2, the memo entry spaces from Task 6, and the regenerated `rtrrl/catalog.json` from Task 7. `test_experiments.py` reads the AAAI entry's space out of that file (`memo/tests/test_experiments.py:52`), so Task 7 must be finished before this one can pass.
- Produces: no experiment file names any of the seven; every file has an `environment` and a `budget` section.

- [ ] **Step 1: Write the failing test**

Append to `memo/tests/test_experiments.py`:

```python
RESERVED = frozenset(
    {
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
    }
)


def test_it_keeps_the_environment_and_the_budget_out_of_the_space(named):
    assert not RESERVED & named


def test_it_has_an_environment_and_a_budget(experiment):
    assert experiment["environment"]["id"]
    assert experiment["budget"]["total_steps"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
git add memo/tests/test_experiments.py
git commit -m "test(experiments): the environment and the budget are sections"
git push
gh run watch
```

Expected: fails for all 25 files.

- [ ] **Step 3: Convert every file**

For each file under `experiments/`, move the seven keys out of `space` into two new top-level sections placed after `storage`. The Hopper files all become:

```yaml
environment:
  id: brax::hopper
  backend: spring
  observed: [0, 1, 2, 3, 4]
  num_envs: 1

budget:
  total_steps: 2000000
  epoch_steps: 100000
  eval_steps: 1000
```

with `total_steps`, `epoch_steps`, `eval_steps` and `num_envs` copied from that file's own former space values rather than from the block above.

`observed` is derived from the file's former `env_mode` and `environment`. These are the index sets the deleted boolean table held, one row per task and mode. `F` keeps everything, so an `F` file omits `observed` entirely rather than listing every index.

| task | width | `P` | `V` |
|---|---|---|---|
| hopper | 11 | `[0, 1, 2, 3, 4]` | `[5, 6, 7, 8, 9, 10]` |
| walker2d | 17 | `[0, 1, 2, 3, 4, 5, 6, 7]` | `[8, 9, 10, 11, 12, 13, 14, 15, 16]` |
| halfcheetah | 17 | `[0, 1, 2, 3, 8, 9, 10, 11, 12]` | `[4, 5, 6, 7, 13, 14, 15, 16]` |
| ant | 27 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]` | `[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]` |

- [ ] **Step 4: Run the tests to verify they pass**

```bash
git add experiments/
git commit -m "exp: move the environment and the budget out of every search space"
git push
gh run watch
```

Expected: `test_experiments.py` passes for all 25 files, and the failure set is exactly the five known golden failures.

- [ ] **Step 5: Verify the control plane accepts one of them offline**

```bash
gh workflow run tests.yml --ref $(git branch --show-current)
gh run watch
```

Expected: all three jobs pass, including `test_examples.py` if it loads files from `experiments/`.

---

## What this plan does not do

No image is built and no experiment is launched. `CONTRACT_VERSION` is 3 as of Task 1, so both images must be rebuilt and their job definitions re-registered before any launch, and that happens after phase 2 rather than between these tasks.
