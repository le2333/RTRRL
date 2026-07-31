# Training and Evaluation Sections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `environment + budget + space.seed` layout with `environment + training + evaluation + space`, so seed, training loop shape, and evaluation shape are injected through the manifest rather than sampled as algorithm parameters.

**Architecture:** This is a contract migration from version 4 to version 5. `EnvironmentConfig` holds task identity, observation selection, and seed; `TrainingConfig` holds training streams and step budget; `EvaluationConfig` holds evaluation rollout length and evaluation streams. The control plane validates these sections, archives them, and writes them into `RunConfig`; entries read them from `config.environment`, `config.training`, and `config.evaluation`.

**Tech Stack:** Python 3.12, pydantic v2, Optuna, JAX/Flax, uv, GitHub Actions.

## Global Constraints

- Never run pytest or docker on this machine. Tests are written here and executed in GitHub Actions.
- Static checks are allowed: run `uv run ruff check` for the changed packages named in each task.
- Commit and push before triggering remote tests; `workflow_dispatch` runs against the remote ref.
- Work on `feature/rtrrl-lru-paper-parity`; do not commit to `main`.
- Stage explicit paths only. Do not use `git add -A` or `git add .`.
- Do not add rationale comments to code or configuration files.
- `CONTRACT_VERSION` becomes `5` in this plan.
- Existing historical files under `rtrrl/infra/control-plane/archive/` are not migrated.
- This plan does not introduce `param()`, `EntryDescriptor.parameters`, structure trees, conditional sampling, or OBGD decomposition. Those are handled by `2026-07-31-parameter-catalog-and-conditional-sampling.md`.

---

## File Structure

- `training-sdk/src/training_sdk/contract.py`: owns the shared wire contract. Replace `BudgetConfig` with `TrainingConfig` and `EvaluationConfig`; move `num_envs` from environment to training; add `seed` to environment.
- `rtrrl/infra/control-plane/src/trainer_infra/experiment.py`: mirrors the experiment YAML schema for preflight.
- `rtrrl/infra/control-plane/src/trainer_infra/preflight.py`: compares score windows against `training.total_steps`.
- `rtrrl/infra/control-plane/src/trainer_infra/launch.py`: archives `training` and `evaluation`; builds `RunConfig` with new sections.
- `memo/runner/loop.py`: keep `drive(reporter, *, init_fn, train_fn, evaluate_fn, total_steps, epoch_steps, eval_steps, num_envs, seed, training_report, eval_reward=None)` unchanged; callers translate from `TrainingConfig` and `EvaluationConfig`.
- `memo/entries/*.py`, `rtrrl/entries/rtrrl_aaai.py`, `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py`: stop reading seed and loop fields from `params`.
- `experiments/*.yaml`: migrate from `budget` to `training/evaluation`; move `space.seed` to `environment.seed`.
- Checked-in catalogs: `rtrrl/catalog.json` and `rtrrl/infra/mock-trainer/catalog.json` regenerated or minimally updated after entry space changes.

---

### Task 1: Shared Contract Version 5

**Files:**
- Modify: `training-sdk/src/training_sdk/contract.py`
- Modify: `training-sdk/tests/test_contract.py`
- Modify: `training-sdk/tests/test_reporter.py`
- Modify: `training-sdk/tests/test_worker.py`
- Modify: `training-sdk/tests/test_aim_sink.py`

**Interfaces:**
- Consumes: contract version 4.
- Produces: `EnvironmentConfig(id: str, backend: str, seed: int, observed: tuple[int, ...] | None)`, `TrainingConfig(num_envs: int, total_steps: int, epoch_steps: int, chunk_steps: int | None = None, early_stop_patience: int | None = None)`, `EvaluationConfig(steps: int, num_envs: int)`, and a `RunConfig` with `environment`, `training`, `evaluation`, and `params` fields.

- [ ] **Step 1: Write the failing contract tests**

In `training-sdk/tests/test_contract.py`, update the imports to replace `BudgetConfig` with `TrainingConfig` and `EvaluationConfig`, then add these tests:

```python
def test_contract_version_is_five() -> None:
    assert CONTRACT_VERSION == 5


def test_environment_carries_seed_but_not_training_streams() -> None:
    environment = EnvironmentConfig(
        id="brax::hopper", backend="spring", seed=7, observed=(0, 1, 2, 3, 4)
    )

    assert environment.seed == 7
    assert environment.observed == (0, 1, 2, 3, 4)
    assert "num_envs" not in environment.model_dump()


def test_training_must_divide_into_whole_epochs_and_stream_rounds() -> None:
    with pytest.raises(ValidationError, match="total_steps 1000"):
        TrainingConfig(total_steps=1000, epoch_steps=300, num_envs=1)

    with pytest.raises(ValidationError, match="epoch_steps 1000"):
        TrainingConfig(total_steps=2000, epoch_steps=1000, num_envs=3)


def test_chunk_steps_must_divide_total_and_epoch_when_present() -> None:
    with pytest.raises(ValidationError, match="chunk_steps"):
        TrainingConfig(total_steps=2000, epoch_steps=1000, num_envs=1, chunk_steps=300)

    training = TrainingConfig(
        total_steps=2000, epoch_steps=1000, num_envs=1, chunk_steps=1000
    )
    assert training.chunk_steps == 1000


def test_evaluation_names_rollout_length_and_parallel_streams() -> None:
    evaluation = EvaluationConfig(steps=1000, num_envs=10)

    assert evaluation.steps == 1000
    assert evaluation.num_envs == 10
```

- [ ] **Step 2: Commit and run the remote red check**

```bash
git add training-sdk/tests/test_contract.py
git commit -m "test(contract): require split training and evaluation sections"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: the training-sdk job fails because `TrainingConfig` and `EvaluationConfig` do not exist and `CONTRACT_VERSION` is still `4`.

- [ ] **Step 3: Implement the contract models**

In `training-sdk/src/training_sdk/contract.py`, set:

```python
CONTRACT_VERSION = 5
```

Replace `EnvironmentConfig` with:

```python
class EnvironmentConfig(_Frozen):
    id: str
    backend: str
    seed: int
    observed: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _usable(self) -> "EnvironmentConfig":
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        if self.observed is None:
            return self
        if not self.observed:
            raise ValueError("observed must name at least one index")
        if len(set(self.observed)) != len(self.observed):
            raise ValueError("observed must not repeat an index")
        if any(index < 0 for index in self.observed):
            raise ValueError("observed indices must not be negative")
        return self
```

Delete `BudgetConfig` and add:

```python
class TrainingConfig(_Frozen):
    num_envs: int
    total_steps: int
    epoch_steps: int
    chunk_steps: int | None = None
    early_stop_patience: int | None = None

    @model_validator(mode="after")
    def _whole(self) -> "TrainingConfig":
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.epoch_steps < 1:
            raise ValueError("epoch_steps must be positive")
        if self.total_steps % self.epoch_steps:
            raise ValueError(
                f"total_steps {self.total_steps} is not whole epochs of "
                f"{self.epoch_steps}"
            )
        if self.epoch_steps % self.num_envs:
            raise ValueError(
                f"epoch_steps {self.epoch_steps} is not "
                f"{self.num_envs} streams' worth"
            )
        if self.chunk_steps is not None:
            if self.chunk_steps < 1:
                raise ValueError("chunk_steps must be positive")
            per_chunk = self.chunk_steps * self.num_envs
            if self.total_steps % per_chunk or self.epoch_steps % per_chunk:
                raise ValueError(
                    f"chunk_steps {self.chunk_steps} over {self.num_envs} streams "
                    "must divide total_steps and epoch_steps"
                )
        if self.early_stop_patience is not None and self.early_stop_patience < 0:
            raise ValueError("early_stop_patience must not be negative")
        return self


class EvaluationConfig(_Frozen):
    steps: int
    num_envs: int

    @model_validator(mode="after")
    def _usable(self) -> "EvaluationConfig":
        if self.steps < 0:
            raise ValueError("evaluation steps must not be negative")
        if self.num_envs < 1:
            raise ValueError("evaluation num_envs must be positive")
        return self
```

In `RunConfig`, replace:

```python
    budget: BudgetConfig
```

with:

```python
    training: TrainingConfig
    evaluation: EvaluationConfig
```

and delete the `_epochs_hold_whole_rounds_of_streams` validator because `TrainingConfig` now owns it.

- [ ] **Step 4: Update training-sdk fixtures**

In `training-sdk/tests/test_contract.py`, update `run_config_kwargs()` to:

```python
        "environment": {
            "id": "brax::hopper",
            "backend": "spring",
            "seed": 0,
        },
        "training": {"num_envs": 1, "total_steps": 100, "epoch_steps": 100},
        "evaluation": {"steps": 0, "num_envs": 1},
```

and keep `params` as algorithm-only:

```python
        "params": {"learning_rate": 0.0003},
```

Update every `RunConfig` fixture under `training-sdk/tests/` by replacing `budget` with `training/evaluation` and moving any `environment.num_envs` to `training.num_envs`.

- [ ] **Step 5: Run static checks and commit**

```bash
uv run ruff check training-sdk
git add training-sdk/src/training_sdk/contract.py training-sdk/tests/test_contract.py training-sdk/tests/test_reporter.py training-sdk/tests/test_worker.py training-sdk/tests/test_aim_sink.py
git commit -m "feat(contract): split training and evaluation run sections"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: training-sdk tests pass. Control-plane and mock-trainer jobs may fail until Tasks 2 and 3 are complete.

---

### Task 2: Control Plane Experiment and Launch Schema

**Files:**
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/experiment.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/preflight.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/launch.py`
- Modify: `rtrrl/infra/control-plane/tests/helpers.py`
- Modify: `rtrrl/infra/control-plane/tests/test_experiment.py`
- Modify: `rtrrl/infra/control-plane/tests/test_preflight_offline.py`
- Modify: `rtrrl/infra/control-plane/tests/test_launch.py`
- Modify: `rtrrl/infra/control-plane/tests/data/experiment.yaml`

**Interfaces:**
- Consumes: `TrainingConfig` and `EvaluationConfig` from Task 1.
- Produces: `Experiment.environment.seed`, `Experiment.training`, `Experiment.evaluation`; `LaunchPlan.space` remains the old flat resolved space for this plan.

- [ ] **Step 1: Write failing experiment-model tests**

In `rtrrl/infra/control-plane/tests/test_experiment.py`, replace budget assertions with training/evaluation assertions:

```python
def test_an_experiment_carries_environment_training_and_evaluation(tmp_path):
    document = _document()
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    experiment = load_experiment(path)

    assert experiment.environment.seed == 0
    assert experiment.environment.observed == (0, 1, 2, 3, 4)
    assert experiment.training.total_steps == 2000
    assert experiment.evaluation.steps == 100
```

Add a reserved-name test that includes all old and new non-space names:

```python
@pytest.mark.parametrize(
    "reserved",
    [
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "seed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
        "chunk_steps",
        "early_stop_patience",
        "eval_envs",
    ],
)
def test_a_space_may_not_name_non_algorithm_fields(tmp_path: Path, reserved: str) -> None:
    document = _document()
    document["space"][reserved] = [1]
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValidationError) as raised:
        load_experiment(path)

    assert reserved in str(raised.value)
```

- [ ] **Step 2: Commit and run the remote red check**

```bash
git add rtrrl/infra/control-plane/tests/test_experiment.py
git commit -m "test(control-plane): require training and evaluation sections"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: control-plane tests fail because `Experiment` still expects `budget`.

- [ ] **Step 3: Implement the experiment model**

In `rtrrl/infra/control-plane/src/trainer_infra/experiment.py`, update `RESERVED` to:

```python
RESERVED = frozenset(
    {
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "seed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
        "chunk_steps",
        "early_stop_patience",
        "eval_envs",
    }
)
```

Replace `Environment` with the same fields and validation as `EnvironmentConfig` from Task 1. Delete `Budget` and add `Training` and `Evaluation` models with the same validation as `TrainingConfig` and `EvaluationConfig`.

In `Experiment`, replace:

```python
    budget: Budget
```

with:

```python
    training: Training
    evaluation: Evaluation
```

and remove the `budget.epoch_steps % environment.num_envs` check from the experiment validator.

- [ ] **Step 4: Update control-plane helpers and sample YAML**

In `rtrrl/infra/control-plane/tests/helpers.py`, make `_document()` use:

```python
        "environment": {
            "id": "brax::hopper",
            "backend": "spring",
            "seed": 0,
            "observed": [0, 1, 2, 3, 4],
        },
        "training": {"num_envs": 1, "total_steps": 2000, "epoch_steps": 1000},
        "evaluation": {"steps": 100, "num_envs": 1},
```

Update `rtrrl/infra/control-plane/tests/data/experiment.yaml` with the same section shape. Its `space` must not contain `seed`, `num_envs`, `total_steps`, `epoch_steps`, or `eval_steps`.

- [ ] **Step 5: Update preflight and launch**

In `rtrrl/infra/control-plane/src/trainer_infra/preflight.py`, replace budget reads with:

```python
    if experiment.score.window_steps[1] > experiment.training.total_steps:
        raise PreflightError(
            f"score window upper bound {experiment.score.window_steps[1]} exceeds "
            f"the training total_steps ({experiment.training.total_steps})"
        )
```

In `rtrrl/infra/control-plane/src/trainer_infra/launch.py`, archive:

```python
        "environment": experiment.environment.model_dump(mode="json"),
        "training": experiment.training.model_dump(mode="json"),
        "evaluation": experiment.evaluation.model_dump(mode="json"),
```

and build `RunConfig` with:

```python
        environment=EnvironmentConfig(
            id=experiment.environment.id,
            backend=experiment.environment.backend,
            seed=experiment.environment.seed,
            observed=experiment.environment.observed,
        ),
        training=TrainingConfig(
            num_envs=experiment.training.num_envs,
            total_steps=experiment.training.total_steps,
            epoch_steps=experiment.training.epoch_steps,
            chunk_steps=experiment.training.chunk_steps,
            early_stop_patience=experiment.training.early_stop_patience,
        ),
        evaluation=EvaluationConfig(
            steps=experiment.evaluation.steps,
            num_envs=experiment.evaluation.num_envs,
        ),
```

Update imports from `training_sdk.contract` accordingly.

- [ ] **Step 6: Update control-plane tests**

Update assertions in `test_experiment.py`, `test_preflight_offline.py`, and `test_launch.py`:

```python
assert experiment.training.total_steps == 2000
assert config.training.total_steps == 2000
assert config.evaluation.steps == 100
assert archived["training"]["epoch_steps"] == 1000
assert archived["evaluation"]["steps"] == 100
```

Remove assertions that read `experiment.budget` or `config.budget`.

- [ ] **Step 7: Run static checks and commit**

```bash
uv run ruff check rtrrl/infra/control-plane
git add rtrrl/infra/control-plane/src/trainer_infra/experiment.py rtrrl/infra/control-plane/src/trainer_infra/preflight.py rtrrl/infra/control-plane/src/trainer_infra/launch.py rtrrl/infra/control-plane/tests rtrrl/infra/control-plane/tests/data/experiment.yaml
git commit -m "feat(control-plane): model training and evaluation sections"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: control-plane tests pass. Mock-trainer and entry tests may still fail until Tasks 3 and 4.

---

### Task 3: Mock Trainer Consumes Injected Seed and Training Budget

**Files:**
- Modify: `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py`
- Modify: `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/space.py`
- Modify: `rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py`
- Modify: `rtrrl/infra/mock-trainer/tests/test_train.py`
- Modify: `rtrrl/infra/mock-trainer/catalog.json`

**Interfaces:**
- Consumes: `RunConfig.environment.seed`, `RunConfig.training.num_envs`, `RunConfig.training.total_steps`.
- Produces: mock acceptance `SPACE` without `seed` or `total_steps`.

- [ ] **Step 1: Write failing mock-trainer tests**

In `rtrrl/infra/mock-trainer/tests/test_train.py`, add:

```python
def test_acceptance_config_reads_seed_and_budget_from_run_sections(run_config):
    config = AcceptanceConfig.from_run_config(run_config)

    assert config.seed == run_config.environment.seed
    assert config.num_envs == run_config.training.num_envs
    assert config.num_timesteps == run_config.training.total_steps
```

Update the `run_config` fixture to use `environment.seed`, `training`, and `evaluation`, and remove `seed` from `params`.

- [ ] **Step 2: Commit and run the remote red check**

```bash
git add rtrrl/infra/mock-trainer/tests/test_train.py
git commit -m "test(mock-trainer): read injected seed and training budget"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: mock-trainer tests fail because `AcceptanceConfig.from_run_config()` still reads `params["seed"]`, `config.environment.num_envs`, and `config.budget.total_steps`.

- [ ] **Step 3: Update mock-trainer config mapping**

In `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py`, replace the algorithm and budget construction inside `from_run_config()` with:

```python
        algorithm = {
            "learning_rate": params["learning_rate"],
            "num_envs": config.training.num_envs,
            "episode_length": params.get("episode_length", 32),
            "failure_mode": params.get("failure_mode", "none"),
        }
        parameters = {
            "runtime": {"seed": config.environment.seed},
            "algorithm": algorithm,
        }
        training_budget = {"env_steps": config.training.total_steps}
```

Remove every read of `config.environment.num_envs`, `config.budget.total_steps`, and `params["seed"]`.

- [ ] **Step 4: Remove non-algorithm fields from mock catalog source**

In `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/space.py`, delete `seed` and `total_steps` from `SPACE`. Keep `learning_rate`, `episode_length`, and `failure_mode`.

Regenerate the mock catalog:

```bash
uv run --project rtrrl/infra/mock-trainer python rtrrl/infra/mock-trainer/scripts/build_catalog.py
```

Expected diff: `rtrrl/infra/mock-trainer/catalog.json` has `"contract": 5`, and the acceptance entry no longer declares `seed` or `total_steps`.

- [ ] **Step 5: Update runtime fixture**

In `rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py`, change the JSON fixture:

```json
"environment": {
  "id": "brax::inverted_pendulum",
  "backend": "generalized",
  "seed": 7
},
"training": {
  "num_envs": 4,
  "total_steps": 128,
  "epoch_steps": 128
},
"evaluation": {
  "steps": 0,
  "num_envs": 1
},
"params": {
  "learning_rate": 0.0003,
  "episode_length": 32,
  "failure_mode": "none"
}
```

- [ ] **Step 6: Run static checks and commit**

```bash
uv run ruff check rtrrl/infra/mock-trainer
git add rtrrl/infra/mock-trainer/src rtrrl/infra/mock-trainer/tests rtrrl/infra/mock-trainer/catalog.json
git commit -m "feat(mock-trainer): consume injected run shape"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: the mock-trainer job passes.

---

### Task 4: Memo and AAAI Entries Stop Declaring Injected Fields

**Files:**
- Modify: `memo/entries/rtrrl.py`
- Modify: `memo/entries/stream_ac.py`
- Modify: `memo/entries/upstream_stream_ac.py`
- Modify: `rtrrl/entries/rtrrl_aaai.py`
- Modify: `memo/tests/test_entries.py`
- Modify: `rtrrl/tests/test_entry.py`
- Modify: `rtrrl/catalog.json`

**Interfaces:**
- Consumes: `config.environment.seed`, `config.training`, and `config.evaluation`.
- Produces: entry `SPACE` without `seed`; AAAI `SPACE` additionally without `scan_steps`, `eval_envs`, or `patience`.

- [ ] **Step 1: Write failing entry tests**

In `memo/tests/test_entries.py`, update `RESERVED` to include `seed`, `chunk_steps`, `early_stop_patience`, and `eval_envs`:

```python
RESERVED = frozenset(
    {
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "seed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
        "chunk_steps",
        "early_stop_patience",
        "eval_envs",
    }
)
```

In `rtrrl/tests/test_entry.py`, use the same reserved set and add:

```python
def test_aaai_entry_declares_no_injected_runtime_fields():
    from entries import rtrrl_aaai

    assert not RESERVED & set(rtrrl_aaai.SPACE)
```

- [ ] **Step 2: Commit and run the remote red checks**

```bash
git add memo/tests/test_entries.py rtrrl/tests/test_entry.py
git commit -m "test(entries): injected run fields are not parameters"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
gh workflow run build-aaai-image.yml --ref "$(git branch --show-current)"
```

Expected: memo entry tests fail on `seed`; AAAI entry tests fail on `seed`, `scan_steps`, `eval_envs`, and `patience`.

- [ ] **Step 3: Update memo entries**

In each of `memo/entries/rtrrl.py`, `memo/entries/stream_ac.py`, and `memo/entries/upstream_stream_ac.py`:

Delete `"seed"` from `SPACE`.

In `run(reporter, config)`, pass injected fields to `drive()`:

```python
        total_steps=config.training.total_steps,
        epoch_steps=config.training.epoch_steps,
        eval_steps=config.evaluation.steps,
        num_envs=config.training.num_envs,
        seed=config.environment.seed,
```

In each `build(params, environment)` call site, use `config.training.num_envs` where the agent config needs training stream count. The cleanest signature is:

```python
def build(params: Mapping[str, Any], environment, training) -> StreamAC:
```

and the caller:

```python
agent = build(config.params, config.environment, config.training)
```

For `rtrrl.py`, use the same pattern:

```python
def build(params: Mapping[str, Any], environment, training) -> RTRRL:
```

and set `num_envs=training.num_envs` in `RTRRLConfig`.

- [ ] **Step 4: Update AAAI entry**

In `rtrrl/entries/rtrrl_aaai.py`, delete `"seed"`, `"scan_steps"`, `"eval_envs"`, and `"patience"` from `SPACE`.

Replace `settings()` with:

```python
def settings(params: Mapping[str, Any], environment, training, evaluation) -> dict[str, Any]:
    """Every field of theirs this entry sets, as plain values."""

    chunk_steps = int(training.chunk_steps or training.epoch_steps)
    total = iterations(
        total_steps=training.total_steps,
        scan_steps=chunk_steps,
        num_envs=training.num_envs,
    )
    per_epoch = iterations(
        total_steps=training.epoch_steps,
        scan_steps=chunk_steps,
        num_envs=training.num_envs,
    )
    return {
        "seed": environment.seed,
        "episodes": total,
        "steps": chunk_steps,
        "patience": 0 if training.early_stop_patience is None else training.early_stop_patience,
        "eval_every": per_epoch,
        "eval_steps": evaluation.steps,
        "eval_batch_size": evaluation.num_envs,
        "rnn_model": str(params["backbone"]),
        "gradient_mode": str(params["gradient_mode"]),
        "hidden_size": int(params["hidden_dim"]),
        "meta_rl": bool(params["meta_rl"]),
        "f_align": bool(params["f_align"]),
        "mlp_actor": bool(params["mlp_actor"]),
        "layer_norm": bool(params["layer_norm"]),
        "normalize_obs": bool(params["normalize_observation"]),
        "normalize_reward": bool(params["normalize_reward"]),
        "trace_mode": str(params["trace_mode"]),
        "gamma": float(params["gamma"]),
        "lambda_pi": float(params["lambda_pi"]),
        "lambda_v": float(params["lambda_v"]),
        "lambda_rnn": float(params["lambda_rnn"]),
        "eta_pi": float(params["eta_pi"]),
        "eta_f": float(params["eta_f"]),
        "entropy_rate": float(params["entropy_rate"]),
        "update_period": float(params["update_period"]),
        "update_trace_before_td": bool(params["update_trace_before_td"]),
        "environment": {
            "env_name": environment.id.replace("::", "-"),
            "batch_size": training.num_envs,
            "max_ep_length": MAX_EPISODE_LENGTH,
            "render": False,
            "obs_mask": tuple(environment.observed) if environment.observed else None,
            "env_kwargs": {"backend": environment.backend},
        },
        "td": {"opt_name": "adam", "learning_rate": float(params["td_lr"])},
        "rnn": {
            "opt_name": "adam",
            "learning_rate": float(params["rnn_lr"]),
            "gradient_clip": float(params["rnn_grad_clip"]),
        },
    }
```

Replace `parameters()` with:

```python
def parameters(params: Mapping[str, Any], environment, training, evaluation) -> RTRRLParams:
    """Assemble the dataclass their training function takes."""

    from envs.environments import EnvironmentParams
    from optimizers import OptimizerConfig
    from rtrrl import RTRRLParams

    chosen = dict(settings(params, environment, training, evaluation))
    return RTRRLParams(
        env_params=EnvironmentParams(**chosen.pop("environment")),
        optimizer_params_td=OptimizerConfig(**chosen.pop("td")),
        optimizer_params_rnn=OptimizerConfig(**chosen.pop("rnn")),
        **chosen,
    )
```

Replace the body of `run()` with:

```python
    logger: Any = ReporterLogger(reporter)
    train_rtrrl(
        parameters(config.params, config.environment, config.training, config.evaluation),
        logger,
    )
```

- [ ] **Step 5: Regenerate the AAAI catalog**

```bash
cd rtrrl
uv run python scripts/build_catalog.py
cd ..
```

Expected diff: `rtrrl/catalog.json` has `"contract": 5`; `rtrrl_aaai` no longer declares `seed`, `scan_steps`, `eval_envs`, or `patience`.

- [ ] **Step 6: Run static checks and commit**

```bash
uv run ruff check memo rtrrl/entries rtrrl/tests
git add memo/entries memo/tests/test_entries.py rtrrl/entries/rtrrl_aaai.py rtrrl/tests/test_entry.py rtrrl/catalog.json
git commit -m "feat(entries): consume injected seed and run shape"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
gh workflow run build-aaai-image.yml --ref "$(git branch --show-current)"
```

Expected: entry and AAAI checks pass, except for the already sanctioned memo golden failures if Memo CI runs.

---

### Task 5: Experiment YAML Migration

**Files:**
- Modify: `experiments/*.yaml`
- Modify: `memo/tests/test_experiments.py`
- Modify: `rtrrl/infra/control-plane/tests/test_experiments.py`
- Modify: `docs/trainerctl-manual.md`

**Interfaces:**
- Consumes: Experiment schema from Task 2 and entry spaces from Task 4.
- Produces: every trainerctl experiment file uses `environment.seed`, `training`, and `evaluation`; no `space.seed`, `budget`, `scan_steps`, `eval_envs`, or `patience` remains in `experiments/*.yaml`.

- [ ] **Step 1: Write failing YAML tests**

In `memo/tests/test_experiments.py`, replace the existing reserved set with:

```python
RESERVED = frozenset(
    {
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "seed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
        "chunk_steps",
        "early_stop_patience",
        "eval_envs",
    }
)
```

Add:

```python
def test_it_has_training_and_evaluation_sections(experiment):
    assert "budget" not in experiment
    assert experiment["environment"]["seed"] >= 0
    assert experiment["training"]["total_steps"] > 0
    assert experiment["evaluation"]["num_envs"] > 0
```

- [ ] **Step 2: Commit and run the remote red check**

```bash
git add memo/tests/test_experiments.py
git commit -m "test(experiments): require injected seed and run sections"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: experiment tests fail because files still use `budget` and `space.seed`.

- [ ] **Step 3: Convert every experiment file**

For each file in `experiments/*.yaml`, copy the actual scalar values already present in that file. A standard two-million-step Hopper file currently shaped as:

```yaml
environment:
  id: brax::hopper
  backend: spring
  observed: [0, 1, 2, 3, 4]
  num_envs: 16

budget:
  total_steps: 2000000
  epoch_steps: 100000
  eval_steps: 1000

space:
  seed: [0]
```

becomes:

```yaml
environment:
  id: brax::hopper
  backend: spring
  observed: [0, 1, 2, 3, 4]
  seed: 0

training:
  num_envs: 16
  total_steps: 2000000
  epoch_steps: 100000

evaluation:
  steps: 1000
  num_envs: 1

space: {}
```

If the file has different `total_steps`, `epoch_steps`, `eval_steps`, `num_envs`, or `seed`, use that file's values exactly.

and remove `seed` from `space`.

For `experiments/rtrrl-hopper-aaai.yaml`, also move:

```yaml
  scan_steps: [1000]
  eval_envs: [10]
  patience: [0]
```

to:

```yaml
training:
  chunk_steps: 1000
  early_stop_patience: 0

evaluation:
  num_envs: 10
```

using the existing `training.total_steps`, `training.epoch_steps`, and `evaluation.steps` values in that file.

For seed sweep files such as `experiments/streamac-hopper-seeds-ours-point-on-ours.yaml`, do not keep multiple seeds in `space`. Create one YAML per seed by copying the file and suffixing `-seed0` through `-seed4`, each with one `environment.seed`. Delete the original multi-seed file after the copies exist. This plan deliberately does not define multi-seed aggregation.

- [ ] **Step 4: Update docs**

In `docs/trainerctl-manual.md`, update every example that shows `budget` or `space.seed` to the new shape:

```yaml
environment:
  id: brax::hopper
  backend: spring
  observed: [0, 1, 2, 3, 4]
  seed: 0

training:
  num_envs: 16
  total_steps: 2000000
  epoch_steps: 100000

evaluation:
  steps: 1000
  num_envs: 1

space: {}
```

- [ ] **Step 5: Run static checks and commit**

```bash
uv run ruff check memo rtrrl/infra/control-plane
git add -- experiments/*.yaml memo/tests/test_experiments.py rtrrl/infra/control-plane/tests/test_experiments.py docs/trainerctl-manual.md
git commit -m "exp: move seed and run shape out of search spaces"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: `tests.yml` succeeds for training-sdk, control-plane, and mock-trainer. Memo CI may still report the known five golden failures and no others.

---

## Self-Review

**Spec coverage.** This plan covers spec §1's four YAML sections, `environment.seed`, `training.chunk_steps`, `training.early_stop_patience`, `evaluation.num_envs`, score-window validation against `training.total_steps`, and the decision that these values are not HPO dimensions. It covers §3's statement that seed and run-shape fields are not algorithm declarations. It does not cover `param()` declarations, catalog `parameters`, conditional sampling, `eps` splitting, OBGD decomposition, or metrics.

**Placeholder scan.** No steps contain open-ended implementation text. Each task names exact files, explicit signatures, and expected remote verification results.

**Type consistency.** The same names are used throughout: `EnvironmentConfig.seed`, `TrainingConfig.num_envs`, `TrainingConfig.total_steps`, `TrainingConfig.epoch_steps`, `TrainingConfig.chunk_steps`, `TrainingConfig.early_stop_patience`, `EvaluationConfig.steps`, and `EvaluationConfig.num_envs`.
