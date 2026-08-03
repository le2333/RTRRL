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

