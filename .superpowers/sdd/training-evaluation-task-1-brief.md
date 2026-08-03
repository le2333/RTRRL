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

