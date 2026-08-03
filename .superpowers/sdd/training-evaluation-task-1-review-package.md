# Review package: training/evaluation Task 1

## Commits
```
9740b67 docs(sdd): report training evaluation contract migration
51e21a2 feat(contract): split training and evaluation run sections
2db4224 test(contract): require split training and evaluation sections
```

## Name status
```
A	.superpowers/sdd/training-evaluation-task-1-report.md
M	training-sdk/src/training_sdk/contract.py
M	training-sdk/tests/test_aim_sink.py
M	training-sdk/tests/test_contract.py
M	training-sdk/tests/test_reporter.py
M	training-sdk/tests/test_worker.py
```

## Stat
```
 .../sdd/training-evaluation-task-1-report.md       | 66 ++++++++++++++++++++++
 training-sdk/src/training_sdk/contract.py          | 61 +++++++++++++-------
 training-sdk/tests/test_aim_sink.py                |  2 +-
 training-sdk/tests/test_contract.py                | 52 +++++++++++------
 training-sdk/tests/test_reporter.py                |  9 +--
 training-sdk/tests/test_worker.py                  |  2 +-
 6 files changed, 150 insertions(+), 42 deletions(-)
```

## Diff
```diff
diff --git a/.superpowers/sdd/training-evaluation-task-1-report.md b/.superpowers/sdd/training-evaluation-task-1-report.md
new file mode 100644
index 0000000..73e81b1
--- /dev/null
+++ b/.superpowers/sdd/training-evaluation-task-1-report.md
@@ -0,0 +1,66 @@
+# Training/Evaluation Task 1 Report
+
+## Scope and files changed
+
+- `training-sdk/src/training_sdk/contract.py`
+  - Raised `CONTRACT_VERSION` from 4 to 5.
+  - Replaced environment stream count with the required non-negative `seed`.
+  - Replaced `BudgetConfig` with `TrainingConfig` and `EvaluationConfig`, including
+    whole-epoch, stream-round, chunk, and early-stop validation.
+  - Split `RunConfig.budget` into `training` and `evaluation` and removed the
+    validator whose ownership moved into `TrainingConfig`.
+- `training-sdk/tests/test_contract.py`
+  - Added the v5 contract tests and updated the shared RunConfig fixture.
+- `training-sdk/tests/test_reporter.py`
+  - Updated the RunConfig fixture to v5 fields and algorithm-only params.
+- `training-sdk/tests/test_worker.py`
+  - Updated the catalog fixture to contract version 5.
+- `training-sdk/tests/test_aim_sink.py`
+  - Updated the Aim assertion for the algorithm-only parameter fixture.
+
+## Test and check results
+
+| Command | Result | Full summary |
+| --- | --- | --- |
+| `uv run pytest training-sdk/tests/test_contract.py` (repository root) | FAIL | `pytest` was not found in the root environment: `Failed to spawn: pytest; program not found`. |
+| `uv run pytest tests/test_contract.py` (`training-sdk`) | FAIL | uv could not read its shared cache: `C:\Users\le233\AppData\Local\uv\cache\sdists-v9\.git: Access denied`. |
+| `uv sync --all-groups` (`training-sdk`) | NOT RUN | The required sandbox escalation was rejected because the machine instructions prohibit local dependency synchronization. |
+| `uv run ruff check .; uv run pytest tests/test_contract.py tests/test_reporter.py tests/test_worker.py tests/test_aim_sink.py` (`training-sdk`) | FAIL | uv downloaded CPython and created `.venv`, but dependency resolution stopped before either command ran: `aimrocks==0.5.2` has no source distribution or wheel for Windows `win_amd64`. |
+| `uvx ruff check .` (`training-sdk`) | FAIL | uv again could not read its shared cache (`sdists-v9\.git: Access denied`). |
+| `git diff --check` | PASS | Exit code 0; no whitespace errors reported. |
+
+The red test phase was written before production changes. Its focused pytest
+invocation could not reach collection because the local Python environment was
+unavailable, so its expected missing-import failure could not be observed.
+
+## Commits
+
+- `2db4224` — `test(contract): require split training and evaluation sections`
+- `51e21a2` — `feat(contract): split training and evaluation run sections`
+
+## Remote CI and push
+
+`git push origin HEAD` was attempted and failed because the sandbox could not
+connect to GitHub. A retry requesting external access was rejected because the
+destination and export were not explicitly authorized by the environment.
+Consequently, no `gh workflow run tests.yml` invocation was attempted and no
+remote CI result is available.
+
+## Self-review notes
+
+- `EnvironmentConfig` exactly carries `id`, `backend`, `seed`, and optional
+  `observed`; `num_envs` is rejected by the model's existing `extra="forbid"`
+  policy.
+- `TrainingConfig` owns all training stream divisibility checks, including
+  `chunk_steps * num_envs` dividing both totals.
+- `EvaluationConfig` uses the requested rollout `steps` and its own `num_envs`.
+- All listed test fixtures now construct v5 `RunConfig` objects with separate
+  `training` and `evaluation` fields and no training budget in `params`.
+
+## Concerns
+
+- The required pytest and ruff checks could not execute locally because the
+  locked Aim dependency does not support Windows and the shared uv cache is
+  inaccessible in the sandbox.
+- Remote validation remains pending because GitHub push/workflow authorization
+  was denied.
diff --git a/training-sdk/src/training_sdk/contract.py b/training-sdk/src/training_sdk/contract.py
index ba9df63..17f7ca0 100644
--- a/training-sdk/src/training_sdk/contract.py
+++ b/training-sdk/src/training_sdk/contract.py
@@ -1,17 +1,17 @@
 from __future__ import annotations
 
 from typing import Annotated, Literal, TypeAlias
 
 from pydantic import BaseModel, ConfigDict, Field, model_validator
 
-CONTRACT_VERSION = 4
+CONTRACT_VERSION = 5
 
 Scalar: TypeAlias = int | float | str | bool
 
 
 class _Frozen(BaseModel):
     model_config = ConfigDict(frozen=True, extra="forbid")
 
 
 class FloatSpec(_Frozen):
     type: Literal["float"]
@@ -79,56 +79,87 @@ class EntryDescriptor(_Frozen):
 
 
 class Catalog(_Frozen):
     contract: int
     entries: dict[str, EntryDescriptor]
 
 
 class EnvironmentConfig(_Frozen):
     id: str
     backend: str
-    num_envs: int
+    seed: int
     observed: tuple[int, ...] | None = None
 
     @model_validator(mode="after")
     def _usable(self) -> "EnvironmentConfig":
-        if self.num_envs < 1:
-            raise ValueError("num_envs must be positive")
+        if self.seed < 0:
+            raise ValueError("seed must not be negative")
         if self.observed is None:
             return self
         if not self.observed:
             raise ValueError("observed must name at least one index")
         if len(set(self.observed)) != len(self.observed):
             raise ValueError("observed must not repeat an index")
         if any(index < 0 for index in self.observed):
             raise ValueError("observed indices must not be negative")
         return self
 
 
-class BudgetConfig(_Frozen):
+class TrainingConfig(_Frozen):
+    num_envs: int
     total_steps: int
     epoch_steps: int
-    eval_steps: int
+    chunk_steps: int | None = None
+    early_stop_patience: int | None = None
 
     @model_validator(mode="after")
-    def _whole(self) -> "BudgetConfig":
+    def _whole(self) -> "TrainingConfig":
+        if self.num_envs < 1:
+            raise ValueError("num_envs must be positive")
         if self.total_steps < 1:
             raise ValueError("total_steps must be positive")
         if self.epoch_steps < 1:
             raise ValueError("epoch_steps must be positive")
-        if self.eval_steps < 0:
-            raise ValueError("eval_steps must not be negative")
         if self.total_steps % self.epoch_steps:
             raise ValueError(
                 f"total_steps {self.total_steps} is not whole epochs of "
                 f"{self.epoch_steps}"
             )
+        if self.epoch_steps % self.num_envs:
+            raise ValueError(
+                f"epoch_steps {self.epoch_steps} is not "
+                f"{self.num_envs} streams' worth"
+            )
+        if self.chunk_steps is not None:
+            if self.chunk_steps < 1:
+                raise ValueError("chunk_steps must be positive")
+            per_chunk = self.chunk_steps * self.num_envs
+            if self.total_steps % per_chunk or self.epoch_steps % per_chunk:
+                raise ValueError(
+                    f"chunk_steps {self.chunk_steps} over {self.num_envs} streams "
+                    "must divide total_steps and epoch_steps"
+                )
+        if self.early_stop_patience is not None and self.early_stop_patience < 0:
+            raise ValueError("early_stop_patience must not be negative")
+        return self
+
+
+class EvaluationConfig(_Frozen):
+    steps: int
+    num_envs: int
+
+    @model_validator(mode="after")
+    def _usable(self) -> "EvaluationConfig":
+        if self.steps < 0:
+            raise ValueError("evaluation steps must not be negative")
+        if self.num_envs < 1:
+            raise ValueError("evaluation num_envs must be positive")
         return self
 
 
 class LoggingConfig(_Frozen):
     aim: str
     every_steps: int
     rerun_s3: str | None = None
     rerun_every_episodes: int | None = None
 
 
@@ -150,23 +181,15 @@ class ScoreConfig(_Frozen):
 class RunConfig(_Frozen):
     contract: int
     run_id: str
     experiment: str
     name: str
     launch_id: str
     trial: int
     entry: str
     digest: str
     environment: EnvironmentConfig
-    budget: BudgetConfig
+    training: TrainingConfig
+    evaluation: EvaluationConfig
     params: dict[str, Scalar]
     logging: LoggingConfig
     score: ScoreConfig
-
-    @model_validator(mode="after")
-    def _epochs_hold_whole_rounds_of_streams(self) -> "RunConfig":
-        if self.budget.epoch_steps % self.environment.num_envs:
-            raise ValueError(
-                f"epoch_steps {self.budget.epoch_steps} is not "
-                f"{self.environment.num_envs} streams' worth"
-            )
-        return self
diff --git a/training-sdk/tests/test_aim_sink.py b/training-sdk/tests/test_aim_sink.py
index bb7bd4a..b191747 100644
--- a/training-sdk/tests/test_aim_sink.py
+++ b/training-sdk/tests/test_aim_sink.py
@@ -39,21 +39,21 @@ def test_run_is_named_by_run_id_and_carries_launch_fields(tmp_path: Path) -> Non
 
     repo = Repo.from_path(repo_path)
     runs = list(repo.iter_runs())
     assert len(runs) == 1
     run = runs[0]
     assert run.name == config.run_id
     assert run["launch_id"] == config.launch_id
     assert run["trial"] == config.trial
     assert run["entry"] == config.entry
     assert run["digest"] == config.digest
-    assert run["params"]["total_steps"] == 4
+    assert run["params"]["learning_rate"] == 0.0003
     values = list(run.metrics())
     assert values, "the metric sequence must exist"
     close_aim_run(run)
 
 
 def test_read_only_run_close_raises_sequence_infos_attribute_error(tmp_path: Path) -> None:
     # Documents Aim 3.28.0: Run.close() on a read-only run fails because
     # RunTracker never creates sequence_infos. If this test fails, Aim fixed
     # the bug and close_aim_run() can be retired.
     repo_path = str(tmp_path / "aim")
diff --git a/training-sdk/tests/test_contract.py b/training-sdk/tests/test_contract.py
index 9783fea..db34b9c 100644
--- a/training-sdk/tests/test_contract.py
+++ b/training-sdk/tests/test_contract.py
@@ -1,58 +1,60 @@
 import pytest
 from pydantic import ValidationError
 
 from training_sdk.contract import (
-    BudgetConfig,
     CONTRACT_VERSION,
     Catalog,
     ChoiceSpec,
     EntryDescriptor,
     EnvironmentConfig,
+    EvaluationConfig,
     FloatSpec,
     IntSpec,
     RunConfig,
     ScoreConfig,
+    TrainingConfig,
 )
 
 
 def run_config_kwargs() -> dict:
     return {
-        "contract": 4,
+        "contract": 5,
         "run_id": "sweep-20260725-051400-t7",
         "experiment": "locomotion",
         "name": "sweep",
         "launch_id": "20260725-051400",
         "trial": 7,
         "entry": "brax_ppo",
         "digest": "registry.example/trainer@sha256:" + "a" * 64,
         "environment": {
             "id": "brax::hopper",
             "backend": "spring",
-            "num_envs": 1,
+            "seed": 0,
         },
-        "budget": {"total_steps": 100, "epoch_steps": 100, "eval_steps": 0},
-        "params": {"total_steps": 128, "learning_rate": 0.0003},
+        "training": {"num_envs": 1, "total_steps": 100, "epoch_steps": 100},
+        "evaluation": {"steps": 0, "num_envs": 1},
+        "params": {"learning_rate": 0.0003},
         "logging": {"aim": "aim://127.0.0.1:53801", "every_steps": 1},
         "score": {
             "metric": "episode_return",
             "window_steps": [0, 128],
             "reduce": "mean",
             "direction": "maximize",
             "non_finite": "worst",
             "s3": "s3://bucket/score.json",
         },
     }
 
 
-def test_contract_version_is_four() -> None:
-    assert CONTRACT_VERSION == 4
+def test_contract_version_is_five() -> None:
+    assert CONTRACT_VERSION == 5
 
 
 def test_catalog_parses_float_int_and_choice_entries() -> None:
     catalog = Catalog.model_validate(
         {
             "contract": 2,
             "entries": {
                 "brax_ppo": {
                     "command": ["python", "-m", "brax_ppo.train"],
                     "metrics": ["episode_return"],
@@ -175,43 +177,59 @@ def test_a_run_config_has_no_source_hash() -> None:
 
 
 def test_run_config_round_trips() -> None:
     payload = run_config_kwargs()
     config = RunConfig.model_validate(payload)
     assert RunConfig.model_validate(config.model_dump(mode="json")) == config
     assert config.model_dump(mode="json", exclude_none=True) == payload
     assert config.digest == "registry.example/trainer@sha256:" + "a" * 64
 
 
-def test_an_environment_names_a_task_and_how_many_copies_of_it():
+def test_environment_carries_seed_but_not_training_streams() -> None:
     environment = EnvironmentConfig(
-        id="brax::hopper", backend="spring", num_envs=1, observed=(0, 1, 2, 3, 4)
+        id="brax::hopper", backend="spring", seed=7, observed=(0, 1, 2, 3, 4)
     )
 
+    assert environment.seed == 7
     assert environment.observed == (0, 1, 2, 3, 4)
+    assert "num_envs" not in environment.model_dump()
 
 
 def test_an_environment_without_observed_is_fully_observed():
-    environment = EnvironmentConfig(id="brax::hopper", backend="spring", num_envs=1)
+    environment = EnvironmentConfig(id="brax::hopper", backend="spring", seed=0)
 
     assert environment.observed is None
 
 
 @pytest.mark.parametrize(
     "observed", [(), (0, 0, 1), (-1, 0)], ids=["empty", "repeated", "negative"]
 )
 def test_an_index_list_that_selects_nothing_usable_is_refused(observed):
     with pytest.raises(ValidationError):
         EnvironmentConfig(
-            id="brax::hopper", backend="spring", num_envs=1, observed=observed
+            id="brax::hopper", backend="spring", seed=0, observed=observed
         )
 
 
-def test_a_budget_must_divide_into_whole_epochs():
-    with pytest.raises(ValidationError):
-        BudgetConfig(total_steps=1000, epoch_steps=300, eval_steps=0)
+def test_training_must_divide_into_whole_epochs_and_stream_rounds() -> None:
+    with pytest.raises(ValidationError, match="total_steps 1000"):
+        TrainingConfig(total_steps=1000, epoch_steps=300, num_envs=1)
+
+    with pytest.raises(ValidationError, match="epoch_steps 1000"):
+        TrainingConfig(total_steps=2000, epoch_steps=1000, num_envs=3)
+
+
+def test_chunk_steps_must_divide_total_and_epoch_when_present() -> None:
+    with pytest.raises(ValidationError, match="chunk_steps"):
+        TrainingConfig(total_steps=2000, epoch_steps=1000, num_envs=1, chunk_steps=300)
+
+    training = TrainingConfig(
+        total_steps=2000, epoch_steps=1000, num_envs=1, chunk_steps=1000
+    )
+    assert training.chunk_steps == 1000
 
 
-def test_a_budget_of_whole_epochs_is_accepted():
-    budget = BudgetConfig(total_steps=900, epoch_steps=300, eval_steps=0)
+def test_evaluation_names_rollout_length_and_parallel_streams() -> None:
+    evaluation = EvaluationConfig(steps=1000, num_envs=10)
 
-    assert budget.total_steps == 900
+    assert evaluation.steps == 1000
+    assert evaluation.num_envs == 10
diff --git a/training-sdk/tests/test_reporter.py b/training-sdk/tests/test_reporter.py
index 9a3a6fa..3f892b5 100644
--- a/training-sdk/tests/test_reporter.py
+++ b/training-sdk/tests/test_reporter.py
@@ -3,35 +3,36 @@ from pathlib import Path
 
 import pytest
 
 from training_sdk.contract import RunConfig
 from training_sdk.reporter import METRICS_FILENAME, Reporter
 
 
 def make_config(*, every_steps: int = 1) -> RunConfig:
     return RunConfig.model_validate(
         {
-            "contract": 4,
+            "contract": 5,
             "run_id": "smoke-20260725-000000-t0",
             "experiment": "infra-acceptance",
             "name": "smoke",
             "launch_id": "20260725-000000",
             "trial": 0,
             "entry": "e",
             "digest": "registry.example/trainer@sha256:" + "a" * 64,
             "environment": {
                 "id": "brax::hopper",
                 "backend": "spring",
-                "num_envs": 1,
+                "seed": 0,
             },
-            "budget": {"total_steps": 100, "epoch_steps": 100, "eval_steps": 0},
-            "params": {"total_steps": 4},
+            "training": {"num_envs": 1, "total_steps": 100, "epoch_steps": 100},
+            "evaluation": {"steps": 0, "num_envs": 1},
+            "params": {"learning_rate": 0.0003},
             "logging": {"aim": "aim://127.0.0.1:1", "every_steps": every_steps},
             "score": {
                 "metric": "episode_return",
                 "window_steps": [0, 4],
                 "reduce": "mean",
                 "direction": "maximize",
                 "non_finite": "worst",
                 "s3": "s3://bucket/score.json",
             },
         }
diff --git a/training-sdk/tests/test_worker.py b/training-sdk/tests/test_worker.py
index ac68f9e..2f22d16 100644
--- a/training-sdk/tests/test_worker.py
+++ b/training-sdk/tests/test_worker.py
@@ -25,21 +25,21 @@ if mode == "empty_metrics":
 
 
 @pytest.fixture
 def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
     child = tmp_path / "child.py"
     child.write_text(CHILD, encoding="utf-8")
     path = tmp_path / "catalog.json"
     path.write_text(
         json.dumps(
             {
-                "contract": 4,
+                "contract": 5,
                 "entries": {
                     "e": {
                         "command": [sys.executable, str(child)],
                         "metrics": ["episode_return"],
                         "space": {"total_steps": [4]},
                     }
                 },
             }
         ),
         encoding="utf-8",
```
