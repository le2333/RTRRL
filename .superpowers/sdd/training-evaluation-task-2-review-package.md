# Review package: training/evaluation Task 2

## Commits
```
0c7327c feat(control-plane): model training and evaluation sections
827dec1 test(control-plane): require training and evaluation sections
```

## Name status
```
M	rtrrl/infra/control-plane/src/trainer_infra/experiment.py
M	rtrrl/infra/control-plane/src/trainer_infra/launch.py
M	rtrrl/infra/control-plane/src/trainer_infra/preflight.py
M	rtrrl/infra/control-plane/tests/data/experiment.yaml
M	rtrrl/infra/control-plane/tests/helpers.py
M	rtrrl/infra/control-plane/tests/test_experiment.py
M	rtrrl/infra/control-plane/tests/test_launch.py
M	rtrrl/infra/control-plane/tests/test_preflight_offline.py
```

## Stat
```
 .../control-plane/src/trainer_infra/experiment.py  | 61 ++++++++++++++++------
 .../control-plane/src/trainer_infra/launch.py      | 22 +++++---
 .../control-plane/src/trainer_infra/preflight.py   |  4 +-
 .../infra/control-plane/tests/data/experiment.yaml | 11 ++--
 rtrrl/infra/control-plane/tests/helpers.py         |  7 +--
 rtrrl/infra/control-plane/tests/test_experiment.py | 30 ++++++-----
 rtrrl/infra/control-plane/tests/test_launch.py     | 10 ++--
 .../control-plane/tests/test_preflight_offline.py  |  6 +--
 8 files changed, 100 insertions(+), 51 deletions(-)
```

## Diff
```diff
diff --git a/rtrrl/infra/control-plane/src/trainer_infra/experiment.py b/rtrrl/infra/control-plane/src/trainer_infra/experiment.py
index 651c639..8da1e7d 100644
--- a/rtrrl/infra/control-plane/src/trainer_infra/experiment.py
+++ b/rtrrl/infra/control-plane/src/trainer_infra/experiment.py
@@ -64,94 +64,125 @@ class LoggingSpec(_Frozen):
     every_steps: int
     rerun_every_episodes: int | None = None
 
 
 RESERVED = frozenset(
     {
         "environment",
         "env_mode",
         "env_backend",
         "observed",
+        "seed",
         "num_envs",
         "total_steps",
         "epoch_steps",
         "eval_steps",
+        "chunk_steps",
+        "early_stop_patience",
+        "eval_envs",
     }
 )
 
 
 class Environment(_Frozen):
     id: str
     backend: str
-    num_envs: int
+    seed: int
     observed: tuple[int, ...] | None = None
 
     @model_validator(mode="after")
     def _usable(self) -> "Environment":
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
 
 
-class Budget(_Frozen):
+class Training(_Frozen):
+    num_envs: int
     total_steps: int
     epoch_steps: int
-    eval_steps: int
+    chunk_steps: int | None = None
+    early_stop_patience: int | None = None
 
     @model_validator(mode="after")
-    def _whole(self) -> "Budget":
+    def _whole(self) -> "Training":
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
+class Evaluation(_Frozen):
+    steps: int
+    num_envs: int
+
+    @model_validator(mode="after")
+    def _usable(self) -> "Evaluation":
+        if self.steps < 0:
+            raise ValueError("evaluation steps must not be negative")
+        if self.num_envs < 1:
+            raise ValueError("evaluation num_envs must be positive")
         return self
 
 
 class Experiment(_Frozen):
     experiment: str
     name: str
     description: Blank = ""
     image: str
     entry: str
     storage: str
     environment: Environment
-    budget: Budget
+    training: Training
+    evaluation: Evaluation
     compute: Compute
     hpo: Hpo
     space: dict[str, SpaceEntry]
     score: ScoreSpec
     logging: LoggingSpec
 
     @model_validator(mode="after")
     def _space_is_only_algorithm(self) -> "Experiment":
         taken = sorted(RESERVED & set(self.space))
         if taken:
             raise ValueError(
                 f"space names {', '.join(taken)}, which belong to the environment "
-                "and budget sections and are not searched"
-            )
-        if self.budget.epoch_steps % self.environment.num_envs:
-            raise ValueError(
-                f"epoch_steps {self.budget.epoch_steps} is not "
-                f"{self.environment.num_envs} streams' worth"
+                "and training or evaluation sections and are not searched"
             )
         return self
 
 
 def load_experiment(path: Path) -> Experiment:
     document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
     return Experiment.model_validate(document)
diff --git a/rtrrl/infra/control-plane/src/trainer_infra/launch.py b/rtrrl/infra/control-plane/src/trainer_infra/launch.py
index 64fb023..f9d35eb 100644
--- a/rtrrl/infra/control-plane/src/trainer_infra/launch.py
+++ b/rtrrl/infra/control-plane/src/trainer_infra/launch.py
@@ -2,26 +2,27 @@ from __future__ import annotations
 
 import json
 from collections.abc import Mapping
 from dataclasses import dataclass
 from datetime import datetime
 from pathlib import Path
 
 from training_sdk import objects
 from training_sdk.contract import (
     CONTRACT_VERSION,
-    BudgetConfig,
+    EvaluationConfig,
     EnvironmentConfig,
     LoggingConfig,
     RunConfig,
     Scalar,
     ScoreConfig,
+    TrainingConfig,
 )
 
 from trainer_infra.preflight import LaunchPlan
 
 
 @dataclass(frozen=True)
 class Launch:
     plan: LaunchPlan
     launch_id: str
     archive: Path
@@ -47,21 +48,22 @@ def create_launch(
         for key, spec in plan.space.items()
     }
     launch_payload = {
         "contract": CONTRACT_VERSION,
         "experiment": experiment.experiment,
         "name": experiment.name,
         "description": experiment.description,
         "launch_id": launch_id,
         "entry": plan.entry_name,
         "environment": experiment.environment.model_dump(mode="json"),
-        "budget": experiment.budget.model_dump(mode="json"),
+        "training": experiment.training.model_dump(mode="json"),
+        "evaluation": experiment.evaluation.model_dump(mode="json"),
         "digest": plan.digest,
         "queue": plan.queue,
         "job_definition": plan.job_definition,
         "sampler": experiment.hpo.sampler,
         "rounds": experiment.hpo.rounds,
         "trials_per_round": experiment.hpo.trials_per_round,
         "parallel_jobs": experiment.hpo.parallel_jobs,
     }
     documents = {
         "experiment.yaml": Path(source).read_bytes(),
@@ -93,27 +95,33 @@ def build_run_config(
         run_id=f"{experiment.name}-{launch.launch_id}-t{trial}",
         experiment=experiment.experiment,
         name=experiment.name,
         launch_id=launch.launch_id,
         trial=trial,
         entry=launch.plan.entry_name,
         digest=launch.plan.digest,
         environment=EnvironmentConfig(
             id=experiment.environment.id,
             backend=experiment.environment.backend,
-            num_envs=experiment.environment.num_envs,
+            seed=experiment.environment.seed,
             observed=experiment.environment.observed,
         ),
-        budget=BudgetConfig(
-            total_steps=experiment.budget.total_steps,
-            epoch_steps=experiment.budget.epoch_steps,
-            eval_steps=experiment.budget.eval_steps,
+        training=TrainingConfig(
+            num_envs=experiment.training.num_envs,
+            total_steps=experiment.training.total_steps,
+            epoch_steps=experiment.training.epoch_steps,
+            chunk_steps=experiment.training.chunk_steps,
+            early_stop_patience=experiment.training.early_stop_patience,
+        ),
+        evaluation=EvaluationConfig(
+            steps=experiment.evaluation.steps,
+            num_envs=experiment.evaluation.num_envs,
         ),
         params=dict(params),
         logging=LoggingConfig(
             aim=experiment.logging.aim,
             every_steps=experiment.logging.every_steps,
             rerun_s3=rerun_s3,
             rerun_every_episodes=experiment.logging.rerun_every_episodes,
         ),
         score=ScoreConfig(
             metric=experiment.score.metric,
diff --git a/rtrrl/infra/control-plane/src/trainer_infra/preflight.py b/rtrrl/infra/control-plane/src/trainer_infra/preflight.py
index 6f5f6a1..d877b24 100644
--- a/rtrrl/infra/control-plane/src/trainer_infra/preflight.py
+++ b/rtrrl/infra/control-plane/src/trainer_infra/preflight.py
@@ -42,24 +42,24 @@ def check_offline(experiment: Experiment, catalog: Catalog) -> dict[str, SpaceEn
             f"image does not declare entry {experiment.entry!r}; available: {available}"
         )
     if experiment.score.metric not in entry.metrics:
         reported = ", ".join(entry.metrics)
         raise PreflightError(
             f"entry {experiment.entry} does not report metric "
             f"{experiment.score.metric!r}; it reports: {reported}"
         )
     space = resolve_space(entry, experiment.space)
     check_sampler(experiment.hpo.sampler, distributions(space))
-    if experiment.score.window_steps[1] > experiment.budget.total_steps:
+    if experiment.score.window_steps[1] > experiment.training.total_steps:
         raise PreflightError(
             f"score window upper bound {experiment.score.window_steps[1]} exceeds "
-            f"the budget's total_steps ({experiment.budget.total_steps})"
+            f"the training total_steps ({experiment.training.total_steps})"
         )
     return space
 
 
 def connect(host: str, port: int) -> None:
     try:
         with socket.create_connection((host, port), timeout=5):
             return
     except OSError as error:
         raise PreflightError(f"aim endpoint aim://{host}:{port} is not reachable: {error}") from error
diff --git a/rtrrl/infra/control-plane/tests/data/experiment.yaml b/rtrrl/infra/control-plane/tests/data/experiment.yaml
index 476decb..de3262e 100644
--- a/rtrrl/infra/control-plane/tests/data/experiment.yaml
+++ b/rtrrl/infra/control-plane/tests/data/experiment.yaml
@@ -2,42 +2,45 @@ experiment: infra-acceptance
 name: brax-ppo-smoke
 description: Fixture for the control-plane suite; not a shipped example
 
 image: 007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:1111111111111111111111111111111111111111111111111111111111111111
 entry: brax_ppo_acceptance
 storage: s3://rtrrl-artifacts-007122174918/trainer
 
 environment:
   id: brax::hopper
   backend: spring
-  num_envs: 1
+  seed: 0
   observed: [0, 1, 2, 3, 4]
 
-budget:
+training:
+  num_envs: 1
   total_steps: 128
   epoch_steps: 128
-  eval_steps: 100
+
+evaluation:
+  steps: 100
+  num_envs: 1
 
 compute:
   instance_type: c7a.medium
   timeout_minutes: 60
 
 hpo:
   sampler: tpe
   rounds: 2
   trials_per_round: 2
   parallel_jobs: 2
 
 space:
   env: [inverted_pendulum]
   backend: [generalized]
-  seed: [0]
   learning_rate: {type: float, low: 1.0e-4, high: 1.0e-3, log: true}
 
 score:
   metric: episode_return
   window_steps: [0, 128]
   reduce: mean
   direction: maximize
   non_finite: worst
 
 logging:
diff --git a/rtrrl/infra/control-plane/tests/helpers.py b/rtrrl/infra/control-plane/tests/helpers.py
index c833e3b..203491b 100644
--- a/rtrrl/infra/control-plane/tests/helpers.py
+++ b/rtrrl/infra/control-plane/tests/helpers.py
@@ -9,21 +9,21 @@ EXAMPLE = Path("tests/data/experiment.yaml")
 """The suite's own experiment, deliberately not one of the shipped examples.
 
 The examples under `examples/` name real image digests and the real Aim host, and
 they change whenever an image is rebuilt. Asserting against them made every image
 push a test failure; `tests/test_examples.py` keeps them honest instead.
 """
 
 
 CATALOG = Catalog.model_validate(
     {
-        "contract": 4,
+        "contract": 5,
         "entries": {
             "brax_ppo_acceptance": {
                 "command": ["python", "-m", "brax_ppo_acceptance"],
                 "metrics": ["episode_return", "episode_length"],
                 "space": {
                     "env": ["inverted_pendulum"],
                     "backend": ["generalized"],
                     "total_steps": {"type": "int", "low": 1, "high": 100000},
                     "seed": {"type": "int", "low": 0, "high": 1000},
                     "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2},
@@ -37,24 +37,25 @@ CATALOG = Catalog.model_validate(
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
-            "num_envs": 1,
+            "seed": 0,
             "observed": [0, 1, 2, 3, 4],
         },
-        "budget": {"total_steps": 2000, "epoch_steps": 1000, "eval_steps": 100},
+        "training": {"num_envs": 1, "total_steps": 2000, "epoch_steps": 1000},
+        "evaluation": {"steps": 100, "num_envs": 1},
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
diff --git a/rtrrl/infra/control-plane/tests/test_experiment.py b/rtrrl/infra/control-plane/tests/test_experiment.py
index 8f77614..f57cb28 100644
--- a/rtrrl/infra/control-plane/tests/test_experiment.py
+++ b/rtrrl/infra/control-plane/tests/test_experiment.py
@@ -14,21 +14,21 @@ def _modified_example(tmp_path: Path, old: str, new: str) -> Path:
     path.write_text(text, encoding="utf-8")
     return path
 
 
 def test_example_file_loads() -> None:
     experiment = load_experiment(EXAMPLE)
     assert experiment.experiment == "infra-acceptance"
     assert experiment.entry == "brax_ppo_acceptance"
     assert experiment.compute.instance_type == "c7a.medium"
     assert experiment.hpo.trials_per_round >= experiment.hpo.parallel_jobs
-    assert experiment.budget.total_steps == 128
+    assert experiment.training.total_steps == 128
 
 
 def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
     path = tmp_path / "bad.yaml"
     path.write_text(EXAMPLE.read_text() + "\ngroups: {}\n", encoding="utf-8")
     with pytest.raises(ValidationError, match="groups"):
         load_experiment(path)
 
 
 def test_the_job_timeout_must_be_positive(tmp_path: Path) -> None:
@@ -59,36 +59,37 @@ def test_parallel_jobs_may_not_exceed_trials_per_round(tmp_path: Path) -> None:
     with pytest.raises(ValidationError, match="parallel_jobs"):
         load_experiment(path)
 
 
 def test_score_window_steps_must_be_ordered(tmp_path: Path) -> None:
     path = _modified_example(tmp_path, "window_steps: [0, 128]", "window_steps: [128, 0]")
     with pytest.raises(ValidationError, match="window_steps must be ordered"):
         load_experiment(path)
 
 
-def test_an_experiment_carries_its_environment_and_budget(tmp_path):
+def test_an_experiment_carries_environment_training_and_evaluation(tmp_path):
     document = _document()
     path = tmp_path / "experiment.yaml"
     path.write_text(yaml.safe_dump(document), encoding="utf-8")
 
     experiment = load_experiment(path)
 
+    assert experiment.environment.seed == 0
     assert experiment.environment.observed == (0, 1, 2, 3, 4)
-    assert experiment.budget.total_steps == 2000
+    assert experiment.training.total_steps == 2000
+    assert experiment.evaluation.steps == 100
 
 
 @pytest.mark.parametrize(
     ("field", "value", "message"),
     [
-        ("num_envs", 0, "num_envs must be positive"),
-        ("num_envs", -1, "num_envs must be positive"),
+        ("seed", -1, "seed must not be negative"),
         ("observed", [], "observed must name at least one index"),
         ("observed", [0, 0, 1], "observed must not repeat an index"),
         ("observed", [-1, 0], "observed indices must not be negative"),
     ],
 )
 def test_an_environment_must_be_usable(
     tmp_path: Path, field: str, value: object, message: str
 ) -> None:
     document = _document()
     document["environment"][field] = value
@@ -108,74 +109,77 @@ def test_an_omitted_observed_field_means_fully_observed(tmp_path: Path) -> None:
     experiment = load_experiment(path)
 
     assert experiment.environment.observed is None
 
 
 @pytest.mark.parametrize(
     ("field", "value", "message"),
     [
         ("total_steps", 0, "total_steps must be positive"),
         ("epoch_steps", 0, "epoch_steps must be positive"),
-        ("eval_steps", -1, "eval_steps must not be negative"),
         ("epoch_steps", 300, "total_steps 2000 is not whole epochs of 300"),
     ],
 )
-def test_a_budget_must_describe_whole_positive_epochs(
+def test_training_must_describe_whole_positive_epochs(
     tmp_path: Path, field: str, value: int, message: str
 ) -> None:
     document = _document()
-    document["budget"][field] = value
+    document["training"][field] = value
     path = tmp_path / "experiment.yaml"
     path.write_text(yaml.safe_dump(document), encoding="utf-8")
 
     with pytest.raises(ValidationError, match=message):
         load_experiment(path)
 
 
-def test_eval_steps_may_be_zero(tmp_path: Path) -> None:
+def test_evaluation_steps_may_be_zero(tmp_path: Path) -> None:
     document = _document()
-    document["budget"]["eval_steps"] = 0
+    document["evaluation"]["steps"] = 0
     path = tmp_path / "experiment.yaml"
     path.write_text(yaml.safe_dump(document), encoding="utf-8")
 
     experiment = load_experiment(path)
 
-    assert experiment.budget.eval_steps == 0
+    assert experiment.evaluation.steps == 0
 
 
 @pytest.mark.parametrize(
     "reserved",
     [
         "environment",
         "env_mode",
         "env_backend",
         "observed",
+        "seed",
         "num_envs",
         "total_steps",
         "epoch_steps",
         "eval_steps",
+        "chunk_steps",
+        "early_stop_patience",
+        "eval_envs",
     ],
 )
-def test_a_space_may_not_name_the_environment_or_the_budget(
+def test_a_space_may_not_name_non_algorithm_fields(
     tmp_path: Path, reserved: str
 ) -> None:
     document = _document()
     document["space"][reserved] = [2000]
     path = tmp_path / "experiment.yaml"
     path.write_text(yaml.safe_dump(document), encoding="utf-8")
 
     with pytest.raises(ValidationError) as raised:
         load_experiment(path)
 
     assert reserved in str(raised.value)
 
 
 def test_epoch_steps_must_be_a_whole_number_of_environment_streams(
     tmp_path: Path,
 ) -> None:
     document = _document()
-    document["environment"]["num_envs"] = 3
+    document["training"]["num_envs"] = 3
     path = tmp_path / "experiment.yaml"
     path.write_text(yaml.safe_dump(document), encoding="utf-8")
 
     with pytest.raises(ValidationError, match="epoch_steps 1000"):
         load_experiment(path)
diff --git a/rtrrl/infra/control-plane/tests/test_launch.py b/rtrrl/infra/control-plane/tests/test_launch.py
index 9dd2ae8..9cde5af 100644
--- a/rtrrl/infra/control-plane/tests/test_launch.py
+++ b/rtrrl/infra/control-plane/tests/test_launch.py
@@ -39,21 +39,21 @@ def test_launch_id_is_a_utc_timestamp(s3_base: str, tmp_path: Path) -> None:
     assert launch.prefix == f"{s3_base}/infra-acceptance/brax-ppo-smoke/20260725-051400"
 
 
 def test_launch_metadata_is_written_to_archive_and_s3(s3_base: str, tmp_path: Path) -> None:
     source_bytes = SOURCE.read_bytes()
     launch = create_launch(make_plan(s3_base), tmp_path, SOURCE, WHEN)
     assert (launch.archive / "experiment.yaml").read_bytes() == source_bytes
     assert objects.get_bytes(f"{launch.prefix}/experiment.yaml") == source_bytes
     archived = json.loads((launch.archive / "launch.json").read_text())
     assert archived["digest"] == "sha256:" + "a" * 64
-    assert archived["contract"] == 4
+    assert archived["contract"] == 5
     remote = json.loads(objects.get_bytes(f"{launch.prefix}/launch.json"))
     assert remote == archived
     space = json.loads(objects.get_bytes(f"{launch.prefix}/space.json"))
     assert space["total_steps"]["type"] == "int"
 
 
 @pytest.mark.parametrize("trial", [3, 7])
 def test_run_config_uses_trial_params_verbatim(
     s3_base: str, tmp_path: Path, trial: int
 ) -> None:
@@ -90,27 +90,29 @@ def test_run_config_disables_rerun_when_not_configured(
     s3_base: str, tmp_path: Path
 ) -> None:
     launch = create_launch(
         make_plan(s3_base, rerun_enabled=False), tmp_path, SOURCE, WHEN
     )
     config = build_run_config(launch, 7, TRIAL_PARAMS)
     assert config.logging.rerun_s3 is None
     assert config.score.s3 == f"{launch.prefix}/trials/t7/score.json"
 
 
-def test_the_run_config_carries_the_environment_and_the_budget(tmp_path):
+def test_the_run_config_carries_environment_training_and_evaluation(tmp_path):
     launch = _launch(tmp_path)
 
     config = build_run_config(launch, trial=0, params={"learning_rate": 0.001})
 
     assert config.environment.id == "brax::hopper"
     assert config.environment.observed == (0, 1, 2, 3, 4)
-    assert config.budget.total_steps == 2000
+    assert config.training.total_steps == 2000
+    assert config.evaluation.steps == 100
 
 
 def test_the_archived_launch_records_both_sections(tmp_path):
     launch = _launch(tmp_path)
 
     archived = json.loads((launch.archive / "launch.json").read_text(encoding="utf-8"))
 
     assert archived["environment"]["observed"] == [0, 1, 2, 3, 4]
-    assert archived["budget"]["epoch_steps"] == 1000
+    assert archived["training"]["epoch_steps"] == 1000
+    assert archived["evaluation"]["steps"] == 100
diff --git a/rtrrl/infra/control-plane/tests/test_preflight_offline.py b/rtrrl/infra/control-plane/tests/test_preflight_offline.py
index 63527f4..62dd0e7 100644
--- a/rtrrl/infra/control-plane/tests/test_preflight_offline.py
+++ b/rtrrl/infra/control-plane/tests/test_preflight_offline.py
@@ -34,36 +34,36 @@ def _catalog() -> Catalog:
                     "space": {"learning_rate": ChoiceSpec(choices=(0.001,))},
                 }
             },
         }
     )
 
 
 def _written(tmp_path, *, window, total_steps):
     document = _document()
     document["score"]["window_steps"] = list(window)
-    document["budget"]["total_steps"] = total_steps
+    document["training"]["total_steps"] = total_steps
     path = tmp_path / "experiment.yaml"
     path.write_text(yaml.safe_dump(document), encoding="utf-8")
     return load_experiment(path)
 
 
-def test_a_score_window_past_the_budget_is_refused(tmp_path):
+def test_a_score_window_past_training_is_refused(tmp_path):
     experiment = _written(tmp_path, window=(0, 4000), total_steps=2000)
 
     with pytest.raises(PreflightError) as raised:
         check_offline(experiment, _catalog())
 
     assert "4000" in str(raised.value)
 
 
-def test_a_score_window_inside_the_budget_is_accepted(tmp_path):
+def test_a_score_window_inside_training_is_accepted(tmp_path):
     experiment = _written(tmp_path, window=(0, 2000), total_steps=2000)
 
     assert "learning_rate" in check_offline(experiment, _catalog())
 
 
 def test_example_passes_offline_checks() -> None:
     space = check_offline(load_experiment(EXAMPLE), CATALOG)
     assert space["total_steps"].model_dump() == {
         "type": "int",
         "low": 1,
```
