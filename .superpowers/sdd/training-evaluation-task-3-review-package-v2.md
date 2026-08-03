# Review package: training/evaluation Task 3 re-review

## Commits
```
98a050d docs(sdd): record task 3 review fix verification
8e8cc93 docs: record catalog seed review fix
cddee86 test(mock-trainer): reserve catalog seed field
cc01246 docs(sdd): record task 3 local verification
17b3788 docs(sdd): record task 3 verification
bd4c072 feat(mock-trainer): consume injected run shape
e86a13f test(mock-trainer): read injected seed and training budget
```

## Name status
```
A	.superpowers/sdd/training-evaluation-task-3-report.md
M	rtrrl/infra/mock-trainer/catalog.json
M	rtrrl/infra/mock-trainer/scripts/build_catalog.py
M	rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py
M	rtrrl/infra/mock-trainer/tests/test_catalog.py
M	rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py
M	rtrrl/infra/mock-trainer/tests/test_train.py
```

## Stat
```
 .../sdd/training-evaluation-task-3-report.md       | 83 ++++++++++++++++++++++
 rtrrl/infra/mock-trainer/catalog.json              |  9 +--
 rtrrl/infra/mock-trainer/scripts/build_catalog.py  |  1 -
 .../mock-trainer/src/brax_ppo_acceptance/config.py |  6 +-
 rtrrl/infra/mock-trainer/tests/test_catalog.py     |  1 +
 rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py | 10 +--
 rtrrl/infra/mock-trainer/tests/test_train.py       | 23 ++++--
 7 files changed, 111 insertions(+), 22 deletions(-)
```

## Diff
```diff
diff --git a/.superpowers/sdd/training-evaluation-task-3-report.md b/.superpowers/sdd/training-evaluation-task-3-report.md
new file mode 100644
index 0000000..1a0f795
--- /dev/null
+++ b/.superpowers/sdd/training-evaluation-task-3-report.md
@@ -0,0 +1,83 @@
+# Task 3: Mock trainer injected run shape
+
+## Files changed
+
+- `rtrrl/infra/mock-trainer/tests/test_train.py`
+  - Migrated the reusable run-config fixture to contract 5 with `environment.seed`,
+    `training`, and `evaluation` sections.
+  - Removed `seed` from algorithm params and added the specified mapping test.
+- `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py`
+  - Reads seed from `environment`, and `num_envs` plus `total_steps` from
+    `training`.
+- `rtrrl/infra/mock-trainer/scripts/build_catalog.py`
+  - This repository has no `src/brax_ppo_acceptance/space.py`; its catalog source
+    is `scripts/build_catalog.py`. Removed `seed` from that source.
+- `rtrrl/infra/mock-trainer/catalog.json`
+  - Updated to contract 5 and removed `seed`.
+- `rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py`
+  - Migrated the subprocess configuration fixture to contract 5.
+
+## Checks
+
+- `uv run --project rtrrl/infra/mock-trainer pytest tests/test_train.py -k acceptance_config_reads_seed_and_budget_from_run_sections`
+  - Could not start: Windows `uv` cache denied access.
+- WSL discovery succeeded after approval, but creation of the requested temporary
+  WSL test environment was rejected by the execution safety gate because it would
+  install dependencies and run pytest on this micro instance.
+- `uv run ruff check rtrrl/infra/mock-trainer`
+  - Could not start: same Windows `uv` cache access denial.
+- `UV_CACHE_DIR=<workspace cache> uv run --project rtrrl/infra/mock-trainer ruff check .`
+  - Could not install the Linux-only `aimrocks==0.5.2` dependency on Windows.
+- `python -m py_compile ...`
+  - Could not start: the available Windows Python executable reported that its
+    login session had been terminated.
+- `git diff --check`
+  - Passed before final staging.
+
+## Commits
+
+- `e86a13f test(mock-trainer): read injected seed and training budget`
+- `bd4c072 feat(mock-trainer): consume injected run shape`
+
+## Self-review
+
+- `AcceptanceConfig.from_run_config()` has no remaining reads of
+  `config.environment.num_envs`, `config.budget.total_steps`, or `params["seed"]`.
+- The generated catalog content matches the edited generator source and exposes
+  only algorithm parameters (`learning_rate`, `episode_length`, and
+  `failure_mode`).
+- The requested red-check push and remote workflow trigger were attempted but
+  rejected by the external-action safety gate; no remote run was created.
+
+## Controller follow-up verification
+
+After the user confirmed this checkout can install and run tests locally, the
+controller installed the mock-trainer environment inside WSL. The virtualenv and
+cache were kept outside the repository:
+
+- `UV_PROJECT_ENVIRONMENT=/tmp/streaming-rtrrl-mock-trainer-venv`
+- `UV_CACHE_DIR=/tmp/streaming-rtrrl-uv-cache`
+
+Additional commands:
+
+| Command | Result | Full summary |
+| --- | --- | --- |
+| `uv sync --all-groups` (`rtrrl/infra/mock-trainer`, WSL) | PASS | Installed 129 Linux-compatible packages, including `brax`, `jax`, `training-sdk`, `pytest`, and `ruff`. |
+| `uv run ruff check .` (`rtrrl/infra/mock-trainer`, WSL) | PASS | `All checks passed!` |
+| `uv run pytest tests/test_config.py tests/test_catalog.py -q` (`rtrrl/infra/mock-trainer`, WSL) | PASS | 69 tests passed in 2.74s. |
+| `uv run pytest tests/test_train.py::test_acceptance_config_reads_seed_and_budget_from_run_sections tests/test_train.py::test_launcher_uses_run_budget_instead_of_total_steps_param -q` (`rtrrl/infra/mock-trainer`, WSL) | PASS | 2 tests passed with dependency warnings in 15.21s. |
+| `uv run pytest tests/test_runtime_cpu.py -q` (`rtrrl/infra/mock-trainer`, WSL) | PASS | 1 test passed with dependency warnings in 27.37s. |
+| `uv run pytest tests/test_train.py -q` (`rtrrl/infra/mock-trainer`, WSL) | TIMEOUT | Timed out after 124 seconds; this file includes real Brax/JAX training-path tests outside the targeted mapping change. |
+| `uv run python scripts/build_catalog.py` (`rtrrl/infra/mock-trainer`, WSL) | PASS | Regenerated `catalog.json`; Windows `git diff --ignore-cr-at-eol` showed no content diff, only WSL line-ending noise, which was restored. |
+
+## Minor review fix: catalog reserved fields
+
+- Added `seed` to `tests/test_catalog.py`'s `RESERVED` set, so the catalog
+  contract check rejects a reintroduced seed algorithm parameter.
+
+| Command | Result | Summary |
+| --- | --- | --- |
+| `uv run ruff check .` (WSL mock-trainer) | PASS | `All checks passed!` |
+| `uv run pytest tests/test_catalog.py -q` (WSL mock-trainer) | PASS | 1 test passed in 0.84s. |
+
+Commit: `cddee8662d1c2a1f475b3a3931189ea3c97e3935` (`test(mock-trainer): reserve catalog seed field`).
diff --git a/rtrrl/infra/mock-trainer/catalog.json b/rtrrl/infra/mock-trainer/catalog.json
index 5c252df..c9f0c2d 100644
--- a/rtrrl/infra/mock-trainer/catalog.json
+++ b/rtrrl/infra/mock-trainer/catalog.json
@@ -1,12 +1,12 @@
 {
-  "contract": 4,
+  "contract": 5,
   "entries": {
     "brax_ppo_acceptance": {
       "command": [
         "python",
         "-m",
         "brax_ppo_acceptance"
       ],
       "metrics": [
         "episode_return",
         "episode_length"
@@ -20,22 +20,15 @@
         "failure_mode": {
           "choices": [
             "none"
           ]
         },
         "learning_rate": {
           "high": 0.01,
           "log": false,
           "low": 1e-06,
           "type": "float"
-        },
-        "seed": {
-          "high": 1000,
-          "log": false,
-          "low": 0,
-          "step": 1,
-          "type": "int"
         }
       }
     }
   }
 }
diff --git a/rtrrl/infra/mock-trainer/scripts/build_catalog.py b/rtrrl/infra/mock-trainer/scripts/build_catalog.py
index 18ec441..53c870a 100644
--- a/rtrrl/infra/mock-trainer/scripts/build_catalog.py
+++ b/rtrrl/infra/mock-trainer/scripts/build_catalog.py
@@ -16,21 +16,20 @@ PACKAGE_ROOT = Path(__file__).resolve().parents[1]
 CATALOG_PATH = PACKAGE_ROOT / "catalog.json"
 ENTRY_NAME = "brax_ppo_acceptance"
 
 
 def build_entry() -> EntryDescriptor:
     return EntryDescriptor.model_validate(
         {
             "command": ["python", "-m", "brax_ppo_acceptance"],
             "metrics": ["episode_return", "episode_length"],
             "space": {
-                "seed": {"type": "int", "low": 0, "high": 1000},
                 "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2},
                 "episode_length": [32],
                 "failure_mode": ["none"],
             },
         }
     )
 
 
 def build_catalog() -> Catalog:
     return Catalog(
diff --git a/rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py b/rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py
index 745ba9b..6b548f6 100644
--- a/rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py
+++ b/rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py
@@ -106,29 +106,29 @@ class AcceptanceConfig:
             raise ValueError("environment.id must be a qualified Brax environment")
         if config.environment.observed is not None:
             raise ValueError("environment.observed is not supported")
         environment = {
             "name": environment_name,
             "options": {"backend": config.environment.backend},
         }
         logging = {"aim_every_env_steps": 1, "rerun_every_episodes": 1}
         algorithm = {
             "learning_rate": params["learning_rate"],
-            "num_envs": config.environment.num_envs,
+            "num_envs": config.training.num_envs,
             "episode_length": params.get("episode_length", 32),
             "failure_mode": params.get("failure_mode", "none"),
         }
         parameters = {
-            "runtime": {"seed": params["seed"]},
+            "runtime": {"seed": config.environment.seed},
             "algorithm": algorithm,
         }
-        training_budget = {"env_steps": config.budget.total_steps}
+        training_budget = {"env_steps": config.training.total_steps}
 
         _exact_keys(
             {"environment": environment, "logging": logging, "parameters": parameters, "training_budget": training_budget},
             {"environment", "logging", "parameters", "training_budget"},
             "params",
         )
         _exact_keys(environment, {"name", "options"}, "environment")
         if environment["name"] != "inverted_pendulum":
             raise ValueError("environment.name must be exactly 'inverted_pendulum'")
         options = _mapping(environment["options"], "environment.options")
diff --git a/rtrrl/infra/mock-trainer/tests/test_catalog.py b/rtrrl/infra/mock-trainer/tests/test_catalog.py
index 7c59a85..afa2b92 100644
--- a/rtrrl/infra/mock-trainer/tests/test_catalog.py
+++ b/rtrrl/infra/mock-trainer/tests/test_catalog.py
@@ -9,20 +9,21 @@ RESERVED = frozenset(
         "env",
         "backend",
         "environment",
         "env_mode",
         "env_backend",
         "observed",
         "num_envs",
         "total_steps",
         "epoch_steps",
         "eval_steps",
+        "seed",
     }
 )
 
 
 def test_catalog_declares_current_contract_and_only_algorithm_parameters() -> None:
     catalog = Catalog.model_validate(json.loads(CATALOG.read_text()))
     assert catalog.contract == CONTRACT_VERSION
     entry = catalog.entries["brax_ppo_acceptance"]
     taken = RESERVED & set(entry.space)
     assert not taken, f"brax_ppo_acceptance still declares {sorted(taken)}"
diff --git a/rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py b/rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py
index 7c2c545..4f06854 100644
--- a/rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py
+++ b/rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py
@@ -19,40 +19,40 @@ def test_installed_module_runs_real_cpu_ppo_in_subprocess(tmp_path: Path) -> Non
     assert {device.platform for device in jax.devices()} == {"cpu"}
 
     scratch = tmp_path / "scratch"
     scratch.mkdir()
     aim_repo = tmp_path / "aim"
     Repo.from_path(str(aim_repo), init=True)
     config_path = tmp_path / "run-config.json"
     config_path.write_text(
         json.dumps(
             {
-                "contract": 4,
+                "contract": 5,
                 "run_id": "runtime-cpu-1",
                 "experiment": "brax-runtime-cpu",
                 "name": "runtime",
                 "launch_id": "20260725-000000",
                 "trial": 0,
                 "entry": "brax_ppo_acceptance",
                 "digest": "registry.example/trainer@sha256:" + "a" * 64,
                 "environment": {
                     "id": "brax::inverted_pendulum",
                     "backend": "generalized",
-                    "num_envs": 4,
+                    "seed": 7,
                 },
-                "budget": {
+                "training": {
+                    "num_envs": 4,
                     "total_steps": 128,
                     "epoch_steps": 128,
-                    "eval_steps": 0,
                 },
+                "evaluation": {"steps": 0, "num_envs": 1},
                 "params": {
-                    "seed": 7,
                     "learning_rate": 0.0003,
                     "episode_length": 32,
                     "failure_mode": "none",
                 },
                 "logging": {"aim": str(aim_repo), "every_steps": 1},
                 "score": {
                     "metric": "episode_return",
                     "window_steps": [0, 128],
                     "reduce": "mean",
                     "direction": "maximize",
diff --git a/rtrrl/infra/mock-trainer/tests/test_train.py b/rtrrl/infra/mock-trainer/tests/test_train.py
index f10868c..ca2a25b 100644
--- a/rtrrl/infra/mock-trainer/tests/test_train.py
+++ b/rtrrl/infra/mock-trainer/tests/test_train.py
@@ -37,21 +37,20 @@ VALID: dict[str, Any] = {
             "episode_length": 32,
             "failure_mode": "none",
         },
     },
     "training_budget": {"env_steps": 128},
 }
 
 
 def default_params(**overrides: Any) -> dict[str, Any]:
     params = {
-        "seed": 7,
         "learning_rate": 0.0003,
         "episode_length": 32,
         "failure_mode": "none",
     }
     params.update(overrides)
     return params
 
 
 def test_environ(*, fast: bool = True) -> dict[str, str]:
     environ = {"BRAX_ACCEPTANCE_TEST_MODE": "1"}
@@ -87,57 +86,71 @@ class FailingCloseSink(RecordingRerun):
 def make_run_config(
     tmp_path: Path,
     params: dict[str, Any],
     *,
     include_rerun: bool = True,
     total_steps: int = 128,
 ) -> RunConfig:
     trial_prefix = f"s3://bucket/trials/t{params.get('trial', 0)}"
     return RunConfig.model_validate(
         {
-            "contract": 4,
+            "contract": 5,
             "run_id": "run-1",
             "experiment": "brax-acceptance",
             "name": "acceptance",
             "launch_id": "20260725-000000",
             "trial": 0,
             "entry": "brax_ppo_acceptance",
             "digest": "registry.example/trainer@sha256:" + "a" * 64,
             "environment": {
                 "id": "brax::inverted_pendulum",
                 "backend": "generalized",
-                "num_envs": 4,
+                "seed": 7,
             },
-            "budget": {
+            "training": {
+                "num_envs": 4,
                 "total_steps": total_steps,
                 "epoch_steps": total_steps,
-                "eval_steps": 0,
             },
+            "evaluation": {"steps": 0, "num_envs": 1},
             "params": params,
             "logging": {
                 "aim": str(tmp_path / "aim"),
                 "every_steps": 1,
                 "rerun_s3": f"{trial_prefix}/episodes/" if include_rerun else None,
                 "rerun_every_episodes": 1 if include_rerun else None,
             },
             "score": {
                 "metric": "episode_return",
                 "window_steps": [0, total_steps],
                 "reduce": "mean",
                 "direction": "maximize",
                 "non_finite": "worst",
                 "s3": f"{trial_prefix}/score.json",
             },
         }
     )
 
 
+@pytest.fixture
+def run_config(tmp_path: Path) -> RunConfig:
+    return make_run_config(tmp_path, default_params())
+
+
+def test_acceptance_config_reads_seed_and_budget_from_run_sections(run_config: RunConfig) -> None:
+    config = AcceptanceConfig.from_run_config(run_config)
+
+    assert config.seed == run_config.environment.seed
+    assert config.num_envs == run_config.training.num_envs
+    assert config.num_timesteps == run_config.training.total_steps
+
+
 def make_reporter(
     tmp_path: Path,
     *,
     failure_mode: str = "none",
     fast: bool = True,
     rerun: RecordingRerun | None = None,
     extra_sinks: list[Any] | None = None,
 ) -> tuple[Reporter, AcceptanceConfig, RecordingRerun, Path]:
     params = default_params(failure_mode=failure_mode)
     scratch = tmp_path / "scratch"
```
