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

