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

