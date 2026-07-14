# memorax-rtrl HPO data

This project's HPO data for the shared engine (`../../infra/hpo`). The engine is
project-agnostic; this dir holds only memorax-rtrl's specs/results.

```text
hpo/
├── specs/       # study specs (paths inside are relative to the spec file)
├── runs/        # generated rounds (engine output)
├── studies/     # per-study Optuna SQLite (gitignored)
├── snapshots/   # Aim repo snapshots (gitignored)
└── targets.py   # OPTIONAL: register_target("name", fn) for memorax objectives
```

Run the shared engine against this project:

```bash
cd ../infra/hpo && uv sync
uv run python src/hpo_control/scheduler.py \
  --project-root ../../memorax-rtrl \
  suggest --spec ../../memorax-rtrl/hpo/specs/<spec>.yaml -n 4
```

The paper reproduction currently uses fixed configs (`config/*.yml`), not HPO;
add specs here when tuning is needed. Define custom objectives (e.g. a
MemoryChain solve-rate) in `targets.py`:

```python
def _memorychain_solve_rate(run):
    ...  # -> float | None

register_target("memorychain_solve_rate", _memorychain_solve_rate)
```
