# HPO Control Plane (shared engine)

Jump-host-only HPO advisor/scheduler, **project-agnostic** and part of the shared
infra repo (`trainer/infra/hpo`). Training workers stay simple: they run a
generated config and log to Aim. The engine reads Aim, syncs completed runs into
an Optuna SQLite study, writes next-round configs, and can submit them to AWS
Batch via the shared `../submit.sh`.

## Project layout

The engine is shared; each training project owns its HPO **data** under
`<project>/hpo/`:

```text
<project>/hpo/
├── specs/       # study specs (paths inside are relative to the spec file)
├── runs/        # generated rounds: <study>/round_XXX/{plan.json,report.md,configs/}
├── studies/     # per-study Optuna SQLite (gitignored)
├── snapshots/   # Aim repo snapshots (gitignored)
└── targets.py   # OPTIONAL: register_target("name", fn) for project objectives
```

Point the engine at a project with `--project-root PATH` or `$HPO_PROJECT_ROOT`;
if omitted it is inferred from the `--spec` / `--plan` path (which lives under
`<project>/hpo/`). Built-in objectives (`hc_r20`, `hc_target_score`,
`rtrrl_hop_r1m`) are registered in the engine; a project adds its own by shipping
`<project>/hpo/targets.py` that calls `register_target(name, fn)`.

## Install / Run

The engine is a `uv` project (deps: aim, optuna, pandas, pyyaml):

```bash
cd infra/hpo
uv sync
# Run against a specific project (spec lives under <project>/hpo/specs):
uv run python src/hpo_control/scheduler.py \
  --project-root ../../streaming-rtrrl \
  suggest --spec ../../streaming-rtrrl/hpo/specs/ppo_hc038.yaml -n 4
```

## Study Scope

The PPO HalfCheetah spec is scoped by data, not by instance type:

- Run name is only a coarse index: `PPO-HC-*` / `ppo-hc-*`.
- Archived Aim runs are skipped by default. Platform benchmark runs should be
  archived in Aim so they do not enter HPO history.
- Structured hparams must match the spec filters: PPO, `brax-halfcheetah`, the
  configured observation mask, and the `spring` backend.
- `num_timesteps` is not a membership filter. Short and long runs can both seed
  the study, as long as they describe the same PPO-HC task.

The spec does not invent Aim experiment names. Generated configs keep the base
config `run_name`; edit the generated config before submitting if the next run
needs a new `PPO-HC-xxx ...` name.

## Workflow

First sync completed Aim runs into the local Optuna study:

```bash
cd infra/hpo
uv run python src/hpo_control/scheduler.py sync-aim \
  --spec specs/ppo_hc038.yaml
```

Generate the next round from Aim history:

```bash
cd infra/hpo
uv run python src/hpo_control/scheduler.py suggest \
  --spec specs/ppo_hc038.yaml \
  -n 4
```

This writes:

```text
<project>/hpo/runs/<study>/round_XXX/
  plan.json
  report.md
  configs/config_001.yml
  configs/config_002.yml
```

Before submitting, review `report.md`, then inspect/edit the generated configs.
In particular, set `run_name` to the intended Aim experiment name if this is a
real run.

Review the report, then dry-run submission:

```bash
uv run python src/hpo_control/scheduler.py submit \
  --plan runs/<study>/round_XXX/plan.json
```

Submit for real:

```bash
uv run python src/hpo_control/scheduler.py submit \
  --plan runs/<study>/round_XXX/plan.json \
  --yes
```

## Storage

- Aim remains the source of truth for full experiment data and curves.
- Optuna uses per-study SQLite files under `<project>/hpo/studies/<study>/`.
- Workers do not need Optuna or database credentials.
- `suggest` also syncs Aim before proposing candidates, so `sync-aim` is mainly
  a migration/status command.

## Inspecting Trials

List all Optuna trials for a study:

```bash
cd infra/hpo
uv run python - <<'PY'
import optuna

study_name = "ppo-hc-v1"
storage = f"sqlite:///studies/{study_name}/optuna.db"
study = optuna.load_study(study_name=study_name, storage=storage)

for trial in study.trials:
    print(trial.number, trial.state.name, trial.value, trial.params, trial.user_attrs)
PY
```

List the Aim runs that the spec would import:

```bash
cd infra/hpo
uv run python - <<'PY'
from pathlib import Path
from src.hpo_control.scheduler import import_aim_runs, load_yaml

spec_path = Path("specs/ppo_hc038.yaml").resolve()
spec = load_yaml(spec_path)

for row in import_aim_runs(spec, spec_path):
    print(row.aim_hash, row.name, row.target, row.value, row.params)
PY
```

## Duplicate Control

Candidate diversity is controlled in the spec:

```yaml
constraints:
  min_distance_to_history: 0.08
  min_distance_in_batch: 0.12
```

`distance_to_history` measures how close a candidate is to imported Aim history.
`distance_to_batch` measures how close it is to other candidates in the same
round. Distances are computed after normalizing the configured search space;
log-scaled floats use log space and categorical values use their choice index.

The generated `plan.json` and `report.md` show these distances for each
candidate. A `null` batch distance means the candidate is the first item in that
round.

## Submitting Batch Jobs

`submit` is dry-run by default:

```bash
uv run python src/hpo_control/scheduler.py submit \
  --plan runs/ppo-hc-v1/round_001/plan.json
```

Submit for real only after reviewing names/configs:

```bash
uv run python src/hpo_control/scheduler.py submit \
  --plan runs/ppo-hc-v1/round_001/plan.json \
  --yes
```

The plan stores a sanitized `job_name` for AWS Batch. That is separate from the
Aim `run_name` inside the generated config.

## Objective Metric

The PPO HalfCheetah spec optimizes `hc_target_score` for new runs. Older Aim
runs that predate that metric can still seed Optuna through the
`eval/best_eval_reward` fallback target.

## Resource Policy

Resource selection is declared in the spec. The PPO HalfCheetah spec currently
uses `c7a.medium` by default and switches to `g6.xlarge` when
`ppo_overrides.num_envs > 64`.

## Search Space Compatibility

Optuna can only import historical trials whose parameter values fit the current
`search_space` distributions. If a real PPO-HC history run should seed the
study, keep its observed categorical values in the spec choices and keep numeric
bounds wide enough to include it. Archived benchmark runs are skipped before this
check.

## Aim Snapshots

Do not copy a live Aim repo while experiments are running unless you accept a
possibly inconsistent snapshot. The command refuses by default if Aim appears to
be running:

```bash
uv run python src/hpo_control/scheduler.py snapshot-aim
```

After experiments stop, run the same command to copy `logs/aim` into
`<project>/hpo/snapshots/`.

## Notes

The first version intentionally keeps automation conservative:

- `suggest` generates configs and a human-readable report.
- `submit` is dry-run by default; `--yes` is required to spend money.
- Resource selection is rule/config driven via the spec file.
- No daemon, no automatic infinite loop, no worker-side Optuna.
