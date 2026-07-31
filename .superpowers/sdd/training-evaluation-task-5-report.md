# Task 5 report: experiment YAML migration

## Changes

- Rewrote all 27 `experiments/*.yaml` from `environment` + `budget` +
  `space.seed` to `environment` (with `seed`) + `training` + `evaluation`.
  Comments travelled with the keys they describe, so the prose about
  sixteen streams now sits above `training.num_envs` rather than above the
  seed that replaced it.
- Split the four seed sweeps into one file per seed, 20 files in total.
  Each names its own seed, carries a distinct `name` so the five do not share
  a storage prefix, and drops `trials_per_round`/`parallel_jobs` to 1 because
  its space is now a single grid point.
- `rtrrl-hopper-aaai.yaml`'s `scan_steps`, `patience` and `eval_envs` became
  `training.chunk_steps`, `training.early_stop_patience` and
  `evaluation.num_envs`.
- Added four guards to `memo/tests/test_experiments.py`: the sections exist,
  `space` names none of the twelve injected fields, and `environment` no longer
  carries `num_envs`.
- Fixed `memo/tests/test_hopper_reproduction.py` — the `build()` call left open
  by Task 4, and a budget assertion that read `experiment["budget"]`.
- Fixed `rtrrl/tests/test_experiment.py`, which still imported `BudgetConfig`
  and so failed collection for the whole module. It now builds the three config
  objects from the experiment file itself instead of restating their values.
- Rewrote the `environment`/`budget` half of `docs/trainerctl-manual.md`:
  the complete example, the field tables, the reserved-name list, the score
  window's unit, and the error table.

## Two deviations from the plan

**`evaluation.num_envs`.** The plan says to write `1` for the memo files.
`drive()` evaluates on the same stream count it trained on
(`memo/runner/loop.py:133` passes the training `num_envs` to
`report_evaluation`), and memo entries never read `config.evaluation.num_envs`.
Writing `1` beside `training.num_envs: 16` would put a value in the manifest
that nothing reads and that contradicts what runs. Each memo file's
`evaluation.num_envs` equals its `training.num_envs`. The AAAI file keeps 10,
which is a real second stream count its fork uses for `eval_batch_size`.

**The seed sweeps' `hpo` block.** The plan describes copying the file per seed
and does not mention `hpo`. The originals ran `sampler: grid` with
`trials_per_round: 5` — the five grid points were the five seeds. After the
split each file's space is one point, so five trials would ask a grid sampler
to enumerate a grid of one five times. Each copy is now `trials_per_round: 1`,
`parallel_jobs: 1`, matching the single-point cell files beside them.

## Commands and results

Run in WSL with `UV_PROJECT_ENVIRONMENT` and `UV_CACHE_DIR` outside the
repository.

| Command | Result |
| --- | --- |
| `ruff check` on the `memo-ci.yml` path list (`memo`) | PASS — `All checks passed!` |
| `pytest tests` (`memo`) | 5 failed, and they are the 5 sanctioned `test_stream_ac_golden.py` failures. Nothing else. |
| `ruff check .` (`rtrrl/infra/control-plane`) | PASS |
| `pytest` (`rtrrl/infra/control-plane`) | PASS — 203 passed |
| `ruff check entries tests scripts` (`rtrrl`) | PASS |
| `pytest tests` (`rtrrl`) | PASS — 26 passed |
| `pytest` (`rtrrl/infra/mock-trainer`) | PASS — 100 passed |
| `pytest` (`training-sdk`) | PASS — 79 passed, unchanged by this task |

Before this task memo had 36 not-passing tests and control-plane 27. Both are
now at their baseline: memo's five golden failures, and nothing in the control
plane.

## Commits

- `73aeda6` — `test(experiments): require injected seed and run sections`
  (red: 108 failures, 27 files × 4 new guards)
- `98db37a` — `exp: move seed and run shape out of search spaces`

## Not done here

Remote GitHub Actions were not triggered. Everything above is local. The
merge gate is still a green remote run.
