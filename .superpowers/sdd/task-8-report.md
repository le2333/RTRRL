# Task 8 Report: Experiment Contract Migration

## Status

BLOCKED

The requested configuration migration and its focused tests are complete, but
the required Memo CI workflow remains red on five pre-existing
`tests/test_stream_ac_golden.py` failures.

## Commits

- `3944bb5e834e6ef87242ec89be8a955c57659135` — `test(control-plane): load repository experiments`
- `cf7d20e40196f3067b386361eb5f7fd34f8e006c` — `chore(experiments): separate task and budget settings`
- `8f8fa2514c46ca8f4dee16571d39688255467349` — `test(memo): read reproduction controls from contract`

## TDD Evidence

The test-first commit was pushed before any experiment was migrated.

- RED: Tests run
  [30589965649](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30589965649)
  concluded `failure` at
  `3944bb5e834e6ef87242ec89be8a955c57659135`. The new control-plane test
  failed for exactly the 26 unmigrated experiment files with pydantic
  validation errors for missing top-level `environment` and `budget`.
- GREEN: Tests run
  [30590778795](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30590778795)
  concluded `success` at
  `cf7d20e40196f3067b386361eb5f7fd34f8e006c`.
- Final GREEN: Tests run
  [30591412214](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30591412214)
  concluded `success` at
  `8f8fa2514c46ca8f4dee16571d39688255467349`.

The final Memo CI algorithm-suite log shows
`tests/test_experiments.py` fully green and
`tests/test_hopper_reproduction.py` fully green.

## Migration

Exactly 26 files under `experiments/` were migrated. The already-migrated
`experiments/rtrrl-hopper-aaai.yaml` was unchanged. Five migrated files retain
`num_envs: 1`; the other 21 retain `num_envs: 16`.

## Workflow Conclusions

- Tests
  [30591412214](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30591412214):
  `success`.
- Memo CI
  [30590791044](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30590791044):
  `failure`; it exposed the stale reproduction-test reads plus the five
  pre-existing golden failures.
- Memo CI
  [30591410645](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30591410645):
  `failure`; the reproduction and experiment tests are green, and only the
  same five `tests/test_stream_ac_golden.py` failures remain.

The five golden failures also occurred before this migration in Memo CI run
[30587722521](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30587722521).
They are unrelated to the experiment YAML contract and were not altered.

## Judgement Calls

- Existing comments attached to `num_envs` or budget keys in
  `rtrrl-hopper-smoke.yaml`, `streamac-hopper-smoke.yaml`, and
  `streamac-hopper-reproduce.yaml` moved with their associated keys.
- The redundant `# environment` and `# budget` labels in the four RTRRL search
  files were removed when those groups became named top-level sections.
- Memo CI revealed that `test_hopper_reproduction.py` still read control
  settings from `space`; it was updated to read the new `environment` and
  `budget` sections.

## Concerns

The global requirement that both CI workflows pass is not met. Tests is green,
and all Task 8-specific Memo tests are green, but Memo CI is blocked by the
pre-existing golden snapshot failures described above.
