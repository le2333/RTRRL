# Infra-only Acceptance Task 2 Report

## Scope

- Repository: `/home/ubuntu/trainer/streaming-rtrrl-worktrees/trainer-infra`
- Starting commit: `cfa955334c9ad01781869d470c6dfd2394bade7e`
- Live merge base: `1551fda2ecb92dc6351113fb3ee77e55bfe56cd0`
- Reference worktree and `reference/memo-sdk-2026-07-23` were not modified.
- No AWS or Docker command was run.

## TDD evidence

RED:

- Added `tests/test_infra_merge_boundary.py` before the gate existed.
- Ran `uv run --project rtrrl/infra/control-plane pytest tests/test_infra_merge_boundary.py -q`.
- Result: `2 failed`; both failures were caused by the missing
  `scripts/check-infra-merge-boundary.sh`.

GREEN:

- Added executable `scripts/check-infra-merge-boundary.sh`.
- The reusable interface is
  `scripts/check-infra-merge-boundary.sh [BASE]`; without `BASE`, it uses
  `git merge-base main HEAD`.
- The gate checks `git diff --raw` and `git diff --name-status` for `memo`,
  `.github/workflows/build-memo-image.yml`, and
  `.github/workflows/memo-ci.yml`, covering path, type, mode, and blob changes.
- Re-ran the targeted test: `2 passed in 0.03s`.

Minor follow-up:

- Added a real temporary-git-repository negative test, parameterized over
  protected-tree addition, deletion, content/blob modification, and executable
  mode modification.
- Each case copies and invokes the real gate with an explicit baseline commit,
  commits its mutation, verifies the temporary repository is clean, and proves
  a nonzero exit plus `protected tree differs`.
- The test never modifies the repository's real `memo/` tree and does not
  depend on the repository's uncommitted state.
- Re-ran the targeted test: `6 passed`.

## Restored from the live merge base

The workflow `.github/workflows/build-memo-image.yml` was restored to blob
`40ac3783fab03c8ac7b9a6d25dea8dbad97b360a`.

The following previously modified memo files were restored to their merge-base
blobs and modes:

- `memo/.pre-commit-config.yaml`
- `memo/experiments/base/experiment.py`
- `memo/experiments/rtrrl_hopper/run.py`
- `memo/experiments/stream_ac_mujoco_masked/run.py`
- `memo/infra/docker/Dockerfile`
- `memo/infra/docker/Dockerfile.gpu`
- `memo/logging_util.py`
- `memo/memorax/algorithms/__init__.py`
- `memo/memorax/environments/brax.py`
- `memo/memorax/environments/kmemory_chain.py`
- `memo/memorax/environments/memory_chain.py`
- `memo/memorax/environments/wrappers/mask_observation.py`
- `memo/memorax/environments/wrappers/record_episode_statistics.py`
- `memo/pyproject.toml`
- `memo/uv.lock`

## Removed because absent at the live merge base

- `.github/workflows/memo-ci.yml`
- `memo/config/independent_rtrrl_hopper_maskP_lru.yml`
- `memo/docs/superpowers/plans/2026-07-17-composable-online-recurrent-ac-plan.md`
- `memo/docs/superpowers/specs/2026-07-17-composable-online-recurrent-ac-design.md`
- `memo/docs/superpowers/specs/2026-07-19-independent-dual-path-rtrrl-hpo-design.md`
- `memo/experiments/base/facility.py`
- `memo/experiments/memo_rtrrl/run.py`
- `memo/experiments/memo_stream_ac/run.py`
- `memo/infra/docker/Dockerfile.facility`
- `memo/infra/docker/Dockerfile.facility.gpu`
- `memo/infra/scripts/index.yaml`
- `memo/infra/scripts/memo_rtrrl.yaml`
- `memo/infra/scripts/memo_stream_ac.yaml`
- `memo/memorax/algorithms/independent_rtrrl.py`
- `memo/memorax/online_ac/`
- `memo/tests/fixtures/facility_rtrrl.yml`
- `memo/tests/fixtures/facility_stream_ac.yml`
- `memo/tests/online_ac/`
- `memo/tests/test_experiment_observability.py`
- `memo/tests/test_facility_catalog.py`
- `memo/tests/test_facility_launchers.py`
- `memo/tests/test_independent_rtrrl.py`
- `memo/tests/test_logging_compat.py`

No merge-base file was deleted: the protected-tree raw and name-status diffs
against the live merge base are both empty after restoration.

## Historical document handling

Only a superseded notice was inserted after the title in:

- `docs/superpowers/specs/2026-07-21-complete-training-facility-design.md`
- `docs/superpowers/plans/2026-07-21-complete-training-facility.md`

Existing reports and
`docs/acceptance/2026-07-23-complete-facility-task7-phase-a.md` are unchanged.
No pre-existing file under `.superpowers/sdd/` was modified; this report is the
only Task 2 addition there.

## Verification

- Targeted pytest after the Minor follow-up: `6 passed`.
- Gate: `protected tree matches
  1551fda2ecb92dc6351113fb3ee77e55bfe56cd0 by path, blob, and mode`.
- Direct `git diff --raw` assertion against the live merge base: empty.
- Direct `git diff --name-status` assertion against the live merge base: empty.
- Build workflow blob assertion: exact match.
- `git diff --check`: clean.
- IDE lint for the gate, test, and two superseded documents: no errors.

## Commit

- Task 2 implementation: `b9859d6f9e5138ee5bdab84dbdd2aa330f430e5e`
  (`fix(infra): restore algorithm merge boundary`).
- The Minor follow-up is a separate, non-amended commit; its hash is recorded
  in the final follow-up status after commit creation.

## Attention points

- The default gate result follows the current live `main` merge base; callers
  may pass an explicit base commit for deterministic checks.
- The commit intentionally contains a large memo deletion/restoration diff
  relative to `cfa9553`; this is the removal of branch-only memo adaptations,
  not loss of any path present at the live merge base.
- Historical reports and acceptance evidence remain untouched.
