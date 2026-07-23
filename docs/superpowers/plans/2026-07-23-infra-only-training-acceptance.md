# Infra-Only Training Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace memo-coupled facility acceptance with an infrastructure-owned Brax PPO trainer while preserving the complete facility and proving that the mergeable branch has no memo or memo-workflow change.

**Architecture:** Freeze the current memo-plus-SDK work at source commit `82b6edb` on a non-merge reference branch before reverting anything. On the mergeable branch, restore the algorithm boundary to merge base `1551fda2`, add a self-contained `rtrrl/infra/mock-trainer` package consumed by the existing worker/controller/SDK contracts, and migrate local, container, deployment, and AWS acceptance to that package. AWS writes remain separately authorized phases, and the final branch gate compares Git trees by blob ID and mode rather than trusting a clean working tree.

**Tech Stack:** Python 3.12, Brax 0.14.2, JAX/JAXlib 0.10.0, `jax[cuda12]` 0.10.0, `training-sdk`, Pydantic 2, Optuna 4, pytest 8, Ruff, uv, Docker, AWS ECR/Batch/S3, Aim 3.28, Rerun.

## Global Constraints

- Approved specification: `docs/superpowers/specs/2026-07-23-infra-only-training-acceptance-design.md`.
- Planning baseline: source commit `82b6edb9d7050a8f299020c076584617b1deb5d7`; merge base `1551fda2ecb92dc6351113fb3ee77e55bfe56cd0`.
- The reference branch is exactly `reference/memo-sdk-2026-07-23`; its worktree is `/home/ubuntu/trainer/streaming-rtrrl-worktrees/trainer-infra-memo-sdk-reference`; it is never merged to `main`.
- The mergeable branch must have zero tree difference from `git merge-base main HEAD` under `memo/`, `.github/workflows/build-memo-image.yml`, and `.github/workflows/memo-ci.yml`; both blob IDs and file modes are compared.
- Historical reports under `.superpowers/sdd/` and `docs/acceptance/` remain unchanged. The old complete-facility spec and plan may receive only a supersession notice.
- Acceptance package root is exactly `rtrrl/infra/mock-trainer/`; import package and script identity are exactly `brax_ppo_acceptance`.
- CPU and GPU images expose the same protocol-version-1 catalog containing exactly the script key `brax_ppo_acceptance`.
- The acceptance package imports `training_sdk`, JAX, and Brax; it must not import `memo` or `trainer_infra`.
- The package runtime test and real AWS run perform real Brax PPO training and real JAX operations. The repeated fake-controller E2E may set `BRAX_ACCEPTANCE_E2E_FAST=1`; this still performs a real jitted Brax reset/step and complete SDK lifecycle but skips PPO optimization to avoid ten independent CPU compilations. The descriptor and normal runtime cannot enable this mode.
- The launcher records resolved parameters/configuration, completed episode summaries, finite objective `eval/episode_return`, one complete evaluation `training_sdk.Episode`, one checkpoint, and exactly one terminal success or failure transition.
- Test-only failure modes are exactly `none`, `before_training`, `after_training`, and `after_checkpoint`; descriptors expose only `none`, and non-`none` values require `BRAX_ACCEPTANCE_TEST_MODE=1`. `BRAX_ACCEPTANCE_E2E_FAST=1` is accepted only when `BRAX_ACCEPTANCE_TEST_MODE=1` is also present.
- Profiles remain exactly `c7am`, `c7al`, `c7ax`, and `g6x`; formal queues remain `run-cpu-c7am-queue`, `run-cpu-c7al-queue`, `run-cpu-c7ax-queue`, and `run-gpu-queue`.
- Real acceptance uses two independent groups, five trials per group, `configs_per_batch: 2`, `runs_per_job: 2`, rounds `2+2+1`, three `c7am` jobs, three `g6x` jobs, and six jobs total.
- AWS native retry attempts and facility submission attempts remain exactly one. No resubmit, cancellation, continuation, or implicit cleanup is introduced.
- The existing shared account is `007122174918`, region is `eu-north-1`, ECR repository is `rtrrl`, S3 bucket is `rtrrl-artifacts-007122174918`, and experiment root is `experiments/`.
- Test labels are exactly `infra-acceptance-brax-ppo-cpu-20260723` and `infra-acceptance-brax-ppo-gpu-20260723`; they do not replace memo or historical image tags.
- No AWS mutation, image push, job-definition registration, paid Batch run, or deletion is executed until the user explicitly authorizes that exact phase after reviewing its dry-run/read-only evidence.
- The current host can access Docker only through `sudo -n docker`; Docker operations use an explicit `DockerRunner` prefix and an isolated Docker config that is removed with the same privilege. Python/AWS commands never run under sudo.
- The optional memo reference image is additional evidence only. Its build, push, or execution cannot block local gates, container gates, the infrastructure merge, or generic AWS acceptance.
- Never commit credentials, registry tokens, local Aim state, `/tmp` reports, Docker credentials, generated artifacts, or acceptance output.
- Every task below ends in one independent commit on the named branch, except a task that is stopped awaiting authorization; do not combine task commits.

## File Map

- `docs/reference/memo-sdk-2026-07-23.md`: immutable-reference identity and non-merge warning, committed only on the reference branch.
- `scripts/check-infra-merge-boundary.sh`: reusable blob-and-mode tree comparison against the live merge base.
- `tests/test_infra_merge_boundary.py`: regression test for the boundary script and protected path set.
- `rtrrl/infra/mock-trainer/pyproject.toml`, `uv.lock`: isolated acceptance dependencies and CPU/GPU extra.
- `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py`: strict concrete-config loading and test-mode validation.
- `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/train.py`: real PPO execution, evaluation, SDK publication, checkpointing, and lifecycle.
- `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/__main__.py`: `python -m brax_ppo_acceptance --config PATH` entry point.
- `rtrrl/infra/mock-trainer/scripts/index.yaml`, `brax_ppo_acceptance.yaml`: single-script protocol-version-1 catalog.
- `rtrrl/infra/mock-trainer/docker/Dockerfile.cpu`, `Dockerfile.gpu`: repository-root builds containing only acceptance runtime, SDK, worker, and catalog.
- `rtrrl/infra/mock-trainer/tests/`: package, real-runtime, SDK artifact, catalog, and image contract tests.
- `rtrrl/infra/control-plane/tests/test_end_to_end.py`: real worker plus real acceptance launcher fake-service lifecycle.
- `rtrrl/infra/control-plane/examples/experiment-smoke.yaml`: exact two-group/six-job acceptance experiment.
- `rtrrl/infra/control-plane/scripts/deploy_facility.py`: acceptance image build/verify/push and digest-bound registration.
- `rtrrl/infra/control-plane/scripts/facility_preflight.py`: acceptance tags and read-only readiness.
- `rtrrl/infra/control-plane/scripts/cleanup_acceptance.py`: exact-prefix, explicitly confirmed S3/Aim scratch cleanup only.
- `rtrrl/infra/control-plane/config/facility.yaml`: test-labelled CPU/GPU tags.
- `docs/acceptance/2026-07-23-infra-only-training-acceptance-*.md`: phase-specific evidence, each written only after its commands run.
- `infra/README.md`: final authoritative generic facility manual.

---

### Task 1: Freeze the Memo SDK Reference Before Any Revert

**Branch:** `reference/memo-sdk-2026-07-23` only.

**Files:**
- Create: `docs/reference/memo-sdk-2026-07-23.md`

**Interfaces:**
- Consumes: exact source commit `82b6edb9d7050a8f299020c076584617b1deb5d7`.
- Produces: immutable non-merge reference branch and worktree with recorded source tree.

- [ ] **Step 1: Prove the source identity before creating anything**

Run:

```bash
cd /home/ubuntu/trainer/streaming-rtrrl-worktrees/trainer-infra
SOURCE_COMMIT=82b6edb9d7050a8f299020c076584617b1deb5d7
git cat-file -e "$SOURCE_COMMIT^{commit}"
test -z "$(git diff --name-only "$SOURCE_COMMIT" -- \
  memo .github/workflows/build-memo-image.yml .github/workflows/memo-ci.yml)"
git show-ref --verify --quiet refs/heads/reference/memo-sdk-2026-07-23 && exit 1 || true
test ! -e /home/ubuntu/trainer/streaming-rtrrl-worktrees/trainer-infra-memo-sdk-reference
```

Expected: exit 0; the protected reference surfaces still match the recorded
source commit; neither reference branch nor worktree exists. Later
design/plan commits do not alter the frozen memo surfaces.

- [ ] **Step 2: Create the reference branch and worktree before any restore/revert**

Run:

```bash
cd /home/ubuntu/trainer/streaming-rtrrl-worktrees/trainer-infra
git branch reference/memo-sdk-2026-07-23 82b6edb9d7050a8f299020c076584617b1deb5d7
git worktree add /home/ubuntu/trainer/streaming-rtrrl-worktrees/trainer-infra-memo-sdk-reference reference/memo-sdk-2026-07-23
```

Expected: `Preparing worktree (checking out 'reference/memo-sdk-2026-07-23')`; `git rev-parse HEAD` in the new worktree prints the exact source commit.

- [ ] **Step 3: Write the reference identity record**

Create `docs/reference/memo-sdk-2026-07-23.md` in the reference worktree with exactly:

```markdown
# Memo SDK Reference — 2026-07-23

- Source commit: `82b6edb9d7050a8f299020c076584617b1deb5d7`
- Source branch at capture: `feature/trainer-infra`
- Reference branch: `reference/memo-sdk-2026-07-23`
- Status: non-mergeable algorithm-side example

This branch preserves the two memo facility launchers, host-side SDK calls,
episode conversion, evaluation trace handling, and success/failure lifecycle.
It must not be merged into `main` or used as the infrastructure acceptance
catalog. After this identity commit the branch is immutable, except for a
separately reviewed evidence-only amendment identifying a test-labelled image
digest and its acceptance result.
```

- [ ] **Step 4: Verify the frozen tree contains the reference surfaces**

Run:

```bash
cd /home/ubuntu/trainer/streaming-rtrrl-worktrees/trainer-infra-memo-sdk-reference
test -f memo/experiments/memo_stream_ac/run.py
test -f memo/experiments/memo_rtrrl/run.py
test -f memo/experiments/base/facility.py
test -f memo/infra/scripts/index.yaml
test -f memo/infra/docker/Dockerfile.facility
test -f memo/infra/docker/Dockerfile.facility.gpu
git diff --quiet 82b6edb9d7050a8f299020c076584617b1deb5d7 -- memo
```

Expected: exit 0; the memo tree is byte-for-byte the captured source tree before adding the identity document.

- [ ] **Step 5: Commit only the reference record**

```bash
cd /home/ubuntu/trainer/streaming-rtrrl-worktrees/trainer-infra-memo-sdk-reference
git add docs/reference/memo-sdk-2026-07-23.md
git commit -m "docs(reference): freeze memo SDK example"
git status --short
```

Expected: one commit on `reference/memo-sdk-2026-07-23`; final status is empty. Return to `/home/ubuntu/trainer/streaming-rtrrl-worktrees/trainer-infra` for every later task.

---

### Task 2: Restore the Merge Boundary and Supersede Memo-Coupled Documents

**Branch:** mergeable infrastructure branch.

**Files:**
- Create: `scripts/check-infra-merge-boundary.sh`
- Create: `tests/test_infra_merge_boundary.py`
- Modify: `docs/superpowers/specs/2026-07-21-complete-training-facility-design.md:1-8`
- Modify: `docs/superpowers/plans/2026-07-21-complete-training-facility.md:1-9`
- Restore from merge base: `memo/`
- Restore from merge base: `.github/workflows/build-memo-image.yml`
- Delete because absent at merge base: `.github/workflows/memo-ci.yml`

**Interfaces:**
- Produces: `scripts/check-infra-merge-boundary.sh [BASE]`, exit 0 only when protected trees have identical path, mode, type, and blob.

- [ ] **Step 1: Write the failing boundary tests**

Create `tests/test_infra_merge_boundary.py`:

```python
from pathlib import Path
import subprocess

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "check-infra-merge-boundary.sh"


def test_boundary_script_protects_memo_and_both_memo_workflows() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "memo" in source
    assert ".github/workflows/build-memo-image.yml" in source
    assert ".github/workflows/memo-ci.yml" in source
    assert "git diff --raw" in source
    assert "git diff --name-status" in source


def test_current_head_has_zero_protected_tree_diff() -> None:
    result = subprocess.run(
        [str(SCRIPT)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "protected tree matches" in result.stdout
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run --project rtrrl/infra/control-plane pytest tests/test_infra_merge_boundary.py -q`

Expected: FAIL because `scripts/check-infra-merge-boundary.sh` does not exist.

- [ ] **Step 3: Implement the exact blob-and-mode gate**

Create executable `scripts/check-infra-merge-boundary.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BASE="${1:-$(git merge-base main HEAD)}"
PROTECTED=(
  memo
  .github/workflows/build-memo-image.yml
  .github/workflows/memo-ci.yml
)

raw="$(git diff --raw "$BASE" -- "${PROTECTED[@]}")"
names="$(git diff --name-status "$BASE" -- "${PROTECTED[@]}")"
if [[ -n "$raw" || -n "$names" ]]; then
  printf '%s\n' "protected tree differs from $BASE" >&2
  [[ -z "$raw" ]] || printf '%s\n' "$raw" >&2
  [[ -z "$names" ]] || printf '%s\n' "$names" >&2
  exit 1
fi
printf '%s\n' "protected tree matches $BASE by path, blob, and mode"
```

- [ ] **Step 4: Restore every protected path from the live merge base**

Run:

```bash
cd /home/ubuntu/trainer/streaming-rtrrl-worktrees/trainer-infra
BASE="$(git merge-base main HEAD)"
test "$BASE" = "1551fda2ecb92dc6351113fb3ee77e55bfe56cd0"
git restore --source="$BASE" --staged --worktree -- memo .github/workflows/build-memo-image.yml
git rm -f .github/workflows/memo-ci.yml
chmod +x scripts/check-infra-merge-boundary.sh
```

Expected: all modified memo files become their base blobs, all branch-added memo files disappear, `build-memo-image.yml` becomes blob `40ac3783fab03c8ac7b9a6d25dea8dbad97b360a`, and `memo-ci.yml` is absent.

- [ ] **Step 5: Add supersession notices without rewriting historical evidence**

Insert after each old document title:

```markdown
> **Superseded on 2026-07-23:** Memo registration, memo launchers, and memo
> facility images in this document are replaced by
> `docs/superpowers/specs/2026-07-23-infra-only-training-acceptance-design.md`.
> The generic control-plane, worker, and SDK contracts remain historical design
> context. Reports produced from this document are historical evidence and are
> intentionally unchanged.
```

Do not edit any file under `.superpowers/sdd/` or the existing
`docs/acceptance/2026-07-23-complete-facility-task7-phase-a.md`.

- [ ] **Step 6: Run the boundary tests and direct tree assertions**

Run:

```bash
uv run --project rtrrl/infra/control-plane pytest tests/test_infra_merge_boundary.py -q
scripts/check-infra-merge-boundary.sh
BASE="$(git merge-base main HEAD)"
test -z "$(git diff --raw "$BASE" -- memo .github/workflows/build-memo-image.yml .github/workflows/memo-ci.yml)"
test -z "$(git diff --name-status "$BASE" -- memo .github/workflows/build-memo-image.yml .github/workflows/memo-ci.yml)"
git diff --check
```

Expected: `2 passed`; both diff substitutions are empty; boundary script reports a match.

- [ ] **Step 7: Commit**

```bash
git add memo .github/workflows/build-memo-image.yml .github/workflows/memo-ci.yml \
  scripts/check-infra-merge-boundary.sh tests/test_infra_merge_boundary.py \
  docs/superpowers/specs/2026-07-21-complete-training-facility-design.md \
  docs/superpowers/plans/2026-07-21-complete-training-facility.md
git commit -m "fix(infra): restore algorithm merge boundary"
```

Expected: one mergeable-branch commit; running the gate against the new merge base still exits 0.

---

### Task 3: Define the Standalone Acceptance Package and Strict Configuration

**Files:**
- Create: `rtrrl/infra/mock-trainer/pyproject.toml`
- Create: `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/__init__.py`
- Create: `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/config.py`
- Create: `rtrrl/infra/mock-trainer/tests/test_config.py`
- Generate: `rtrrl/infra/mock-trainer/uv.lock`

**Interfaces:**
- `FailureMode = Literal["none", "before_training", "after_training", "after_checkpoint"]`
- `AcceptanceConfig.load(path: str | Path, *, environ: Mapping[str, str] = os.environ) -> AcceptanceConfig`
- Fields: `protocol_version: Literal["1"]`, `environment`, `logging`, `parameters`, `training_budget`.
- Resolved parameters: `seed: int`, `learning_rate: float`, `num_envs: int`, `episode_length: int`, `num_timesteps: int`, `failure_mode: FailureMode`.

- [ ] **Step 1: Write strict RED tests**

Create tests covering this valid payload and rejection of every extra/missing field:

```python
VALID = {
    "protocol_version": "1",
    "environment": {
        "name": "inverted_pendulum",
        "options": {"backend": "generalized"},
    },
    "logging": {"aim_every_env_steps": 1, "rerun_every_episodes": 1},
    "parameters": {
        "runtime": {"seed": 7},
        "algorithm": {
            "learning_rate": 0.0003,
            "num_envs": 4,
            "episode_length": 32,
            "failure_mode": "none",
        },
    },
    "training_budget": {"env_steps": 128},
}


def test_non_none_failure_requires_test_environment(tmp_path: Path) -> None:
    payload = copy.deepcopy(VALID)
    payload["parameters"]["algorithm"]["failure_mode"] = "after_training"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="BRAX_ACCEPTANCE_TEST_MODE=1"):
        AcceptanceConfig.load(path, environ={})
    assert AcceptanceConfig.load(
        path, environ={"BRAX_ACCEPTANCE_TEST_MODE": "1"}
    ).failure_mode == "after_training"
```

Also assert environment is exactly `inverted_pendulum`, backend is exactly
`generalized`, positive integral budgets are enforced, floating values are
finite, and `env_steps == num_timesteps`.

- [ ] **Step 2: Run and verify RED**

Run: `uv run --project rtrrl/infra/mock-trainer pytest tests/test_config.py -q`

Expected: FAIL because the project and `AcceptanceConfig` do not exist.

- [ ] **Step 3: Add the isolated project contract**

Create `pyproject.toml` with:

```toml
[project]
name = "brax-ppo-acceptance"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "brax==0.14.2",
  "boto3>=1.35,<2",
  "jax==0.10.0",
  "jaxlib==0.10.0",
  "numpy>=2",
  "pyyaml>=6,<7",
  "training-sdk",
]

[project.optional-dependencies]
cuda12 = ["jax[cuda12]==0.10.0"]

[dependency-groups]
dev = ["pytest>=8,<9", "ruff>=0.9,<1"]

[tool.uv.sources]
training-sdk = { path = "../../../training-sdk" }

[tool.pytest.ini_options]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
```

Implement `AcceptanceConfig` as a frozen dataclass with explicit key-set checks, finite-number checks, and immutable nested values. Do not add Pydantic or a second facility schema.

- [ ] **Step 4: Lock and verify isolation**

Run:

```bash
uv lock --project rtrrl/infra/mock-trainer
uv sync --project rtrrl/infra/mock-trainer --frozen
uv run --project rtrrl/infra/mock-trainer pytest tests/test_config.py -q
uv run --project rtrrl/infra/mock-trainer ruff check src tests
! rg -n '(^|[[:space:]])(from|import)[[:space:]]+(memo|trainer_infra)' \
  rtrrl/infra/mock-trainer/src rtrrl/infra/mock-trainer/tests
```

Expected: all config tests pass; lock is unchanged under `uv lock --check`; forbidden-import search has no matches.

- [ ] **Step 5: Commit**

```bash
git add rtrrl/infra/mock-trainer
git commit -m "feat(infra): define isolated acceptance trainer"
```

---

### Task 4: Implement Real Brax PPO and the Complete SDK Lifecycle

**Files:**
- Create: `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/train.py`
- Create: `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/__main__.py`
- Create: `rtrrl/infra/mock-trainer/tests/test_train.py`
- Create: `rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py`

**Interfaces:**
- `train(config: AcceptanceConfig, run: training_sdk.TrainingRun) -> TrainingResult`
- `rollout_episode(environment: brax.envs.Env, policy: Callable, seed: int, episode_length: int, phase: Literal["train", "eval"]) -> tuple[training_sdk.Episode, float]`
- `main(argv: Sequence[str] | None = None) -> int`
- `TrainingResult(objective: float, checkpoint: Path, platform: str, device_kind: str)`

- [ ] **Step 1: Write lifecycle RED tests around a real SDK run**

Use `training_sdk.MemorySpool`, a recording Aim sink, and a recording Rerun sink. Assert:

```python
result = train(AcceptanceConfig.load(config_path), training_run)
assert math.isfinite(result.objective)
assert result.platform == "cpu"
assert result.checkpoint.name == "ppo-params.npz"
assert result.checkpoint.exists()
assert [event.stream for event in spool.events].count("episode_summary") >= 1
finals = [event for event in spool.events if event.kind == "final"]
assert finals[-1].data["objective_metric"] == "eval/episode_return"
assert finals[-1].data["finalized"] is True
assert len(rerun.episodes) == 1
episode = rerun.episodes[0]
assert episode.phase == "eval"
assert len(episode.observations) == len(episode.actions) + 1
assert episode.terminals[-1] or episode.truncations[-1]
```

Parameterize all three injected failure points and assert `run.fail(error)` is called once, no finalized objective is emitted, and artifacts created before the failure remain visible for the worker.

- [ ] **Step 2: Run and verify RED**

Run: `JAX_PLATFORM_NAME=cpu uv run --project rtrrl/infra/mock-trainer pytest tests/test_train.py -q`

Expected: FAIL because `train`, `rollout_episode`, and the CLI are absent.

- [ ] **Step 3: Implement the real PPO path**

The implementation must:

```python
environment = envs.get_environment(
    env_name=config.environment_name,
    backend=config.backend,
)
make_inference_fn, params, metrics = ppo_train.train(
    environment=environment,
    num_timesteps=config.num_timesteps,
    episode_length=config.episode_length,
    num_envs=config.num_envs,
    learning_rate=config.learning_rate,
    unroll_length=4,
    batch_size=4,
    num_minibatches=1,
    num_updates_per_batch=1,
    seed=config.seed,
    num_evals=1,
    normalize_observations=True,
    reward_scaling=1.0,
)
```

Run `jax.jit(lambda x: jnp.sin(x) + jnp.cos(x))(jnp.arange(1024.0)).block_until_ready()` before training so both images prove a real selected-device operation. Convert only host-visible rollout values to NumPy. The normal path passes the four fixed micro-PPO values shown above so 128 steps are valid and bounded. When both `BRAX_ACCEPTANCE_TEST_MODE=1` and `BRAX_ACCEPTANCE_E2E_FAST=1` are present, replace only PPO optimization with one real jitted environment reset/step and an explicit zero-action policy; retain rollout, SDK, checkpoint, and lifecycle behavior. Reject the fast flag in every other environment.

After PPO returns, roll out one real `phase="train"` episode with seed
`config.seed`, call
`log_episode_summary(env_steps=config.num_timesteps,
episode_return=train_return,
episode_length=len(train_episode.actions))`, then roll out a separate
deterministic `phase="eval"` episode with seed `config.seed + 1`, call
`log_episode(eval_episode)`, and use its finite return as
`eval/episode_return`. Set truncation when either rollout reaches its fixed
budget without environment termination, so both are complete `Episode`
values. Write `ppo-params.npz` through `jax.tree_util.tree_flatten` plus
`numpy.savez`, then call `register_checkpoint`. The fast path uses an explicit
zero-action policy and a one-array checkpoint rather than pretending to return
PPO parameters.

The launcher order is exactly:

```python
run = bootstrap_from_environment()
if run is None:
    raise RuntimeError("TRAINER_RUN_CONTEXT_PATH is required")
try:
    run.start()
    result = train(config, run)
    run.finish(
        {
            "eval/episode_return": result.objective,
            "runtime/device_count": len(jax.devices()),
        }
    )
except BaseException as error:
    run.fail(error)
    raise
```

- [ ] **Step 4: Add the real CPU runtime test**

`test_runtime_cpu.py` launches the module in a subprocess with a real SDK context and asserts:

```python
assert jax.default_backend() == "cpu"
assert {device.platform for device in jax.devices()} == {"cpu"}
assert math.isfinite(read_final_objective(artifact_directory))
assert exactly_one_rerun_file(artifact_directory)
assert (artifact_directory / "checkpoints" / "ppo-params.npz").is_file()
```

Use `num_timesteps: 128`, `num_envs: 4`, and `episode_length: 32`; do not assert PPO convergence.

- [ ] **Step 5: Verify GREEN and failure behavior**

Run:

```bash
JAX_PLATFORM_NAME=cpu uv run --project rtrrl/infra/mock-trainer \
  pytest tests/test_train.py tests/test_runtime_cpu.py -q
uv run --project rtrrl/infra/mock-trainer ruff check src tests
```

Expected: all tests pass; CPU backend is real; objective is finite; exactly one checkpoint and Rerun episode exist; each injected failure exits nonzero without a finalized objective.

- [ ] **Step 6: Commit**

```bash
git add rtrrl/infra/mock-trainer
git commit -m "feat(infra): run observable Brax PPO acceptance"
```

---

### Task 5: Package the Shared Catalog and CPU/GPU Images

**Files:**
- Create: `rtrrl/infra/mock-trainer/scripts/index.yaml`
- Create: `rtrrl/infra/mock-trainer/scripts/brax_ppo_acceptance.yaml`
- Create: `rtrrl/infra/mock-trainer/docker/Dockerfile.cpu`
- Create: `rtrrl/infra/mock-trainer/docker/Dockerfile.gpu`
- Create: `rtrrl/infra/mock-trainer/tests/test_catalog.py`
- Create: `rtrrl/infra/mock-trainer/tests/test_image_contract.py`
- Modify: `.dockerignore`

**Interfaces:**
- Catalog label: `org.rtrrl.trainer.scripts.v1`.
- Image paths: `/opt/trainer/worker.py`, `/opt/trainer/scripts/index.yaml`, `/opt/trainer/scripts/brax_ppo_acceptance.yaml`, `/opt/acceptance`.
- Launcher argv: `python -m brax_ppo_acceptance --config {config_path}`.

- [ ] **Step 1: Write RED catalog and Docker contract tests**

Assert both Dockerfiles use repository root context inputs, copy no `memo` or control-plane source, and share the same catalog. Decode the generated label with `trainer_infra.image_catalog.decode_catalog` and assert:

```python
assert catalog.protocol_version == "1"
assert set(catalog.scripts) == {"brax_ppo_acceptance"}
descriptor = catalog.scripts["brax_ppo_acceptance"]
assert descriptor.argv == (
    "python", "-m", "brax_ppo_acceptance", "--config", "{config_path}"
)
assert descriptor.sdk_protocol_version == "1"
assert descriptor.objective.metric == "eval/episode_return"
assert descriptor.objective.direction == "maximize"
assert descriptor.environments == ("inverted_pendulum",)
assert descriptor.fields["failure_mode"].choices == ("none",)
```

- [ ] **Step 2: Run and verify RED**

Run: `uv run --project rtrrl/infra/control-plane pytest rtrrl/infra/mock-trainer/tests/test_catalog.py rtrrl/infra/mock-trainer/tests/test_image_contract.py -q`

Expected: FAIL because catalog and Dockerfiles do not exist.

- [ ] **Step 3: Create the descriptor and index**

Create `scripts/index.yaml`:

```yaml
protocol_version: "1"
scripts:
  - brax_ppo_acceptance.yaml
```

Create the descriptor with:

```yaml
name: brax_ppo_acceptance
argv: [python, -m, brax_ppo_acceptance, --config, "{config_path}"]
sdk_protocol_version: "1"
environments: [inverted_pendulum]
defaults:
  environment:
    name: inverted_pendulum
    options: {backend: generalized}
  training_budget: {env_steps: 128}
  logging: {aim_every_env_steps: 1, rerun_every_episodes: 1}
objective:
  metric: eval/episode_return
  direction: maximize
  reduction: last
fields:
  seed:
    path: runtime.seed
    type: int
    default: 0
    constraints: {ge: 0}
  learning_rate:
    path: algorithm.learning_rate
    type: float
    default: 0.0003
    searchable: true
    constraints: {gt: 0}
    default_search: {values: [0.0002, 0.0003, 0.0004, 0.0005, 0.0006]}
  num_envs:
    path: algorithm.num_envs
    type: int
    default: 4
    choices: [4]
  episode_length:
    path: algorithm.episode_length
    type: int
    default: 32
    choices: [32]
  failure_mode:
    path: algorithm.failure_mode
    type: str
    default: none
    choices: [none]
```

- [ ] **Step 4: Build minimal root-context Dockerfiles**

Create `Dockerfile.cpu`:

```dockerfile
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
WORKDIR /workspace/rtrrl/infra/mock-trainer
COPY training-sdk /workspace/training-sdk
COPY rtrrl/infra/mock-trainer /workspace/rtrrl/infra/mock-trainer
RUN uv sync --frozen --no-dev --no-editable && uv cache clean

FROM python:3.12-slim AS runtime
ARG TRAINER_SCRIPT_CATALOG
RUN test -n "${TRAINER_SCRIPT_CATALOG}"
LABEL org.rtrrl.trainer.scripts.v1="${TRAINER_SCRIPT_CATALOG}"
COPY --from=builder /opt/venv /opt/venv
COPY rtrrl/infra/mock-trainer /opt/acceptance
COPY rtrrl/infra/worker/worker.py /opt/trainer/worker.py
COPY rtrrl/infra/mock-trainer/scripts /opt/trainer/scripts
ENV PATH="/opt/venv/bin:${PATH}" PYTHONUNBUFFERED=1 JAX_PLATFORM_NAME=cpu
CMD ["python", "/opt/trainer/worker.py", "--help"]
```

Create `Dockerfile.gpu` with the same stages, changing the sync line to
`RUN uv sync --frozen --no-dev --no-editable --extra cuda12 && uv cache clean`
and omitting `JAX_PLATFORM_NAME=cpu` from the final `ENV`. Both files therefore
copy only `training-sdk`, the mock trainer, worker, and one-script catalog.

Extend `.dockerignore` allowlisting only:

```text
!training-sdk
!training-sdk/**
!rtrrl/infra/worker/worker.py
!rtrrl/infra/mock-trainer
!rtrrl/infra/mock-trainer/**
```

Keep secret, VCS, cache, Aim, and artifact exclusions. Remove the memo-specific allowlist because the mergeable acceptance images do not consume memo.

- [ ] **Step 5: Build both images and run the host-supported container contracts**

Run:

```bash
CATALOG="$(uv run --project rtrrl/infra/control-plane trainer-image-catalog encode rtrrl/infra/mock-trainer/scripts/index.yaml)"
sudo -n docker build --platform linux/amd64 \
  --build-arg "TRAINER_SCRIPT_CATALOG=$CATALOG" \
  -f rtrrl/infra/mock-trainer/docker/Dockerfile.cpu \
  -t brax-ppo-acceptance:cpu .
sudo -n docker build --platform linux/amd64 \
  --build-arg "TRAINER_SCRIPT_CATALOG=$CATALOG" \
  -f rtrrl/infra/mock-trainer/docker/Dockerfile.gpu \
  -t brax-ppo-acceptance:gpu .
sudo -n docker run --rm brax-ppo-acceptance:cpu \
  python -c 'import brax_ppo_acceptance,training_sdk,jax; assert jax.default_backend()=="cpu"'
sudo -n docker run --rm --entrypoint /opt/venv/bin/python brax-ppo-acceptance:gpu \
  -c 'import importlib.util,jax; assert importlib.util.find_spec("jax_cuda12_plugin") is not None; print(jax.__version__)'
sudo -n docker run --rm brax-ppo-acceptance:cpu \
  python -c 'import importlib.util; assert importlib.util.find_spec("memo") is None; assert importlib.util.find_spec("trainer_infra") is None'
sudo -n docker run --rm --entrypoint /opt/venv/bin/python brax-ppo-acceptance:gpu \
  python -c 'import importlib.util; assert importlib.util.find_spec("memo") is None; assert importlib.util.find_spec("trainer_infra") is None'
```

Expected: both builds succeed; CPU executes a CPU JAX operation; the GPU image
contains the CUDA 12 JAX plugin and imports without requiring a host GPU;
neither runtime imports memo or control-plane. This host has no authorized
direct Docker access and no GPU, so L4 selection and the real CUDA JIT
operation are deferred explicitly to the authorized `g6x` Batch acceptance.

- [ ] **Step 6: Run tests and commit**

```bash
uv run --project rtrrl/infra/control-plane pytest \
  rtrrl/infra/mock-trainer/tests/test_catalog.py \
  rtrrl/infra/mock-trainer/tests/test_image_contract.py -q
git add .dockerignore rtrrl/infra/mock-trainer
git commit -m "feat(infra): package acceptance trainer images"
```

---

### Task 6: Migrate Fake End-to-End Acceptance and Examples Off Memo

**Files:**
- Rewrite: `rtrrl/infra/control-plane/tests/test_end_to_end.py`
- Delete: `rtrrl/infra/control-plane/tests/test_memo_catalog.py`
- Delete: `rtrrl/infra/control-plane/tests/test_memo_image_contract.py`
- Modify: `rtrrl/infra/control-plane/tests/test_facility_concrete_contract.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/controller.py`
- Modify: `rtrrl/infra/control-plane/tests/test_controller.py`
- Rewrite: `rtrrl/infra/control-plane/examples/experiment-smoke.yaml`

**Interfaces:**
- Fake E2E invokes the real `/opt/trainer/worker.py` logic and real `python -m brax_ppo_acceptance`.
- Experiment groups: `cpu` with `c7am`, `gpu` with `g6x`; both use `brax_ppo_acceptance`.
- `ExperimentReport.experiment_metadata: Mapping[str, JsonValue]` preserves the resolved immutable experiment metadata in both success and failure reports.

- [ ] **Step 1: Rewrite the E2E expectations first and verify RED**

Replace memo fixtures with the acceptance catalog and subprocess environment. Assert:

```python
assert report.completed_runs == 10
assert len(report.submitted_job_ids) == 6
assert [len(call) for call in batch.query_calls] == [2, 2, 2]
assert [len(bundle.runs) for bundle in batch.submitted] == [2, 2, 2, 2, 1, 1]
assert {run.run_context["group"] for bundle in batch.submitted for run in bundle.runs} == {
    "cpu", "gpu"
}
assert sum(bundle.resource_profile == "c7am" for bundle in batch.submitted) == 3
assert sum(bundle.resource_profile == "g6x" for bundle in batch.submitted) == 3
```

Retain every fail-fast parameter: Batch failed/timeout, child nonzero, artifact upload, marker missing/tampered, Aim failed/timeout/nonfinite, input/job upload, partial submit, and final persistence errors. Assert no resubmit, retry, or cancel call.

Add report tests asserting both success and failure `report.json` contain:

```python
assert report.experiment_name == "infra-brax-ppo-acceptance"
assert report.experiment_metadata == {"purpose": "infra-acceptance"}
```

Run: `uv run --project rtrrl/infra/control-plane pytest tests/test_end_to_end.py -q`

Expected: FAIL while the harness still imports memo and expects five mixed-image jobs.

- [ ] **Step 2: Rewire the harness to the real package**

Set `PYTHONPATH` to `training-sdk/src:rtrrl/infra/mock-trainer/src`, set
`JAX_PLATFORM_NAME=cpu`, `BRAX_ACCEPTANCE_TEST_MODE=1`, and
`BRAX_ACCEPTANCE_E2E_FAST=1` for repeated local execution of both logical
profiles. The fast path must still execute a real jitted Brax reset/step,
rollouts, SDK publication, and worker artifact upload. Package runtime tests
unset the fast flag and execute real PPO once. Use non-`none` failure modes
only in injected child-failure cases. Do not import anything under `memo/`.

The fake ECR returns separate immutable CPU and GPU digests carrying the same one-script catalog. Fake preflight returns matching definitions for `c7am` and `g6x`, preserving image/profile partitioning into six jobs.

- [ ] **Step 3: Replace the committed smoke experiment**

Use exact defaults:

```yaml
experiment:
  name: infra-brax-ppo-acceptance
  description: Infrastructure-owned CPU and GPU facility acceptance
  metadata: {purpose: infra-acceptance}
defaults:
  image: 007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl:infra-acceptance-brax-ppo-cpu-20260723
  environment: {name: inverted_pendulum, options: {backend: generalized}}
  training_budget: {env_steps: 128}
  logging: {aim_every_env_steps: 1, rerun_every_episodes: 1}
  resources: {profile: c7am}
  hpo:
    total_trials: 5
    configs_per_batch: 2
    parameter_policy: explicit_scan
  execution:
    runs_per_job: 2
    aim_result_timeout_seconds: 600
groups:
  cpu:
    script: brax_ppo_acceptance
    parameters:
      learning_rate: {values: [0.0002, 0.0003, 0.0004, 0.0005, 0.0006]}
  gpu:
    script: brax_ppo_acceptance
    image: 007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl:infra-acceptance-brax-ppo-gpu-20260723
    resources: {profile: g6x}
    parameters:
      learning_rate: {values: [0.0002, 0.0003, 0.0004, 0.0005, 0.0006]}
```

- [ ] **Step 4: Prove all local lifecycle and isolation expectations**

Run:

```bash
uv run --project rtrrl/infra/control-plane pytest \
  tests/test_end_to_end.py \
  tests/test_facility_concrete_contract.py \
  tests/test_controller.py -q
! rg -n 'memo_stream_ac|memo_rtrrl|MEMO_ROOT|memo/infra' \
  rtrrl/infra/control-plane/tests/test_end_to_end.py \
  rtrrl/infra/control-plane/tests/test_facility_concrete_contract.py \
  rtrrl/infra/control-plane/examples/experiment-smoke.yaml
```

Expected: tests pass; search returns no matches; success path has ten completed trials, `2+2+1` rounds per group, and exactly six jobs.

- [ ] **Step 5: Commit**

```bash
git add rtrrl/infra/control-plane/tests rtrrl/infra/control-plane/examples/experiment-smoke.yaml
git commit -m "test(infra): accept facility without memo"
```

---

### Task 7: Migrate Deployment, Preflight, and Exact Cleanup

**Files:**
- Modify: `rtrrl/infra/control-plane/scripts/deploy_facility.py`
- Modify: `rtrrl/infra/control-plane/scripts/facility_preflight.py`
- Create: `rtrrl/infra/control-plane/scripts/cleanup_acceptance.py`
- Modify: `rtrrl/infra/control-plane/config/facility.yaml`
- Modify: `rtrrl/infra/control-plane/tests/test_facility_deployment.py`
- Modify: `rtrrl/infra/control-plane/tests/test_facility_deploy_review.py`
- Modify: `rtrrl/infra/control-plane/tests/test_facility_preflight_review.py`
- Create: `rtrrl/infra/control-plane/tests/test_cleanup_acceptance.py`

**Interfaces:**
- `DOCKERFILES = {"cpu": ROOT / "rtrrl/infra/mock-trainer/docker/Dockerfile.cpu", "gpu": ...}`
- `DockerRunner(prefix: tuple[str, ...], config_directory: Path | None)` prefixes every Docker invocation; `--docker-via-sudo` selects `("sudo", "-n", "docker")`.
- `_verify_image(kind: Literal["cpu", "gpu"], image: str) -> None`; CPU proves backend `cpu`, while GPU proves the CUDA plugin/import contract before push and defers device execution to `g6x`.
- `CleanupRequest(experiment_id: str, confirm_prefix: str | None, execute: bool)`.
- `cleanup(request, *, control, s3, aim_repo) -> CleanupReport`.

- [ ] **Step 1: Write RED deployment/preflight tests**

Assert dry-run planned tags are exactly the two test labels, image verification imports `brax_ppo_acceptance` and `training_sdk`, catalog key is exactly `brax_ppo_acceptance`, and source contains no memo path or memo script identity.

Assert the pre-push GPU runtime verification imports JAX and requires
`jax_cuda12_plugin` without creating a device. Assert the separate Batch
acceptance command executes:

```python
devices = jax.devices()
assert jax.default_backend() == "gpu"
assert any("NVIDIA L4" in device.device_kind for device in devices)
jax.jit(lambda x: x @ x)(jnp.eye(64)).block_until_ready()
```

CPU verification requires backend `cpu`. Both require no import spec for
`memo` and `trainer_infra`. Tests also prove `--docker-via-sudo` prefixes every
build/inspect/run/login/push command, uses `docker --config TEMP` rather than
the default config, and removes TEMP with the same runner in `finally`.

- [ ] **Step 2: Write RED exact-cleanup tests**

Tests must prove dry-run performs no delete, execute requires:

```python
expected_prefix = (
    "s3://rtrrl-artifacts-007122174918/experiments/"
    f"{experiment_id}/"
)
assert request.confirm_prefix == expected_prefix
assert str(uuid.UUID(experiment_id)) == experiment_id
```

Only S3 keys under that prefix and Aim runs whose
`context.experiment_id` equals the exact ID may be deleted. An ID is an
acceptance ID only when its canonical `report.json` exists under the same
prefix and contains both
`experiment_name == "infra-brax-ppo-acceptance"` and
`experiment_metadata.purpose == "infra-acceptance"`; UUID shape alone is not
evidence. Reject empty IDs, `..`, slashes, missing/mismatched reports, prefix
mismatches, main Aim repo, and any request to remove images, job definitions,
queues, compute environments, ECR repositories, or buckets.

- [ ] **Step 3: Run and verify RED**

Run:

```bash
uv run --project rtrrl/infra/control-plane pytest \
  tests/test_facility_deployment.py \
  tests/test_facility_deploy_review.py \
  tests/test_facility_preflight_review.py \
  tests/test_cleanup_acceptance.py -q
```

Expected: failures mention memo paths/catalog and missing cleanup script.

- [ ] **Step 4: Implement generic deployment and read-only preflight**

Change only acceptance Dockerfiles, tags, label checks, and runtime checks.
`_verify_image("gpu", image)` must prove the CUDA plugin and imports without
requesting a host GPU. The real `g6x` child command must require an NVIDIA L4
and complete the JIT matrix operation before training. Preserve account/region
confirmation, isolated Docker credentials, digest-only ECR verification, four
profile registrations, retry attempts one, and absence of Batch
submit/cleanup from `deploy_facility.py`. Python and boto3 remain unprivileged;
only Docker subprocesses use the selected runner.

Set:

```yaml
cpu_image_tag: infra-acceptance-brax-ppo-cpu-20260723
gpu_image_tag: infra-acceptance-brax-ppo-gpu-20260723
```

Keep all other facility values unchanged.

- [ ] **Step 5: Implement cleanup as a separate guarded command**

CLI:

```text
python scripts/cleanup_acceptance.py --control config/facility.yaml \
  --experiment-id ID
python scripts/cleanup_acceptance.py --control config/facility.yaml \
  --experiment-id ID \
  --confirm-prefix s3://rtrrl-artifacts-007122174918/experiments/ID/ \
  --execute
```

Dry-run emits canonical JSON listing exact S3 keys and exact Aim run hashes with `writes_performed:false`. Execute re-lists immediately, refuses if the set changed, deletes only listed S3 objects and exact Aim hashes, then verifies both sets are empty. It does not expose ECR, Batch registration, queue, CE, IAM, bucket, or filesystem-recursive deletion APIs.

- [ ] **Step 6: Verify GREEN and commit**

```bash
uv run --project rtrrl/infra/control-plane pytest \
  tests/test_facility_deployment.py \
  tests/test_facility_deploy_review.py \
  tests/test_facility_preflight_review.py \
  tests/test_cleanup_acceptance.py -q
uv run --project rtrrl/infra/control-plane ruff check src tests scripts
git add rtrrl/infra/control-plane
git commit -m "feat(infra): deploy generic acceptance images"
```

Expected: all tests pass; deployment has no submit/cleanup path; cleanup cannot target shared or historical resources.

---

### Task 8: Add the Local Full-Suite and Whole-Branch Merge Gate

**Files:**
- Create: `scripts/verify-infra-only-acceptance.sh`
- Create: `tests/test_verify_infra_only_acceptance.py`

**Interfaces:**
- `scripts/verify-infra-only-acceptance.sh` is the single local pre-merge command.

- [ ] **Step 1: Write RED command-contract tests**

Assert the script invokes, in order: boundary gate; SDK tests/Ruff; mock-trainer tests/Ruff/lock check; control-plane tests/Ruff/lock check; Docker catalog/image contract tests; forbidden-import scans; `git diff --check`; and final boundary gate.

- [ ] **Step 2: Run and verify RED**

Run: `uv run --project rtrrl/infra/control-plane pytest tests/test_verify_infra_only_acceptance.py -q`

Expected: FAIL because the verification script is absent.

- [ ] **Step 3: Implement the gate**

The executable script uses:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
scripts/check-infra-merge-boundary.sh
uv lock --project training-sdk --check
uv run --project training-sdk pytest -q
uv run --project training-sdk ruff check src tests
uv lock --project rtrrl/infra/mock-trainer --check
JAX_PLATFORM_NAME=cpu uv run --project rtrrl/infra/mock-trainer pytest -q
uv run --project rtrrl/infra/mock-trainer ruff check src tests
uv lock --project rtrrl/infra/control-plane --check
uv run --project rtrrl/infra/control-plane pytest -q
uv run --project rtrrl/infra/control-plane ruff check src tests scripts
! rg -n '(^|[[:space:]])(from|import)[[:space:]]+(memo|trainer_infra)' \
  rtrrl/infra/mock-trainer/src
! rg -n 'memo_stream_ac|memo_rtrrl|memo/infra' \
  rtrrl/infra/control-plane/examples \
  rtrrl/infra/control-plane/scripts \
  rtrrl/infra/control-plane/src \
  rtrrl/infra/control-plane/tests/test_end_to_end.py \
  rtrrl/infra/control-plane/tests/test_facility_concrete_contract.py
git diff --check
scripts/check-infra-merge-boundary.sh
```

- [ ] **Step 4: Run the complete gate**

Run: `scripts/verify-infra-only-acceptance.sh`

Expected: every suite and Ruff/lock check passes; both boundary checks report identical blob/mode trees; no forbidden import/reference is found.

- [ ] **Step 5: Commit**

```bash
git add scripts/verify-infra-only-acceptance.sh tests/test_verify_infra_only_acceptance.py
git commit -m "test(infra): gate infra-only acceptance branch"
```

---

### Task 9: Record Read-Only AWS Phase A

**Files:**
- Create after execution: `docs/acceptance/2026-07-23-infra-only-training-acceptance-phase-a.md`

**Interfaces:**
- Canonical runtime report: `/tmp/infra-only-training-acceptance-phase-a.json`; never commit this generated file.

- [ ] **Step 1: Run local gates before contacting AWS**

Run: `scripts/verify-infra-only-acceptance.sh`

Expected: exit 0.

- [ ] **Step 2: Run only read-only preflight**

Run:

```bash
cd rtrrl/infra/control-plane
uv run python scripts/facility_preflight.py --control config/facility.yaml \
  | tee /tmp/infra-only-training-acceptance-phase-a.json
cd ../../../..
```

Expected: account `007122174918`, region `eu-north-1`, all four profiles ready, S3 visible, Aim scratch isolated on port 53801, `writes_performed:false`. Missing test labels may be reported as pending before the push phase.

- [ ] **Step 3: Record exact evidence**

The report records HEAD, merge base, SHA-256 of the canonical JSON, caller ARN, profile results, image visibility, Aim identity, local gate output, and explicit counters: zero ECR pushes, zero registrations, zero Batch submissions, zero S3 writes/deletes.

- [ ] **Step 4: Commit**

```bash
git add docs/acceptance/2026-07-23-infra-only-training-acceptance-phase-a.md
git commit -m "docs(infra): record generic acceptance preflight"
```

---

### Task 10: Build and Push Test-Labelled CPU/GPU Images After Authorization

**Files:**
- Create after execution: `docs/acceptance/2026-07-23-infra-only-training-acceptance-images.md`

**Interfaces:**
- Produces immutable `CPU_DIGEST` and `GPU_DIGEST` in `007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl`.

- [ ] **Step 1: Produce a no-write deployment preview**

Run:

```bash
cd rtrrl/infra/control-plane
uv run python scripts/deploy_facility.py --control config/facility.yaml
```

Expected: mode `dry-run`, exact two test labels, profiles `c7am,c7al,c7ax,g6x`, `submission_supported:false`, and no AWS write.

- [ ] **Step 2: Stop and request explicit authorization for exactly two ECR test-label pushes**

Do not run `--push` until the user authorizes pushing:

```text
infra-acceptance-brax-ppo-cpu-20260723
infra-acceptance-brax-ppo-gpu-20260723
```

No memo tag is included in this authorization.

- [ ] **Step 3: After authorization, build, runtime-test, and push**

Run:

```bash
cd rtrrl/infra/control-plane
uv run python scripts/deploy_facility.py \
  --control config/facility.yaml \
  --build --push --docker-via-sudo --confirm-account 007122174918 \
  | tee /tmp/infra-only-training-acceptance-images.json
```

Expected: CPU and GPU images build; catalog decodes to one shared
`brax_ppo_acceptance` identity; CPU runtime uses CPU; the GPU image contains
the CUDA 12 JAX plugin and imports on the non-GPU builder; neither image
contains memo/control-plane; push returns one immutable digest per label.
NVIDIA L4 selection and the real CUDA JIT operation are not claimed until the
authorized `g6x` job in Task 12.

- [ ] **Step 4: Verify ECR digests read-only**

Use `aws ecr batch-get-image` for each exact tag and compare returned digests with deployment output. Record digests, image IDs, label decode result, local runtime checks, and zero Batch submissions.

- [ ] **Step 5: Commit evidence**

```bash
git add docs/acceptance/2026-07-23-infra-only-training-acceptance-images.md
git commit -m "docs(infra): record acceptance image evidence"
```

---

### Task 11: Register Four Digest-Bound Job Definitions After Separate Authorization

**Files:**
- Create after execution: `docs/acceptance/2026-07-23-infra-only-training-acceptance-definitions.md`

**Interfaces:**
- Consumes: immutable `CPU_DIGEST` and `GPU_DIGEST` from Task 10.
- Produces: four job-definition ARNs, each retry attempts one.

- [ ] **Step 1: Verify digests and registration request without mutation**

Run read-only ECR queries and confirm CPU digest is used by `c7am`, `c7al`, `c7ax`; GPU digest is used by `g6x`; roles and resources match `config/facility.yaml` and `aws_profiles.py`.

- [ ] **Step 2: Stop and request authorization for four registrations**

Do not call `RegisterJobDefinition` until the user authorizes exactly four new digest-bound definitions. This authorization does not authorize Batch submission.

- [ ] **Step 3: Register after authorization**

Run:

```bash
cd rtrrl/infra/control-plane
uv run python scripts/deploy_facility.py \
  --control config/facility.yaml \
  --register --confirm-account 007122174918 \
  --cpu-digest "$CPU_DIGEST" \
  --gpu-digest "$GPU_DIGEST" \
  | tee /tmp/infra-only-training-acceptance-definitions.json
```

Expected: four nonempty ARNs; image references are immutable digests; `retryStrategy.attempts == 1`; worker protocol is `1`; no jobs submitted.

- [ ] **Step 4: Read back and record definitions**

Use `aws batch describe-job-definitions --status ACTIVE` for each exact ARN. Record image digest, resources, roles, command, protocol, and retry count.

- [ ] **Step 5: Commit evidence**

```bash
git add docs/acceptance/2026-07-23-infra-only-training-acceptance-definitions.md
git commit -m "docs(infra): record acceptance definitions"
```

---

### Task 12: Run the Six-Job CPU/GPU AWS Acceptance After Paid-Run Authorization

**Files:**
- Create after execution: `docs/acceptance/2026-07-23-infra-only-training-acceptance-run.md`

**Interfaces:**
- Produces one canonical lowercase UUID experiment ID and exact `s3://rtrrl-artifacts-007122174918/experiments/{experiment_id}/` prefix.

- [ ] **Step 1: Validate without writes**

Run:

```bash
cd rtrrl/infra/control-plane
uv run trainerctl validate examples/experiment-smoke.yaml
```

Expected: two groups; five trials each; `configs_per_batch:2`; `runs_per_job:2`; estimated jobs three per group; profiles `c7am` and `g6x`; both digest-bound catalogs expose `brax_ppo_acceptance`.

- [ ] **Step 2: Stop and request paid-job authorization**

Request authorization for exactly six jobs: three `c7am` and three `g6x`, each native attempt one, with child counts `2,2,1` per group. Do not run `trainerctl run` before approval.

- [ ] **Step 3: Run once in the foreground after authorization**

Run:

```bash
cd rtrrl/infra/control-plane
uv run trainerctl run examples/experiment-smoke.yaml \
  | tee /tmp/infra-only-training-acceptance-run.json
```

Expected: success report with ten completed runs and six submitted job IDs. Any failure stops the command; do not retry or continue.

- [ ] **Step 4: Verify exact acceptance evidence read-only**

For all six job IDs verify:

- three `c7am`, three `g6x`;
- no job has more than one attempt;
- first two rounds submit one job per group with two serial children; final round submits one job per group with one child;
- GPU CloudWatch output reports backend `gpu`, device kind containing `NVIDIA L4`, and successful real JAX matrix operation;
- CPU output reports backend `cpu`;
- all ten completion markers have exit code zero;
- all ten Aim runs have finite `eval/episode_return` and finalized marker;
- all ten runs have one checkpoint and one complete Rerun evaluation episode;
- both Optuna studies contain five COMPLETE trials and received one `study.tell()` per trial;
- no resubmit, retry, cancellation, or seventh job exists.

- [ ] **Step 5: Commit evidence**

```bash
git add docs/acceptance/2026-07-23-infra-only-training-acceptance-run.md
git commit -m "docs(infra): record six-job acceptance"
```

The optional reference memo image may be tested only after this task passes and under its own authorization. Do not amend this generic acceptance result if the optional evidence fails or is skipped.

---

### Task 13: Clean Only the Exact Scratch Experiment After Separate Authorization

**Files:**
- Create after execution: `docs/acceptance/2026-07-23-infra-only-training-acceptance-cleanup.md`

**Interfaces:**
- Consumes: exact experiment ID and prefix from Task 12.
- Preserves: ECR images/tags, all four job definitions, queues, compute environments, roles, bucket, Aim main repository, historical HPO data, and every other experiment.

- [ ] **Step 1: Generate and save a dry-run manifest**

Run:

```bash
cd rtrrl/infra/control-plane
uv run python scripts/cleanup_acceptance.py \
  --control config/facility.yaml \
  --experiment-id "$EXPERIMENT_ID" \
  | tee /tmp/infra-only-training-acceptance-cleanup-dry-run.json
```

Expected: `writes_performed:false`; prefix equals `s3://rtrrl-artifacts-007122174918/experiments/$EXPERIMENT_ID/`; only exact S3 keys and Aim run hashes from the ten acceptance runs appear.

- [ ] **Step 2: Stop and request authorization for the exact manifest**

Present the experiment ID, exact prefix, S3 key count, Aim run hashes, and SHA-256 of the dry-run manifest. Do not delete before the user authorizes that exact set.

- [ ] **Step 3: Execute exact cleanup after authorization**

Run:

```bash
cd rtrrl/infra/control-plane
PREFIX="s3://rtrrl-artifacts-007122174918/experiments/$EXPERIMENT_ID/"
uv run python scripts/cleanup_acceptance.py \
  --control config/facility.yaml \
  --experiment-id "$EXPERIMENT_ID" \
  --confirm-prefix "$PREFIX" \
  --execute \
  | tee /tmp/infra-only-training-acceptance-cleanup.json
```

Expected: only the approved manifest is deleted; post-delete exact-prefix listing and exact Aim query are empty.

- [ ] **Step 4: Verify preserved shared resources read-only**

Confirm both test labels still resolve, four registered definitions remain active, eight queues/four CEs are unchanged, bucket exists, Aim main repository is untouched, and no other `experiments/` prefix count changed.

- [ ] **Step 5: Commit evidence**

```bash
git add docs/acceptance/2026-07-23-infra-only-training-acceptance-cleanup.md
git commit -m "docs(infra): record exact acceptance cleanup"
```

---

### Task 14: Publish the Authoritative Generic User Manual and Final Gate

**Files:**
- Rewrite: `infra/README.md`

**Interfaces:**
- Documentation covers only generic image/descriptor/SDK contracts and verified facility operations.
- Memo-specific guidance links to branch `reference/memo-sdk-2026-07-23` and labels it non-mergeable example code.

- [ ] **Step 1: Write documentation assertions before editing the manual**

Extend `tests/test_verify_infra_only_acceptance.py` to assert the manual contains:

```python
required = {
    "brax_ppo_acceptance",
    "org.rtrrl.trainer.scripts.v1",
    "trainerctl validate",
    "trainerctl run",
    "run-cpu-c7am-queue",
    "run-gpu-queue",
    "Aim",
    "Rerun",
    "checkpoints",
    "reference/memo-sdk-2026-07-23",
    "non-mergeable",
}
assert required <= set_of_present_terms
assert "memo_stream_ac" not in primary_acceptance_sections
assert "memo_rtrrl" not in primary_acceptance_sections
```

Run the test and expect failure against the old manual.

- [ ] **Step 2: Rewrite the manual from verified commands**

Document:

- algorithm-independent boundaries and the standalone acceptance trainer;
- exact protocol-version-1 descriptor and repository-level SDK integration contract;
- four profiles, four dev queues at priority 10, four run queues at priority 100, shared capacity, and non-preemption;
- local install/test commands and `scripts/verify-infra-only-acceptance.sh`;
- CPU/GPU root-context image build, label decode, runtime tests, and test tags;
- `trainerctl validate/run`, foreground lifecycle, one-attempt/fail-fast semantics, `2+2+1`, serial children, and artifact layout;
- Aim/Rerun/checkpoint/completion-marker/S3 lookup commands;
- separate authorization boundaries for push, register, paid run, and exact cleanup;
- preserved historical commands and resources;
- the memo reference branch as an optional non-mergeable example, never a production catalog or acceptance blocker.

Do not document an unexecuted AWS command as successful; mark skipped optional memo evidence as optional.

- [ ] **Step 3: Execute every documented safe command**

Run all `--help`, dry-run, validate, local test, label decode, and read-only lookup commands copied into the manual. For mutating commands, compare text against the already executed phase reports instead of rerunning them.

Expected: every safe command exits as documented; every mutation example includes the exact authorization warning.

- [ ] **Step 4: Run the final local and whole-branch gates**

Run:

```bash
scripts/verify-infra-only-acceptance.sh
BASE="$(git merge-base main HEAD)"
test -z "$(git diff --raw "$BASE" HEAD -- memo .github/workflows/build-memo-image.yml .github/workflows/memo-ci.yml)"
test -z "$(git diff --name-status "$BASE" HEAD -- memo .github/workflows/build-memo-image.yml .github/workflows/memo-ci.yml)"
git diff --check "$BASE"..HEAD
git status --short
```

Expected: all suites pass; protected raw/name-status outputs are empty; no whitespace errors; only the intended uncommitted manual/test edits are present before commit.

- [ ] **Step 5: Commit the final documentation task**

```bash
git add infra/README.md tests/test_verify_infra_only_acceptance.py
git commit -m "docs(infra): publish generic facility manual"
```

- [ ] **Step 6: Re-run the post-commit merge gate**

Run:

```bash
scripts/verify-infra-only-acceptance.sh
scripts/check-infra-merge-boundary.sh
git status --short
```

Expected: both gates exit 0 and status is empty. The mergeable branch is ready for review without any memo or memo-workflow tree change.

