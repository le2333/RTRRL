# Batch Heavy-Test Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run each memory-intensive JAX test file in an isolated, explicitly selected AWS Batch profile.

**Architecture:** A small control-plane module owns immutable test profiles, read-only drift checks, digest-bound test job definitions, and one-file-per-job submission. A shell builder creates CPU/GPU overlay images from the current worktree without changing formal image tags.

**Tech Stack:** Python 3.12, boto3, Pydantic 2, AWS Batch, ECR, Docker, pytest.

## Global Constraints

- Profiles are exactly `c7am`, `c7ax`, and `g6x`.
- `g6x` is exactly `g6.xlarge` with one NVIDIA L4.
- Existing c7am/g6x resources are read-only.
- Only missing dedicated c7ax resources may be created.
- Each Batch job runs exactly one pytest file.
- Test paths must resolve under `memo/tests/`.
- Images and jobs use unique test labels and never overwrite formal tags.
- No Aim or formal experiment S3 data is written.
- Do not push git commits or Docker `latest` tags.

---

### Task 1: Exact Test Profiles and Dedicated c7ax Queue

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/heavy_tests.py`
- Create: `rtrrl/infra/control-plane/tests/test_heavy_tests.py`

**Interfaces:**
- `TEST_PROFILES: Mapping[str, HeavyTestProfile]`
- `validate_test_profile(batch, name: str) -> ValidatedTestProfile`
- `create_c7ax_if_missing(batch, settings: AwsNetworkSettings) -> None`

- [ ] Write failing tests for all exact names and drift fields.

```python
def test_profiles_are_exact():
    assert set(TEST_PROFILES) == {"c7am", "c7ax", "g6x"}
    assert TEST_PROFILES["c7ax"].instance_type == "c7a.xlarge"
    assert TEST_PROFILES["g6x"].instance_type == "g6.xlarge"
    assert TEST_PROFILES["g6x"].gpu_model == "NVIDIA L4"


def test_existing_profile_drift_is_never_mutated(fake_batch):
    fake_batch.compute_environment.instance_types = ["c7a.large"]
    with pytest.raises(ProfileDriftError, match="instanceTypes"):
        validate_test_profile(fake_batch, "c7am")
    assert fake_batch.update_calls == []
```

- [ ] Run `uv run pytest tests/test_heavy_tests.py -v`; expect missing module.

- [ ] Implement immutable profiles.

```python
TEST_PROFILES = {
    "c7am": HeavyTestProfile(
        queue="rtrrl-cpu-c7am-queue", compute_environment="rtrrl-cpu-c7am-ce",
        instance_type="c7a.medium", vcpus=1, memory_mib=1600, gpus=0,
    ),
    "c7ax": HeavyTestProfile(
        queue="rtrrl-cpu-c7ax-queue", compute_environment="rtrrl-cpu-c7ax-ce",
        instance_type="c7a.xlarge", vcpus=4, memory_mib=7168, gpus=0,
    ),
    "g6x": HeavyTestProfile(
        queue="rtrrl-gpu-g6x-queue", compute_environment="rtrrl-gpu-g6x-ce",
        instance_type="g6.xlarge", vcpus=4, memory_mib=12000, gpus=1,
        gpu_model="NVIDIA L4",
    ),
}
```

- [ ] Implement create-only-if-absent c7ax behavior. An existing resource always goes through the same exact validator and is never updated.

- [ ] Run full control-plane tests, Ruff, and `git diff --check`.

- [ ] Commit with `feat(infra): add Batch heavy-test profiles`.

---

### Task 2: Current-Worktree Test Images

**Files:**
- Create: `infra/batch/heavy-tests/Dockerfile`
- Create: `infra/batch/heavy-tests/build-image.sh`
- Test: `rtrrl/infra/control-plane/tests/test_heavy_tests.py`

**Interfaces:**
- `build-image.sh --profile c7am|c7ax|g6x`
- Prints one final JSON object containing `tag`, `digest`, and `image`.

- [ ] Write a failing static integration test proving the builder excludes `.venv`, caches, logs, and `.git`.

- [ ] Add the overlay Dockerfile.

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY training-sdk /workspace/training-sdk
COPY memo /app
RUN /opt/venv/bin/python -m pip install /workspace/training-sdk pytest
WORKDIR /app
ENV PYTHONPATH=/workspace/training-sdk/src:/app \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MALLOC_ARENA_MAX=2
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
```

- [ ] Implement a temporary build context containing only filtered `memo/`, `training-sdk/`, and the Dockerfile. Resolve the chosen base image and pushed test image to ECR digests.

- [ ] Verify a CPU image can execute `python -m pytest --version`.

- [ ] Commit with `feat(infra): build isolated Batch test images`.

---

### Task 3: One-File-Per-Job Submission and Evidence

**Files:**
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/heavy_tests.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/heavy_test_cli.py`
- Modify: `rtrrl/infra/control-plane/pyproject.toml`
- Modify: `rtrrl/infra/control-plane/tests/test_heavy_tests.py`

**Interfaces:**
- `trainer-heavy-test submit --profile PROFILE --image IMAGE@DIGEST TEST_FILE...`
- `trainer-heavy-test wait JOB_ID...`

- [ ] Write failing path, job-shape, and failure aggregation tests.

```python
def test_one_job_per_exact_test_file(runner, image):
    jobs = runner.submit(
        profile="c7ax",
        image=image,
        tests=["memo/tests/online_ac/test_eval_trace.py",
               "memo/tests/online_ac/test_jit_contract.py"],
    )
    assert len(jobs) == 2
    assert all(" /usr/bin/time -v " in job.command_text for job in jobs)


@pytest.mark.parametrize("path", ["memo/pyproject.toml", "../tests/x.py", "rtrrl/tests/x.py"])
def test_rejects_non_memo_test_path(runner, path):
    with pytest.raises(ValueError, match="memo/tests"):
        runner.submit(profile="c7ax", image="repo@sha256:" + "a" * 64, tests=[path])
```

- [ ] Register or reuse an exact digest-bound test job definition per profile and image.

- [ ] Submit command:

```bash
/usr/bin/time -v env \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MALLOC_ARENA_MAX=2 \
  python -m pytest memo/tests/online_ac/test_eval_trace.py -q
```

- [ ] For g6x prepend a probe that prints `jax.devices()` and
  `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`; fail unless
  the output names NVIDIA L4.

- [ ] Wait for terminal states, collect CloudWatch log stream names and maximum
  RSS lines, and fail the aggregate if any job is not `SUCCEEDED`.

- [ ] Run tests, Ruff, and `git diff --check`.

- [ ] Commit with `feat(infra): submit isolated Batch test files`.

---

## Initial Use

After all three tasks:

```bash
CPU_JSON="$(infra/batch/heavy-tests/build-image.sh --profile c7ax)"
CPU_IMAGE="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["image"])' "${CPU_JSON}")"
GPU_JSON="$(infra/batch/heavy-tests/build-image.sh --profile g6x)"
GPU_IMAGE="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["image"])' "${GPU_JSON}")"

trainer-heavy-test submit --profile c7ax --image "${CPU_IMAGE}" \
  memo/tests/online_ac/test_eval_trace.py
trainer-heavy-test submit --profile g6x --image "${GPU_IMAGE}" \
  memo/tests/online_ac/test_jit_contract.py
```

Retain evidence until memo-first Task 3 review is complete, then deregister
test job-definition revisions and delete temporary ECR tags.
