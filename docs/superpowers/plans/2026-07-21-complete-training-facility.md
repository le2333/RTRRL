# Complete Training Facility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute one task at a time, write tests first, commit each task, and require an independent review before the next task.

**Goal:** Complete the foreground `trainerctl run` path from experiment YAML to memo CPU/GPU Batch execution, Aim/Rerun collection, Optuna completion, and a verified user manual.

**Architecture:** Extend the existing Task 1–3 control-plane contracts with immutable execution bundles, a pure submit/query Batch adapter, an S3-backed fail-fast worker, two observable memo launchers, and a foreground controller. Formal job definitions are deployed separately from runtime. Historical entries remain untouched.

**Tech Stack:** Python 3.10+, Pydantic 2, Optuna 4, boto3, Aim 3.28, Rerun, JAX/memo, pytest, Ruff, Docker, AWS Batch/ECR/S3.

## Global Constraints

- Profiles are exactly `c7am`, `c7al`, `c7ax`, and `g6x`.
- Formal queues are exactly `run-cpu-c7am-queue`, `run-cpu-c7al-queue`, `run-cpu-c7ax-queue`, and `run-gpu-queue`.
- Batch submission and AWS native retry attempts are exactly one.
- The adapter exposes no queue/CE mutation, retry, resubmit, cancel, rollback, or cleanup API.
- Any job or child failure terminates the experiment and prevents future HPO batches.
- One `trainerctl run` authorizes every trial in its YAML.
- Only `memo_stream_ac/rtu_rtrl` and `memo_rtrrl/shared` are added to the memo facility catalog.
- SDK and NumPy calls remain outside JIT.
- Historical commands, descriptors, workflows, images, job definitions, and data are preserved.
- Real AWS writes, image pushes, job-definition registration, paid jobs, and cleanup require separate authorization.

---

### Task 1: Execution Contracts and Four Profiles

**Files:**
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/models.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/materialize.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/heavy_tests.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/aws_profiles.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/execution.py`
- Modify/Test: corresponding `tests/test_models.py`, `test_materialize.py`, `test_heavy_tests.py`
- Create: `rtrrl/infra/control-plane/tests/test_execution.py`

**Interfaces:**
- `ResourceProfileName = Literal["c7am", "c7al", "c7ax", "g6x"]`
- `profile(name: ResourceProfileName) -> AwsProfile`
- `RunBundle`, `JobBundle`, `CompletionMarker`, `JobQuery`
- `build_run_context(experiment_id, group, concrete_run, artifact_prefix) -> training_sdk.RunContext`

- [ ] Write RED tests for the four exact resources, run queues, job resources, dev priority 10, and rejection of legacy/unknown profile names.
- [ ] Write RED tests proving execution retry fields are absent and an execution attempt is always zero.
- [ ] Write RED round-trip tests for job/run bundles, canonical hashes, completion markers, and every mandatory SDK RunContext field.
- [ ] Implement immutable profile and execution records using the existing canonical JSON/YAML/hash helpers.
- [ ] Extend materialization only enough to produce complete execution context; do not duplicate experiment resolution.
- [ ] Change heavy-test expected/created queue priority from 1 to 10 without changing any other runner behavior.
- [ ] Run targeted tests, full control-plane tests, Ruff, IDE lint, and `git diff --check`.
- [ ] Commit `feat(infra): define formal execution contracts`.

---

### Task 2: S3, ECR, Batch Adapter, and Worker

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/adapters/__init__.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/adapters/protocols.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/adapters/s3.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/adapters/aws_batch.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/ecr.py`
- Create: `rtrrl/infra/worker/worker.py`
- Create: `rtrrl/infra/control-plane/tests/test_s3.py`
- Create: `rtrrl/infra/control-plane/tests/test_aws_batch.py`
- Create: `rtrrl/infra/control-plane/tests/test_ecr.py`
- Create: `rtrrl/infra/control-plane/tests/test_worker.py`

**Interfaces:**
- `ObjectStore.put_bytes/get_bytes/put_json/get_json/put_file`
- `BotoEcrCatalogReader.resolve_and_fetch(reference) -> (ResolvedImage, ScriptCatalog)`
- `AwsBatchAdapter.submit(job_bundle, profile, job_definition) -> SubmittedJob`
- `AwsBatchAdapter.query(job_ids) -> tuple[JobQuery, ...]`
- Worker command: `python /opt/trainer/worker.py --bundle-s3-uri URI`

- [ ] Write RED S3 tests for exact experiment prefix, canonical JSON, hashes, missing/tampered objects, and artifact uploads.
- [ ] Write RED ECR tests for one tag resolution, digest-only catalog fetch, malformed/missing image metadata, and no second tag lookup.
- [ ] Write RED Batch tests for four run queues, exact resource overrides, preconfigured job definition, `attempts=1`, 100-ID query chunking, and raw failed states.
- [ ] Assert the adapter fake records no register/update/delete/cancel/retry calls and exposes no such methods.
- [ ] Write RED worker tests for hash verification, RunContext file injection, `shell=False`, deterministic order, marker/artifact upload, and stop-after-first-child-failure.
- [ ] Implement the smallest adapters and worker satisfying those contracts; propagate AWS errors unchanged.
- [ ] Run targeted/full control-plane tests, Ruff, lint, and diff check.
- [ ] Commit `feat(infra): execute immutable Batch job bundles`.

---

### Task 3: Memo Facility Launchers and Observability

**Files:**
- Create: `memo/experiments/base/facility.py`
- Create: `memo/experiments/memo_stream_ac/run.py`
- Create: `memo/experiments/memo_rtrrl/run.py`
- Modify: `memo/experiments/base/experiment.py`
- Modify: `memo/memorax/online_ac/build.py`
- Modify: `memo/memorax/environments/brax.py`
- Create: `memo/tests/test_facility_launchers.py`
- Create: `memo/tests/test_experiment_observability.py`
- Create: focused fixtures under `memo/tests/fixtures/`

**Interfaces:**
- `FacilityInput.load(path) -> FacilityInput`
- `emit_training_summaries(training_run, completed_stats, env_steps) -> None`
- `episode_from_trace(trace, environment_index) -> training_sdk.Episode`
- launcher argv: `python /app/experiments/memo_stream_ac/run.py --config PATH`
  or `python /app/experiments/memo_rtrrl/run.py --config PATH`

- [ ] Write RED loader tests for strict concrete YAML, supported environment maps, and fixed topology/agent constraints.
- [ ] Write RED budget tests proving real `state.step` equals total environment interactions and `max_episode_steps` is configurable.
- [ ] Write RED host-observability tests for each completed training episode, exact env steps, complete evaluation Episode, cadence, and incomplete-trace rejection.
- [ ] Add JIT-boundary tests proving SDK/bootstrap/NumPy conversion execute only on the host.
- [ ] Implement both launchers through existing memo builders; call `bootstrap_from_environment()` once and close/finalize the run.
- [ ] Expose evaluation traces to the host without moving training transitions out of JIT.
- [ ] Run lightweight memo tests locally and memory-heavy parity/trace tests through the dev Batch runner.
- [ ] Commit `feat(memo): add observable facility launchers`.

---

### Task 4: Memo Catalog and Formal Images

**Files:**
- Create: `memo/infra/scripts/index.yaml`
- Create: `memo/infra/scripts/memo_stream_ac.yaml`
- Create: `memo/infra/scripts/memo_rtrrl.yaml`
- Modify: memo CPU/GPU Dockerfiles and `memo/pyproject.toml`/`uv.lock` as required
- Modify: `.github/workflows/build-memo-image.yml`
- Create/Modify: repository-root `.dockerignore`
- Create: catalog/image tests

**Interfaces:**
- Memo catalog protocol version 1 with exactly two script files.
- Image paths: `/opt/trainer/worker.py`, `/opt/trainer/scripts/*`, shared `training_sdk`.
- Label: `org.rtrrl.trainer.scripts.v1`.

- [ ] Write RED descriptor tests for exact launcher argv, environment choices, fields, objectives, and catalog scope.
- [ ] Write RED Docker context tests proving memo, SDK, worker, and descriptors are present while caches/history are excluded.
- [ ] Change memo builds to repository-root context and a lock-consistent two-stage install.
- [ ] Embed the deterministic nonempty catalog label and copy descriptors/worker to fixed paths.
- [ ] Keep all legacy descriptors, Dockerfiles, workflows, and image tags unchanged.
- [ ] Run catalog codec, lock, CPU image, and GPU static/runtime smoke tests available without paid Batch work.
- [ ] Commit `feat(memo): package facility catalog and worker images`.

---

### Task 5: Aim Reader, Controller, and CLI

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/aim_reader.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/controller.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/cli.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/__main__.py`
- Create: `rtrrl/infra/control-plane/config/control.example.yaml`
- Create: `rtrrl/infra/control-plane/examples/experiment-smoke.yaml`
- Create: `test_aim_reader.py`, `test_controller.py`, `test_cli.py`

**Interfaces:**
- `AimReader.wait_for_result(run_id, objective, timeout) -> float`
- `ExperimentController.validate(path) -> ValidationReport`
- `ExperimentController.run(path) -> ExperimentReport`
- CLI: `trainerctl validate EXPERIMENT.yaml`, `trainerctl run EXPERIMENT.yaml`

- [ ] Write RED Aim tests for exact run ID, finalized marker, finite expected objective, timeout, and spool replay.
- [ ] Write RED controller test for two independent groups completing automatic `2+2+1`; assert only controller thread calls `study.tell()`.
- [ ] Write RED failure tests proving one failed job/child/marker/Aim result stops all future submissions and returns submitted IDs.
- [ ] Write RED CLI tests for read-only validate, foreground run, stable JSON/text errors, and the absence of status/resume/history commands.
- [ ] Implement dependency-injected controller orchestration using existing resolver, sampler, materializer, tracker, adapters, and Aim reader.
- [ ] Generate a fresh experiment ID once per invocation and persist final report/state under its experiment prefix.
- [ ] Run targeted/full tests, Ruff, lint, and diff check.
- [ ] Commit `feat(infra): run complete foreground experiments`.

---

### Task 6: Local End-to-End and Historical Compatibility

**Files:**
- Create: `rtrrl/infra/control-plane/tests/test_end_to_end.py`
- Create: `rtrrl/infra/control-plane/tests/test_historical_entries.py`
- Modify only implementation defects exposed by these tests.

- [ ] Build a fake ECR/S3/Batch/Aim harness that executes the real worker and two memo launcher fixtures.
- [ ] Prove resolve-to-tell `2+2+1`, mixed groups, multiple jobs, serial children, objective collection, Rerun/S3 artifact identities, and finite-space exit.
- [ ] Inject each failure boundary and prove no future batch is submitted and no resubmit/cancel occurs.
- [ ] Assert every historical command, descriptor, workflow, and HPO data entry still exists; exercise existing help/dry-run where safe.
- [ ] Run control-plane, SDK, memo targeted suites, Ruff, lock checks, and diff check.
- [ ] Commit `test(infra): verify complete facility lifecycle`.

---

### Task 7: Deployment and Real AWS Acceptance

**Files:**
- Create: explicit deployment/preflight scripts under `rtrrl/infra/control-plane/scripts/`
- Modify: IAM policies only after separate authorization
- Produce: test-labelled acceptance report under `docs/`

- [ ] Restore Aim scratch on port 53801 and verify it is isolated from the main repo.
- [ ] Run read-only account/region/profile/IAM/S3/ECR/Aim preflight.
- [ ] Request authorization, then build/push immutable memo CPU/GPU images and verify worker/SDK/catalog labels.
- [ ] Request authorization, then register four digest-bound single-attempt trainer job definitions through the deployment script.
- [ ] Request paid-job authorization, then run the small two-launcher CPU/GPU experiment through `trainerctl run`.
- [ ] Verify parallel Batch jobs, serial children, L4/JAX, Aim summaries/objectives, Rerun episodes, S3 markers/artifacts, Optuna completion, and no retries.
- [ ] Request separate cleanup authorization and remove only the scratch experiment data; preserve historical entries and all shared AWS resources.
- [ ] Commit deployment scripts and the evidence report; never commit secrets.

---

### Task 8: Authoritative Infra User Manual

**Files:**
- Rewrite: `infra/README.md`

- [ ] Document current component boundaries and the verified four-CE/eight-queue topology.
- [ ] Document profile selection, shared capacity, run priority, and non-preemption.
- [ ] Provide tested install, image build, `trainerctl validate/run`, heavy-test, Aim/Rerun/S3 lookup, and troubleshooting commands.
- [ ] Explain foreground lifecycle, single-command HPO authorization, fail-fast/no-retry behavior, and artifact locations.
- [ ] List historical entries and their compatibility status without deleting or presenting them as the new primary path.
- [ ] Execute every help/dry-run command and cross-check every path, link, queue, profile, and port.
- [ ] Run Markdown/link checks available in the repository.
- [ ] Commit `docs(infra): document the complete training facility`.
