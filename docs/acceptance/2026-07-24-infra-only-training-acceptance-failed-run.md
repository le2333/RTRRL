# Infrastructure-Only Training Acceptance — Failed Run

## Attempt identity and scope

- Attempt started at (UTC): `2026-07-24T21:30:24Z`
- Experiment ID: `b03f9ff7-4290-4f68-939a-a5c646782dbb`
- First-round CPU job: `0ea76b3c-d095-4dc2-a795-9e17d4499c0b`
- First-round GPU job: `423dfb14-3d75-48a6-a391-c0858558d3bb`
- AWS result for both jobs: `SUCCEEDED`
- Native attempts for both jobs: `1`

This document records read-only evidence from the already authorized attempt.
No AWS, GitHub, Docker, cleanup, cancellation, resubmission, or retry operation
was performed while documenting the failure.

The Batch result was re-checked read-only with:

```bash
aws batch describe-jobs \
  --region eu-north-1 \
  --jobs \
    0ea76b3c-d095-4dc2-a795-9e17d4499c0b \
    423dfb14-3d75-48a6-a391-c0858558d3bb \
  --query \
    'jobs[].{jobId:jobId,status:status,attempts:length(attempts),exitCode:container.exitCode,queue:jobQueue,definition:jobDefinition,logStream:container.logStreamName}'
```

It returned `SUCCEEDED`, one attempt, and exit code zero for both jobs. The CPU
job used `run-cpu-c7am-queue` and definition
`trainer-c7am-ec1ae1e426313dbbda1567dc48eb942f7b76c1b0d91b3738584ef5a9fda91a08:1`.
The GPU job used `run-gpu-queue` and definition
`trainer-g6x-938fb1bfce6131ecd3f40a56475946e8865eed6f902b74f15c3e808f9da97e6a:1`.

## Completed child evidence

The first CPU job completed two serial children and the GPU job completed one
child. All three children wrote the expected S3 evidence:

- Aim buffer;
- checkpoint;
- complete Rerun evaluation artifact;
- terminal status artifact.

The three final `eval/episode_return` objective values were `24`, `23`, and
`23`. GPU logs identified an `NVIDIA L4`. Thus the paid worker executions and
their three child artifact sets completed; the acceptance failed later in the
controller read path.

The read-only S3 listing used bucket `rtrrl-artifacts-007122174918` and prefix
`experiments/b03f9ff7-4290-4f68-939a-a5c646782dbb/`. For each of these exact
run prefixes, it returned non-empty `aim-buffer/events.jsonl`,
`checkpoints/ppo-params.npz`, `rerun/.../episode-000002.rrd`, and
`status/attempt-0.json` keys:

```text
groups/cpu/runs/b03f9ff7-4290-4f68-939a-a5c646782dbb:cpu:0001/
groups/cpu/runs/b03f9ff7-4290-4f68-939a-a5c646782dbb:cpu:0002/
groups/gpu/runs/b03f9ff7-4290-4f68-939a-a5c646782dbb:gpu:0001/
```

The three checkpoint sizes were `1014805`, `1014456`, and `1018656` bytes; the
three Rerun sizes were `33402`, `33262`, and `33260` bytes. Filtering the exact
GPU CloudWatch stream
`trainer-g6x-938fb1bfce6131ecd3f40a56475946e8865eed6f902b74f15c3e808f9da97e6a/default/2485b749b873471c9f39af54cb7eedca`
for `NVIDIA L4` returned:

```json
{"device_kind":"NVIDIA L4","device_platforms":["gpu"],"platform":"gpu"}
```

## Controller failure

After reading the first finalized objective, the controller entered the
`finally` block and called `run.close()` on an Aim 3.28.0 read-only run. Aim
3.28.0 creates `RunTracker.sequence_infos` only for non-read-only trackers, but
`Run.close()` clears that attribute unconditionally. The close therefore raised:

```text
AttributeError: 'RunTracker' object has no attribute 'sequence_infos'
```

The emitted report correctly recorded `status:failed`, `completed_runs:0`, the
two submitted job IDs above, and the `AttributeError`. The launch command had
been piped through `tee` without `pipefail`, so its enclosing shell reported
exit zero despite the failed `trainerctl` report. Task 12 now uses separate
stdout and stderr redirection so a future command preserves the real
`trainerctl` exit status.

The authoritative S3 report key is
`experiments/b03f9ff7-4290-4f68-939a-a5c646782dbb/report.json`; its exact
payload is:

```json
{"completed_runs":0,"error":"AttributeError: 'RunTracker' object has no attribute 'sequence_infos'","experiment_id":"b03f9ff7-4290-4f68-939a-a5c646782dbb","experiment_metadata":{"purpose":"infra-acceptance"},"experiment_name":"infra-brax-ppo-acceptance","status":"failed","submitted_job_ids":["0ea76b3c-d095-4dc2-a795-9e17d4499c0b","423dfb14-3d75-48a6-a391-c0858558d3bb"]}
```

## Offline regression evidence

An independent direct call to Aim 3.28.0, without using `AimReader`, confirmed
the upstream mismatch on a temporary repository. From the repository root:

```bash
cd rtrrl/infra/control-plane
uv run --offline python - <<'PY'
from tempfile import TemporaryDirectory

from aim import Run

with TemporaryDirectory() as repo:
    writer = Run(repo=repo)
    run_hash = writer.hash
    writer["probe"] = True
    writer.close()

    reader = Run(repo=repo, run_hash=run_hash, read_only=True)
    print(
        {
            "aim_read_only": reader.read_only,
            "has_sequence_infos": hasattr(reader._tracker, "sequence_infos"),
        }
    )
    reader.close()
PY
```

It exited one with:

```text
{'aim_read_only': True, 'has_sequence_infos': False}
AttributeError: 'RunTracker' object has no attribute 'sequence_infos'
```

That reproducer exited one at `aim/sdk/run.py:720`. The focused real-repository
regression then passed with the compatibility fix:

```bash
cd rtrrl/infra/control-plane
uv run --offline pytest \
  tests/test_aim_reader.py::test_reader_safely_closes_real_read_only_aim_run -q
```

Result: `1 passed` on Aim `3.28.0`. This proof was run after the fix to verify
both the raw upstream failure and the guarded `AimReader` behavior; it is
separate from the implementation agent's original RED/GREEN cycle.

## Fail-fast and authorization boundary

The second-round CPU job was not submitted. There was no retry, resubmit,
cancellation, or fourth job. This is the expected fail-fast boundary after the
controller failure, but it is not a passing three-job acceptance.

Task 12 remains in progress. The Aim read-only close compatibility fix must be
landed and verified offline first. Any new paid AWS acceptance attempt requires
fresh explicit authorization; this failed attempt does not authorize a retry.
