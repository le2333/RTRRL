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

## Fail-fast and authorization boundary

The second-round CPU job was not submitted. There was no retry, resubmit,
cancellation, or fourth job. This is the expected fail-fast boundary after the
controller failure, but it is not a passing three-job acceptance.

Task 12 remains in progress. The Aim read-only close compatibility fix must be
landed and verified offline first. Any new paid AWS acceptance attempt requires
fresh explicit authorization; this failed attempt does not authorize a retry.
