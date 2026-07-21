# Complete Facility Task 2 Report

## Status

Implemented Task 2 only: strict S3 exchange storage, fail-closed ECR catalog
reads, read-only Batch preflight, submit/query-only Batch runtime behavior, and
the serial fail-fast image worker.

## Delivered

- `ObjectStore` is a minimal protocol for bytes, canonical JSON, and file
  transfer.
- `S3ObjectStore` accepts only exact configured
  `s3://<bucket>/experiments/<experiment-id>/...` objects. Uploads record a
  SHA-256 metadata value; downloads verify stored and caller-provided hashes.
  AWS exceptions are not wrapped.
- `BotoEcrCatalogReader` resolves a tag with one `BatchGetImage`, then reads the
  canonical digest manifest and config label without another tag lookup.
  Missing, ambiguous, malformed, and ECR-declared failure responses fail
  closed. It exposes no ECR mutation.
- `AwsBatchPreflight` read-only validates all four Task 1 compute environments,
  run queues, and active digest-bound job definitions. Job-definition pages
  are consumed until completion.
- `AwsBatchAdapter` exposes only `submit` and `query`. Submission uses the
  profile's exact run queue, a validated preconfigured definition, the worker
  bundle URI command, exact resource requirements, and AWS retry attempts one.
  Queries are chunked at 100 IDs and preserve raw `FAILED` status and reason.
- The standalone worker imports the Task 1 contracts from the control-plane
  package, verifies canonical bundle/input hashes, writes each complete SDK
  context to a temporary JSON file, exports `TRAINER_RUN_CONTEXT_PATH`, invokes
  children with `shell=False`, uploads sorted Aim/Rerun/checkpoint artifacts
  and completion markers, and stops after the first nonzero child.

## TDD Evidence

RED was observed before implementation for each new module and again for
stored S3 metadata verification, canonical ECR config digests, digest-bound
Batch definition identity, and deterministic worker artifact upload order.

## Verification

- Task 2 targeted suite: 30 passed.
- Full control-plane suite: 326 passed.
- Ruff: passed.
- `git diff --check`: passed.
- IDE lint for changed implementation and tests: no findings.

No memo, JAX, Docker, or real AWS command was run.

## Follow-up Integration Note

The formal memo image must install the control-plane package alongside
`training-sdk` before copying `/opt/trainer/worker.py`; the worker intentionally
reuses `JobBundle`, `RunBundle`, and `CompletionMarker` instead of copying Task
1 contract types.
