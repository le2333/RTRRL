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
- Execution records and strict canonical/hash parsing live in
  `training_sdk.execution`; `trainer_infra.execution` re-exports them and keeps
  only the `ConcreteRun` to `RunContext` bridge.
- One strict `training_sdk.storage` S3 parser is shared by the control-plane
  S3/Batch paths and the worker.
- The standalone worker imports only `training_sdk` contracts, generates both
  config and context paths inside a per-run temporary directory, invokes
  children with `shell=False`, rejects escaping/symlink/nonregular artifacts,
  uploads sorted artifacts and completion markers, and stops after the first
  failure. Artifact upload failures still cause one failed marker write; marker
  write failures remain visible as the original storage exception.
- ECR reads bind every call to the expected registry account/region, verify the
  digest response identity, and verify downloaded config bytes against the
  manifest digest.
- Batch preflight uses an explicit expected contract and fail-closed checks for
  queue priority, complete compute shape/network/AMI, and active job-definition
  worker protocol, roles, logging, image, and resources.

## TDD Evidence

RED was observed for SDK protocol extraction, control-plane re-export identity,
shared URI parsing, canonical JSON reads, ECR registry/digest/blob checks,
strict Batch infrastructure contracts, isolated worker imports, generated
temporary paths, artifact failure markers, and unsafe artifact rejection.

## Verification

- Review-fix targeted suite: 106 passed.
- Full training-sdk suite: 120 passed.
- Full control-plane suite: 370 passed.
- Ruff: passed.
- `git diff --check`: passed.
- IDE lint for changed implementation and tests: no findings.

No memo, JAX, Docker, or real AWS command was run.

The formal worker image requires `training-sdk` and boto3 only; an isolated
subprocess import test blocks every `trainer_infra` import.
