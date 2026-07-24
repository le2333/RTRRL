# Infra-Only Training Acceptance Design

**Date:** 2026-07-23  
**Status:** Approved design; implementation pending

## Purpose

The training facility must be developed, tested, and merged without changing
the `memo` algorithm package or its existing entry points on `main`. The
facility remains algorithm-independent. Algorithm repositories integrate the
facility SDK only through separately reviewed algorithm-side work.

The existing memo SDK integration in the `trainer-infra` worktree is useful as
a reference implementation, but it is not part of the infrastructure merge.

## Non-Negotiable Merge Boundary

The infrastructure branch that is eventually proposed for merge must:

- contain no changes under `memo/`;
- contain no changes to the existing memo image workflow or memo image tags;
- preserve every historical algorithm entry point;
- contain the generic `training-sdk`, worker, controller, adapters, CLI,
  infrastructure tests, and deployment tooling;
- use an infrastructure-owned acceptance trainer for repeatable tests.

Before integration, the final diff against the branch base must prove that
`memo/` and the existing memo build workflow are unchanged.

No operation in this design modifies `main` directly.

## Reference Memo Integration

Before removing memo changes from the infrastructure branch, the current
memo-plus-SDK implementation is frozen on a dedicated reference branch and
worktree.

The reference implementation:

- is explicitly labelled as non-mergeable example code;
- keeps the two launchers, host-side SDK calls, episode conversion, evaluation
  trace handling, and success/failure lifecycle examples;
- may be used to build a test-labelled image for one-time realistic acceptance;
- is not a production catalog and is not merged into `main`;
- gives the memo maintainers an implementation reference for the refactored
  training code.

The reference branch is immutable after acceptance except for documentation
that identifies the source commit and test evidence.

## Infrastructure-Owned Acceptance Trainer

A standalone package is created under `rtrrl/infra/mock-trainer/`. It does not
import memo or any other algorithm project.

It contains:

- a small Brax PPO launcher;
- CPU and GPU image definitions;
- a protocol-version-1 script catalog and descriptor;
- deterministic small-budget experiment presets;
- explicit test-only failure modes;
- integration with the repository-level `training-sdk`.

The launcher records:

- parameters and configuration;
- completed training episode summaries;
- a finite objective metric;
- one complete evaluation episode for Rerun;
- a checkpoint;
- correct success or failure lifecycle state.

The GPU variant runs a real JAX/Brax operation and verifies that the selected
device is an NVIDIA L4 on `g6.xlarge`. The acceptance budget is intentionally
too small to assess PPO convergence; it validates the facility data path and
GPU runtime only.

## Test and Acceptance Layers

### Local deterministic tests

The fake ECR/S3/Batch/Aim harness executes the real worker and acceptance
launcher. It proves:

- resolver-to-materializer-to-controller-to-worker-to-SDK flow;
- independent HPO groups and `2+2+1` rounds;
- parallel jobs and serial child runs;
- objective collection and `study.tell()`;
- Aim, Rerun, checkpoint, completion-marker, and S3 identities;
- all fail-fast boundaries without retries, resubmission, or cancellation.

### Container tests

CPU and GPU images are built on isolated GitHub Actions runners from the
infrastructure acceptance package. The development host does not build or run
Docker images. Before push, each remote image job inspects and executes its
image to prove:

- fixed worker and catalog paths;
- importability of `training_sdk` and the launcher;
- catalog-label decoding;
- expected CPU or CUDA JAX runtime;
- absence of the control-plane package and memo.

The CPU and GPU matrix jobs use separate ephemeral disks. A build-only workflow
requires no AWS credentials and cannot push. ECR login and push are enabled
only by a separately authorized workflow input after the exact commit and tags
are reviewed. Actual NVIDIA L4 device execution remains a `g6.xlarge` Batch
acceptance check because standard GitHub-hosted runners do not provide L4 GPUs.

### Real AWS acceptance

AWS mutation remains split into separately authorized phases:

1. push test-labelled CPU and GPU images;
2. register four digest-bound, single-attempt job definitions;
3. run six paid jobs: three `c7am` and three `g6x`;
4. separately delete only the exact scratch experiment prefix.

The real experiment uses two five-trial groups, two configurations per HPO
round, and two serial children per job. Any failed Batch job, child run,
completion marker, Aim result, or objective terminates the foreground command.
There is no retry or continuation.

The one-time reference memo image may be run as additional evidence after the
generic acceptance trainer passes. It is not required for the infrastructure
merge.

## Documentation Corrections

The existing complete-facility specification and implementation plan are
revised so that memo registration is marked as superseded by this design.
Task reports remain historical evidence and are not rewritten to imply that
the original boundary was correct.

The user manual documents only the generic image/descriptor/SDK contract.
Memo-specific integration guidance points to the non-mergeable reference
branch and is clearly labelled as an example.

## Cleanup and Safety

- No shared Batch queue, compute environment, ECR repository, image, Aim main
  repository, historical HPO data, or historical entry point is deleted.
- Test images and job definitions remain until separately authorized cleanup.
- Scratch S3 and Aim data are deleted only with exact-prefix authorization.
- Exact cleanup first stops and validates the dedicated Aim scratch server,
  then requires the saved dry-run manifest and its separately authorized
  SHA-256. That manifest remains recovery authority if the final S3 sentinel
  deletion has an uncertain network result; recovery may delete only subsets
  of the original manifest and rejects all newly observed keys or Aim hashes.
- No credentials, tokens, local Aim state, or generated acceptance artifacts
  are committed.

## Acceptance Criteria

This correction is complete when:

1. the reference memo integration exists on a dedicated non-merge branch;
2. the infrastructure branch has no memo or memo-workflow diff;
3. the standalone acceptance trainer passes local and container contracts;
4. authorized AWS acceptance proves CPU/GPU execution and all artifacts;
5. the final whole-branch review confirms the main merge changes infra only.
