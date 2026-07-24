# Infrastructure-Only Training Acceptance — Images

## Scope and source identity

- Branch: `feature/trainer-infra`
- Source commit for every run:
  `358f23f022f2c61f1bba9a23ac873a47f43229c9`
- Repository: `le2333/RTRRL-AAAI25`
- AWS account: `007122174918`
- Region: `eu-north-1`
- Existing repository:
  `007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl`
- Phase A prerequisite:
  `docs/acceptance/2026-07-23-infra-only-training-acceptance-phase-a.md`

The development host did not build or run Docker. GitHub-hosted runners built
and verified the images. This record contains no registry or OIDC token.

## Bootstrap build-only run

Run
[`30075805991`](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30075805991)
was triggered by `push`, used the exact source commit above, and concluded
`success`. Its jobs were:

- `Build and verify cpu`, job `89426150116`: `success`
- `Build and verify gpu`, job `89426150234`: `success`
- push matrix gate, job `89427252186`: `skipped`

Both build jobs passed catalog encoding, local image build, actual-label
decode/equality, required-path checks, worker/import checks, runtime checks,
JSON evidence creation, and artifact upload. The CPU image selected the CPU
backend and completed a real JAX addition. For the GPU image,
the `jax_cuda12_plugin` module spec was discoverable and JAX imported on
non-GPU runner. The runtime checks also proved that neither image exposed an
importable `memo` package or the control-plane `trainer_infra` package. No AWS
credential, ECR login, ECR query, or image push step ran.

Downloaded evidence:

- CPU artifact `infra-acceptance-build-cpu-evidence`: artifact ID
  `8589833229`; artifact API digest
  `sha256:8653fde0687f52fe0817e8be7a19b21097d57b0c8095b52ed21cd74f5f4cd18f`;
  downloaded JSON SHA-256
  `00db168cd9d707c808207194707fafa948e3f292d53787bb0ebb8cf1e312b05f`.
- GPU artifact `infra-acceptance-build-gpu-evidence`: artifact ID
  `8589922921`; artifact API digest
  `sha256:ef07e46b511c4d126e650daf30f8b9ac1696bf71499448c7ef176a818cf97dbf`;
  downloaded JSON SHA-256
  `cff16722f286837d3836c8551ddae4d8220983f374f2498e382cd8f552804ea5`.

The artifact API digest covers the uploaded artifact archive; the downloaded
JSON SHA-256 covers the extracted evidence file and is therefore intentionally
a different value.

The build-only JSON recorded:

- CPU image ID
  `sha256:f811e34bc79e264a34842a17a50aa1e3b5958038b3f5785e027c3643fe2f0e88`,
  size `1706924299` bytes, and `immutable_digest: null`.
- GPU image ID
  `sha256:ea04c22bf51e4a16529ca20af46c9138a7ee75e2d65c27d173fcdc3e623b9e9c`,
  size `6759991369` bytes, and `immutable_digest: null`.

## First authorized push attempt and fail-closed boundary

Run
[`30076390833`](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30076390833)
was an authorized `workflow_dispatch` push attempt at the same source commit.
The CPU and GPU build jobs (`89427955961` and `89427955886`) succeeded. Both
rebuild jobs (`89429085567` CPU and `89429085613` GPU) then rebuilt and passed
their label/path/runtime checks, but failed at `Confirm account and role
boundary`.

The repository variable had been created under the wrong name,
`AWS_ROLE_ARN`, while the workflow deliberately reads
`INFRA_ACCEPTANCE_AWS_ROLE_ARN`. The boundary step consequently observed an
empty role value and rejected it as outside the approved account/role boundary.
For both matrix jobs, every later step was `skipped`: configure push-only AWS
credentials, authenticated-account verification, existing-repository
verification, ECR login, fixed-tag push, push evidence creation, and push
evidence upload. Therefore this attempt configured no AWS credentials, made no
ECR call, and pushed `0` images.

Only build evidence was uploaded:

- CPU artifact ID `8590059356`, artifact API digest
  `sha256:cea4cf0d36f25c23c8335477ad030b392376eea25319d99ced609501b7c89e08`.
- GPU artifact ID `8590148092`, artifact API digest
  `sha256:a13675a59ede91bf60028c6f1901fc3ce94d238422ea51d928f43d325d7aebc1`.

The mistaken `AWS_ROLE_ARN` repository variable was deleted and the expected
`INFRA_ACCEPTANCE_AWS_ROLE_ARN` variable was set. A read-only repository
variable listing after correction showed only
`INFRA_ACCEPTANCE_AWS_ROLE_ARN`, updated at `2026-07-24T07:56:58Z`; no variable
value is reproduced here.

## Successful authorized push run

Run
[`30077184004`](https://github.com/le2333/RTRRL-AAAI25/actions/runs/30077184004)
was dispatched after that correction, used the same source commit, and
concluded `success`. All four matrix jobs succeeded:

- `Build and verify cpu`, job `89430426690`
- `Build and verify gpu`, job `89430426752`
- `Rebuild, verify, and push cpu`, job `89431552461`
- `Rebuild, verify, and push gpu`, job `89431552463`

For both variants, initial build and runtime verification succeeded. The push
jobs then succeeded at rebuild, repeated runtime verification, exact
account/role boundary confirmation, OIDC credential configuration,
authenticated account `007122174918` verification, existing `rtrrl`
repository verification, ECR login, exact fixed-tag push, evidence creation,
and evidence upload.

The shared actual image label decoded to protocol version `1` with exactly the
`brax_ppo_acceptance` script identity and matched the committed catalog. Both
push JSON files recorded catalog SHA-256
`c3881b0fcdc4881aabcb1ef1f41e28518150efe22b4074946fe130d148be1a2a`
and these two pinned Dockerfile base digests:

- `sha256:bb38f0ebd7d5d42c60d46a72cc8dbf2ed66c7263212f3b94a8ddfe2b60f7f8ca`
- `sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49`

CPU evidence:

- Exact tag:
  `infra-acceptance-brax-ppo-cpu-20260723`
- Immutable image:
  `007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:ec1ae1e426313dbbda1567dc48eb942f7b76c1b0d91b3738584ef5a9fda91a08`
- Image ID:
  `sha256:822355dd071847c9b23672c71079441f5fc04fd86751564807831a1da09d7e9c`
- Image size: `1706924299` bytes
- Push artifact ID: `8590526376`
- Push artifact API digest:
  `sha256:ec0fcdba2f0baa9607cf9d7c05568b32846e6ace95e19e918b0b849fe44c3997`
- Downloaded push JSON SHA-256:
  `2d6d080261388d239715d15b61b15677b3828b1b5a19bf60a45ad18c9d398864`
- Initial-build artifact ID: `8590362141`; artifact API digest:
  `sha256:99bd9ca5792b4815faae8694b41823fc2b95f3a5226d32a59ad5b2b0a840dba1`

GPU evidence:

- Exact tag:
  `infra-acceptance-brax-ppo-gpu-20260723`
- Immutable image:
  `007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:938fb1bfce6131ecd3f40a56475946e8865eed6f902b74f15c3e808f9da97e6a`
- Image ID:
  `sha256:e87d0f39d499135675f48986d7d0747213f3a4ed738821f273af8a6f886a6b90`
- Image size: `6759991369` bytes
- Push artifact ID: `8590687074`
- Push artifact API digest:
  `sha256:6f56b88d3119fd71d8001b249a7569d6f3bd42d2c2c32c36e1fca4a74b0a481d`
- Downloaded push JSON SHA-256:
  `eb612633ac05b5153d603e57d1b9394e108d924ffec196debde9f8fffb3521e1`
- Initial-build artifact ID: `8590445771`; artifact API digest:
  `sha256:b47d447249a397ef4065630307e8e3247c85b2a2182fdb66a6e540004bbb19f8`

GitHub emitted Node.js 20 and `punycode` deprecation warnings while forcing
affected actions to Node.js 24. These were warnings, not failures; every
required job and step in the successful run concluded `success`.

## Read-only ECR verification

A local `ecr:DescribeImages` request failed with `AccessDeniedException` for
the controller role. No permission was changed. The authorized read-only
`aws ecr batch-get-image` request for both exact tags succeeded and returned:

- CPU:
  `sha256:ec1ae1e426313dbbda1567dc48eb942f7b76c1b0d91b3738584ef5a9fda91a08`
- GPU:
  `sha256:938fb1bfce6131ecd3f40a56475946e8865eed6f902b74f15c3e808f9da97e6a`
- `failures: []`

Both read-back digests exactly match the push output and downloaded push JSON.

## Claim boundary and mutation counters

This phase proves image construction and non-GPU-runner runtime contracts. It
does **not** claim NVIDIA L4 selection, a real CUDA JIT operation, a paid Batch
run, or Task 12 acceptance.

The workflow's runtime verification proved no importable `memo` or
control-plane package in either image. The branch protection gate and direct
raw/name-status comparisons showed zero tree difference from merge base
`1551fda2ecb92dc6351113fb3ee77e55bfe56cd0` under `memo/`,
`.github/workflows/build-memo-image.yml`, and
`.github/workflows/memo-ci.yml`.

Task 10 mutation counters:

- ECR image pushes: `2` total, exactly the two authorized test labels
- ECR image pushes in failed run `30076390833`: `0`
- Batch job-definition registrations: `0`
- Batch job submissions: `0`
- S3 object writes: `0`
- S3 object deletes: `0`

No `/tmp` report, downloaded artifact, credential, or token is committed.
