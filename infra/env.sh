#!/usr/bin/env bash
# Shared configuration for the AWS infra scripts (project-agnostic).
#
# This infra/ tree is a STANDALONE shared repo (jax-free orchestration + HPO)
# used by multiple training projects (streaming-rtrrl, memorax-rtrl, ...). The
# orchestration layer never imports jax, so the projects' incompatible JAX
# versions never meet here — only the per-project training IMAGE carries jax.
#
# Layering:
#   1. This file sets the COMMON AWS resources (shared across all projects:
#      ECR repo, Batch queues/roles, S3, IAM, subnets/SG, Aim server, W&B secret).
#   2. It then sources the calling PROJECT's `project.env` (found via PROJECT_DIR,
#      default: $PWD) which sets the per-project bits: PROJECT_NAME, IMAGE_TAG,
#      GPU_IMAGE_TAG, DEFAULT_ENTRY, WANDB_PROJECT.
#   3. Finally it derives IMAGE/GPU_IMAGE from the (project-overridden) tags.
#
# So scripts do:  source "$(dirname "$0")/env.sh"   (run from a project dir, or
# with PROJECT_DIR=/path/to/project set / passed via --project).

# ---- Core -------------------------------------------------------------------
export REGION="eu-north-1"
export ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null)}"

# ---- Storage / image (shared repo; projects differ only by tag) -------------
export S3_BUCKET="rtrrl-artifacts-007122174918"
export S3_PREFIX="runs"                            # keys: s3://$S3_BUCKET/$S3_PREFIX/<run>/
export ECR_REPO="rtrrl"                            # one ECR repo; per-project image = tag
export ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"

# ---- Batch (EC2-backed compute environment) ---------------------------------
export COMPUTE_ENV="rtrrl-cpu-ce"
export JOB_QUEUE="rtrrl-cpu-queue"
export JOB_DEF="rtrrl-cpu-job"
export INSTANCE_TYPE="c7a.xlarge"                  # 4 vCPU / 8 GiB
export MAX_VCPUS="16"                              # cap on concurrent compute
export PROVISIONING="EC2"                          # EC2 or SPOT
export ECS_INSTANCE_ROLE="rtrrl-ecs-instance-role" # instance profile name
export BATCH_JOB_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/rtrrl-batch-job-role"
# Execution role lets the job definition inject secrets (WANDB_API_KEY) into the
# container from Secrets Manager. Created by infra/iam/setup-iam.sh.
export BATCH_EXEC_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/rtrrl-batch-execution-role"

# Networking. The jump host is in the DEFAULT VPC (vpc-0a403420fd30ecb83),
# a PUBLIC subnet (172.31.0.0/16). Running Batch in the same VPC's public
# subnet(s) needs no NAT/VPC endpoints (instances get public IPs -> ECR/S3),
# and the Aim server is reachable over the jump host private IP below.
export SUBNET_IDS="subnet-08127d1c5d4de6ac2,subnet-0b8c68ea0a9784758,subnet-01a2aa195678f8411"  # public subnets 1a/1b/1c (auto public IP, c7a available)
export SECURITY_GROUP_ID="sg-0c0ed6b927c5113dc"    # rtrrl-sg: egress all, reaches jump host Aim 53800

# ---- Job resources ----------------------------------------------------------
export JOB_VCPUS="4"
export JOB_MEMORY_MB="7168"                         # leave headroom under 8 GiB

# ---- GPU Batch (optional; created by create-batch.sh --gpu) ------------------
# Brax/JAX are GPU-first; a single mid GPU (A10G/L4) beats many CPU cores. These
# defaults target g5.2xlarge (8 vCPU / 32 GiB / 1x A10G). The GPU image (tag from
# project.env) is built from the project's infra/docker/Dockerfile.gpu.
export GPU_COMPUTE_ENV="rtrrl-gpu-ce"
export GPU_JOB_QUEUE="rtrrl-gpu-queue"
export GPU_JOB_DEF="rtrrl-gpu-job"
export GPU_INSTANCE_TYPE="g5.2xlarge"               # A10G; g6.2xlarge = L4
export GPU_MAX_VCPUS="8"                            # one whole instance at a time
export GPU_JOB_VCPUS="8"
export GPU_JOB_MEMORY_MB="28000"                    # headroom under 32 GiB
export GPU_PER_JOB="1"                              # GPUs per job

# ---- Aim remote tracking server (on the jump host; shared by all projects) --
# Batch containers send live metrics here. Jump host private IP (default VPC).
export AIM_SERVER="aim://172.31.62.192:53800"

# ---- Logging + Weights & Biases ---------------------------------------------
# Default logging backend(s) for submitted jobs: "aim", "wandb", or "aim+wandb".
export LOGGING="${LOGGING:-aim}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
# FULL ARN of the Secrets Manager secret holding the W&B API key (shared).
export WANDB_SECRET_ARN="${WANDB_SECRET_ARN:-arn:aws:secretsmanager:eu-north-1:007122174918:secret:rtrrl/wandb-api-key-ewMYy3}"

# ---- Per-project overrides --------------------------------------------------
# Locate the calling project and source its project.env. PROJECT_DIR may be set
# by the caller (or --project via the scripts); otherwise default to $PWD.
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
if [ -f "${PROJECT_DIR}/project.env" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/project.env"
else
  echo "WARNING: no project.env under PROJECT_DIR=${PROJECT_DIR}." >&2
  echo "         Set PROJECT_DIR=/path/to/project (or run from a project dir)." >&2
fi

# Fallbacks if project.env is missing / incomplete.
export PROJECT_NAME="${PROJECT_NAME:-unknown}"
export IMAGE_TAG="${IMAGE_TAG:-cpu}"
export GPU_IMAGE_TAG="${GPU_IMAGE_TAG:-gpu}"
export DEFAULT_ENTRY="${DEFAULT_ENTRY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-${PROJECT_NAME}}"

# ---- Derived (after project overrides) --------------------------------------
export IMAGE="${ECR_URI}:${IMAGE_TAG}"
export GPU_IMAGE="${ECR_URI}:${GPU_IMAGE_TAG}"
