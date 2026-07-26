#!/usr/bin/env bash
# Build a project's training image and push it to ECR.
# Run from the jump host (its `controller` role can push to ECR) or anywhere
# with Docker + credentials.
#
#   ../infra/build-and-push.sh [--project PATH] [--gpu]
#
# The Dockerfile and build context come from the PROJECT (project.env selects
# the image tag): <project>/infra/docker/Dockerfile[.gpu], context = <project>.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
GPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    --project) PROJECT_DIR="$(cd "$2" && pwd)"; shift 2 ;;
    --gpu)     GPU=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
export PROJECT_DIR
source "${HERE}/env.sh"

if [ "${GPU}" -eq 1 ]; then
  DOCKERFILE="${PROJECT_DIR}/infra/docker/Dockerfile.gpu"
  TARGET_IMAGE="${GPU_IMAGE}"
else
  DOCKERFILE="${PROJECT_DIR}/infra/docker/Dockerfile"
  TARGET_IMAGE="${IMAGE}"
fi
[ -f "${DOCKERFILE}" ] || { echo "ERROR: Dockerfile not found: ${DOCKERFILE}" >&2; exit 1; }

echo "Project    : ${PROJECT_NAME} (${PROJECT_DIR})"
echo "Dockerfile : ${DOCKERFILE}"
echo "Image      : ${TARGET_IMAGE}"

# Stage the shared in-container helper into the project build context so it bakes
# into /app (the shared infra dir itself is NOT part of the build context). This
# keeps run_many.py single-sourced here; the staged copy is gitignored.
cp "${HERE}/run_many.py" "${PROJECT_DIR}/run_many.py"

# Descriptor-aware projects bind their validated catalog into the image label.
# Projects without an index (for example memo) keep the existing build behavior.
BUILD_ARGS=()
CATALOG_INDEX="${PROJECT_DIR}/infra/scripts/index.yaml"
if [ -f "${CATALOG_INDEX}" ]; then
  TRAINER_SCRIPT_CATALOG="$(
    uv run --project "${PROJECT_DIR}/infra/control-plane" \
      trainer-image-catalog "${CATALOG_INDEX}"
  )"
  BUILD_ARGS+=(--build-arg "TRAINER_SCRIPT_CATALOG=${TRAINER_SCRIPT_CATALOG}")
fi

# Create the ECR repo if it does not exist (needs ecr:CreateRepository; if the
# jump host role lacks it, create the repo once with admin creds instead).
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${ECR_REPO}" --region "${REGION}" >/dev/null

# Log Docker in to ECR.
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Build (linux/amd64 to match Batch EC2 instances) and push.
docker build \
  --platform linux/amd64 \
  -f "${DOCKERFILE}" \
  -t "${TARGET_IMAGE}" \
  "${BUILD_ARGS[@]}" \
  "${PROJECT_DIR}"

docker push "${TARGET_IMAGE}"

echo "Pushed ${TARGET_IMAGE}"
