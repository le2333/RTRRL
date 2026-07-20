#!/usr/bin/env bash
# Build an isolated heavy-test overlay from the current memo and training-sdk trees.
# Usage: build-image.sh --profile c7am|c7ax|g6x
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PROFILE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile)
      [ "$#" -ge 2 ] || { echo "ERROR: --profile requires a value" >&2; exit 2; }
      PROFILE="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      exit 2
      ;;
  esac
done

case "${PROFILE}" in
  c7am|c7ax) BASE_TAG="memorax-rtrl-cpu" ;;
  g6x)       BASE_TAG="memorax-rtrl-gpu" ;;
  *)
    echo "ERROR: --profile must be one of c7am, c7ax, or g6x" >&2
    exit 2
    ;;
esac

export PROJECT_DIR="${REPOSITORY_ROOT}/memo"
# shellcheck disable=SC1091
source "${REPOSITORY_ROOT}/infra/env.sh"

BASE_DIGEST="$(
  aws ecr batch-get-image \
    --repository-name "${ECR_REPO}" \
    --image-ids "imageTag=${BASE_TAG}" \
    --region "${REGION}" \
    --query 'images[0].imageId.imageDigest' \
    --output text
)"
if [[ ! "${BASE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ERROR: could not resolve ${ECR_URI}:${BASE_TAG} to an ECR digest" >&2
  exit 1
fi
BASE_IMAGE="${ECR_URI}@${BASE_DIGEST}"

BUILD_CONTEXT="$(mktemp -d)"
cleanup() {
  rm -rf "${BUILD_CONTEXT}"
}
trap cleanup EXIT

mkdir -p "${BUILD_CONTEXT}/memo" "${BUILD_CONTEXT}/training-sdk"
TAR_EXCLUDES=(
  --exclude='.git'
  --exclude='.venv'
  --exclude='__pycache__'
  --exclude='.cache'
  --exclude='cache'
  --exclude='logs'
  --exclude='*.log'
)
tar -C "${REPOSITORY_ROOT}/memo/" "${TAR_EXCLUDES[@]}" -cf - . \
  | tar -C "${BUILD_CONTEXT}/memo/" -xf -
tar -C "${REPOSITORY_ROOT}/training-sdk/" "${TAR_EXCLUDES[@]}" -cf - . \
  | tar -C "${BUILD_CONTEXT}/training-sdk/" -xf -
cp "${SCRIPT_DIR}/Dockerfile" "${BUILD_CONTEXT}/Dockerfile"

GIT_REVISION="$(git -C "${REPOSITORY_ROOT}" rev-parse --short=12 HEAD)"
UNIQUE_SUFFIX="$(
  /usr/bin/env python3 -c \
    'import secrets, time; print(f"{time.time_ns()}-{secrets.token_hex(4)}")'
)"
TAG="heavy-test-${PROFILE}-${GIT_REVISION}-${UNIQUE_SUFFIX}"
TAGGED_IMAGE="${ECR_URI}:${TAG}"

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  else
    echo "ERROR: Docker daemon is unavailable" >&2
    exit 1
  fi
fi

aws ecr get-login-password --region "${REGION}" \
  | "${DOCKER[@]}" login \
      --username AWS \
      --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com" >/dev/null

# docker build receives only the isolated temporary context.
"${DOCKER[@]}" build \
  --platform linux/amd64 \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${TAGGED_IMAGE}" \
  -f "${BUILD_CONTEXT}/Dockerfile" \
  "${BUILD_CONTEXT}"
"${DOCKER[@]}" push "${TAGGED_IMAGE}"

PUSHED_DIGEST="$(
  aws ecr batch-get-image \
    --repository-name "${ECR_REPO}" \
    --image-ids "imageTag=${TAG}" \
    --region "${REGION}" \
    --query 'images[0].imageId.imageDigest' \
    --output text
)"
if [[ ! "${PUSHED_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "ERROR: could not resolve pushed image ${TAGGED_IMAGE} to a digest" >&2
  exit 1
fi

if [ "${PROFILE}" != "g6x" ]; then
  "${DOCKER[@]}" run --rm \
    --entrypoint /opt/venv/bin/python \
    "${TAGGED_IMAGE}" \
    -m pytest --version
fi

DIGEST_IMAGE="${ECR_URI}@${PUSHED_DIGEST}"
printf '{"tag":"%s","digest":"%s","image":"%s"}\n' \
  "${TAG}" "${PUSHED_DIGEST}" "${DIGEST_IMAGE}"
