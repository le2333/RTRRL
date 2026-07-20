#!/usr/bin/env bash
# Build an isolated heavy-test overlay from the current memo and training-sdk trees.
# Usage: build-image.sh --profile c7am|c7ax|g6x
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

stage_context() {
  local source_root="$1"
  local destination="$2"
  local dockerfile="$3"
  local excluded
  local -a tar_excludes=(--exclude='*.log')

  for excluded in \
    .git .venv __pycache__ .cache .pytest_cache .ruff_cache cache log logs
  do
    tar_excludes+=(--exclude="${excluded}" --exclude="*/${excluded}")
  done

  mkdir -p "${destination}/memo" "${destination}/training-sdk"
  tar -C "${source_root}/memo/" "${tar_excludes[@]}" -cf - . \
    | tar -C "${destination}/memo/" -xf -
  tar -C "${source_root}/training-sdk/" "${tar_excludes[@]}" -cf - . \
    | tar -C "${destination}/training-sdk/" -xf -
  cp "${dockerfile}" "${destination}/Dockerfile"
}

resolve_ecr_digest() {
  local tag="$1"
  local attempt response digest
  local max_attempts="${ECR_MAX_ATTEMPTS:-5}"
  local retry_delay="${ECR_RETRY_DELAY_SECONDS:-1}"

  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if response="$(
      aws ecr batch-get-image \
        --repository-name "${ECR_REPO}" \
        --image-ids "imageTag=${tag}" \
        --region "${REGION}" \
        --output json 2>&1
    )" && digest="$(
      printf '%s' "${response}" | /usr/bin/env python3 -c '
import json
import re
import sys

try:
    payload = json.load(sys.stdin)
    failures = payload.get("failures")
    images = payload.get("images")
    digest = images[0]["imageId"]["imageDigest"]
except (AttributeError, IndexError, KeyError, TypeError, json.JSONDecodeError):
    raise SystemExit(1)
if failures != [] or not isinstance(images, list) or not images:
    raise SystemExit(1)
if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
    raise SystemExit(1)
print(digest)
'
    )"; then
      printf '%s\n' "${digest}"
      return 0
    fi

    if [ "${attempt}" -lt "${max_attempts}" ]; then
      echo "ECR digest lookup for tag ${tag} failed (attempt ${attempt}/${max_attempts}); retrying" >&2
      sleep "${retry_delay}"
    fi
  done

  echo "ERROR: ECR BatchGetImage failed for tag ${tag} after ${max_attempts} attempts" >&2
  echo "ECR response: ${response:-<no response>}" >&2
  return 1
}

main() {
  local profile=""
  local base_tag base_digest base_image build_context
  local git_revision unique_suffix tag tagged_image pushed_digest digest_image
  local -a docker_command

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --profile)
        [ "$#" -ge 2 ] || { echo "ERROR: --profile requires a value" >&2; return 2; }
        profile="$2"
        shift 2
        ;;
      *)
        echo "ERROR: unknown option: $1" >&2
        return 2
        ;;
    esac
  done

  case "${profile}" in
    c7am|c7ax) base_tag="memorax-rtrl-cpu" ;;
    g6x)       base_tag="memorax-rtrl-gpu" ;;
    *)
      echo "ERROR: --profile must be one of c7am, c7ax, or g6x" >&2
      return 2
      ;;
  esac

  export PROJECT_DIR="${REPOSITORY_ROOT}/memo"
  # shellcheck disable=SC1091
  source "${REPOSITORY_ROOT}/infra/env.sh"

  base_digest="$(resolve_ecr_digest "${base_tag}")"
  base_image="${ECR_URI}@${base_digest}"

  build_context="$(mktemp -d)"
  trap "rm -rf -- $(printf '%q' "${build_context}")" EXIT
  stage_context "${REPOSITORY_ROOT}" "${build_context}" "${SCRIPT_DIR}/Dockerfile"

  git_revision="$(git -C "${REPOSITORY_ROOT}" rev-parse --short=12 HEAD)"
  unique_suffix="$(
    /usr/bin/env python3 -c \
      'import secrets, time; print(f"{time.time_ns()}-{secrets.token_hex(4)}")'
  )"
  tag="heavy-test-${profile}-${git_revision}-${unique_suffix}"
  tagged_image="${ECR_URI}:${tag}"

  docker_command=(docker)
  if ! docker info >/dev/null 2>&1; then
    if sudo -n docker info >/dev/null 2>&1; then
      docker_command=(sudo docker)
    else
      echo "ERROR: Docker daemon is unavailable" >&2
      return 1
    fi
  fi

  aws ecr get-login-password --region "${REGION}" \
    | "${docker_command[@]}" login \
        --username AWS \
        --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com" >&2

  "${docker_command[@]}" build \
    --platform linux/amd64 \
    --build-arg "BASE_IMAGE=${base_image}" \
    -t "${tagged_image}" \
    -f "${build_context}/Dockerfile" \
    "${build_context}" >&2
  "${docker_command[@]}" push "${tagged_image}" >&2

  pushed_digest="$(resolve_ecr_digest "${tag}")"

  if [ "${profile}" != "g6x" ]; then
    "${docker_command[@]}" run --rm \
      --entrypoint /opt/venv/bin/python \
      "${tagged_image}" \
      -m pytest --version >&2
  fi

  digest_image="${ECR_URI}@${pushed_digest}"
  printf '{"tag":"%s","digest":"%s","image":"%s"}\n' \
    "${tag}" "${pushed_digest}" "${digest_image}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
