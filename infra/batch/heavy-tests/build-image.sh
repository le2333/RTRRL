#!/usr/bin/env bash
# Build an isolated heavy-test overlay from the current memo and training-sdk trees.
# Usage: build-image.sh --profile c7am|c7ax|g6x [--gpu-rebase-from IMAGE@DIGEST]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CRANE_VERSION="0.21.7"
CRANE_ARCHIVE_SHA256="1a57bc98207fa1c0d04bf760699099e26f8383499bfd55b99c1b919a928a7230"
CRANE_WORK_DIR=""
CRANE_BIN_RESOLVED=""

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

validate_digest_reference() {
  /usr/bin/env python3 - "$1" <<'PY'
import re
import sys

component = r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
label = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
registry = rf"(?:localhost|{label}(?:\.{label})*)(?::[0-9]+)?"
pattern = re.compile(
    rf"(?:{registry}/)?{component}(?:/{component})*@sha256:[0-9a-f]{{64}}"
)
if pattern.fullmatch(sys.argv[1]) is None:
    raise SystemExit("image must be an immutable registry/repository@sha256 digest")
PY
}

ensure_crane() {
  if [ -n "${CRANE_BIN:-}" ]; then
    [ -n "${CRANE_BIN_SHA256:-}" ] || {
      echo "ERROR: CRANE_BIN_SHA256 is required when CRANE_BIN is set" >&2
      return 1
    }
    printf '%s\n' "${CRANE_BIN_SHA256}" | /usr/bin/env python3 -c '
import re
import sys
if re.fullmatch(r"[0-9a-f]{64}", sys.stdin.read().strip()) is None:
    raise SystemExit("CRANE_BIN_SHA256 must be 64 lowercase hexadecimal characters")
'
    local override_path
    override_path="$(command -v -- "${CRANE_BIN}")" || {
      echo "ERROR: CRANE_BIN is not an executable command: ${CRANE_BIN}" >&2
      return 1
    }
    CRANE_BIN_RESOLVED="$(readlink -f -- "${override_path}")"
    [ -f "${CRANE_BIN_RESOLVED}" ] && [ -x "${CRANE_BIN_RESOLVED}" ] || {
      echo "ERROR: resolved CRANE_BIN is not an executable file" >&2
      return 1
    }
    local actual_checksum
    actual_checksum="$(sha256sum "${CRANE_BIN_RESOLVED}")"
    actual_checksum="${actual_checksum%% *}"
    [ "${actual_checksum}" = "${CRANE_BIN_SHA256}" ] || {
      echo "ERROR: CRANE_BIN checksum mismatch" >&2
      return 1
    }
  else
    CRANE_WORK_DIR="$(mktemp -d)"
    local archive="${CRANE_WORK_DIR}/go-containerregistry.tar.gz"
    curl -fsSL \
      "https://github.com/google/go-containerregistry/releases/download/v${CRANE_VERSION}/go-containerregistry_Linux_x86_64.tar.gz" \
      -o "${archive}"
    printf '%s  %s\n' "${CRANE_ARCHIVE_SHA256}" "${archive}" | sha256sum --check --status
    tar -xzf "${archive}" -C "${CRANE_WORK_DIR}" crane
    CRANE_BIN_RESOLVED="${CRANE_WORK_DIR}/crane"
  fi
  local actual_version
  actual_version="$("${CRANE_BIN_RESOLVED}" version)"
  [ "${actual_version}" = "${CRANE_VERSION}" ] || {
    echo "ERROR: crane version ${actual_version} does not match ${CRANE_VERSION}" >&2
    return 1
  }
}

validate_gpu_config() {
  /usr/bin/env python3 -c '
import json
import sys

payload = json.load(sys.stdin)
config = payload.get("config", payload)
environment = {}
for item in config.get("Env", []):
    key, separator, value = item.partition("=")
    if separator:
        environment[key] = value
if environment.get("JAX_PLATFORM_NAME") != "gpu":
    raise SystemExit("rebased GPU config must set JAX_PLATFORM_NAME=gpu")
if environment.get("XLA_FLAGS") != "":
    raise SystemExit("rebased GPU config must clear XLA_FLAGS")
'
}

rebase_gpu_image() {
  local overlay_source="$1"
  local tag="$2"
  local old_cpu_digest new_gpu_digest old_cpu_base new_gpu_base
  local overlay_manifest_digest overlay_manifest overlay_config_digest
  local intermediate_tag tagged_image final_config pushed_digest digest_image

  validate_digest_reference "${overlay_source}"
  ensure_crane
  old_cpu_digest="$(resolve_ecr_digest "memorax-rtrl-cpu")"
  new_gpu_digest="$(resolve_ecr_digest "memorax-rtrl-gpu")"
  old_cpu_base="${ECR_URI}@${old_cpu_digest}"
  new_gpu_base="${ECR_URI}@${new_gpu_digest}"
  intermediate_tag="${ECR_URI}:${tag}-base"
  tagged_image="${ECR_URI}:${tag}"

  aws ecr get-login-password --region "${REGION}" \
    | "${CRANE_BIN_RESOLVED}" auth login \
        --username AWS \
        --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com" >&2

  overlay_manifest_digest="$(
    "${CRANE_BIN_RESOLVED}" digest --platform linux/amd64 "${overlay_source}"
  )"
  overlay_manifest="$(
    "${CRANE_BIN_RESOLVED}" manifest --platform linux/amd64 "${overlay_source}"
  )"
  overlay_config_digest="$(
    printf '%s' "${overlay_manifest}" | /usr/bin/env python3 -c \
      'import json,sys; print(json.load(sys.stdin)["config"]["digest"])'
  )"

  "${CRANE_BIN_RESOLVED}" rebase \
    --platform linux/amd64 \
    --old_base "${old_cpu_base}" \
    --new_base "${new_gpu_base}" \
    --tag "${intermediate_tag}" \
    "${overlay_source}" >&2
  "${CRANE_BIN_RESOLVED}" mutate \
    --platform linux/amd64 \
    --env JAX_PLATFORM_NAME=gpu \
    --env XLA_FLAGS= \
    --tag "${tagged_image}" \
    "${intermediate_tag}" >&2

  final_config="$("${CRANE_BIN_RESOLVED}" config --platform linux/amd64 "${tagged_image}")"
  printf '%s' "${final_config}" | validate_gpu_config
  pushed_digest="$("${CRANE_BIN_RESOLVED}" digest --platform linux/amd64 "${tagged_image}")"
  printf '%s\n' "${pushed_digest}" | /usr/bin/env python3 -c '
import re
import sys
digest = sys.stdin.read().strip()
if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
    raise SystemExit("crane returned an invalid digest")
'
  digest_image="${ECR_URI}@${pushed_digest}"

  /usr/bin/env python3 - \
    "${tag}" "${pushed_digest}" "${digest_image}" \
    "${CRANE_VERSION}" "${overlay_source}" "${overlay_manifest_digest}" \
    "${overlay_config_digest}" "${old_cpu_base}" "${new_gpu_base}" <<'PY'
import json
import sys

(
    tag,
    digest,
    image,
    crane_version,
    overlay_source,
    overlay_manifest_digest,
    overlay_config_digest,
    old_cpu_base,
    new_gpu_base,
) = sys.argv[1:]
print(json.dumps({
    "mode": "registry-rebase",
    "tag": tag,
    "digest": digest,
    "image": image,
    "crane_version": crane_version,
    "overlay_source_image": overlay_source,
    "overlay_manifest_digest": overlay_manifest_digest,
    "overlay_config_digest": overlay_config_digest,
    "old_cpu_base": old_cpu_base,
    "new_gpu_base": new_gpu_base,
    "config_environment": {
        "JAX_PLATFORM_NAME": "gpu",
        "XLA_FLAGS": "",
    },
}, separators=(",", ":"), sort_keys=True))
PY
}

main() {
  local profile=""
  local gpu_rebase_from=""
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
      --gpu-rebase-from)
        [ "$#" -ge 2 ] || { echo "ERROR: --gpu-rebase-from requires a value" >&2; return 2; }
        gpu_rebase_from="$2"
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

  git_revision="$(git -C "${REPOSITORY_ROOT}" rev-parse --short=12 HEAD)"
  unique_suffix="$(
    /usr/bin/env python3 -c \
      'import secrets, time; print(f"{time.time_ns()}-{secrets.token_hex(4)}")'
  )"
  tag="heavy-test-${profile}-${git_revision}-${unique_suffix}"
  trap 'if [ -n "${build_context:-}" ]; then rm -rf -- "${build_context}"; fi; if [ -n "${CRANE_WORK_DIR:-}" ]; then rm -rf -- "${CRANE_WORK_DIR}"; fi' EXIT

  if [ -n "${gpu_rebase_from}" ]; then
    [ "${profile}" = "g6x" ] || {
      echo "ERROR: --gpu-rebase-from is only valid with --profile g6x" >&2
      return 2
    }
    rebase_gpu_image "${gpu_rebase_from}" "${tag}"
    return
  fi

  base_digest="$(resolve_ecr_digest "${base_tag}")"
  base_image="${ECR_URI}@${base_digest}"

  build_context="$(mktemp -d)"
  stage_context "${REPOSITORY_ROOT}" "${build_context}" "${SCRIPT_DIR}/Dockerfile"

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
