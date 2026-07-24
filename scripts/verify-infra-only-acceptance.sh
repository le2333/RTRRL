#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
export UV_OFFLINE=1

run() {
  local seconds="$1"
  shift
  /usr/bin/time -v /usr/bin/timeout \
    --signal=TERM --kill-after=30s "${seconds}s" "$@"
}

run 120 scripts/check-infra-merge-boundary.sh

run 300 env -u PYTHONPATH -u BRAX_ACCEPTANCE_TEST_MODE \
  -u BRAX_ACCEPTANCE_E2E_FAST -u CUDA_VISIBLE_DEVICES \
  uv lock --offline --project training-sdk --check
run 600 env -u PYTHONPATH -u BRAX_ACCEPTANCE_TEST_MODE \
  -u BRAX_ACCEPTANCE_E2E_FAST -u CUDA_VISIBLE_DEVICES \
  uv run --offline --directory training-sdk pytest -q
run 300 env -u PYTHONPATH -u BRAX_ACCEPTANCE_TEST_MODE \
  -u BRAX_ACCEPTANCE_E2E_FAST -u CUDA_VISIBLE_DEVICES \
  uv run --offline --directory training-sdk ruff check src tests

run 300 env -u PYTHONPATH -u BRAX_ACCEPTANCE_TEST_MODE \
  -u BRAX_ACCEPTANCE_E2E_FAST -u CUDA_VISIBLE_DEVICES \
  uv lock --offline --project rtrrl/infra/mock-trainer --check
run 1200 env -u PYTHONPATH -u BRAX_ACCEPTANCE_TEST_MODE \
  -u BRAX_ACCEPTANCE_E2E_FAST -u CUDA_VISIBLE_DEVICES \
  JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 \
  uv run --offline --directory rtrrl/infra/mock-trainer \
  --with-editable ../control-plane pytest -q
run 300 env -u PYTHONPATH -u BRAX_ACCEPTANCE_TEST_MODE \
  -u BRAX_ACCEPTANCE_E2E_FAST -u CUDA_VISIBLE_DEVICES \
  uv run --offline --directory rtrrl/infra/mock-trainer ruff check src tests

run 300 env -u PYTHONPATH -u BRAX_ACCEPTANCE_TEST_MODE \
  -u BRAX_ACCEPTANCE_E2E_FAST -u CUDA_VISIBLE_DEVICES \
  uv lock --offline --project rtrrl/infra/control-plane --check
run 7200 env -u PYTHONPATH -u BRAX_ACCEPTANCE_TEST_MODE \
  -u BRAX_ACCEPTANCE_E2E_FAST -u CUDA_VISIBLE_DEVICES \
  JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 \
  uv run --offline --directory rtrrl/infra/control-plane pytest -q
run 300 env -u PYTHONPATH -u BRAX_ACCEPTANCE_TEST_MODE \
  -u BRAX_ACCEPTANCE_E2E_FAST -u CUDA_VISIBLE_DEVICES \
  uv run --offline --directory rtrrl/infra/control-plane ruff check src tests scripts

if run 120 rg -n \
  '^[[:space:]]*(from[[:space:]]+(memo|trainer_infra)([.]|[[:space:]])|import[[:space:]]+(memo|trainer_infra)([.]|[[:space:],]|$))' \
  rtrrl/infra/mock-trainer/src
then
  printf '%s\n' "forbidden mock-trainer import found" >&2
  exit 1
else
  status=$?
  if (( status != 1 )); then
    printf '%s\n' "mock-trainer import scan failed with status $status" >&2
    exit "$status"
  fi
fi

if run 120 rg -n 'memo_stream_ac|memo_rtrrl|memo/infra' \
  rtrrl/infra/control-plane/examples \
  rtrrl/infra/control-plane/scripts \
  rtrrl/infra/control-plane/src \
  rtrrl/infra/control-plane/tests/test_end_to_end.py \
  rtrrl/infra/control-plane/tests/test_facility_concrete_contract.py
then
  printf '%s\n' "forbidden control-plane reference found" >&2
  exit 1
else
  status=$?
  if (( status != 1 )); then
    printf '%s\n' "control-plane reference scan failed with status $status" >&2
    exit "$status"
  fi
fi

run 120 git diff --check
run 120 scripts/check-infra-merge-boundary.sh
