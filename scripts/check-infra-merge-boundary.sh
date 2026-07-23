#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BASE="${1:-$(git merge-base main HEAD)}"
PROTECTED=(
  memo
  .github/workflows/build-memo-image.yml
  .github/workflows/memo-ci.yml
)

raw="$(git diff --raw "$BASE" -- "${PROTECTED[@]}")"
names="$(git diff --name-status "$BASE" -- "${PROTECTED[@]}")"
if [[ -n "$raw" || -n "$names" ]]; then
  printf '%s\n' "protected tree differs from $BASE" >&2
  [[ -z "$raw" ]] || printf '%s\n' "$raw" >&2
  [[ -z "$names" ]] || printf '%s\n' "$names" >&2
  exit 1
fi
printf '%s\n' "protected tree matches $BASE by path, blob, and mode"
