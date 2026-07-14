#!/usr/bin/env bash
# Back up the shared "main" Aim repo to S3 (durable copy of all experiment data).
#
# The monorepo root holds the live Aim repo at ./.aim, served by the jump-host
# Aim server (see infra/README). This script mirrors it to
#   s3://$S3_BUCKET/aim/
# with `aws s3 sync` (incremental: only changed/new objects are uploaded).
#
# Design notes:
#   * NO --delete. S3 is the safety net; we never remove a backed-up run just
#     because it vanished locally. Prune S3 manually if you ever need to.
#   * The "test"/debug repo (.aim-scratch, AIM_SCRATCH_REPO) and regenerable HPO
#     studies (optuna.db) are intentionally NOT backed up.
#   * Syncing a live repo is fine: Aim data is append-mostly and sync is a
#     best-effort snapshot. Run it on a schedule (cron) or after a batch of runs.
#
# Usage (from anywhere):
#   infra/backup-aim.sh                 # sync ./.aim (repo root) -> s3
#   infra/backup-aim.sh --repo /path/.aim
#   DRY_RUN=1 infra/backup-aim.sh       # show what would upload, change nothing
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"   # monorepo root (parent of infra/)

# Pull S3_BUCKET / REGION from the shared env (core section, before project.env).
# PROJECT_DIR is irrelevant here; point it at ROOT to silence the missing-project
# warning without affecting anything we use.
export PROJECT_DIR="${PROJECT_DIR:-$ROOT}"
# shellcheck disable=SC1091
source "${HERE}/env.sh" >/dev/null 2>&1 || true
: "${S3_BUCKET:?S3_BUCKET not set (env.sh failed to load)}"
: "${REGION:=eu-north-1}"

AIM_DIR="${ROOT}/.aim"
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) AIM_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ ! -d "${AIM_DIR}" ]; then
  echo "ERROR: Aim repo not found at ${AIM_DIR}" >&2
  exit 1
fi

DEST="s3://${S3_BUCKET}/aim/"
EXTRA=()
[ -n "${DRY_RUN:-}" ] && EXTRA+=(--dryrun)

echo "[backup-aim] ${AIM_DIR}  ->  ${DEST}  (region ${REGION}${DRY_RUN:+, DRY RUN})"
aws s3 sync "${AIM_DIR}" "${DEST}" --region "${REGION}" --no-progress "${EXTRA[@]}"
echo "[backup-aim] done."
