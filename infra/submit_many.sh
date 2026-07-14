#!/usr/bin/env bash
# Submit one AWS Batch job that runs multiple configs sequentially in one container.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Pre-scan for --project so env.sh sources the right project.env (default $PWD).
for ((i=1; i<=$#; i++)); do
  if [ "${!i}" = "--project" ]; then j=$((i+1)); export PROJECT_DIR="$(cd "${!j}" && pwd)"; fi
done
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
source "${HERE}/env.sh"

ENTRY="${DEFAULT_ENTRY:-rtrrl.py}"
NAME=""
MODE="${LOGGING}"
QUEUE="${JOB_QUEUE}"
JOBDEF="${JOB_DEF}"
FAIL_FAST=0
CONFIGS=()
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --project)  shift 2 ;;
    --config)   CONFIGS+=("$2"); shift 2 ;;
    --entry)    ENTRY="$2"; shift 2 ;;
    --name)     NAME="$2"; shift 2 ;;
    --logging)  MODE="$2"; shift 2 ;;
    --queue)    QUEUE="$2"; shift 2 ;;
    --job-def)  JOBDEF="$2"; shift 2 ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

[ "${#CONFIGS[@]}" -gt 0 ] || { echo "ERROR: at least one --config is required" >&2; exit 1; }
for cfg in "${CONFIGS[@]}"; do
  [ -f "${cfg}" ] || { echo "ERROR: config not found: ${cfg}" >&2; exit 1; }
done

if [ -z "${NAME}" ]; then
  first="$(basename "${CONFIGS[0]}" .yml)"
  last="$(basename "${CONFIGS[$((${#CONFIGS[@]} - 1))]}" .yml)"
  NAME="$(printf 'many_%s_to_%s' "${first}" "${last}" | tr -c 'A-Za-z0-9_.-' '_')"
fi

CONFIGS_B64=$(
  python3 - "${CONFIGS[@]}" <<'PY'
import base64
import json
import pathlib
import sys

items = []
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    items.append(
        {
            "path": str(path),
            "name": path.name,
            "config_b64": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
    )
payload = json.dumps(items, separators=(",", ":")).encode("utf-8")
print(base64.b64encode(payload).decode("ascii"))
PY
)

# run_many.py is staged into the image at /app/run_many.py by build-and-push.sh.
CMD=(python run_many.py --entry "${ENTRY}" --logging "${MODE}")
case "${MODE}" in
  *aim*) CMD+=(--log_repo "${AIM_SERVER}") ;;
esac
[ "${FAIL_FAST}" = "1" ] && CMD+=(--fail-fast)
if [ "${#EXTRA[@]}" -gt 0 ]; then
  CMD+=(-- "${EXTRA[@]}")
fi

OVERRIDES=$(
  python3 - "${CONFIGS_B64}" "${CMD[@]}" <<'PY'
import json
import sys

configs_b64 = sys.argv[1]
cmd = sys.argv[2:]
print(
    json.dumps(
        {
            "command": cmd,
            "environment": [
                {"name": "RUN_MANY_CONFIGS_B64", "value": configs_b64},
            ],
        }
    )
)
PY
)

echo "Run name : ${NAME}"
echo "Configs  : ${#CONFIGS[@]}"
printf '  %s\n' "${CONFIGS[@]}"
echo "Queue    : ${QUEUE}  (job def ${JOBDEF})"
echo "Logging  : ${MODE}"
echo "Command  : ${CMD[*]}"

aws batch submit-job \
  --region "${REGION}" \
  --job-name "${NAME}" \
  --job-queue "${QUEUE}" \
  --job-definition "${JOBDEF}" \
  --container-overrides "${OVERRIDES}" \
  --query "{name:jobName, id:jobId}" --output table
