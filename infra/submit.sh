#!/usr/bin/env bash
# Submit one training run (or a W&B sweep agent) to AWS Batch.
#
# Run from a project dir (PROJECT_DIR=$PWD) or pass --project PATH. The project's
# project.env selects the image tag / default entry / W&B project.
#
# Normal run (from the project dir):
#   ../infra/submit.sh --config config/rtrrl_brax_hopper_paral1.yml [options] [-- extra]
#
# W&B sweep agent (the sweep selects hyperparameters; see infra/sweep.yaml):
#   ../infra/submit.sh --sweep <entity/project/sweep_id> --config config/ppo_X.yml \
#     --count 1 --name sweep_w1
#
# Options:
#   --project PATH    project root (default: $PWD); selects project.env
#   --config PATH     YAML config to run (required; base config for sweeps too)
#   --entry FILE      entry script (default: DEFAULT_ENTRY from project.env)
#   --name NAME       job/run name (default: derived from the config filename)
#   --logging MODE    aim | wandb | aim+wandb (default: $LOGGING from env.sh)
#   --test            log to the TEST Aim repo (.aim-scratch, UI :43801) instead
#                     of the backed-up main repo; use for smoke/validation runs
#   --sweep ID        run `wandb agent ID` instead of a single training command
#   --count N         trials per agent job in sweep mode (default 1)
#   -- ...            extra args appended verbatim to the training command
#                     (normal mode only; e.g. -- --seed 1)
#
# Each --name gets its own job; submit in a loop for sweeps/seed sweeps. The
# config is base64-injected (CONFIG_B64) and decoded to /tmp/run-config.yml in
# the container. WANDB_API_KEY is injected by the job definition (Secrets
# Manager); see infra/env.sh / infra/iam/setup-iam.sh.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Pre-scan for --project so env.sh sources the right project.env. Default
# PROJECT_DIR to $PWD (run from a project dir). This must happen BEFORE env.sh.
for ((i=1; i<=$#; i++)); do
  if [ "${!i}" = "--project" ]; then j=$((i+1)); export PROJECT_DIR="$(cd "${!j}" && pwd)"; fi
done
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
source "${HERE}/env.sh"

ENTRY="${DEFAULT_ENTRY:-rtrrl.py}"
CONFIG=""
NAME=""
MODE="${LOGGING}"
SWEEP=""
COUNT="1"
QUEUE="${JOB_QUEUE}"
JOBDEF="${JOB_DEF}"
# Which Aim repo to log to (when MODE includes aim): main by default, or the
# test repo (.aim-scratch, UI :43801) with --test so smoke runs stay out of the
# backed-up main repo.
AIM_TARGET="${AIM_SERVER}"
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --project)  shift 2 ;;
    --config)   CONFIG="$2"; shift 2 ;;
    --entry)    ENTRY="$2"; shift 2 ;;
    --name)     NAME="$2"; shift 2 ;;
    --logging)  MODE="$2"; shift 2 ;;
    --test)     AIM_TARGET="${AIM_SERVER_TEST}"; shift ;;
    --sweep)    SWEEP="$2"; shift 2 ;;
    --count)    COUNT="$2"; shift 2 ;;
    --queue)    QUEUE="$2"; shift 2 ;;
    --job-def)  JOBDEF="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

[ -z "${CONFIG}" ] && { echo "ERROR: --config is required" >&2; exit 1; }
[ -f "${CONFIG}" ] || { echo "ERROR: config not found: ${CONFIG}" >&2; exit 1; }

# Run name: default to the config basename (sanitized).
if [ -z "${NAME}" ]; then
  NAME="$(basename "${CONFIG}" .yml | tr -c 'A-Za-z0-9_.-' '_')"
fi

# The config is injected (not baked into the image): base64-encode the YAML; the
# entrypoint decodes it to /tmp/run-config.yml. base64 -w0 is JSON-safe.
CONFIG_B64=$(base64 -w0 "${CONFIG}")

if [ -n "${SWEEP}" ]; then
  # Sweep mode: the agent pulls hyperparameters from the W&B sweep controller and
  # runs the program defined in infra/sweep.yaml (which reads /tmp/run-config.yml).
  CMD=(wandb agent --count "${COUNT}" "${SWEEP}")
else
  # Normal mode: run the training command directly.
  CMD=(python "${ENTRY}" --config_path /tmp/run-config.yml --logging "${MODE}")
  case "${MODE}" in
    *aim*) CMD+=(--log_repo "${AIM_TARGET}") ;;
  esac
  CMD+=("${EXTRA[@]}")
fi

# JSON array for the container command override.
CMD_JSON=$(printf '"%s",' "${CMD[@]}"); CMD_JSON="[${CMD_JSON%,}]"

echo "Run name : ${NAME}"
echo "Config   : ${CONFIG}"
echo "Queue    : ${QUEUE}  (job def ${JOBDEF})"
[ -n "${SWEEP}" ] && echo "Sweep    : ${SWEEP} (count ${COUNT})" || echo "Logging  : ${MODE}$(case "${MODE}" in *aim*) echo " -> ${AIM_TARGET}";; esac)"
echo "Command  : ${CMD[*]}"

aws batch submit-job \
  --region "${REGION}" \
  --job-name "${NAME}" \
  --job-queue "${QUEUE}" \
  --job-definition "${JOBDEF}" \
  --container-overrides "{\"command\": ${CMD_JSON}, \"environment\": [{\"name\": \"CONFIG_B64\", \"value\": \"${CONFIG_B64}\"}]}" \
  --query "{name:jobName, id:jobId}" --output table
