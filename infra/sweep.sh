#!/usr/bin/env bash
# Backup path: create a W&B sweep and launch parallel agents on AWS Batch.
# Main HPO now lives in control/hpo (Optuna scheduler + Aim history).
#
#   infra/sweep.sh create [sweep.yaml]
#       Create the sweep on W&B and print its id (entity/project/sweep_id).
#       Run on the jump host after `wandb login` (or with WANDB_API_KEY set).
#
#   infra/sweep.sh launch <entity/project/sweep_id> <num_agents> \
#       --config <base.yml> [--count K] [--entry ppo_baseline.py]
#       Submit <num_agents> Batch jobs, each running `wandb agent`. They pull
#       trials from the sweep controller until it is exhausted.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
source "${HERE}/env.sh"

cmd="${1:-}"; shift || true

case "${cmd}" in
  create)
    YAML="${1:-${HERE}/sweep.yaml}"
    ENTITY_ARG=()
    [ -n "${WANDB_ENTITY}" ] && ENTITY_ARG=(--entity "${WANDB_ENTITY}")
    echo "NOTE: W&B sweeps are a backup path. Prefer control/hpo for main HPO."
    echo "Creating sweep from ${YAML} in project ${WANDB_PROJECT}..."
    uv run wandb sweep --project "${WANDB_PROJECT}" "${ENTITY_ARG[@]}" "${YAML}"
    echo
    echo "Copy the sweep id above, then launch agents, e.g.:"
    echo "  infra/sweep.sh launch <entity/project/sweep_id> 4 --config config/ppo_hopper_default_2m.yml"
    ;;
  launch)
    SWEEP="${1:-}"; shift || true
    N="${1:-}"; shift || true
    [ -z "${SWEEP}" ] && { echo "ERROR: need <entity/project/sweep_id>" >&2; exit 1; }
    [ -z "${N}" ] && { echo "ERROR: need <num_agents>" >&2; exit 1; }
    echo "NOTE: W&B sweeps are a backup path. Prefer control/hpo for main HPO."
    SHORT="${SWEEP##*/}"
    for i in $(seq 1 "${N}"); do
      echo "== agent ${i}/${N} =="
      "${HERE}/submit.sh" --sweep "${SWEEP}" --name "sweep_${SHORT}_a${i}" "$@"
    done
    ;;
  *)
    echo "usage: infra/sweep.sh {create [sweep.yaml] | launch <sweep_id> <num_agents> --config <base.yml> [--count K]}" >&2
    exit 1
    ;;
esac
