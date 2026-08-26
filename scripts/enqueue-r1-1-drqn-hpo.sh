#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=${ROOT:-/home/ubuntu/trainer/streaming-rtrrl-names-deploy}
readonly PYTHON=/home/ubuntu/trainer/streaming-rtrrl/rtrrl/infra/control-plane/.venv/bin/python
readonly STATE="$ROOT/runs/study-scheduler.sqlite"
readonly DATABASES="$ROOT/runs/r1-1-drqn-hpo-2m-auc-system"
readonly EXCLUDE=${1:-}

readonly -a CONFIGS=(
  r1-1-minesweeper-drqn-lstm-hpo.yml
  r1-1-minesweeper-drqn-rtu-hpo.yml
  r1-1-noisy-stateless-cartpole-drqn-lstm-hpo.yml
  r1-1-noisy-stateless-cartpole-drqn-rtu-hpo.yml
  r1-1-repeat-first-drqn-lstm-hpo.yml
  r1-1-repeat-first-drqn-rtu-hpo.yml
  r1-1-stateless-cartpole-drqn-lstm-hpo.yml
  r1-1-stateless-cartpole-drqn-rtu-hpo.yml
  r1-1-umbrella-chain-drqn-lstm-hpo.yml
  r1-1-umbrella-chain-drqn-rtu-hpo.yml
)

mkdir -p "$DATABASES"
for config in "${CONFIGS[@]}"; do
  if [[ "$config" == "$EXCLUDE" ]]; then
    continue
  fi
  stem=${config%.yml}
  PYTHONPATH="$ROOT/infra/src" "$PYTHON" -m trainer_infra.scheduler_cli \
    --state "$STATE" add "$ROOT/config/$config" \
    --catalog "$ROOT/memo/catalog.json" \
    --database "$DATABASES/$stem.sqlite"
done
