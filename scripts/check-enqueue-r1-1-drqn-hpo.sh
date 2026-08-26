#!/usr/bin/env bash
set -euo pipefail

root=${1:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
script="$root/scripts/enqueue-r1-1-drqn-hpo.sh"

test -f "$script"
bash -n "$script"
test "$(grep -Ec '^  r1-1-.*-drqn-(lstm|rtu)-hpo.yml$' "$script")" -eq 10
! grep -Fq 'discounting-chain' "$script"
! grep -Fq 'systemd-run' "$script"
grep -Fq 'runs/study-scheduler.sqlite' "$script"
grep -Fq 'runs/r1-1-drqn-hpo-2m-auc-system' "$script"
