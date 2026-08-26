#!/usr/bin/env bash
set -euo pipefail

root=${1:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
unit="$root/infra/systemd/r1-study-scheduler.service"
installer="$root/scripts/install-study-scheduler.sh"

test -f "$unit"
test -f "$installer"
grep -Fq 'Restart=on-failure' "$unit"
grep -Fq 'RestartSec=5' "$unit"
grep -Fq 'AWS_CONFIG_FILE=/dev/null' "$unit"
grep -Fq 'AWS_SHARED_CREDENTIALS_FILE=/dev/null' "$unit"
grep -Fq 'runs/study-scheduler.sqlite' "$unit"
grep -Fq -- '--max-concurrent 4' "$unit"
bash -n "$installer"
