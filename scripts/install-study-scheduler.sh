#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=${1:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
readonly USER_SYSTEMD_ROOT=${XDG_CONFIG_HOME:-"$HOME/.config"}/systemd/user
readonly UNIT=r1-study-scheduler.service

mkdir -p "$USER_SYSTEMD_ROOT"
install -m 0644 "$ROOT/infra/systemd/$UNIT" "$USER_SYSTEMD_ROOT/$UNIT"
systemctl --user daemon-reload
systemctl --user enable --now "$UNIT"
