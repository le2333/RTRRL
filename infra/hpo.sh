#!/usr/bin/env bash
# Thin wrapper around the shared HPO scheduler (infra/hpo/src/hpo_control/scheduler.py).
#
# Saves the `cd infra/hpo && uv run python src/hpo_control/scheduler.py` dance and
# ensures deps are installed once. All args are forwarded verbatim, so every
# subcommand/flag the scheduler supports works here.
#
# The scheduler resolves the project root from --project-root, $HPO_PROJECT_ROOT,
# or (fallback) the --spec/--plan path (which lives under <project>/hpo/). Studies
# and snapshots stay under <project>/hpo/ and are regenerable, so they are NOT
# git-tracked or backed up to S3.
#
# Examples (from anywhere):
#   infra/hpo.sh suggest --spec streaming-rtrrl/hpo/specs/ppo_hc038.yaml -n 4
#   infra/hpo.sh --project-root streaming-rtrrl sync-aim --spec .../ppo_hc038.yaml
#   infra/hpo.sh submit --plan streaming-rtrrl/hpo/runs/<study>/round_003/plan.json --yes
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HPO_DIR="${HERE}/hpo"
cd "${HPO_DIR}"

# First use: create the engine's venv (aim, optuna, pandas, pyyaml). `uv sync`
# is a no-op once the venv matches uv.lock, so this stays cheap on later calls.
if [ ! -d .venv ]; then
  echo "[hpo] first run: uv sync in ${HPO_DIR}" >&2
  uv sync
fi

exec uv run python src/hpo_control/scheduler.py "$@"
