#!/usr/bin/env bash
# Minimal container entrypoint: just run the training command.
#
# S3 checkpoint sync and Aim run resume were intentionally left out for now:
# jobs run on on-demand EC2 (not interrupted), so resume is not needed yet.
# Aim metrics still stream live to the jump-host Aim server (aim://...:53800)
# via the training command's --log_repo argument.
#
# The real command comes from the Batch job definition and is passed as args.
set -uo pipefail

# Persistent JAX compilation cache (rtrrl.py points here); harmless if unused.
mkdir -p /tmp/jax_cache

# Config is injected at submit time (not baked into the image): submit.sh
# base64-encodes the chosen YAML into CONFIG_B64; decode it to a fixed path
# that the training command reads via --config_path /tmp/run-config.yml.
if [ -n "${CONFIG_B64:-}" ]; then
  echo "${CONFIG_B64}" | base64 -d > /tmp/run-config.yml
  echo "[entrypoint] decoded config -> /tmp/run-config.yml ($(wc -c < /tmp/run-config.yml) bytes)"
fi

echo "[entrypoint] running: $*"
exec "$@"
