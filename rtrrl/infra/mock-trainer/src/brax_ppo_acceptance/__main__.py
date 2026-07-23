from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import jax
from training_sdk import bootstrap_from_environment

from brax_ppo_acceptance.config import AcceptanceConfig
from brax_ppo_acceptance.train import train


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Brax PPO acceptance trainer")
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    config = AcceptanceConfig.load(arguments.config)

    run = bootstrap_from_environment()
    if run is None:
        raise RuntimeError("TRAINER_RUN_CONTEXT_PATH is required")
    try:
        result = train(config, run)
        run.finish(
            {
                "eval/episode_return": result.objective,
                "runtime/device_count": len(jax.devices()),
            }
        )
    except BaseException as error:
        run.fail(error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
