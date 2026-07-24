from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
from training_sdk import bootstrap_from_environment

from brax_ppo_acceptance.config import AcceptanceConfig
from brax_ppo_acceptance.train import train


def _verify_device_contract(
    resource_profile: str,
    *,
    environ: Mapping[str, str] = os.environ,
) -> None:
    devices = jax.devices()
    test_only_hardware_skip = (
        environ.get("BRAX_ACCEPTANCE_TEST_MODE") == "1"
        and environ.get("BRAX_ACCEPTANCE_E2E_FAST") == "1"
    )
    if resource_profile == "g6x":
        if not test_only_hardware_skip:
            assert jax.default_backend() == "gpu", "g6x requires the JAX gpu backend"
            assert any("NVIDIA L4" in device.device_kind for device in devices), (
                "g6x requires an NVIDIA L4"
            )
    else:
        assert jax.default_backend() == "cpu", "CPU profiles require the JAX CPU backend"
    jax.jit(lambda x: x @ x)(jnp.eye(64)).block_until_ready()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Brax PPO acceptance trainer")
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    config = AcceptanceConfig.load(arguments.config)

    run = bootstrap_from_environment()
    if run is None:
        raise RuntimeError("TRAINER_RUN_CONTEXT_PATH is required")
    try:
        _verify_device_contract(run.context.resource_profile)
        result = train(config, run)
        run.finish(
            {
                "eval/episode_return": result.objective,
                "runtime/device_count": len(jax.devices()),
            }
        )
        print(
            json.dumps(
                {
                    "device_kind": result.device_kind,
                    "device_platforms": sorted({device.platform for device in jax.devices()}),
                    "platform": result.platform,
                },
                sort_keys=True,
            )
        )
    except BaseException as error:
        run.fail(error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
