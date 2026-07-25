from __future__ import annotations

import json
import os
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from training_sdk.reporter import Reporter

from brax_ppo_acceptance.config import AcceptanceConfig
from brax_ppo_acceptance.train import train


def _infer_profile() -> str:
    if jax.default_backend() == "gpu":
        return "g6x"
    return "cpu"


def _verify_device_contract(
    resource_profile: str,
    *,
    environ: dict[str, str] | None = None,
) -> None:
    devices = jax.devices()
    env = environ if environ is not None else os.environ
    test_only_hardware_skip = (
        env.get("BRAX_ACCEPTANCE_TEST_MODE") == "1"
        and env.get("BRAX_ACCEPTANCE_E2E_FAST") == "1"
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
    import argparse

    parser = argparse.ArgumentParser(description="Run the Brax PPO acceptance trainer")
    parser.parse_args(argv)

    with Reporter.from_env() as reporter:
        config = AcceptanceConfig.from_params(reporter.config.params)
        _verify_device_contract(_infer_profile())
        result = train(config, reporter)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
