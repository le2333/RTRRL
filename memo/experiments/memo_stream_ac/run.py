"""Facility launcher for the supported Stream AC exact-RTRL environments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parents[1]
_ROOT = _EXPERIMENTS.parent
for _path in (_EXPERIMENTS, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from base.experiment import run_experiment  # noqa: E402
from base.facility import FacilityInput, build_stream_ac_config  # noqa: E402
from stream_ac_kmemorychain.run import train as train_kmemory_chain  # noqa: E402
from stream_ac_memorychain.run import train as train_memory_chain  # noqa: E402
from stream_ac_mujoco_masked.run import train as train_mujoco_masked  # noqa: E402
from training_sdk import bootstrap_from_environment  # noqa: E402

_TRAINERS = {
    "memory_chain": train_memory_chain,
    "kmemory_chain": train_kmemory_chain,
    "mujoco_masked": train_mujoco_masked,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    value = FacilityInput.load(args.config)
    config = build_stream_ac_config(value)
    training_run = bootstrap_from_environment()
    if training_run is not None:
        config.logging = "aim"
    trainer = _TRAINERS[value.environment["name"]]
    run_experiment(trainer, config, project_name="memorax-rtrl")


if __name__ == "__main__":
    main()
