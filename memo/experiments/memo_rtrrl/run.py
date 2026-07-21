"""Facility launcher for shared-topology RTRRL on Hopper."""

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
from base.facility import FacilityInput, build_rtrrl_config  # noqa: E402
from rtrrl_hopper.run import train  # noqa: E402
from training_sdk import bootstrap_from_environment  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    value = FacilityInput.load(args.config)
    config = build_rtrrl_config(value)
    training_run = bootstrap_from_environment()
    if training_run is not None:
        config.logging = "aim"
    try:
        run_experiment(train, config, project_name="memorax-rtrl")
    except BaseException as error:
        if training_run is not None:
            training_run.fail(error)
        raise


if __name__ == "__main__":
    main()
