from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from trainer_infra.aim_scratch import launch_aim_scratch
from trainer_infra.facility_control import load_facility_control


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the isolated Task 7 Aim scratch.")
    parser.add_argument("--control", required=True, type=Path)
    arguments = parser.parse_args(argv)
    control = load_facility_control(arguments.control)
    executable = shutil.which("aim")
    if executable is None:
        raise RuntimeError("Aim executable is unavailable")
    report = launch_aim_scratch(control.aim, aim_executable=executable)
    print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
