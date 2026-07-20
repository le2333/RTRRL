"""Write exact pytest collection nodeids without terminal grouping."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest


class NodeIdCollector:
    def __init__(self, output: Path) -> None:
        self.output = output

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.output.write_text(
            json.dumps(
                sorted(item.nodeid for item in session.items),
                indent=2,
            )
            + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.chdir(args.root)
    status = pytest.main(
        [
            "--collect-only",
            "-o",
            "addopts=",
            "-p",
            "no:terminal",
            "tests/online_ac",
        ],
        plugins=[NodeIdCollector(args.output)],
    )
    if status != pytest.ExitCode.OK:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
