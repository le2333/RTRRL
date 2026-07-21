"""Verify Task 12 input archives before any extraction."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


def verify_archives(
    archives: dict[str, tuple[Path, str]],
) -> dict[str, object]:
    records = {}
    for name, (path, expected) in sorted(archives.items()):
        actual = sha256(path.read_bytes()).hexdigest()
        records[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "verified": actual == expected,
        }
    return {
        "all_verified": all(record["verified"] for record in records.values()),
        "archives": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        action="append",
        nargs=3,
        metavar=("NAME", "PATH", "EXPECTED_SHA256"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    archives = {
        name: (Path(path), expected)
        for name, path, expected in arguments.archive
    }
    result = verify_archives(archives)
    arguments.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    if not result["all_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
