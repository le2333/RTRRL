"""Build the current contract catalog baked into the acceptance image."""

from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import json
from pathlib import Path

from pydantic import ValidationError
from training_sdk.contract import CONTRACT_VERSION, Catalog, EntryDescriptor

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PACKAGE_ROOT / "catalog.json"
ENTRY_NAME = "brax_ppo_acceptance"


def build_entry() -> EntryDescriptor:
    return EntryDescriptor.model_validate(
        {
            "command": ["python", "-m", "brax_ppo_acceptance"],
            "metrics": ["episode_return", "episode_length"],
            "space": {
                "seed": {"type": "int", "low": 0, "high": 1000},
                "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2},
                "episode_length": [32],
                "failure_mode": ["none"],
            },
        }
    )


def build_catalog() -> Catalog:
    return Catalog(
        contract=CONTRACT_VERSION,
        entries={ENTRY_NAME: build_entry()},
    )


def encode_label(catalog: Catalog) -> str:
    raw = catalog.model_dump_json(exclude_none=True).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")


def decode_catalog(value: str) -> Catalog:
    try:
        compressed = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("catalog label is not valid base64") from error
    try:
        raw = gzip.decompress(compressed)
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise ValueError("catalog label is not valid gzip data") from error
    try:
        return Catalog.model_validate_json(raw)
    except (ValidationError, ValueError) as error:
        raise ValueError(f"catalog label does not contain a valid catalog: {error}") from error


def write_catalog(path: Path = CATALOG_PATH) -> Catalog:
    catalog = build_catalog()
    path.write_text(
        json.dumps(catalog.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-label",
        action="store_true",
        help="Print the encoded Docker label value after writing catalog.json",
    )
    args = parser.parse_args()
    catalog = write_catalog()
    if args.print_label:
        print(encode_label(catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
