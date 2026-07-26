"""Build the catalog the image carries, from what the topologies declare.

The registry is the single source of both what can be built and what may be
searched, so the catalog is derived rather than written. Nothing here needs
editing when a topology is added.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

from training_sdk.contract import CONTRACT_VERSION, Catalog, EntryDescriptor

from .registry import TOPOLOGIES

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PACKAGE_ROOT / "catalog.json"
SOURCE_ROOTS = (PACKAGE_ROOT / "memorax", PACKAGE_ROOT / "runner")


def source_hash(roots: tuple[Path, ...] = SOURCE_ROOTS) -> str:
    """Identify the algorithm sources, and nothing else.

    Only ``.py`` files count. Bytecode caches come and go with whoever
    imported the package last, so hashing every file would give a checkout and
    the image built from it different answers -- the one thing this value must
    never do, since it is what tells the control plane the algorithm changed.
    """

    digest = hashlib.sha256()
    for root in roots:
        paths = sorted(
            path for path in root.rglob("*.py") if "__pycache__" not in path.parts
        )
        for path in paths:
            digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def build_catalog() -> Catalog:
    revision = source_hash()
    return Catalog(
        contract=CONTRACT_VERSION,
        entries={
            name: EntryDescriptor.model_validate(
                {
                    "command": ["python", "-m", "runner.main"],
                    "source_hash": revision,
                    "metrics": list(topology.metrics),
                    "space": dict(topology.space),
                }
            )
            for name, topology in TOPOLOGIES.items()
        },
    )


def encode_label(catalog: Catalog) -> str:
    raw = catalog.model_dump_json(exclude_none=True).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")


def write_catalog(path: Path = CATALOG_PATH) -> Catalog:
    catalog = build_catalog()
    path.write_text(
        json.dumps(catalog.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-label",
        action="store_true",
        help="Print the encoded Docker label value after writing catalog.json",
    )
    args = parser.parse_args(argv)
    catalog = write_catalog()
    if args.print_label:
        print(encode_label(catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
