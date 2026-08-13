"""Generate the catalog embedded in a training image at build time."""

from __future__ import annotations

import argparse
import base64
import gzip
import importlib
import json
import pkgutil
from pathlib import Path
from typing import Any

import entries
from memorax.parameters import describe

from .contract import CONTRACT_VERSION, Catalog, EntryDescriptor

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PACKAGE_ROOT / "catalog.json"


def discover() -> dict[str, Any]:
    """Import each explicitly exposed entry module and its declarations."""

    found = {}
    for module in sorted(
        info.name
        for info in pkgutil.iter_modules(entries.__path__)
        if not info.name.startswith("_")
    ):
        imported = importlib.import_module(f"{entries.__name__}.{module}")
        missing = [
            name
            for name in ("PARAMETERS", "METRICS", "main")
            if getattr(imported, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"{imported.__name__} declares no {', '.join(missing)}; "
                "an entry needs parameters, metrics, and main"
            )
        found[module] = imported
    if not found:
        raise ValueError("no entries were found, so the image can run nothing")
    return found


def build_catalog() -> Catalog:
    return Catalog(
        contract=CONTRACT_VERSION,
        entries={
            name: EntryDescriptor(
                command=("python", "-m", module.__name__),
                metrics=tuple(module.METRICS),
                parameters=describe(module.PARAMETERS),
            )
            for name, module in discover().items()
        },
    )


def encode_label(catalog: Catalog) -> str:
    raw = catalog.model_dump_json(exclude_none=True).encode()
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
    parser.add_argument("--print-label", action="store_true")
    args = parser.parse_args(argv)
    catalog = write_catalog()
    if args.print_label:
        print(encode_label(catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
