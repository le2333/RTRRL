"""Read what the image can run out of the image, at the time it is built.

The control plane samples parameters on a machine that has none of this
installed, so it cannot import an entry to ask what the entry accepts. It reads
this catalog off the image's label instead, which binds what was sampled to the
digest that will run it: a file edited but not rebuilt cannot quietly widen the
space an experiment is searched over.

Nothing is registered. Every module under ``entries`` that declares
``PARAMETERS`` and ``METRICS`` is one, and its command follows from its name.
"""

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
from worker.contract import CONTRACT_VERSION, Catalog, EntryDescriptor

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PACKAGE_ROOT / "catalog.json"


def discover() -> dict[str, Any]:
    """Import every entry and take what it declares about itself."""

    found = {}
    for module in sorted(info.name for info in pkgutil.iter_modules(entries.__path__)):
        imported = importlib.import_module(f"{entries.__name__}.{module}")
        missing = [
            name
            for name in ("PARAMETERS", "METRICS", "main")
            if getattr(imported, name, None) is None
        ]
        if missing:
            raise ValueError(
                f"{imported.__name__} declares no {', '.join(missing)}; "
                "an entry is a module with parameters, a score, and a way to run"
            )
        found[module] = imported
    if not found:
        raise ValueError("no entries were found, so the image can run nothing")
    return found


def build_catalog() -> Catalog:
    return Catalog(
        contract=CONTRACT_VERSION,
        entries={
            name: EntryDescriptor.model_validate(
                {
                    "command": ["python", "-m", module.__name__],
                    "metrics": list(module.METRICS),
                    "parameters": dict(module.PARAMETERS),
                }
            )
            for name, module in discover().items()
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
