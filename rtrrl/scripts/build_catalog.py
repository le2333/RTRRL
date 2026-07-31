"""Read what this image can run out of this image, at the time it is built.

The control plane samples parameters on a machine that has none of this
installed, so it cannot import an entry to ask what the entry accepts. It reads
the catalog off the image's label instead, which binds what was sampled to the
digest that will run it: a file edited but not rebuilt cannot quietly widen the
space an experiment is searched over.

This is `memo/runner/catalog.py` for the forked project. It differs in what a
module in `entries` has to look like. There, every module is
an entry and one that declares none of `SPACE`, `METRICS` or `main` is a
mistake; here `entries` also holds the logger adapter, and a package that may
not contain a helper is a package that pushes its helpers somewhere worse. A
module declaring none of the three is skipped, and a module declaring some of
them is still the mistake it was.
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

from training_sdk.contract import CONTRACT_VERSION, Catalog, EntryDescriptor

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PACKAGE_ROOT / "catalog.json"
DECLARED = ("SPACE", "METRICS", "main")


def discover() -> dict[str, Any]:
    """Import every entry and take what it declares about itself."""

    import entries

    found = {}
    for module in sorted(info.name for info in pkgutil.iter_modules(entries.__path__)):
        imported = importlib.import_module(f"{entries.__name__}.{module}")
        missing = [name for name in DECLARED if getattr(imported, name, None) is None]
        if len(missing) == len(DECLARED):
            continue
        if missing:
            raise ValueError(
                f"{imported.__name__} declares no {', '.join(missing)}; "
                "an entry is a module with a space, a score, and a way to run"
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
                    "space": dict(module.SPACE),
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
