"""Read what the image can run out of the image, at the time it is built.

The control plane samples parameters on a machine that has none of this
installed, so it cannot import an entry to ask what the entry accepts. It reads
this catalog off the image's label instead, which binds what was sampled to the
digest that will run it: a file edited but not rebuilt cannot quietly widen the
space an experiment is searched over.

Nothing is registered. Every module under ``entries`` that declares ``SPACE``
and ``METRICS`` is one, and its command follows from its name.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import importlib
import json
import pkgutil
from pathlib import Path
from typing import Any

from training_sdk.contract import CONTRACT_VERSION, Catalog, EntryDescriptor

import entries

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PACKAGE_ROOT / "catalog.json"
SOURCE_ROOTS = (
    PACKAGE_ROOT / "memorax",
    PACKAGE_ROOT / "runner",
    PACKAGE_ROOT / "entries",
)


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
            name = f"{root.name}/{path.relative_to(root).as_posix()}"
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def discover() -> dict[str, Any]:
    """Import every entry and take what it declares about itself."""

    found = {}
    for module in sorted(info.name for info in pkgutil.iter_modules(entries.__path__)):
        imported = importlib.import_module(f"{entries.__name__}.{module}")
        missing = [
            name
            for name in ("SPACE", "METRICS", "main")
            if getattr(imported, name, None) is None
        ]
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
    revision = source_hash()
    return Catalog(
        contract=CONTRACT_VERSION,
        entries={
            name: EntryDescriptor.model_validate(
                {
                    "command": ["python", "-m", module.__name__],
                    "source_hash": revision,
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
