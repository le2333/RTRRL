"""What a component can report, and where a run files it.

The same shape as ``parameters.py`` and a different vocabulary, because a
reading is not a parameter: there is no domain to draw one from and nothing
searches over whether to take it. A declaration says which readings exist and
where each lands in ``StepMetrics``; a run's instance of it says which to take,
and every field is a Python bool by the time any arithmetic sees it.

One declaration, two readers. The kernel is handed the instance and gates on its
fields; an entry asks ``taken`` for the paths and publishes those. A name that
the kernel cannot produce therefore cannot be published, which is the failure
this replaces -- the driver drops a series it cannot find and says nothing, and
so a catalog could advertise eighteen readings against a kernel emitting none.
"""

from __future__ import annotations

from dataclasses import field as _field
from dataclasses import fields, is_dataclass
from typing import Any

SPLIT = "split"


def reading(*, at: str, split: bool = False, default: bool = True) -> Any:
    """Declare one reading, and the path it is filed under.

    ``split`` marks one that a network divides into parts -- a norm per position
    group -- whose names are not known until something says how many there are.
    """

    return _field(default=default, metadata={"reading": at, SPLIT: split})


def readings(*, of: type, at: str) -> Any:
    """Declare a nested declaration, and the path everything under it hangs off."""

    return _field(default_factory=of, metadata={"readings": of, "reading": at})


def taken(declaration, *, parts: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Every path this declaration takes, in the order it declares them."""

    if not is_dataclass(declaration):
        raise TypeError(f"{type(declaration).__name__} must be a dataclass")
    found: list[str] = []
    for item in fields(declaration):
        if "reading" not in item.metadata:
            raise ValueError(
                f"{type(declaration).__name__}.{item.name} is not declared with "
                "reading() or readings()"
            )
        value = getattr(declaration, item.name)
        under = item.metadata["reading"]
        if "readings" in item.metadata:
            found += [f"{under}.{name}" for name in taken(value, parts=parts)]
        elif value:
            found += (
                [f"{under}.{part}" for part in parts]
                if item.metadata[SPLIT] and parts
                else [under]
            )
    return tuple(found)


def declared(model: type) -> tuple[str, ...]:
    """Every path the declaration can offer, whatever a run asked for."""

    return taken(_all(model))


def _all(model: type):
    filled = {}
    for item in fields(model):
        if "readings" in item.metadata:
            filled[item.name] = _all(item.metadata["readings"])
        else:
            filled[item.name] = True
    return model(**filled)
