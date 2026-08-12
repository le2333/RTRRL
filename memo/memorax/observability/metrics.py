"""Names and scalar reductions for completed episodes."""

from __future__ import annotations

from collections.abc import Iterable

from memorax.runtime.episode import Episode

WINDOW = "episode"
WINDOWS: tuple[str, ...] = (WINDOW,)
VARIANCE = "_variance"
TOTALS: tuple[str, ...] = ("length", "return")
SPREAD: tuple[str, ...] = ("return_per_step",)


def check_names(names: Iterable[str]) -> None:
    for name in names:
        parts = name.split("/")
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"metric {name!r} is not <phase>/<window>/<quantity>; the middle "
                "part is the window a number was reduced over, and a name "
                "without one says nothing about what it measured"
            )
        if parts[1] not in WINDOWS:
            raise ValueError(
                f"metric {name!r} names the window {parts[1]!r}, which nothing "
                f"reduces over; the windows are {', '.join(WINDOWS)}"
            )


def metric_names(phase: str, series: Iterable[str] = ()) -> tuple[str, ...]:
    names = list(TOTALS)
    for name in (*SPREAD, *series):
        names += [name, f"{name}{VARIANCE}"]
    return tuple(f"{phase}/{WINDOW}/{name}" for name in names)


def statistics(episode: Episode) -> dict[str, float]:
    rewards = [float(one) for one in episode.rewards]
    values: dict[str, float] = {
        "length": float(len(rewards)),
        "return": float(sum(rewards)),
    }
    values |= _moments("return_per_step", rewards)
    for name, series in episode.series.items():
        values |= _moments(name, [float(one) for one in series])
    return {f"{episode.phase}/{WINDOW}/{name}": value for name, value in values.items()}


def _moments(name: str, values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    return {
        name: mean,
        f"{name}{VARIANCE}": sum((one - mean) ** 2 for one in values) / len(values),
    }
