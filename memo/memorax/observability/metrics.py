"""Names and scalar reductions, in each scope a number can be reduced over."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from memorax.runtime.episode import Episode

STEP = "step"
EPISODE = "episode"
WINDOW = "window"
# The middle segment of a name, and the only thing that says how the number
# under it was arrived at: one moment, one episode, or every episode in a
# stretch. A reading whose scope cannot be written down cannot be compared.
SCOPES: tuple[str, ...] = (STEP, EPISODE, WINDOW)
VARIANCE = "_variance"
TOTALS: tuple[str, ...] = ("length", "return")
SPREAD: tuple[str, ...] = ("return_per_step",)


def check_names(names: Iterable[str]) -> None:
    for name in names:
        parts = name.split("/")
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"metric {name!r} is not <phase>/<scope>/<quantity>; the middle "
                "part is the scope a number was reduced over, and a name "
                "without one says nothing about what it measured"
            )
        if parts[1] not in SCOPES:
            raise ValueError(
                f"metric {name!r} names the scope {parts[1]!r}, which nothing "
                f"reduces over; the scopes are {', '.join(SCOPES)}"
            )


def metric_names(
    phase: str, series: Iterable[str] = (), scope: str = EPISODE
) -> tuple[str, ...]:
    if scope not in SCOPES:
        raise ValueError(f"scope {scope!r} is not one of {', '.join(SCOPES)}")
    if scope == STEP:
        # One moment holds one value of each per-transition quantity. It has no
        # spread to report and no total to accumulate: both need a stretch.
        names = list((*SPREAD, *series))
    else:
        names = list(TOTALS)
        for name in (*SPREAD, *series):
            names += [name, f"{name}{VARIANCE}"]
    return tuple(f"{phase}/{scope}/{name}" for name in names)


def per_transition(episode: Episode) -> dict[str, list[float]]:
    """The quantities that have a value at every transition of an episode.

    These are the atoms every scope is built from. A scope that spans more
    than one episode must pool them rather than average what each episode
    reduced to, because episodes differ in length.
    """

    values = {"return_per_step": [float(one) for one in episode.rewards]}
    for name, series in episode.series.items():
        values[name] = [float(one) for one in series]
    return values


def statistics(episode: Episode) -> dict[str, float]:
    """One episode's statistic, which is what the run's record keeps."""

    rewards = [float(one) for one in episode.rewards]
    values: dict[str, float] = {
        "length": float(len(rewards)),
        "return": float(sum(rewards)),
    }
    for name, series in per_transition(episode).items():
        values |= _moments(name, series)
    return {
        f"{episode.phase}/{EPISODE}/{name}": value for name, value in values.items()
    }


def step_statistics(episode: Episode, index: int) -> dict[str, float]:
    """What a reading looks like at the one transition ``index`` names."""

    return {
        f"{episode.phase}/{STEP}/{name}": values[index]
        for name, values in per_transition(episode).items()
    }


class WindowStatistics:
    """Every episode in a stretch, reduced once when the stretch closes.

    A series is pooled over transitions and not over episodes: a mean of
    per-episode means is not the mean when the episodes have unequal lengths,
    and a mean of per-episode variances is not the variance at all. ``return``
    and ``length`` are per-episode quantities to begin with, so their window
    value averages the episodes.

    The transitions are counted rather than kept -- a stretch of a hundred
    thousand steps holds that many values of every series -- so the pooled
    moments come from Welford's recurrence, which needs one pass and does not
    subtract two large numbers to find a small one.
    """

    def __init__(self) -> None:
        self._phase = ""
        self._episodes = 0
        self._totals = {name: 0.0 for name in TOTALS}
        self._counts: dict[str, int] = {}
        self._means: dict[str, float] = {}
        self._squares: dict[str, float] = {}

    def __bool__(self) -> bool:
        return self._episodes > 0

    def add(self, episode: Episode) -> None:
        self._phase = episode.phase
        self._episodes += 1
        rewards = [float(one) for one in episode.rewards]
        self._totals["length"] += float(len(rewards))
        self._totals["return"] += float(sum(rewards))
        for name, values in per_transition(episode).items():
            self._pool(name, values)

    def _pool(self, name: str, values: Sequence[float]) -> None:
        count = self._counts.get(name, 0)
        mean = self._means.get(name, 0.0)
        square = self._squares.get(name, 0.0)
        for value in values:
            count += 1
            delta = value - mean
            mean += delta / count
            square += delta * (value - mean)
        self._counts[name] = count
        self._means[name] = mean
        self._squares[name] = square

    def statistics(self) -> dict[str, float]:
        if not self._episodes:
            raise ValueError("an empty window has nothing to report")
        values = {name: total / self._episodes for name, total in self._totals.items()}
        for name, count in self._counts.items():
            values[name] = self._means[name]
            values[f"{name}{VARIANCE}"] = self._squares[name] / count
        return {
            f"{self._phase}/{WINDOW}/{name}": value for name, value in values.items()
        }


def _moments(name: str, values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    return {
        name: mean,
        f"{name}{VARIANCE}": sum((one - mean) ** 2 for one in values) / len(values),
    }
