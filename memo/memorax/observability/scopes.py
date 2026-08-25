"""What a training reading is taken over, and how often one is taken.

Each scope carries an interval expressed in that scope's own unit, so the
combination that biased the old sampling cannot be written down. That sampling
chose a *step* mark and then reported the *episode* spanning it: a long episode
spans more of the axis, so it was more likely to be picked, and every quantity
that grows with episode length came back inflated -- and inflated by more early
in a run, when the length distribution is widest, than at the end, which
compresses the curve rather than shifting it.

Each scope below selects among objects of one size:

- ``StepScope`` selects a step, and every step is one step.
- ``EpisodeScope`` selects every Nth episode, which is uniform in episode space.
- ``WindowScope`` selects a stretch of the axis, and counts an episode in the
  window it *ends* in, which partitions the episodes between windows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from memorax.runtime.episode import Episode

from .metrics import WindowStatistics, statistics, step_statistics

Reading = tuple[int, Mapping[str, float]]


class Scope(Protocol):
    """One scope's schedule, and the reduction its readings are due from."""

    def take(self, episode: Episode) -> tuple[Reading, ...]:
        """Whatever this completed episode brought due, possibly nothing."""
        ...

    def close(self) -> tuple[Reading, ...]:
        """Whatever the end of the run brought due, possibly nothing."""
        ...

    def suspend(self) -> Any:
        """Whatever this scope is in the middle of, or ``None``."""
        ...

    def resume(self, state: Any) -> None:
        """Take back what ``suspend`` returned in the interrupted process."""
        ...


class StepScope:
    """What a reading looks like at a typical moment, every ``every_steps``.

    A mark selects a step and not an episode, so nothing about how long the
    surrounding episode ran reaches the sample. The step counter numbers every
    stream's every step, so a mark names one stream's one transition -- the
    same rule the trajectory sample follows -- and the episode holding that
    transition is the one that answers for the mark.
    """

    def __init__(self, every_steps: int) -> None:
        if every_steps < 1:
            raise ValueError("step scope every_steps must be positive")
        self._every = every_steps

    def take(self, episode: Episode) -> tuple[Reading, ...]:
        start, end = episode.start_env_steps, episode.end_env_steps
        if end <= start:
            return ()
        width = (end - start) // len(episode.rewards)
        first = max(-(-start // self._every) * self._every, self._every)
        readings = []
        for mark in range(first, end, self._every):
            offset = mark - start
            if offset % width:
                # The mark names another stream's step. That stream reports it
                # from its own episode, so taking it here would report the
                # moment twice and from the wrong transition.
                continue
            readings.append((mark, step_statistics(episode, offset // width)))
        return tuple(readings)

    def close(self) -> tuple[Reading, ...]:
        return ()

    def suspend(self) -> None:
        """Nothing: this scope decides from the episode in front of it."""

        return None

    def resume(self, state: Any) -> None:
        del state


class EpisodeScope:
    """What a typical episode's statistic is, every ``every_episodes``.

    The choice is made among episodes, which are the objects being described,
    so a long one is no likelier to be picked than a short one.
    """

    def __init__(self, every_episodes: int) -> None:
        if every_episodes < 1:
            raise ValueError("episode scope every_episodes must be positive")
        self._every = every_episodes

    def take(self, episode: Episode) -> tuple[Reading, ...]:
        if episode.number % self._every:
            return ()
        return ((episode.end_env_steps, statistics(episode)),)

    def close(self) -> tuple[Reading, ...]:
        return ()

    def suspend(self) -> None:
        """Nothing: this scope decides from the episode in front of it."""

        return None

    def resume(self, state: Any) -> None:
        del state


class WindowScope:
    """What every episode in a stretch averaged, every ``every_steps``.

    A window closes on a multiple of ``every_steps`` and an episode belongs to
    the window its last transition falls in, so each episode is counted once
    whatever its length. ``length_steps`` is how much of the axis before a
    close is kept: the default tiles the axis and uses every episode, and a
    shorter length samples stretches instead -- still unbiased, since a stretch
    is a fixed size -- while keeping the accumulator alive for less of the run.
    """

    def __init__(self, every_steps: int, length_steps: int | None = None) -> None:
        if every_steps < 1:
            raise ValueError("window scope every_steps must be positive")
        length = every_steps if length_steps is None else length_steps
        if not 1 <= length <= every_steps:
            raise ValueError(
                "window scope length_steps must be positive and no longer than "
                "every_steps, or two windows would claim the same episode"
            )
        self._every = every_steps
        self._length = length
        self._closes_at: int | None = None
        self._window = WindowStatistics()

    def take(self, episode: Episode) -> tuple[Reading, ...]:
        end = episode.end_env_steps
        closes_at = -(-end // self._every) * self._every
        readings: tuple[Reading, ...] = ()
        if self._closes_at is not None and closes_at > self._closes_at:
            readings = self.close()
        self._closes_at = closes_at
        if end > closes_at - self._length:
            self._window.add(episode)
        return readings

    def close(self) -> tuple[Reading, ...]:
        """Report the open window, which the end of a run may have cut short.

        A short window is a shorter stretch and still a stretch, so it is
        reported at the close it was scheduled for rather than at the last
        episode that reached it.
        """

        if self._closes_at is None or not self._window:
            return ()
        reading = (self._closes_at, self._window.statistics())
        self._closes_at = None
        self._window = WindowStatistics()
        return (reading,)

    def suspend(self) -> dict[str, Any]:
        """The window that was open, and the close it was waiting for.

        A window spans an interruption exactly as it spans a chunk, and the
        episodes that fell in it before the stop are as much part of the
        stretch as the ones after.
        """

        return {"closes_at": self._closes_at, "window": self._window.suspend()}

    def resume(self, state: Mapping[str, Any]) -> None:
        closes_at = state["closes_at"]
        self._closes_at = None if closes_at is None else int(closes_at)
        self._window = WindowStatistics()
        self._window.resume(state["window"])
