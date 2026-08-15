"""Track complete and sampled episodes across bounded rollout chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from memorax.runtime.episode import Episode, SampledTrajectory
from memorax.runtime.program import ObservationSchema
from memorax.runtime.rollout import read


@dataclass(frozen=True)
class TrackingResult:
    completed: tuple[Episode, ...]
    sampled: tuple[SampledTrajectory, ...]


@dataclass(frozen=True)
class _Walk:
    """One chunk's per-step trajectory, present only when a schema names it."""

    before: np.ndarray
    after: np.ndarray
    actions: np.ndarray


@dataclass
class _OpenEpisode:
    start_env_steps: int
    observations: list[object]
    actions: list[object]
    rewards: list[float]
    terminals: list[bool]
    truncations: list[bool]
    series: dict[str, list[float]]
    post_budget: list[bool]
    samples: list[int]


class EpisodeTracker:
    """Reconstruct one stateful episode per environment stream."""

    def __init__(
        self,
        *,
        observations: ObservationSchema,
        num_envs: int,
        max_episode_steps: int,
        sample_steps: tuple[int, ...] = (),
        first_number: int = 1,
    ) -> None:
        self._observations = observations
        self._num_envs = num_envs
        self._max_episode_steps = max_episode_steps
        self._sample_steps = sample_steps
        self._sample_index = 0
        self._next_number = first_number
        self._slots: list[_OpenEpisode | None] = [None] * num_envs

    @property
    def pending_sample_steps(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                sample
                for slot in self._slots
                if slot is not None
                for sample in slot.samples
            )
        )

    @property
    def next_number(self) -> int:
        return self._next_number

    def consume(
        self,
        summary: object,
        *,
        start_env_steps: int,
        post_budget: bool = False,
        report_completed: bool = True,
    ) -> TrackingResult:
        observations = self._observations
        rewards = np.asarray(read(summary, observations.reward))
        dones = np.asarray(read(summary, observations.done)).astype(bool)
        terminal_values = (
            None
            if observations.terminal is None
            else read(summary, observations.terminal)
        )
        terminals = (
            dones
            if observations.terminal is None
            else np.asarray(terminal_values).astype(bool)
        )
        steps = dones.shape[0]
        series = {}
        for name in observations.series:
            found = read(summary, name)
            if found is None:
                if not post_budget:
                    raise ValueError(f"configured series path {name!r} is missing")
                # A post-budget transition performed no update, so its update
                # readings do not exist. Saying so is the point; a zero here
                # would read as a measurement that was never taken.
                series[name] = np.full((steps, self._num_envs), np.nan)
                continue
            series[name] = _streamed(found, steps, self._num_envs)

        walk = self._walk(summary, observations)

        completed: list[Episode] = []
        sampled: list[SampledTrajectory] = []
        for row in range(steps):
            for stream in range(self._num_envs):
                position = start_env_steps + row * self._num_envs + stream
                self._attach_samples(position, stream, series)
                slot = self._slot(stream, position, series)

                if len(slot.rewards) >= self._max_episode_steps:
                    raise ValueError(
                        "episode exceeded maximum episode length "
                        f"of {self._max_episode_steps} transitions"
                    )

                if walk is not None:
                    slot.observations.append(walk.before[row, stream].tolist())
                    slot.actions.append(walk.actions[row, stream].tolist())
                slot.rewards.append(float(rewards[row, stream]))
                terminal = bool(terminals[row, stream])
                done = bool(dones[row, stream])
                slot.terminals.append(terminal)
                slot.truncations.append(done and not terminal)
                for name, values in series.items():
                    slot.series[name].append(float(values[row, stream]))
                slot.post_budget.append(bool(post_budget))

                if not done:
                    continue

                if walk is not None:
                    slot.observations.append(walk.after[row, stream].tolist())
                episode = Episode(
                    number=self._next_number,
                    phase="train",
                    stream=stream,
                    start_env_steps=slot.start_env_steps,
                    end_env_steps=position + self._num_envs,
                    observations=slot.observations if walk is not None else None,
                    actions=slot.actions if walk is not None else None,
                    rewards=slot.rewards,
                    terminals=slot.terminals,
                    truncations=slot.truncations,
                    series=slot.series,
                )
                if report_completed:
                    completed.append(episode)
                sampled.extend(
                    SampledTrajectory(
                        episode=episode,
                        sample_step=sample,
                        post_budget=tuple(slot.post_budget),
                    )
                    for sample in slot.samples
                )
                self._next_number += 1
                self._slots[stream] = None

        ending_boundary = start_env_steps + steps * self._num_envs
        self._attach_samples(ending_boundary, 0, series)
        return TrackingResult(tuple(completed), tuple(sampled))

    def _walk(self, summary: object, observations: ObservationSchema) -> _Walk | None:
        """This chunk's trajectory, or nothing if the schema names none."""

        before = observations.observation
        after = observations.next_observation
        action = observations.action
        if before is None or after is None or action is None:
            if not (before is None and after is None and action is None):
                raise ValueError(
                    "trajectory schema paths must be all configured or all unset"
                )
            return None
        return _Walk(
            before=self._required(summary, before),
            after=self._required(summary, after),
            actions=self._required(summary, action),
        )

    @staticmethod
    def _required(summary: object, path: str) -> np.ndarray:
        """A trajectory field the schema promised the graph would fill."""

        found = read(summary, path)
        if found is None:
            raise ValueError(f"configured trajectory path {path!r} is missing")
        return np.asarray(found)

    def _slot(
        self,
        stream: int,
        start_env_steps: int,
        series: dict[str, np.ndarray],
    ) -> _OpenEpisode:
        slot = self._slots[stream]
        if slot is None:
            slot = _OpenEpisode(
                start_env_steps=start_env_steps,
                observations=[],
                actions=[],
                rewards=[],
                terminals=[],
                truncations=[],
                series={name: [] for name in series},
                post_budget=[],
                samples=[],
            )
            self._slots[stream] = slot
        return slot

    def _attach_samples(
        self,
        position: int,
        stream: int,
        series: dict[str, np.ndarray],
    ) -> None:
        while (
            self._sample_index < len(self._sample_steps)
            and self._sample_steps[self._sample_index] == position
        ):
            sample = self._sample_steps[self._sample_index]
            self._slot(stream, position, series).samples.append(sample)
            self._sample_index += 1


def _streamed(values: object, steps: int, num_envs: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(values).reshape(steps, -1), (steps, num_envs))
