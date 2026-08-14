"""Serialize the trajectories Runtime selected into local Rerun recordings."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from rerun.any_value import AnyValues
from rerun.archetypes.tensor import Tensor
from rerun.recording_stream import RecordingStream

from memorax.runtime.episode import SampledTrajectory

from ..metadata import RunMetadata


class RerunSink:
    """One RRD per requested sample. It chooses nothing; Runtime chose."""

    def __init__(self, directory: Path, *, metadata: RunMetadata) -> None:
        self._directory = Path(directory)
        self._metadata = metadata

    def log_trajectory(self, trajectory: SampledTrajectory) -> None:
        episode = trajectory.episode
        self._directory.mkdir(parents=True, exist_ok=True)
        name = f"{episode.phase}-sample-{trajectory.sample_step:012d}.rrd"
        self._write(self._directory / name, trajectory)

    def _write(self, path: Path, trajectory: SampledTrajectory) -> None:
        episode = trajectory.episode
        stream = RecordingStream("memorax", recording_id=path.stem)
        stream.save(path)
        stream.log(
            "episode/metadata",
            AnyValues(
                run_id=self._metadata.run_id,
                launch_id=self._metadata.launch_id,
                trial=self._metadata.trial,
                episode=episode.number,
                phase=episode.phase,
                sample_step=trajectory.sample_step,
                stream=episode.stream,
                start_env_steps=episode.start_env_steps,
                end_env_steps=episode.end_env_steps,
            ),
            static=True,
        )
        walked: dict[str, Sequence[object] | None] = {
            "observations": episode.observations,
            "actions": episode.actions,
            "rewards": episode.rewards,
            "terminals": episode.terminals,
            "truncations": episode.truncations,
            # Which transitions were taken after the training budget, so a walk
            # never reads a continuation as if it had trained.
            "post_budget": trajectory.post_budget,
        }
        series: dict[str, Sequence[object]] = {
            entity: values for entity, values in walked.items() if values is not None
        }
        series |= {f"series/{name}": values for name, values in episode.series.items()}
        for entity, values in series.items():
            for index, value in enumerate(values):
                stream.set_time("episode_step", sequence=index)
                stream.log(
                    f"episode/{entity}", Tensor(np.asarray(value, dtype=np.float64))
                )
        stream.flush()
        stream.disconnect()

    def close(self) -> None:
        return None
