"""Sample complete episodes into local Rerun recordings."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from rerun.any_value import AnyValues
from rerun.archetypes.tensor import Tensor
from rerun.recording_stream import RecordingStream

from memorax.runtime.episode import Episode

from ..metadata import RunMetadata


class RerunSink:
    def __init__(
        self,
        directory: Path,
        *,
        every_steps: int,
        num_envs: int,
        metadata: RunMetadata,
    ) -> None:
        self._directory = Path(directory)
        self._every = every_steps
        self._streams = num_envs
        self._metadata = metadata

    def _sampled(self, episode: Episode) -> bool:
        first = -(-episode.start_env_steps // self._every) * self._every
        return any(
            step % self._streams == episode.stream
            for step in range(first, episode.end_env_steps, self._every)
        )

    def log_episode(self, episode: Episode) -> None:
        if not self._sampled(episode):
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{episode.phase}-{episode.number:06d}.rrd"
        self._write(path, episode)

    def _write(self, path: Path, episode: Episode) -> None:
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
