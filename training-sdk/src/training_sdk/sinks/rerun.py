from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import rerun as rr

from training_sdk import objects
from training_sdk.contract import RunConfig
from training_sdk.episode import Episode


class RerunSink:
    def __init__(self, config: RunConfig, scratch: Path) -> None:
        if config.logging.rerun_s3 is None or config.logging.rerun_every_episodes is None:
            raise ValueError("rerun sink requires rerun_s3 and rerun_every_episodes")
        self._prefix = config.logging.rerun_s3.rstrip("/")
        self._every = config.logging.rerun_every_episodes
        self._scratch = Path(scratch)
        self._config = config

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        return None

    def log_episode(self, episode: Episode) -> None:
        if episode.number % self._every:
            return
        name = f"episode-{episode.number:06d}.rrd"
        path = self._scratch / name
        self._write(path, episode)
        objects.put_file(f"{self._prefix}/{name}", path)
        path.unlink()

    def _write(self, path: Path, episode: Episode) -> None:
        stream = rr.RecordingStream("training_sdk", recording_id=path.stem)
        stream.save(path)
        stream.log(
            "episode/metadata",
            rr.AnyValues(
                run_id=self._config.run_id,
                launch_id=self._config.launch_id,
                trial=self._config.trial,
                episode=episode.number,
                phase=episode.phase,
                start_env_steps=episode.start_env_steps,
                end_env_steps=episode.end_env_steps,
            ),
            static=True,
        )
        series: dict[str, Sequence[object]] = {
            "observations": episode.observations,
            "actions": episode.actions,
            "rewards": episode.rewards,
            "terminals": episode.terminals,
            "truncations": episode.truncations,
        }
        for entity, values in series.items():
            for index, value in enumerate(values):
                stream.set_time("episode_step", sequence=index)
                stream.log(f"episode/{entity}", rr.Tensor(np.asarray(value, dtype=np.float64)))
        stream.flush()
        stream.disconnect()

    def close(self) -> None:
        return None
