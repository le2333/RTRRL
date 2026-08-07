from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import rerun as rr

from memorax.runtime.episode import Episode
from worker import objects
from worker.contract import RunConfig


class RerunSink:
    def __init__(self, config: RunConfig, scratch: Path) -> None:
        if config.logging.rerun_s3 is None or config.logging.rerun_every_steps is None:
            raise ValueError("rerun sink requires rerun_s3 and rerun_every_steps")
        self._prefix = config.logging.rerun_s3.rstrip("/")
        self._every = config.logging.rerun_every_steps
        self._streams = config.training.num_envs
        self._scratch = Path(scratch)
        self._config = config

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        return None

    def _sampled(self, episode: Episode) -> bool:
        """Whether a sample step falls inside this episode, and is its stream's.

        The step counter numbers every stream's every step, so step S is row
        S // streams of stream S % streams: a sample step names one stream and
        therefore at most one episode. An episode covering no interval -- an
        evaluation rollout, dated at the boundary it measured rather than
        spending steps of its own -- has no step inside it and is never taken.
        """

        first = -(-episode.start_env_steps // self._every) * self._every
        return any(
            step % self._streams == episode.stream
            for step in range(first, episode.end_env_steps, self._every)
        )

    def log_episode(self, episode: Episode) -> None:
        if not self._sampled(episode):
            return
        name = f"{episode.phase}-{episode.number:06d}.rrd"
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
                    f"episode/{entity}", rr.Tensor(np.asarray(value, dtype=np.float64))
                )
        stream.flush()
        stream.disconnect()

    def close(self) -> None:
        return None
