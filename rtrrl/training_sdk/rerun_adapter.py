from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import rerun as rr

from .context import RunContext
from .types import Episode


class Recording(Protocol):
    def log_properties(self, properties: dict[str, str | int]) -> None: ...

    def log_series(
        self, name: str, values: np.ndarray, times: Sequence[int]
    ) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


RecordingFactory = Callable[[Path], Recording]


def _safe_component(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("artifact path components must be strings")
    if not value:
        return "%"
    safe = bytearray(
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    )
    return "".join(
        chr(byte) if byte in safe else f"%{byte:02X}"
        for byte in value.encode("utf-8")
    )


def _numeric_array(name: str, values: Sequence[object]) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    if array.dtype == object or not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must contain only numeric values")
    return array


class _RerunRecording:
    def __init__(self, path: Path) -> None:
        self._recording = rr.RecordingStream(
            "training_sdk", recording_id=path.stem
        )
        self._recording.save(path)

    def log_properties(self, properties: dict[str, str | int]) -> None:
        self._recording.log(
            "episode/metadata",
            rr.AnyValues(**properties),
            static=True,
        )

    def log_series(
        self, name: str, values: np.ndarray, times: Sequence[int]
    ) -> None:
        for time, value in zip(times, values, strict=True):
            self._recording.set_time("episode_step", sequence=time)
            if np.issubdtype(np.asarray(value).dtype, np.bool_):
                value = np.asarray(value, dtype=np.uint8)
            self._recording.log(f"episode/{name}", rr.Tensor(value))

    def flush(self) -> None:
        self._recording.flush()

    def close(self) -> None:
        self._recording.disconnect()


class RerunAdapter:
    def __init__(
        self,
        context: RunContext,
        *,
        every_episodes: int = 1,
        root: Path | None = None,
        factory: RecordingFactory | None = None,
    ) -> None:
        if type(every_episodes) is not int:
            raise TypeError("every_episodes must be an integer")
        if every_episodes <= 0:
            raise ValueError("every_episodes must be positive")
        self.context = context
        self.every_episodes = every_episodes
        self.root = Path(root or context.artifact_directory)
        self.factory = factory or _RerunRecording

    def log_episode(self, episode: Episode) -> Path | None:
        if episode.number % self.every_episodes:
            return None

        target = (
            self.root
            / _safe_component(self.context.experiment_name)
            / _safe_component(self.context.run_name)
            / f"episode-{episode.number:06d}.rrd"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"episode artifact already exists: {target}")

        arrays = {
            "observations": _numeric_array(
                "observations", episode.observations
            ),
            "actions": _numeric_array("actions", episode.actions),
            "rewards": _numeric_array("rewards", episode.rewards),
            "terminals": _numeric_array("terminals", episode.terminals),
            "truncations": _numeric_array("truncations", episode.truncations),
        }
        if episode.environment_states:
            arrays["environment_states"] = _numeric_array(
                "environment_states", episode.environment_states
            )

        transition_times = range(len(episode.actions))
        observation_times = range(len(episode.observations))
        times: dict[str, Sequence[int]] = {
            "observations": observation_times,
            "actions": transition_times,
            "rewards": transition_times,
            "terminals": transition_times,
            "truncations": transition_times,
        }
        if "environment_states" in arrays:
            state_count = len(arrays["environment_states"])
            if state_count == len(episode.observations):
                times["environment_states"] = observation_times
            elif state_count == len(episode.actions):
                times["environment_states"] = transition_times
            else:
                raise ValueError(
                    "environment_states must contain N or N+1 values "
                    "for N transitions"
                )

        properties: dict[str, str | int] = {
            "experiment": self.context.experiment_name,
            "group": self.context.group,
            "script": self.context.script,
            "run_number": self.context.run_number,
            "trial_number": self.context.trial_number,
            "episode_number": episode.number,
            "phase": episode.phase,
            "start_env_steps": episode.start_env_steps,
            "end_env_steps": episode.end_env_steps,
        }
        temporary_path: Path | None = None
        recording: Recording | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            recording = self.factory(temporary_path)
            recording.log_properties(properties)
            for name, values in arrays.items():
                recording.log_series(name, values, times[name])
            recording.flush()
            recording.close()
            recording = None
            if target.exists():
                raise FileExistsError(
                    f"episode artifact already exists: {target}"
                )
            temporary_path.replace(target)
            temporary_path = None
            return target
        finally:
            if recording is not None:
                recording.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
