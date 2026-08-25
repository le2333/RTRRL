"""Suspend a run to storage and resume it where it stopped.

A long run is not promised the machine it started on: a spot instance is
reclaimed, a job reaches its attempt timeout, a host fails. Without something
written down the only answer is to begin again at step zero, and for a formal
run that is the whole budget spent twice.

*Snapshot* rather than *checkpoint*. In this codebase a checkpoint is the
policy as it stood at an evaluation boundary -- the thing a score belongs to,
and the word the driver and the protocol already use. What is written here is
the whole run: the algorithm's state, the key stream that will continue it,
the episodes still open in the tracker, and what the reporter had already
said. The two are different objects and reusing one name for both would make
every sentence about either of them ambiguous.

What a resumed run owes the interrupted one is *equality*, not similarity. A
run that stopped at a boundary and continued in a second process must produce
the episodes, the readings and the parameters that one uninterrupted process
would have produced. That is why the key stream is carried rather than
re-derived, why the tracker's open episodes are carried rather than dropped,
and why the reporter is asked what it had already written -- an episode that
spans the boundary belongs to neither half on its own.

Only whole boundaries are snapshot points. Between two of them a chunk is in
flight and the state is mid-scan; at a boundary the run is quiet, the
evaluation for that boundary has been taken, and the next chunk is decided by
the schedule rather than by where the interruption fell.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

# Bumped whenever what is written stops being what an older reader expects.
# A resume is only ever attempted within one image, so a mismatch is a
# deployment fault and is refused rather than guessed at.
SNAPSHOT_FORMAT = 1


@runtime_checkable
class Resumable(Protocol):
    """Something with state a snapshot has to carry across the interruption.

    ``suspend`` returns that state as plain data -- lists, dicts, numbers,
    arrays -- and never the object itself, so what is written stays readable
    to something that did not construct the same object graph. ``resume`` puts
    one of those values back into an object that was built the same way its
    predecessor was, which is why nothing here has to serialise construction.
    """

    def suspend(self) -> Any: ...
    def resume(self, state: Any) -> None: ...


def suspend(subject: object) -> Any:
    """What ``subject`` carries across an interruption, or nothing."""

    return subject.suspend() if isinstance(subject, Resumable) else None


def resume(subject: object, state: Any) -> None:
    """Put ``state`` back, refusing a subject that cannot take it.

    A destination that was resumable when the snapshot was written and is not
    now is a changed deployment, and continuing would silently repeat whatever
    it had already reported.
    """

    if state is None:
        return
    if not isinstance(subject, Resumable):
        raise ValueError(
            f"{type(subject).__name__} was given state to resume from but "
            "does not implement suspend/resume"
        )
    subject.resume(state)


@dataclass(frozen=True)
class _TypedKey:
    """A PRNG key array on its way through a file, which is not an array.

    ``np.asarray`` refuses a typed key and a run that keeps one inside its
    state -- an environment that carries its own stream, say -- would fail to
    be written at all. The raw key data plus the implementation that stamped
    it is what ``wrap_key_data`` needs to hand back the same key.
    """

    data: np.ndarray
    impl: str


def _detached(leaf: Any) -> Any:
    if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        return _TypedKey(np.asarray(jax.random.key_data(leaf)), _impl(leaf))
    return np.asarray(leaf)


def _attached(leaf: Any) -> Any:
    if isinstance(leaf, _TypedKey):
        return jax.random.wrap_key_data(jnp.asarray(leaf.data), impl=leaf.impl)
    return jnp.asarray(leaf)


def detach(tree: Any) -> Any:
    """One pytree with its leaves read off the device, ready to be written.

    The structure is untouched: ``jax.tree.map`` rebuilds the same nodes, so a
    state made of ``struct.PyTreeNode``s comes back as those, and what is
    pickled is their fields rather than a device handle.
    """

    return jax.tree.map(_detached, tree)


def attach(tree: Any) -> Any:
    """The inverse of :func:`detach`, putting the leaves back on the device."""

    return jax.tree.map(
        _attached, tree, is_leaf=lambda leaf: isinstance(leaf, _TypedKey)
    )


def _impl(key: jax.Array) -> str:
    return str(jax.random.key_impl(key))


@dataclass(frozen=True)
class RunSnapshot:
    """One run at one boundary, in a form that outlives the process.

    ``trackers`` and ``destinations`` are tuples because a scheduler may be
    running several members through one graph, and a member's open episodes
    are its own. The single-member driver writes one of each; nothing else
    about the record differs between the two.
    """

    step: int
    eval_number: int
    state: Any
    key_data: np.ndarray
    key_impl: str
    trackers: tuple[Any, ...] = ()
    destinations: tuple[Any, ...] = ()
    format: int = SNAPSHOT_FORMAT

    @classmethod
    def taken(
        cls,
        *,
        step: int,
        eval_number: int,
        state: Any,
        key: jax.Array,
        trackers: tuple[Any, ...] = (),
        destinations: tuple[Any, ...] = (),
    ) -> RunSnapshot:
        """Read a live run into a record, leaving the run itself untouched."""

        return cls(
            step=int(step),
            eval_number=int(eval_number),
            state=detach(state),
            key_data=np.asarray(jax.random.key_data(key)),
            key_impl=_impl(key),
            trackers=tuple(trackers),
            destinations=tuple(destinations),
        )

    def key(self) -> jax.Array:
        """The training key stream, continuing rather than restarting.

        A key rebuilt from the run's seed would replay the transitions the
        interrupted process had already taken; this is the stream as it stood
        after the last chunk before the boundary.
        """

        return jax.random.wrap_key_data(jnp.asarray(self.key_data), impl=self.key_impl)

    def algorithm_state(self) -> Any:
        return attach(self.state)


class SnapshotStore(Protocol):
    """Where a run's snapshots are kept, and which one a resume reads."""

    def latest(self) -> RunSnapshot | None: ...
    def save(self, snapshot: RunSnapshot) -> None: ...


class FileSnapshotStore:
    """Snapshots as files in one directory, the highest step winning.

    The write is to a temporary name in the same directory and then a rename,
    which is atomic on both filesystems this runs on: a process killed while
    writing leaves the previous snapshot as the newest complete one rather
    than a half-written newest. ``keep`` is why more than one survives -- the
    newest is the one that will be read, and the one before it is what is left
    if the newest was truncated on its way to or from durable storage.
    """

    SUFFIX = ".snapshot"

    def __init__(self, directory: Path | str, *, keep: int = 2) -> None:
        if keep < 1:
            raise ValueError("a snapshot store that keeps nothing cannot resume")
        self._directory = Path(directory)
        self._keep = keep

    @property
    def directory(self) -> Path:
        return self._directory

    def save(self, snapshot: RunSnapshot) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        final = self._path(snapshot.step)
        pending = final.with_suffix(final.suffix + ".pending")
        with pending.open("wb") as handle:
            pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, final)
        self._prune()

    def latest(self) -> RunSnapshot | None:
        """The newest snapshot that can be read, or nothing to resume from.

        A file that will not load is passed over rather than raised on: the
        one before it is a correct place to continue from, and refusing to
        start is the one outcome worse than losing an interval.
        """

        for path in self._written():
            snapshot = self._load(path)
            if snapshot is not None:
                return snapshot
        return None

    def _load(self, path: Path) -> RunSnapshot | None:
        try:
            with path.open("rb") as handle:
                snapshot = pickle.load(handle)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
            return None
        if not isinstance(snapshot, RunSnapshot):
            return None
        if snapshot.format != SNAPSHOT_FORMAT:
            raise ValueError(
                f"{path} was written in snapshot format {snapshot.format}, and "
                f"this image reads {SNAPSHOT_FORMAT}"
            )
        return snapshot

    def _written(self) -> list[Path]:
        if not self._directory.is_dir():
            return []
        found = []
        for path in self._directory.glob(f"*{self.SUFFIX}"):
            try:
                found.append((int(path.name[: -len(self.SUFFIX)]), path))
            except ValueError:
                continue
        return [path for _, path in sorted(found, reverse=True)]

    def _prune(self) -> None:
        for path in self._written()[self._keep :]:
            path.unlink(missing_ok=True)

    def _path(self, step: int) -> Path:
        return self._directory / f"{int(step)}{self.SUFFIX}"
