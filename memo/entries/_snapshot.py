"""Keep a run's snapshots somewhere the machine's death does not reach.

:mod:`memorax.runtime.snapshot` writes a run to a directory. A directory is
enough to survive a process and nothing more: a Batch job's scratch goes with
its container, and the interruptions worth surviving -- a reclaimed spot
instance, an attempt timeout, a failed host -- take the container with them.
So the directory is mirrored into object storage, which is where the next
attempt looks.

What is mirrored is the run *directory*, not the snapshot alone. The snapshot
records how many bytes of the metrics artifact had been written when it was
taken, and cutting a file back to a length only means something if the file
came back too. Artifacts are otherwise uploaded once, at the end of a
successful run, so without this the resumed process would find an empty
directory beside a snapshot describing a record that is not there.

One object per run, overwritten. A PUT either replaces the object or leaves
the previous one whole, so an upload killed halfway costs the newest interval
rather than the ability to resume at all -- and finding what to resume from is
then a single GET rather than a listing whose consistency would have to be
reasoned about.

A group is one object for the same reason it is one graph: its members' keys
are rows of one key array and their parameters are leaves of one tree, so
there is no such thing as resuming one member of it.
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from memorax.runtime import FileSnapshotStore, RunSnapshot, SnapshotStore
from worker import objects

from ._contract import RunSpec

ARTIFACTS = "artifacts"
SNAPSHOTS = "snapshots"
MEMBERS_FILENAME = "members.json"
# Under the artifact root of the run it belongs to, and under a name no sink
# can take: artifacts are written into the run's directory by name, and this
# is written into the run's prefix by the entry.
RESUME_FILENAME = "resume.tar.gz"

# What of a scratch directory travels. The rest of it -- the configuration the
# worker wrote there, whatever a framework cached -- the next attempt rebuilds
# from the same manifest, and carrying it would only make the object bigger.
CARRIED = (ARTIFACTS, SNAPSHOTS)


@dataclass(frozen=True)
class Resumption:
    """One run or one group, and the single object that outlives its machine.

    ``directories`` are the scratch directories under the run ids they belong
    to, because a group's members arrive in whatever order the manifest listed
    them and the archive must read the same under any of them. ``anchor`` is
    the member whose scratch holds the snapshots and whose artifact root names
    the object: the lowest run id, which is a property of the group rather
    than of the order it arrived in.
    """

    uri: str
    directories: Mapping[str, Path]
    anchor: str

    def store(self) -> SnapshotStore:
        """Where the runtime writes, and what mirrors it as it does."""

        local = FileSnapshotStore(self.directories[self.anchor] / SNAPSHOTS)
        return _MirroredStore(local, self)

    def restore(self) -> bool:
        """Put back what the interrupted attempt left, if it left anything.

        An archive that names different runs is not this group's. It is
        refused rather than unpacked, because the alternative is one run's
        record continued under another run's name -- a wrong number that
        nothing downstream could detect.
        """

        if not objects.exists(self.uri):
            return False
        with tarfile.open(
            fileobj=io.BytesIO(objects.get_bytes(self.uri)), mode="r:gz"
        ) as archive:
            names = _members(archive)
            if names != sorted(self.directories):
                raise ValueError(
                    f"{self.uri} holds runs {names}, and this attempt is "
                    f"{sorted(self.directories)}"
                )
            for run_id, directory in self.directories.items():
                _extract(archive, run_id, Path(directory))
        return True

    def publish(self) -> None:
        """Write every member's carried directories out as one object.

        Through a file rather than through memory. What travels here is the
        run's whole state, which for a replay-based algorithm is its buffer;
        building the archive in a ``BytesIO`` and then handing
        ``getvalue()`` to the upload would hold two copies of that at the one
        moment the run can least afford it.
        """

        anchor = Path(self.directories[self.anchor])
        with tempfile.TemporaryDirectory(dir=anchor.parent) as staging:
            path = Path(staging) / RESUME_FILENAME
            with tarfile.open(path, mode="w:gz") as archive:
                _add_members(archive, sorted(self.directories))
                for run_id, directory in self.directories.items():
                    for carried in CARRIED:
                        source = Path(directory) / carried
                        if source.is_dir():
                            archive.add(source, arcname=f"{run_id}/{carried}")
            objects.put_file(self.uri, path)

    def discard(self) -> None:
        """Drop the object once the run it was insuring has finished.

        A finished run has published its artifacts under their own names, and
        its snapshot describes a budget that is spent. Keeping it would hold a
        second copy of every artifact of every run, and would offer a later
        attempt a snapshot at the end of the budget: a loop that does nothing
        and a run that reports nothing.
        """

        objects.delete(self.uri)


class _MirroredStore:
    """A local snapshot store whose every write is pushed to the object."""

    def __init__(self, local: FileSnapshotStore, resumption: Resumption) -> None:
        self._local = local
        self._resumption = resumption

    def latest(self) -> RunSnapshot | None:
        return self._local.latest()

    def save(self, snapshot: RunSnapshot) -> None:
        # Local first: the archive is built by reading the directory, so the
        # snapshot has to be in it before the directory is read.
        self._local.save(snapshot)
        self._resumption.publish()


def resumption_of(members: Sequence[tuple[RunSpec, Path]]) -> Resumption | None:
    """What this run or group resumes through, or nothing if it does not."""

    if not members:
        raise ValueError("a resumption needs at least one run")
    declared = {spec.training.snapshot_every_evaluations for spec, _ in members}
    if len(declared) != 1:
        raise ValueError(
            f"the group disagrees about snapshot_every_evaluations: "
            f"{sorted(declared)}"
        )
    if not declared.pop():
        return None

    directories = {spec.identity.run_id: Path(scratch) for spec, scratch in members}
    anchor = min(directories)
    root = next(
        spec.artifacts.root.rstrip("/")
        for spec, _ in members
        if spec.identity.run_id == anchor
    )
    return Resumption(
        uri=f"{root}/{RESUME_FILENAME}",
        directories=directories,
        anchor=anchor,
    )


@contextmanager
def resuming(
    members: Sequence[tuple[RunSpec, Path]],
) -> Iterator[SnapshotStore | None]:
    """Restore what the last attempt left, and clear it if this one finishes.

    The object is dropped only on the way out of a block that returned. An
    attempt that raises is exactly the attempt whose snapshot the next one
    needs, so an exception leaves it where it is.
    """

    resumption = resumption_of(members)
    if resumption is None:
        yield None
        return
    resumption.restore()
    yield resumption.store()
    resumption.discard()


def _members(archive: tarfile.TarFile) -> list[str]:
    entry = archive.extractfile(MEMBERS_FILENAME)
    if entry is None:
        raise ValueError("the resume archive names no runs")
    return sorted(json.loads(entry.read().decode("utf-8"))["runs"])


def _add_members(archive: tarfile.TarFile, runs: list[str]) -> None:
    payload = json.dumps({"runs": runs}, sort_keys=True).encode("utf-8")
    info = tarfile.TarInfo(MEMBERS_FILENAME)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _extract(archive: tarfile.TarFile, run_id: str, directory: Path) -> None:
    """Unpack one run's carried directories, and nothing outside them.

    The archive is one this image wrote, but it arrives over the network, and
    unpacking is the one step at which a wrong object could write outside the
    scratch directory. So the destination is composed here from the parts of
    the entry's name rather than taken from it, anything that is not a plain
    file or a directory under a carried name is skipped, and a name holding a
    traversal is refused outright.
    """

    for member in archive.getmembers():
        parts = member.name.split("/")
        if parts[0] != run_id:
            continue
        if any(part in ("", ".", "..") for part in parts):
            raise ValueError(f"{member.name!r} is not a name this archive may hold")
        if len(parts) < 2 or parts[1] not in CARRIED:
            continue
        target = directory.joinpath(*parts[1:])
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            continue
        payload = archive.extractfile(member)
        if payload is None:  # pragma: no cover - isfile already answered this
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload.read())
