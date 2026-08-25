"""The run directory as one object, and what comes back out of it.

A snapshot in a scratch directory survives a process. The interruptions worth
surviving take the container, so the directory itself has to travel -- and it
has to travel *whole*, because the snapshot describes the artifact beside it
by a byte offset and a length is only meaningful against the file it measures.
"""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from entries._snapshot import RESUME_FILENAME, resuming, resumption_of
from memorax.runtime import RunSnapshot
from tests.support.run_config import make_run_config
from worker import objects


def a_run(tmp_path: Path, **training: object):
    """One run document and the scratch directory it reports into."""

    config = make_run_config()
    document = config.model_dump(mode="json")
    document["artifacts"]["root"] = (tmp_path / "storage" / "run").resolve().as_uri()
    document["training"] |= training
    scratch = tmp_path / "scratch"
    (scratch / "artifacts").mkdir(parents=True, exist_ok=True)
    return type(config).model_validate(document), scratch


def snapshot(step: int) -> RunSnapshot:
    return RunSnapshot.taken(
        step=step,
        eval_number=1,
        state={},
        key=jax.random.key(0),
        trackers=({},),
        destinations=(0,),
    )


def test_a_run_that_asks_for_nothing_gets_no_store(tmp_path: Path) -> None:
    """Off is the default, and off is not 'on and thrown away'."""

    config, scratch = a_run(tmp_path)

    assert resumption_of([(config, scratch)]) is None
    with resuming([(config, scratch)]) as store:
        assert store is None


def test_the_directory_travels_with_the_snapshot(tmp_path: Path) -> None:
    """Saving publishes; a later attempt gets the artifacts back too."""

    config, scratch = a_run(tmp_path, snapshot_every_evaluations=100)
    (scratch / "artifacts" / "metrics.jsonl").write_text("row\n", encoding="utf-8")

    resumption = resumption_of([(config, scratch)])
    assert resumption is not None
    resumption.store().save(snapshot(100))

    assert objects.exists(resumption.uri)

    second = tmp_path / "second-attempt"
    (second / "artifacts").mkdir(parents=True)
    later, _ = a_run(tmp_path, snapshot_every_evaluations=100)
    moved = resumption_of([(later, second)])
    assert moved is not None
    assert moved.restore() is True
    assert (second / "artifacts" / "metrics.jsonl").read_text(
        encoding="utf-8"
    ) == "row\n"

    resumed = moved.store().latest()
    assert resumed is not None and resumed.step == 100


def test_a_first_attempt_finds_nothing_and_says_so(tmp_path: Path) -> None:
    config, scratch = a_run(tmp_path, snapshot_every_evaluations=100)
    resumption = resumption_of([(config, scratch)])

    assert resumption is not None
    assert resumption.restore() is False
    assert resumption.store().latest() is None


def test_an_archive_of_other_runs_is_refused(tmp_path: Path) -> None:
    """One run's record continued under another run's name."""

    config, scratch = a_run(tmp_path, snapshot_every_evaluations=100)
    resumption = resumption_of([(config, scratch)])
    assert resumption is not None
    resumption.store().save(snapshot(100))

    other = config.model_dump(mode="json")
    other["identity"]["run_id"] = "somebody-else"
    stranger = type(config).model_validate(other)
    with pytest.raises(ValueError, match="holds runs"):
        resumption_of([(stranger, scratch)]).restore()  # type: ignore[union-attr]


def test_a_finished_run_drops_what_was_insuring_it(tmp_path: Path) -> None:
    """Otherwise storage keeps a second copy of every artifact of every run."""

    config, scratch = a_run(tmp_path, snapshot_every_evaluations=100)
    uri = f"{config.artifacts.root.rstrip('/')}/{RESUME_FILENAME}"

    with resuming([(config, scratch)]) as store:
        assert store is not None
        store.save(snapshot(100))
        assert objects.exists(uri)

    assert objects.exists(uri) is False


def test_an_attempt_that_failed_leaves_its_snapshot_alone(tmp_path: Path) -> None:
    """It is exactly the snapshot the next attempt needs."""

    config, scratch = a_run(tmp_path, snapshot_every_evaluations=100)
    uri = f"{config.artifacts.root.rstrip('/')}/{RESUME_FILENAME}"

    with pytest.raises(RuntimeError, match="the machine went away"):
        with resuming([(config, scratch)]) as store:
            assert store is not None
            store.save(snapshot(100))
            raise RuntimeError("the machine went away")

    assert objects.exists(uri) is True


def test_a_group_is_one_object_under_its_lowest_run(tmp_path: Path) -> None:
    """Whatever order the manifest listed the members in.

    A group is one graph and therefore one snapshot, so the archive's name has
    to be a property of the group rather than of the order it arrived in.
    """

    members = []
    for name in ("b-run", "a-run", "c-run"):
        config, scratch = a_run(tmp_path / name, snapshot_every_evaluations=100)
        document = config.model_dump(mode="json")
        document["identity"]["run_id"] = name
        members.append((type(config).model_validate(document), scratch))

    forwards = resumption_of(members)
    backwards = resumption_of(list(reversed(members)))

    assert forwards is not None and backwards is not None
    assert forwards.uri == backwards.uri
    assert forwards.anchor == backwards.anchor == "a-run"


def test_a_group_that_disagrees_about_the_interval_is_refused(
    tmp_path: Path,
) -> None:
    """Members share the schedule, this included: they share one loop."""

    first, one = a_run(tmp_path / "one", snapshot_every_evaluations=100)
    second, two = a_run(tmp_path / "two")

    with pytest.raises(ValueError, match="disagrees about snapshot_every_evaluations"):
        resumption_of([(first, one), (second, two)])
