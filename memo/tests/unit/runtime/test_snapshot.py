"""What a snapshot store promises the process that comes after the crash.

Two promises, and they pull against each other. The newest complete snapshot
must be found, because resuming from an older one throws away an interval that
was paid for. And a snapshot that is not complete must never be read as one,
because a process killed while writing is the ordinary case here rather than
the exotic one -- it is the case this whole mechanism exists for.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.runtime import FileSnapshotStore, RunSnapshot
from memorax.runtime.snapshot import attach, detach


def taken(step: int, *, seed: int = 0) -> RunSnapshot:
    return RunSnapshot.taken(
        step=step,
        eval_number=step // 10 + 1,
        state={"weights": jnp.arange(3.0) + step, "count": jnp.asarray(step)},
        key=jax.random.key(seed),
        trackers=({"next_number": step},),
        destinations=(step * 2,),
    )


def test_a_snapshot_carries_the_run_and_not_a_reference_to_it(
    tmp_path: Path,
) -> None:
    """Everything the loop would otherwise have to rebuild, read back."""

    store = FileSnapshotStore(tmp_path)
    store.save(taken(40, seed=5))

    resumed = store.latest()

    assert resumed is not None
    assert resumed.step == 40
    assert resumed.eval_number == 5
    assert resumed.trackers == ({"next_number": 40},)
    assert resumed.destinations == (80,)
    assert np.array_equal(
        jax.random.key_data(resumed.key()), jax.random.key_data(jax.random.key(5))
    )
    state = resumed.algorithm_state()
    assert np.array_equal(np.asarray(state["weights"]), np.arange(3.0) + 40)


def test_the_newest_snapshot_is_the_one_a_resume_reads(tmp_path: Path) -> None:
    """Ordered by the step they name, not by the order they were written."""

    store = FileSnapshotStore(tmp_path, keep=4)
    for step in (40, 120, 80):
        store.save(taken(step))

    resumed = store.latest()

    assert resumed is not None and resumed.step == 120


def test_only_the_last_few_snapshots_are_kept(tmp_path: Path) -> None:
    """A run's state is the size of its parameters, once per interval kept."""

    store = FileSnapshotStore(tmp_path, keep=2)
    for step in (10, 20, 30, 40):
        store.save(taken(step))

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "30.snapshot",
        "40.snapshot",
    ]


def test_a_truncated_newest_snapshot_falls_back_to_the_one_before(
    tmp_path: Path,
) -> None:
    """Losing an interval beats refusing to start.

    The write itself is a rename, so a half-written file is not what this
    guards against; an object that arrived over the network incomplete is.
    Either way the previous snapshot is a correct place to continue from and
    the run has somewhere to go.
    """

    store = FileSnapshotStore(tmp_path, keep=2)
    store.save(taken(10))
    store.save(taken(20))
    written = (tmp_path / "20.snapshot").read_bytes()
    (tmp_path / "20.snapshot").write_bytes(written[: len(written) // 2])

    resumed = store.latest()

    assert resumed is not None and resumed.step == 10


def test_an_empty_store_is_a_run_that_has_not_started(tmp_path: Path) -> None:
    assert FileSnapshotStore(tmp_path / "nothing-here").latest() is None
    assert FileSnapshotStore(tmp_path).latest() is None


def test_a_store_that_keeps_nothing_could_never_resume(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keeps nothing"):
        FileSnapshotStore(tmp_path, keep=0)


def test_a_key_inside_the_state_survives_being_written(tmp_path: Path) -> None:
    """A typed key is not an array, and a state may hold one anyway.

    An environment that carries its own stream puts one in the state, and
    ``np.asarray`` refuses it. Writing the raw key data with the
    implementation that stamped it is what makes such a state writable at all.
    """

    state = {"stream": jax.random.key(3), "weights": jnp.ones(2)}

    resumed = attach(detach(state))

    assert jnp.issubdtype(resumed["stream"].dtype, jax.dtypes.prng_key)
    assert np.array_equal(
        jax.random.key_data(resumed["stream"]),
        jax.random.key_data(state["stream"]),
    )
