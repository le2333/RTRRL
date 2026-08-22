"""How a round's runs are packed, and what decides it.

The entry decides. An experiment file asks for a grouped round by naming an
entry that takes groups, and there is no second switch to disagree with the
first.
"""

from __future__ import annotations

import pytest

from trainer_infra.batch import BatchRoundExecutor
from trainer_infra.ensemble import BatchEnsembleExecutor, LocalEnsembleExecutor
from trainer_infra.local import LocalRoundExecutor
from trainer_infra.rounds import RoundError, chunk, grouped_manifests, partition, signature

STATIC = frozenset({"core.kind", "core.lru.hidden_dim"})


def configuration(**parameters) -> dict:
    base = {"core.kind": "lru", "core.lru.hidden_dim": 32, "gamma": 0.99}
    return {"algorithm": {"parameters": {**base, **parameters}}}


def round_of(*configurations) -> tuple[tuple[dict, ...], tuple[str, ...]]:
    uris = tuple(f"s3://x/run-{index}.json" for index in range(len(configurations)))
    return configurations, uris


# ------------------------------------------------- the two channels, side by side


def test_the_parallel_channel_overrides_one_method_and_inherits_the_rest():
    """The measure of how little the two channels differ.

    Publishing the configurations, submitting the jobs, waiting on them and
    reading the scores back are the ordinary channel's and are not overridden.
    If that stops being true, the ordinary channel has grown a second
    implementation that nobody is running.
    """

    for parallel, ordinary, seam in (
        (BatchEnsembleExecutor, BatchRoundExecutor, "_manifests"),
        (LocalEnsembleExecutor, LocalRoundExecutor, "_manifest"),
    ):
        assert issubclass(parallel, ordinary)
        overridden = {
            name
            for name, value in vars(parallel).items()
            if not name.startswith("__") and name in vars(ordinary)
        }
        assert overridden == {seam}


def test_the_ordinary_channel_still_owns_its_own_packing():
    """It was not moved out, only made overridable.

    A round of four across two jobs, as it always was -- this is the behaviour
    a launch depends on, and the seam exists so that it did not have to change.
    """

    executor = BatchRoundExecutor.__new__(BatchRoundExecutor)
    executor.parallel_jobs = 2
    configurations, uris = round_of(*(configuration(gamma=g) for g in (0.9, 0.8, 0.7, 0.6)))
    assert executor._manifests(configurations, uris) == (
        {"runs": [uris[0], uris[1]]},
        {"runs": [uris[2], uris[3]]},
    )


# -------------------------------------------------------------------- grouped


def test_a_grouped_round_of_one_structure_is_one_group():
    configurations, uris = round_of(*(configuration(gamma=g) for g in (0.9, 0.8, 0.7)))
    (body,) = grouped_manifests(
        configurations, uris, parallel_jobs=1, static=STATIC
    )
    assert body == {"groups": [list(uris)]}


def test_members_that_disagree_structurally_are_different_groups():
    """Two cores is two graphs, and a group is what a graph is shared by.

    The entry would refuse them together, naming the leaf. Splitting them here
    is what keeps that refusal a guard rather than an error anyone meets.
    """

    configurations, uris = round_of(
        configuration(),
        configuration(**{"core.kind": "rtu"}),
        configuration(gamma=0.5),
    )
    (body,) = grouped_manifests(
        configurations, uris, parallel_jobs=1, static=STATIC
    )
    assert body == {"groups": [[uris[0], uris[2]], [uris[1]]]}


def test_a_width_that_differs_also_splits_the_group():
    configurations, uris = round_of(
        configuration(**{"core.lru.hidden_dim": 32}),
        configuration(**{"core.lru.hidden_dim": 64}),
    )
    (body,) = grouped_manifests(
        configurations, uris, parallel_jobs=1, static=STATIC
    )
    assert body == {"groups": [[uris[0]], [uris[1]]]}


def test_more_jobs_than_groups_is_fewer_jobs_rather_than_an_error():
    """A round with one structure is meant to be one job.

    Computing its members together on one device is the whole reason to group
    them, so a `parallel_jobs` larger than the round's structures is an upper
    bound that was not reached -- not a file asking for the impossible.
    """

    configurations, uris = round_of(*(configuration(gamma=g) for g in (0.9, 0.8, 0.7)))
    bodies = grouped_manifests(
        configurations, uris, parallel_jobs=4, static=STATIC
    )
    assert bodies == ({"groups": [list(uris)]},)


def test_groups_are_spread_over_the_jobs_that_were_asked_for():
    configurations, uris = round_of(
        *(configuration(**{"core.lru.hidden_dim": width}) for width in (8, 16, 32, 64))
    )
    bodies = grouped_manifests(
        configurations, uris, parallel_jobs=2, static=STATIC
    )
    assert bodies == (
        {"groups": [[uris[0]], [uris[1]]]},
        {"groups": [[uris[2]], [uris[3]]]},
    )


# ------------------------------------------------------------------ signature


def test_only_the_static_leaves_name_a_group():
    """A discount does not separate two members; it is what they sweep."""

    assert signature(configuration(gamma=0.9), STATIC) == signature(
        configuration(gamma=0.1), STATIC
    )
    assert signature(configuration(), STATIC) != signature(
        configuration(**{"core.kind": "rtu"}), STATIC
    )


def test_a_static_leaf_a_configuration_does_not_carry_is_simply_absent():
    """A branch nobody selected declares parameters nobody set.

    So a signature is over what a configuration has, not over what the image
    could have asked for.
    """

    partial = {"algorithm": {"parameters": {"core.kind": "lru"}}}
    assert signature(partial, STATIC) == (("core.kind", "lru"),)


def test_a_round_whose_uris_do_not_match_its_configurations_is_refused():
    with pytest.raises(RoundError, match="2 configurations for 1 uris"):
        partition((configuration(), configuration()), ("s3://x/a.json",), STATIC)


def test_chunk_divides_as_evenly_as_it_can():
    assert chunk(tuple("abcde"), 2) == (("a", "b", "c"), ("d", "e"))
    assert chunk(tuple("abc"), 3) == (("a",), ("b",), ("c",))
