"""Packing a round's runs into groups that can share one graph.

An ordinary round is a list of runs and the worker takes them one at a time.
A grouped round hands several to one process, which computes them as a single
vmapped graph -- so the members of a group have to agree about everything that
graph was built from.

What they must agree about is what the image declares ``static``: the widths
that size an array and the choices that select a branch. A discount is not among
them, which is the point -- that is the axis a sweep moves.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


class RoundError(RuntimeError):
    """A round cannot be packed into the jobs it was asked for."""


def chunk(values: Sequence[Any], count: int) -> tuple[tuple[Any, ...], ...]:
    """``values`` split into ``count`` runs of as even a size as divides."""

    if count < 1 or count > len(values):
        raise RoundError("parallel_jobs must be between one and the number of trials")
    base, remainder = divmod(len(values), count)
    groups = []
    offset = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        groups.append(tuple(values[offset : offset + size]))
        offset += size
    return tuple(groups)


def signature(
    configuration: Mapping[str, Any], static: Iterable[str]
) -> tuple[tuple[str, Any], ...]:
    """What two members of one group must agree about.

    Over what the configuration carries rather than over what the image could
    have asked for: a branch nobody selected declares parameters nobody set.
    """

    parameters = configuration["algorithm"]["parameters"]
    return tuple(
        sorted((name, parameters[name]) for name in static if name in parameters)
    )


def partition(
    configurations: Sequence[Mapping[str, Any]],
    config_uris: Sequence[str],
    static: Iterable[str],
) -> tuple[tuple[str, ...], ...]:
    """The round's runs, gathered into the groups that can share a graph.

    Order is kept: within a group the runs stay in the order the round produced
    them, and the groups themselves in the order their first member appeared.
    A member's own results do not depend on either, but a manifest that reshuffled
    them would make two launches of one experiment harder to compare by eye.
    """

    if len(configurations) != len(config_uris):
        raise RoundError(
            f"{len(configurations)} configurations for {len(config_uris)} uris"
        )
    groups: dict[tuple[tuple[str, Any], ...], list[str]] = {}
    for configuration, uri in zip(configurations, config_uris, strict=True):
        groups.setdefault(signature(configuration, static), []).append(uri)
    return tuple(tuple(members) for members in groups.values())


def grouped_manifests(
    configurations: Sequence[Mapping[str, Any]],
    config_uris: Sequence[str],
    *,
    parallel_jobs: int,
    static: Iterable[str] = (),
) -> tuple[dict[str, Any], ...]:
    """One manifest body per job, with the round's groups spread across them.

    At most ``parallel_jobs`` jobs, and fewer when the round has fewer groups
    than that. Asking for more parallelism than a grouped round contains is not
    an error, as it is for an ungrouped one: a round of a single structure is
    *meant* to be one job, since computing its members together on one device is
    the whole reason to group them.
    """

    groups = partition(configurations, config_uris, static)
    return tuple(
        {"groups": [list(members) for members in job]}
        for job in chunk(groups, min(parallel_jobs, len(groups)))
    )
