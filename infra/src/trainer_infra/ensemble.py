"""The parallel channel: a round's runs handed to one process as a group.

Two channels, side by side. The ordinary one submits a round's runs and the
worker executes them one after another, which is what every launch has done and
what these subclasses leave completely alone. The parallel one packs the same
runs into groups and an ensemble entry computes each group as a single vmapped
graph, filling a device with the sweep rather than with any one run.

Everything except the packing is the ordinary channel's, inherited: the same
configurations published to the same exchange, the same jobs submitted to the
same queue, the same waiting and the same scoring. What a subclass says is one
method long, which is the measure of how little the two differ.

Which channel a launch uses is decided by the entry it names. An image declares
in its catalog whether an entry takes a group, so an experiment asks for the
parallel channel by naming `drqn_ensemble` instead of `drqn`, and there is no
second switch to disagree with the first.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from trainer_infra.batch import BatchRoundExecutor
from trainer_infra.local import LocalRoundExecutor
from trainer_infra.rounds import grouped_manifests


class BatchEnsembleExecutor(BatchRoundExecutor):
    """A Batch round whose jobs each carry groups rather than runs."""

    def __init__(self, *, static: Iterable[str] = (), **arguments: Any) -> None:
        super().__init__(**arguments)
        self.static = frozenset(static)

    def _manifests(
        self,
        configurations: tuple[dict[str, Any], ...],
        config_uris: tuple[str, ...],
    ) -> tuple[dict[str, Any], ...]:
        return grouped_manifests(
            configurations,
            config_uris,
            parallel_jobs=self.parallel_jobs,
            static=self.static,
        )


class LocalEnsembleExecutor(LocalRoundExecutor):
    """A local round run as groups, for the same reason and by the same rule."""

    def __init__(self, *, static: Iterable[str] = (), **arguments: Any) -> None:
        super().__init__(**arguments)
        self.static = frozenset(static)

    def _manifest(
        self,
        configurations: tuple[dict[str, Any], ...],
        config_uris: list[str],
    ) -> dict[str, Any]:
        # One worker here, so one job, whatever a round asked for in parallel.
        (body,) = grouped_manifests(
            configurations, config_uris, parallel_jobs=1, static=self.static
        )
        return body
