"""An experiment file, and the run configurations it turns into.

The names here are the worker's. Five blocks are passed through unchanged, so
this side never learns what ``episode_length`` or ``non_finite`` mean -- it
checks that the file said something and hands it on. That is why there is no
schema negotiation between the two sides: there is one shape, written down in
docs/contract.md, and this file is the half that fills it in.

What this side does own is the coordination: which image, which run is which,
and where each run's result goes. None of it is a preference an experiment
could be missing, so all of it is derived here rather than asked for.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trainer_infra.adapter import resolve_parameter_ranges
from trainer_infra.hpo import HPO

RoundExecutor = Callable[
    [tuple[dict[str, Any], ...]],
    Sequence[Mapping[str, int | float]],
]

# Handed to the worker exactly as written. A block here is the worker's shape,
# not this side's, and the round-trip test is what keeps that true.
PASSED_THROUGH: tuple[str, ...] = (
    "environment",
    "training",
    "evaluation",
    "logging",
    "score",
)

# What a file must say before any container starts. Checking it here rather
# than in the worker is the difference between one failure and a round's worth
# of jobs that each start, read the same missing field, and die.
REQUIRED: Mapping[str, tuple[str, ...]] = {
    "": ("experiment", "name", "image", "entry", "storage", "hpo", "space")
    + PASSED_THROUGH,
    "environment": ("id", "backend", "seed", "episode_length"),
    "training": ("num_envs", "total_steps", "epoch_steps"),
    "evaluation": ("steps",),
    "logging": ("aim",),
    "score": ("metric", "window_steps", "reduce", "non_finite", "direction"),
    "hpo": ("rounds", "trials_per_round", "startup_trials", "seed"),
}


class ExperimentError(ValueError):
    """An experiment file this side cannot turn into run configurations."""


def _absent(experiment: Mapping[str, Any]) -> Iterator[str]:
    for block, names in REQUIRED.items():
        section = experiment if not block else experiment.get(block)
        if not isinstance(section, Mapping):
            continue  # the top-level pass already reports the block itself
        for name in names:
            if name not in section:
                yield f"{block}.{name}" if block else name


def _digest(image: str) -> str:
    """The digest an image reference is pinned to, refusing one that is not.

    A tag can be moved. The catalog binds a search space to the image that
    declared it, and a floating tag would let that space change under a study
    that has already recorded trials against the old one.
    """

    _, pinned, digest = image.partition("@")
    if not pinned:
        raise ExperimentError(
            f"image {image!r} is not pinned to a digest; use name@sha256:..."
        )
    return digest


class ExperimentRunner:
    def __init__(
        self,
        *,
        experiment: Mapping[str, Any],
        catalog: Mapping[str, Any],
        database: Path,
        launch_id: str | None = None,
    ) -> None:
        missing = sorted(_absent(experiment))
        if missing:
            raise ExperimentError(f"the experiment file does not say {missing}")

        self.experiment = experiment
        self.launch_id = launch_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.digest = _digest(experiment["image"])

        entry = experiment["entry"]
        try:
            declared = catalog["entries"][entry]["parameters"]
        except KeyError:
            raise ExperimentError(
                f"the image catalog declares no entry {entry!r}"
            ) from None
        # Echoed, not asserted: what the run configurations must claim is
        # whatever the image that will read them implements.
        self.contract = catalog["contract"]

        hpo = experiment["hpo"]
        self.hpo = HPO(
            name=experiment["name"],
            database=database,
            direction=experiment["score"]["direction"],
            rounds=hpo["rounds"],
            trials_per_round=hpo["trials_per_round"],
            startup_trials=hpo["startup_trials"],
            seed=hpo["seed"],
            parameters=resolve_parameter_ranges(declared, experiment["space"]),
        )

    def next_round(self) -> tuple[dict[str, Any], ...]:
        return self._configurations(self.hpo.ask())

    def run(self, round_executor: RoundExecutor) -> Any:
        def run_round(trials: Any) -> tuple[float, ...]:
            results = round_executor(self._configurations(trials))
            values = {result["trial"]: result["value"] for result in results}
            return tuple(float(values[trial.number]) for trial in trials)

        return self.hpo.run(run_round)

    def _configurations(self, trials: Any) -> tuple[dict[str, Any], ...]:
        return tuple(self._configuration(trial) for trial in trials)

    def _configuration(self, trial: Any) -> dict[str, Any]:
        experiment = self.experiment
        run_id = f"{experiment['name']}-{self.launch_id}-t{trial.number}"
        artifacts = "/".join(
            (
                str(experiment["storage"]).rstrip("/"),
                experiment["experiment"],
                self.launch_id,
                run_id,
            )
        )
        blocks = {name: dict(experiment[name]) for name in PASSED_THROUGH}
        blocks["score"]["s3"] = f"{artifacts}/score.json"
        if blocks["logging"].get("enable_rerun"):
            blocks["logging"].setdefault("rerun_s3", f"{artifacts}/rerun")

        return {
            "contract": self.contract,
            "run_id": run_id,
            "experiment": experiment["experiment"],
            "launch_id": self.launch_id,
            "trial": trial.number,
            "entry": experiment["entry"],
            "digest": self.digest,
            "params": dict(trial.parameters),
            **blocks,
        }
