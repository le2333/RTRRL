"""Turn one experiment and image catalog into nested run specifications."""

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

# What a file must say before any container starts. Checking it here rather
# than in the worker is the difference between one failure and a round's worth
# of jobs that each start, read the same missing field, and die.
REQUIRED: Mapping[str, tuple[str, ...]] = {
    "": (
        "experiment",
        "name",
        "image",
        "entry",
        "storage",
        "hpo",
        "space",
        "environment",
        "training",
        "evaluation",
        "logging",
        "score",
    ),
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
        raise ExperimentError(f"image {image!r} is not pinned to a digest; use name@sha256:...")
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
            descriptor = catalog["entries"][entry]
        except KeyError:
            raise ExperimentError(f"the image catalog declares no entry {entry!r}") from None
        metric = experiment["score"]["metric"]
        if metric not in descriptor["metrics"]:
            raise ExperimentError(f"entry {entry!r} declares no score metric {metric!r}")
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
            parameters=resolve_parameter_ranges(descriptor["parameters"], experiment["space"]),
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
        environment = dict(experiment["environment"])
        seed = environment.pop("seed")
        training = experiment["training"]
        logging = {"aim": {"url": experiment["logging"]["aim"]}}
        if experiment["logging"].get("enable_rerun"):
            logging["rerun"] = {"every_steps": experiment["logging"]["rerun_every_steps"]}

        return {
            "contract": self.contract,
            "identity": {
                "run_id": run_id,
                "experiment": experiment["experiment"],
                "launch_id": self.launch_id,
                "trial": trial.number,
                "digest": self.digest,
            },
            "entry": experiment["entry"],
            "artifacts": {"root": artifacts},
            "algorithm": {
                "environment": environment,
                "num_envs": training["num_envs"],
                "parameters": dict(trial.parameters),
            },
            "runtime": {
                "seed": seed,
                "total_steps": training["total_steps"],
                "epoch_steps": training["epoch_steps"],
                "evaluation_steps": experiment["evaluation"]["steps"],
            },
            "logging": logging,
        }
