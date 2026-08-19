"""Turn one experiment and image catalog into nested run specifications."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trainer_infra.adapter import resolve_parameter_ranges
from trainer_infra.hpo import HPO
from trainer_infra.scoring import ScoreSpec

RoundExecutor = Callable[
    [tuple[dict[str, Any], ...], ScoreSpec],
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
    "training": ("num_envs", "total_steps", "chunk_steps"),
    "evaluation": ("every_steps", "rollout_steps"),
    "logging": ("aim",),
    "score": ("metric", "window_steps", "reduce", "non_finite", "direction"),
    "hpo": ("rounds", "trials_per_round", "startup_trials", "seed"),
}


class ExperimentError(ValueError):
    """An experiment file this side cannot turn into run configurations."""


@dataclass(frozen=True)
class Settlement:
    """What became of one trial that a stopped controller left open."""

    trial: int
    value: float | None = None
    reason: str | None = None


def _formal(experiment: Mapping[str, Any]) -> None:
    """Check the seed a formal launch runs is one of the ones it declared fresh.

    A formal result is a claim about seeds that had no part in choosing the
    configuration, and the way that goes wrong is quiet: a file copied from the
    tuning launch keeps the tuning seed, and five runs of it report a
    hyper-parameter search's own seed as an independent sample. So the file
    names the fresh set and the seed it is launching, and the two are checked
    against each other before anything starts.

    What this does not do is archive the identity of a launch -- which seed
    produced which artifact, kept somewhere a result can be traced back to.
    That belongs with the rest of the formal protocol and is not invented here.
    """

    declared = experiment.get("formal")
    if declared is None:
        return
    missing = [name for name in ("seeds", "tuning_seed") if name not in declared]
    if missing:
        raise ExperimentError(f"the formal block does not say {sorted(missing)}")
    seeds = [int(seed) for seed in declared["seeds"]]
    tuning = int(declared["tuning_seed"])
    if not seeds:
        raise ExperimentError("a formal block declaring no seed measures nothing")
    if len(set(seeds)) != len(seeds):
        raise ExperimentError(f"the formal seeds {seeds} are not distinct")
    if tuning in seeds:
        raise ExperimentError(
            f"formal seed {tuning} is the tuning seed; a configuration cannot be "
            "evaluated on the seed that chose it"
        )
    running = int(experiment["environment"]["seed"])
    if running not in seeds:
        raise ExperimentError(
            f"this launch runs seed {running}, which the formal set {seeds} does "
            "not contain; one launch runs one of the declared fresh seeds"
        )


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
        _formal(experiment)

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
        self.score = ScoreSpec.from_mapping(experiment["score"])

        hpo = experiment["hpo"]
        self.hpo = HPO(
            name=experiment["name"],
            database=database,
            direction=self.score.direction,
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
            results = round_executor(self._configurations(trials), self.score)
            values: dict[int, float] = {}
            for result in results:
                trial = int(result["trial"])
                if trial in values:
                    raise ExperimentError(f"round returned trial {trial} more than once")
                values[trial] = float(result["value"])
            expected = {trial.number for trial in trials}
            if set(values) != expected:
                raise ExperimentError(
                    f"round returned trials {sorted(values)}; expected {sorted(expected)}"
                )
            return tuple(float(values[trial.number]) for trial in trials)

        return self.hpo.run(run_round)

    def settle(self, score_round: RoundExecutor) -> tuple[Settlement, ...]:
        """Read the results of trials the study is still waiting on.

        A controller killed between a worker's last upload and ``study.tell``
        leaves the trial RUNNING with its result already in storage: the
        training is finished and paid for, and only the reading is missing.
        Settling asks the same executor for a score without submitting
        anything, which is what makes re-running the worker unnecessary.

        The launch id must be the one the trials ran under, since it is what
        names their artifacts. Each trial is read on its own, so a trial whose
        work has genuinely not finished is reported and left running rather
        than deciding the outcome for the others.
        """

        settlements = []
        for trial in self.hpo.running():
            configuration = self._configuration(trial)
            try:
                results = score_round((configuration,), self.score)
            except Exception as error:  # noqa: BLE001 - one trial's read decides only it
                reason = f"{type(error).__name__}: {error}"
                settlements.append(Settlement(trial=trial.number, reason=reason))
                continue
            value = float(next(iter(results))["value"])
            self.hpo.tell((trial,), (value,))
            settlements.append(Settlement(trial=trial.number, value=value))
        return tuple(settlements)

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
        evaluation = experiment["evaluation"]
        # A block that is present is a destination that is on. There is no
        # separate switch beside a value for it to disagree with.
        declared = experiment["logging"]
        aim: dict[str, Any] = {"url": declared["aim"]["url"]}
        if "training" in declared["aim"]:
            # A scope block per scope asked for, each with its own interval.
            # Copied one level deeper than the block itself so the emitted
            # configuration shares nothing with the experiment document.
            aim["training"] = {
                scope: dict(interval)
                for scope, interval in declared["aim"]["training"].items()
            }
        logging: dict[str, Any] = {"aim": aim}
        if "rerun" in declared:
            logging["rerun"] = dict(declared["rerun"])

        # Optional, and copied rather than defaulted: a run that says nothing
        # about checkpoints files none, which is the ordinary case. An R2 run
        # that may need forking has to say so before it starts, because the
        # boundary a fork wants is decided from a collapse it has not had yet.
        checkpoint = (
            {"checkpoint": dict(experiment["checkpoint"])}
            if "checkpoint" in experiment
            else {}
        )

        return {
            "contract": self.contract,
            **checkpoint,
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
            "training": {
                "seed": seed,
                "total_steps": training["total_steps"],
                "chunk_steps": training["chunk_steps"],
            },
            "evaluation": {
                "every_steps": evaluation["every_steps"],
                "rollout_steps": evaluation["rollout_steps"],
            },
            "logging": logging,
        }
