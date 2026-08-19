"""Turn one experiment and image catalog into nested run specifications."""

from __future__ import annotations

import statistics
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
    "environment": ("id", "backend", "seeds", "episode_length"),
    "training": ("num_envs", "total_steps", "chunk_steps"),
    "evaluation": ("every_steps", "episodes", "chunk_steps", "seed"),
    "logging": ("aim",),
    "score": ("metric", "window_steps", "reduce", "non_finite", "direction"),
    "hpo": ("rounds", "trials_per_round", "startup_trials", "seed"),
}

# What a formal launch must say about where its configuration came from. The
# block is what turns a launch formal, so it is required only when present.
SELECTION: tuple[str, ...] = ("study", "trial", "tuning_seeds")

TRAINING_PHASE = "train/"


class ExperimentError(ValueError):
    """An experiment file this side cannot turn into run configurations."""


def run_name(configuration: Mapping[str, Any]) -> str:
    """What names one run in an exchange: a configuration and the seed it ran.

    The trial alone stopped being unique when a configuration began running on
    a list of seeds, and two runs writing to one name is one run's result
    reported twice. Padded rather than bare so a directory listing is in the
    order the runs were asked for.
    """

    identity = configuration["identity"]
    return f"trial-{int(identity['trial']):06d}-seed-{int(identity['seed']):06d}"


@dataclass(frozen=True)
class Settlement:
    """What became of one trial that a stopped controller left open."""

    trial: int
    value: float | None = None
    reason: str | None = None


def _seeds(experiment: Mapping[str, Any]) -> tuple[int, ...]:
    """The seeds every configuration is run on, which are not searched.

    A seed is not a hyperparameter: two runs that differ only in it are the
    same configuration measured twice, and letting a sampler draw one would
    have the study spend its budget modelling noise and report the luckiest
    draw as the best setting. So the seeds are listed here, outside ``space``,
    and every configuration is run on all of them.

    The protocol uses that list twice with different lengths. Tuning names one
    seed, and the optimizer is told that run's score directly. The formal
    launch that follows names ten of them, or five on Brax, against a
    configuration already frozen to single values.
    """

    declared = experiment["environment"]["seeds"]
    if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
        raise ExperimentError("environment.seeds must be a list of seeds")
    seeds = tuple(int(seed) for seed in declared)
    if not seeds:
        raise ExperimentError("environment.seeds must name at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ExperimentError(f"environment.seeds repeats a seed: {sorted(seeds)}")
    if any(seed < 0 for seed in seeds):
        raise ExperimentError("environment.seeds must not be negative")
    return seeds


def _frozen(space: Mapping[str, Any]) -> Iterator[str]:
    """Every leaf of a search space that still offers more than one value."""

    for name, node in space.items():
        if isinstance(node, Mapping):
            yield from (f"{name}.{path}" for path in _frozen(node))
        elif (
            not isinstance(node, (str, bytes))
            and isinstance(node, Sequence)
            and len(node) != 1
        ):
            yield name


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
        self.seeds = _seeds(experiment)
        self.selection = self._selected(experiment)
        self.role = "tuning" if self.selection is None else "formal"
        # Which seed produced which score, kept per trial so that a launch
        # reports the runs it paid for and not only what the optimizer heard.
        self.seed_scores: dict[int, dict[int, float]] = {}

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
        self._formal_is_measured()

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
            metadata={
                "role": self.role,
                "seeds": list(self.seeds),
                "evaluation_seed": int(experiment["evaluation"]["seed"]),
                "hpo_seed": int(hpo["seed"]),
                **({} if self.selection is None else {"selection": dict(self.selection)}),
            },
        )

    def _selected(self, experiment: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """The formal launch's provenance, checked against its own seeds.

        A formal launch is one that reports a number, and what makes its
        seeds fresh is that they are not the ones the configuration was chosen
        on. Nothing downstream can tell the two apart -- a seed is an integer
        either way -- so the launch has to say which trial it froze and which
        seeds tuned it, and that claim is what is checked here and archived.
        """

        selection = experiment.get("selection")
        if selection is None:
            return None
        if not isinstance(selection, Mapping):
            raise ExperimentError("selection must name the study, trial and tuning seeds")
        missing = sorted(name for name in SELECTION if name not in selection)
        if missing:
            raise ExperimentError(f"the selection block does not say {missing}")
        tuning = tuple(int(seed) for seed in selection["tuning_seeds"])
        reused = sorted(set(tuning) & set(self.seeds))
        if reused:
            raise ExperimentError(
                f"formal seeds {reused} were already used to tune this configuration; "
                "a formal run measures a choice on seeds that did not make it"
            )
        searched = sorted(_frozen(experiment["space"]))
        if searched:
            raise ExperimentError(
                f"a formal launch runs the configuration it froze, but {searched} "
                "still offer more than one value"
            )
        return selection

    def _formal_is_measured(self) -> None:
        """A formal score is what the policy did when nobody was learning.

        Training return is the learner's own running commentary: it is
        collected under exploration, on the transitions the update just used,
        and it moves with the schedule as much as with the policy. It stays a
        diagnostic, so it cannot be what a formal claim is settled on.
        """

        if self.role == "formal" and self.score.metric.startswith(TRAINING_PHASE):
            raise ExperimentError(
                f"a formal launch cannot be scored on {self.score.metric!r}; "
                "training return is diagnostic, and the primary score is the "
                "fixed evaluation"
            )

    def next_round(self) -> tuple[dict[str, Any], ...]:
        return self._configurations(self.hpo.ask())

    def run(self, round_executor: RoundExecutor) -> Any:
        def run_round(trials: Any) -> tuple[float, ...]:
            results = round_executor(self._configurations(trials), self.score)
            return self._aggregated(trials, results)

        return self.hpo.run(run_round)

    def _aggregated(
        self,
        trials: Any,
        results: Sequence[Mapping[str, int | float]],
    ) -> tuple[float, ...]:
        """One number per trial, from the seeds that configuration was run on.

        The optimizer hears a mean. Under the protocol's tuning launch that is
        one seed's score unchanged; under a formal launch it is the summary of
        a configuration that is no longer being chosen. Either way the seeds'
        own scores are kept, because the mean is the only thing the study
        stores and it is not what a result table reports.
        """

        collected: dict[int, dict[int, float]] = {}
        for result in results:
            trial = int(result["trial"])
            seed = int(result["seed"])
            by_seed = collected.setdefault(trial, {})
            if seed in by_seed:
                raise ExperimentError(
                    f"round returned trial {trial} seed {seed} more than once"
                )
            by_seed[seed] = float(result["value"])
        expected = {trial.number for trial in trials}
        if set(collected) != expected:
            raise ExperimentError(
                f"round returned trials {sorted(collected)}; expected {sorted(expected)}"
            )
        for trial, by_seed in collected.items():
            if set(by_seed) != set(self.seeds):
                raise ExperimentError(
                    f"trial {trial} returned seeds {sorted(by_seed)}; "
                    f"the experiment declares {sorted(self.seeds)}"
                )
        self.seed_scores.update(collected)
        return tuple(
            statistics.fmean(collected[trial.number][seed] for seed in self.seeds)
            for trial in trials
        )

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
            try:
                results = score_round(self._configurations((trial,)), self.score)
                value = self._aggregated((trial,), results)[0]
            except Exception as error:  # noqa: BLE001 - one trial's read decides only it
                reason = f"{type(error).__name__}: {error}"
                settlements.append(Settlement(trial=trial.number, reason=reason))
                continue
            self.hpo.tell((trial,), (value,))
            settlements.append(Settlement(trial=trial.number, value=value))
        return tuple(settlements)

    def _configurations(self, trials: Any) -> tuple[dict[str, Any], ...]:
        """One run per configuration per seed, in the order they were named."""

        return tuple(
            self._configuration(trial, seed) for trial in trials for seed in self.seeds
        )

    def _configuration(self, trial: Any, seed: int) -> dict[str, Any]:
        experiment = self.experiment
        run_id = f"{experiment['name']}-{self.launch_id}-t{trial.number}-s{seed}"
        artifacts = "/".join(
            (
                str(experiment["storage"]).rstrip("/"),
                experiment["experiment"],
                self.launch_id,
                run_id,
            )
        )
        environment = dict(experiment["environment"])
        # The list is the launch's; one run carries the one it was given, and
        # the graph never sees either -- a seed is a budget field.
        environment.pop("seeds")
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
                "seed": seed,
                "role": self.role,
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
                "episodes": evaluation["episodes"],
                "chunk_steps": evaluation["chunk_steps"],
                "seed": evaluation["seed"],
            },
            "logging": logging,
        }
