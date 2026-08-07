from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import optuna

from trainer_infra.adapter import KIND

Direction = Literal["minimize", "maximize"]
Scalar: TypeAlias = bool | int | float | str


@dataclass(frozen=True)
class SampledTrial:
    number: int
    parameters: dict[str, Scalar]


RunRound = Callable[[tuple[SampledTrial, ...]], Sequence[float]]


def sample_parameters(
    trial: optuna.trial.Trial,
    parameters: Mapping[str, Any],
) -> dict[str, Scalar]:
    """Draw one point, descending only the branches this trial chose.

    The tree is the conditional structure, so drawing has to follow it: a
    parameter under a branch exists for the trials that took that branch and
    for no others. Sampling the whole tree instead would put the actor's Adam
    betas in every trial that runs SGD -- dimensions nothing reads, which the
    sampler must still model, and which the run configuration would then carry
    as values that look chosen.

    This is the same walk the worker does when it builds the graph. The only
    difference is where ``kind`` comes from: there it is read out of a
    configuration, here it is drawn a line above the branch it opens.
    """

    drawn: dict[str, Scalar] = {}
    _draw(trial, parameters, prefix="", drawn=drawn)
    return drawn


def _draw(
    trial: optuna.trial.Trial,
    tree: Mapping[str, Any],
    *,
    prefix: str,
    drawn: dict[str, Scalar],
) -> None:
    groups: dict[str, Mapping[str, Any]] = {}
    for name, node in tree.items():
        path = f"{prefix}{name}"
        if "type" in node:
            drawn[path] = _suggest(trial, path, node)
        else:
            groups[name] = node

    chosen = drawn.get(f"{prefix}{KIND}")
    for name, group in groups.items():
        if chosen is not None and name != str(chosen):
            continue
        _draw(trial, group, prefix=f"{prefix}{name}.", drawn=drawn)


def _suggest(
    trial: optuna.trial.Trial,
    name: str,
    parameter_range: Mapping[str, Any],
) -> Scalar:
    if parameter_range["type"] == "choice":
        return trial.suggest_categorical(name, parameter_range["values"])
    if parameter_range["type"] == "float":
        return trial.suggest_float(
            name,
            parameter_range["low"],
            parameter_range["high"],
            log=parameter_range.get("log", False),
        )
    return trial.suggest_int(
        name,
        parameter_range["low"],
        parameter_range["high"],
        step=parameter_range.get("step", 1),
        log=parameter_range.get("log", False),
    )


class HPO:
    """Own a persistent, round-based Optuna optimization lifecycle."""

    def __init__(
        self,
        *,
        name: str,
        database: Path,
        direction: Direction,
        rounds: int,
        trials_per_round: int,
        startup_trials: int,
        parameters: Mapping[str, Any],
        seed: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.name = name
        self.database = Path(database)
        self.direction = direction
        self.rounds = rounds
        self.trials_per_round = trials_per_round
        self.startup_trials = startup_trials
        self.parameters = dict(parameters)
        self.seed = seed
        self.metadata = {} if metadata is None else dict(metadata)
        self._study: optuna.Study | None = None

    def ask(self) -> tuple[SampledTrial, ...]:
        study = self._open()
        trials = tuple(study.ask() for _ in range(self.trials_per_round))
        return tuple(
            SampledTrial(
                number=trial.number,
                parameters=sample_parameters(trial, self.parameters),
            )
            for trial in trials
        )

    def tell(
        self,
        trials: Sequence[SampledTrial],
        values: Sequence[float],
    ) -> None:
        study = self._open()
        for trial, value in zip(trials, values):
            study.tell(trial.number, float(value))

    def run(self, run_round: RunRound) -> optuna.Study:
        """Run each trial round and persist its results before asking the next."""

        study = self._open()
        for _ in range(self.rounds):
            trials = self.ask()
            self.tell(trials, run_round(trials))

        return study

    def _open(self) -> optuna.Study:
        if self._study is not None:
            return self._study
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._study = optuna.create_study(
            study_name=self.name,
            storage=f"sqlite:///{self.database}",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=self.startup_trials,
                seed=self.seed,
            ),
            direction=self.direction,
            load_if_exists=True,
        )
        for key, value in self.metadata.items():
            self._study.set_user_attr(key, value)
        return self._study
