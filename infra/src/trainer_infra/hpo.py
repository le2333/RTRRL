from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import optuna

from trainer_infra.adapter import KIND
from trainer_infra.bindings import Binding, expand

Direction = Literal["minimize", "maximize"]
Scalar: TypeAlias = bool | int | float | str


@dataclass(frozen=True)
class SampledTrial:
    number: int
    parameters: dict[str, Scalar]


RunRound = Callable[[tuple[SampledTrial, ...]], Sequence[float]]


class StudyError(RuntimeError):
    """A study on disk is not the study this launch describes."""


def sample_parameters(
    trial: optuna.trial.Trial,
    parameters: Mapping[str, Any],
    bindings: Sequence[Binding] = (),
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

    A bound leaf is passed over on the way down and filled afterwards, from one
    draw made under the variable's own name. So the study searches one dimension
    and the point that comes back out still carries an ordinary number at each
    destination -- which is the whole of what sharing a parameter means.
    """

    bound = frozenset(path for binding in bindings for path in binding.paths)
    drawn: dict[str, Scalar] = {}
    reached: set[str] = set()
    _draw(trial, parameters, prefix="", drawn=drawn, bound=bound, reached=reached)
    for binding in bindings:
        destinations = [path for path in binding.paths if path in reached]
        if not destinations:
            continue  # a variable nothing live reads is not a dimension either
        value = _suggest(trial, binding.name, binding.domain)
        for path in destinations:
            drawn[path] = value
    return drawn


def _draw(
    trial: optuna.trial.Trial,
    tree: Mapping[str, Any],
    *,
    prefix: str,
    drawn: dict[str, Scalar],
    bound: frozenset[str],
    reached: set[str],
) -> None:
    groups: dict[str, Mapping[str, Any]] = {}
    for name, node in tree.items():
        path = f"{prefix}{name}"
        if "type" not in node:
            groups[name] = node
        elif path in bound:
            reached.add(path)
        else:
            drawn[path] = _suggest(trial, path, node)

    chosen = drawn.get(f"{prefix}{KIND}")
    for name, group in groups.items():
        if chosen is not None and name != str(chosen):
            continue
        _draw(
            trial,
            group,
            prefix=f"{prefix}{name}.",
            drawn=drawn,
            bound=bound,
            reached=reached,
        )


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
        bindings: Sequence[Binding] = (),
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
        self.bindings = tuple(bindings)
        self.seed = seed
        self.metadata = {} if metadata is None else dict(metadata)
        self._study: optuna.Study | None = None

    def ask(self) -> tuple[SampledTrial, ...]:
        study = self._open()
        trials = tuple(study.ask() for _ in range(self.trials_per_round))
        return tuple(
            SampledTrial(
                number=trial.number,
                parameters=sample_parameters(trial, self.parameters, self.bindings),
            )
            for trial in trials
        )

    def running(self) -> tuple[SampledTrial, ...]:
        """The trials this study asked for and never heard an answer to.

        A trial stays RUNNING until something tells the study otherwise, so
        after a controller dies these are exactly the trials whose work may
        already be finished and unread. Optuna recorded their parameters when
        they were drawn, which is what makes them addressable again.

        What it recorded is the point the study searched, so a shared variable
        comes back under its own name and is written out here to the paths it
        stands for. The runs being settled are the runs that were submitted, and
        those carried the destinations.
        """

        study = self._open()
        return tuple(
            SampledTrial(
                number=trial.number,
                parameters=expand(trial.params, self.bindings),
            )
            for trial in study.get_trials(deepcopy=False)
            if trial.state == optuna.trial.TrialState.RUNNING
        )

    def tell(
        self,
        trials: Sequence[SampledTrial],
        values: Sequence[float],
    ) -> None:
        if len(trials) != len(values):
            raise ValueError(f"received {len(values)} values for {len(trials)} asked trials")
        study = self._open()
        converted = tuple(float(value) for value in values)
        for trial, value in zip(trials, converted, strict=True):
            study.tell(trial.number, value)

    def _fail(self, trials: Sequence[SampledTrial]) -> None:
        study = self._open()
        states = {trial.number: trial.state for trial in study.get_trials(deepcopy=False)}
        for trial in trials:
            if states[trial.number] == optuna.trial.TrialState.RUNNING:
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)

    def run(self, run_round: RunRound) -> optuna.Study:
        """Run each trial round and persist its results before asking the next."""

        study = self._open()
        for _ in range(self.rounds):
            trials = self.ask()
            try:
                self.tell(trials, run_round(trials))
            except Exception:
                self._fail(trials)
                raise

        return study

    def _open(self) -> optuna.Study:
        if self._study is not None:
            return self._study
        self.database.parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            study_name=self.name,
            storage=f"sqlite:///{self.database}",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=self.startup_trials,
                seed=self.seed,
            ),
            direction=self.direction,
            load_if_exists=True,
        )
        self._archive(study)
        self._study = study
        return self._study

    def _archive(self, study: optuna.Study) -> None:
        """Stamp what this launch is, refusing to restamp it as something else.

        The study is loaded where it already exists, so this runs again on every
        resume, and an unconditional write would let an edited file rewrite the
        record of trials that had already been drawn under the old one.

        For a shared parameter that record is not a convenience. The study holds
        one dimension per binding and stores it under the variable's name; which
        paths that name stood for is written nowhere else, so a rewritten
        ``bindings`` makes every trial already in the study unreadable while
        leaving it looking complete. The same is true of the seeds a launch was
        measured on and of the selection it froze, so what is compared is
        everything this side archives rather than the bindings alone.
        """

        stored = study.user_attrs
        changed = sorted(
            key for key, value in self.metadata.items() if key in stored and stored[key] != value
        )
        if changed:
            raise StudyError(
                f"study {self.name!r} already records {changed} differently; it describes "
                "the trials it has already drawn, so resume the launch it belongs to or "
                "name a study of your own"
            )
        for key, value in self.metadata.items():
            study.set_user_attr(key, value)
