from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from trainer_infra.adapter import resolve_parameter_ranges
from trainer_infra.hpo import HPO

RoundExecutor = Callable[
    [tuple[dict[str, Any], ...]],
    Sequence[Mapping[str, int | float]],
]


class ExperimentRunner:
    def __init__(
        self,
        *,
        experiment: Mapping[str, Any],
        catalog: Mapping[str, Any],
        database: Path,
    ) -> None:
        self.experiment = experiment
        declared = catalog["algorithms"][experiment["algorithm"]]["parameters"]
        parameters = resolve_parameter_ranges(declared, experiment["parameters"])
        hpo = experiment["hpo"]
        self.hpo = HPO(
            name=experiment["name"],
            database=database,
            direction=experiment["objective"]["direction"],
            rounds=hpo["rounds"],
            trials_per_round=hpo["trials_per_round"],
            startup_trials=hpo["startup_trials"],
            seed=hpo["seed"],
            parameters=parameters,
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
        return tuple(
            {
                "experiment": self.experiment["name"],
                "trial": trial.number,
                "algorithm": self.experiment["algorithm"],
                "run": self.experiment["run"],
                "objective": self.experiment["objective"],
                "loggers": self.experiment["loggers"],
                "parameters": trial.parameters,
            }
            for trial in trials
        )
