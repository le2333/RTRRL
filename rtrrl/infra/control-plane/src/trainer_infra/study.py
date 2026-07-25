from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import optuna
from optuna.distributions import BaseDistribution

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sampler(name: str, space: Mapping[str, BaseDistribution] | None = None):
    if name == "tpe":
        return optuna.samplers.TPESampler()
    if name == "random":
        return optuna.samplers.RandomSampler()
    if name == "grid":
        if space is None:
            raise ValueError("grid sampler requires the search space")
        return optuna.samplers.GridSampler(
            {key: list(dist.choices) for key, dist in space.items()}  # type: ignore[attr-defined]
        )
    raise ValueError(f"unsupported sampler {name!r}; use tpe, random or grid")


def create_study(
    name: str,
    storage_path: Path,
    sampler: str,
    direction: str,
    user_attrs: Mapping[str, object],
    space: Mapping[str, BaseDistribution] | None = None,
) -> optuna.Study:
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{storage_path}",
        sampler=_sampler(sampler, space),
        direction=direction,
    )
    for key, value in user_attrs.items():
        study.set_user_attr(key, value)
    return study


def ask_round(
    study: optuna.Study, distributions: Mapping[str, BaseDistribution], count: int
) -> list[optuna.trial.Trial]:
    return [study.ask(dict(distributions)) for _ in range(count)]


def tell_value(study: optuna.Study, trial: optuna.trial.Trial, value: float) -> None:
    study.tell(trial, value)
