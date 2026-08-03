from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sampler(round_size: int, seed: int | None):
    # TPE samples at random until it has enough results to fit a density to,
    # and its own default for enough is ten. A launch that asks in rounds has
    # exactly one round it could not have modelled -- the first, which has no
    # results yet -- so that is what it spends: a ten-trial default would have
    # spent half of a twenty-trial budget on random points.
    return optuna.samplers.TPESampler(n_startup_trials=round_size, seed=seed)


def create_study(
    name: str,
    storage_path: Path,
    direction: str,
    user_attrs: Mapping[str, object],
    round_size: int,
    seed: int | None = None,
) -> optuna.Study:
    """A study searched by TPE.

    One sampler, because only one of the three optimises. A grid enumerates and
    random draws; neither reads the scores it has already paid for, so neither
    is doing what an HPO loop is for. TPE also takes a range and a fixed set
    alike, which is what lets a declaration say which a parameter is without the
    search having an opinion about it.
    """

    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{storage_path}",
        sampler=_sampler(round_size, seed),
        direction=direction,
    )
    for key, value in user_attrs.items():
        study.set_user_attr(key, value)
    return study


def ask_round(study: optuna.Study, count: int) -> list[optuna.trial.Trial]:
    return [study.ask() for _ in range(count)]


def tell_value(study: optuna.Study, trial: optuna.trial.Trial, value: float) -> None:
    """Record a result.

    This launch is the optimization loop; it is simply spelled with ask and tell
    instead of ``optimize`` so that the trials can run on Batch.
    """

    study.tell(trial, value)
