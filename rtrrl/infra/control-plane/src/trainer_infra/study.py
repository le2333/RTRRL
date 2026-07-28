from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import optuna
from optuna.distributions import BaseDistribution, CategoricalDistribution

optuna.logging.set_verbosity(optuna.logging.WARNING)

SAMPLERS = ("tpe", "random", "grid")


def check_sampler(name: str, space: Mapping[str, BaseDistribution]) -> None:
    """Reject a sampler the space cannot be searched with. Called by preflight."""
    if name not in SAMPLERS:
        raise ValueError(f"unsupported sampler {name!r}; use {', '.join(SAMPLERS)}")
    if name != "grid":
        return
    continuous = sorted(
        key for key, dist in space.items() if not isinstance(dist, CategoricalDistribution)
    )
    if continuous:
        raise ValueError(
            f"the grid sampler needs every parameter to be a fixed list of values, "
            f"but these are ranges: {', '.join(continuous)}; either pin them to lists "
            f"or use the tpe or random sampler"
        )


def _sampler(
    name: str,
    space: Mapping[str, BaseDistribution],
    round_size: int,
    seed: int | None,
):
    check_sampler(name, space)
    if name == "tpe":
        # TPE samples at random until it has enough results to fit a density to,
        # and its own default for enough is ten. A launch that asks in rounds has
        # exactly one round it could not have modelled -- the first, which has no
        # results yet -- so that is what it spends: a ten-trial default would have
        # spent half of a twenty-trial budget on random points.
        return optuna.samplers.TPESampler(n_startup_trials=round_size, seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    # A grid is enumerated rather than drawn, so a seed would only shuffle the
    # order it is walked in, and it is walked to the end either way.
    return optuna.samplers.GridSampler(
        {key: list(dist.choices) for key, dist in space.items()}  # type: ignore[attr-defined]
    )


def create_study(
    name: str,
    storage_path: Path,
    sampler: str,
    direction: str,
    user_attrs: Mapping[str, object],
    space: Mapping[str, BaseDistribution],
    round_size: int,
    seed: int | None = None,
) -> optuna.Study:
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{storage_path}",
        sampler=_sampler(sampler, space, round_size, seed),
        direction=direction,
    )
    for key, value in user_attrs.items():
        study.set_user_attr(key, value)
    return study


def ask_round(
    study: optuna.Study, distributions: Mapping[str, BaseDistribution], count: int
) -> list[optuna.trial.Trial]:
    return [study.ask(dict(distributions)) for _ in range(count)]


def tell_value(study: optuna.Study, trial: optuna.trial.Trial, value: float) -> bool:
    """Record a result, and report whether the sampler considers itself done.

    This launch is the optimization loop; it is simply spelled with ask and tell
    instead of ``optimize`` so that the trials can run on Batch. A sampler that
    has nothing left to suggest ends a search by calling ``Study.stop``, which
    raises unless a loop admits to running, so the flag says what is already
    true for the duration of the call. An exhausted grid then reads as a reason
    to stop asking rather than as a crash on the last result.
    """

    local = study._thread_local
    previous, local.in_optimize_loop = local.in_optimize_loop, True
    study._stop_flag = False
    try:
        study.tell(trial, value)
    finally:
        local.in_optimize_loop = previous
    return bool(study._stop_flag)
