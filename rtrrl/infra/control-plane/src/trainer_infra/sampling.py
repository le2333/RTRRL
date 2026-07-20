from __future__ import annotations

import math
import warnings
from threading import Lock
from typing import Mapping

import optuna
from optuna.study import StudyDirection
from optuna.trial import Trial

from trainer_infra.identities import canonical_json
from trainer_infra.models import (
    ContinuousSearch,
    DiscreteSearch,
    JsonScalar,
    ResolvedGroup,
)


class DuplicateConfigurationError(RuntimeError):
    """The sampled searchable configuration was already allocated."""


class SpaceExhaustedError(RuntimeError):
    """Every configuration in a finite discrete space was allocated."""


class FiniteSpaceTracker:
    def __init__(self, group: ResolvedGroup) -> None:
        self._searchable = tuple(group.searchable_parameters())
        domains = tuple(group.searchable_parameters().values())
        self._finite = all(isinstance(domain, DiscreteSearch) for domain in domains)
        self._capacity = (
            math.prod(
                len({canonical_json(value) for value in domain.values})
                for domain in domains
                if isinstance(domain, DiscreteSearch)
            )
            if self._finite
            else None
        )
        self._reserved: set[str] = set()
        self._lock = Lock()

    @property
    def is_finite(self) -> bool:
        return self._finite

    @property
    def capacity(self) -> int | None:
        return self._capacity

    @property
    def allocated(self) -> int:
        return len(self._reserved)

    @property
    def exhausted(self) -> bool:
        return self._capacity is not None and self.allocated >= self._capacity

    def reserve(self, parameters: Mapping[str, JsonScalar]) -> None:
        try:
            searchable = {name: parameters[name] for name in self._searchable}
        except KeyError as error:
            raise ValueError(f"sampled configuration is missing field {error.args[0]!r}") from error
        key = canonical_json(searchable)
        with self._lock:
            if key in self._reserved:
                raise DuplicateConfigurationError(
                    "sampled configuration has already been allocated"
                )
            if self._capacity is not None and len(self._reserved) >= self._capacity:
                raise SpaceExhaustedError("finite discrete search space is exhausted")
            self._reserved.add(key)


def create_study(
    *,
    study_name: str,
    direction: str | StudyDirection,
    storage: str | optuna.storages.BaseStorage | None = None,
    load_if_exists: bool = False,
) -> optuna.Study:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(constant_liar=True)
    return optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        storage=storage,
        load_if_exists=load_if_exists,
    )


def sample_parameters(
    trial: Trial,
    group: ResolvedGroup,
    *,
    tracker: FiniteSpaceTracker | None = None,
) -> dict[str, JsonScalar]:
    values = dict(group.fixed_parameters)
    for name, domain in group.searchable_parameters().items():
        if isinstance(domain, DiscreteSearch):
            values[name] = trial.suggest_categorical(name, domain.values)
        elif isinstance(domain, ContinuousSearch) and domain.integer:
            values[name] = trial.suggest_int(
                name,
                int(domain.low),
                int(domain.high),
                step=int(domain.step or 1),
            )
        else:
            values[name] = trial.suggest_float(
                name,
                domain.low,
                domain.high,
                log=domain.log,
            )
    if tracker is not None:
        tracker.reserve(values)
    return values
