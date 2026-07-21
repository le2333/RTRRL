from __future__ import annotations

import warnings
from types import MappingProxyType

import optuna
import pytest

from trainer_infra.models import (
    ContinuousSearch,
    DiscreteSearch,
    ExecutionSpec,
    HpoSpec,
    LoggingSpec,
    ObjectiveSpec,
    ResolvedGroup,
    ResolvedParameter,
    ResourcesSpec,
    TrainingBudgetSpec,
    EnvironmentSpec,
)
from trainer_infra.sampling import (
    DuplicateConfigurationError,
    FiniteSpaceTracker,
    SpaceExhaustedError,
    create_study,
    sample_parameters,
)


class FakeTrial:
    def __init__(self, number: int = 0) -> None:
        self.number = number
        self.calls: list[tuple[str, str, object]] = []

    @property
    def suggested_names(self) -> set[str]:
        return {name for _, name, _ in self.calls}

    def suggest_categorical(self, name: str, choices: tuple[object, ...]) -> object:
        self.calls.append(("categorical", name, choices))
        return choices[-1]

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        self.calls.append(("int", name, (low, high, step, log)))
        return high

    def suggest_float(
        self, name: str, low: float, high: float, *, log: bool = False
    ) -> float:
        self.calls.append(("float", name, (low, high, log)))
        return high


def make_group(
    parameters: dict[str, ResolvedParameter] | None = None,
) -> ResolvedGroup:
    return ResolvedGroup(
        name="shared",
        study_key="experiment-123:shared",
        image="repo/image@sha256:" + "a" * 64,
        script="rtrrl",
        argv=("python", "-m", "train"),
        sdk_protocol_version="1",
        objective=ObjectiveSpec(metric="reward", direction="maximize", reduction="last"),
        environment=EnvironmentSpec(
            name="brax-hopper",
            options={
                "backend": "spring",
                "observation_mode": "P",
                "max_episode_steps": 1000,
            },
        ),
        training_budget=TrainingBudgetSpec(env_steps=100),
        logging=LoggingSpec(aim_every_env_steps=10, rerun_every_episodes=2),
        resources=ResourcesSpec(profile="g6x"),
        hpo=HpoSpec(total_trials=10, configs_per_batch=2),
        execution=ExecutionSpec(runs_per_job=2),
        metadata=MappingProxyType({"owner": "test"}),
        parameters=MappingProxyType(
            parameters
            or {
                "seed": ResolvedParameter(fixed_value=7, search_domain=None),
                "topology": ResolvedParameter(
                    fixed_value=None,
                    search_domain=DiscreteSearch(("shared", "dual")),
                ),
                "layers": ResolvedParameter(
                    fixed_value=None,
                    search_domain=ContinuousSearch(
                        low=2,
                        high=8,
                        log=False,
                        integer=True,
                        step=2,
                    ),
                ),
                "learning_rate": ResolvedParameter(
                    fixed_value=None,
                    search_domain=ContinuousSearch(
                        low=1e-5,
                        high=1e-2,
                        log=True,
                        integer=False,
                        step=None,
                    ),
                ),
            }
        ),
    )


def test_sampling_uses_stable_public_optuna_calls_and_never_suggests_fixed_fields() -> None:
    trial = FakeTrial()

    values = sample_parameters(trial, make_group())

    assert values == {
        "seed": 7,
        "topology": "dual",
        "layers": 8,
        "learning_rate": 1e-2,
    }
    assert "seed" not in trial.suggested_names
    assert trial.calls == [
        ("categorical", "topology", ("shared", "dual")),
        ("int", "layers", (2, 8, 2, False)),
        ("float", "learning_rate", (1e-5, 1e-2, True)),
    ]


def test_integer_log_domain_is_forwarded_to_optuna() -> None:
    trial = FakeTrial()
    group = make_group(
        {
            "layers": ResolvedParameter(
                fixed_value=None,
                search_domain=ContinuousSearch(
                    low=1,
                    high=8,
                    log=True,
                    integer=True,
                    step=1,
                ),
            )
        }
    )

    sample_parameters(trial, group)

    assert trial.calls == [("int", "layers", (1, 8, 1, True))]


def test_integer_log_domain_rejects_non_unit_step() -> None:
    with pytest.raises(ValueError, match="log integer domains require step 1"):
        ContinuousSearch(
            low=1,
            high=8,
            log=True,
            integer=True,
            step=2,
        )


@pytest.mark.parametrize("values", [(True, 1), (1, 1.0), ("same", "same")])
def test_discrete_search_rejects_python_equal_categorical_values(
    values: tuple[object, object],
) -> None:
    with pytest.raises(ValueError, match="duplicate categorical values"):
        DiscreteSearch(values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_discrete_search_rejects_non_finite_float(value: float) -> None:
    with pytest.raises(ValueError, match="categorical float values must be finite"):
        DiscreteSearch((value,))


def test_create_study_uses_constant_liar_tpe_sampler() -> None:
    with warnings.catch_warnings(record=True) as caught:
        study = create_study(study_name="experiment-123:shared", direction="maximize")

    assert isinstance(study.sampler, optuna.samplers.TPESampler)
    assert study.sampler._constant_liar is True
    assert caught == []


def test_finite_space_tracker_reserves_unique_combinations_and_reports_exhaustion() -> None:
    group = make_group(
        {
            "seed": ResolvedParameter(fixed_value=7, search_domain=None),
            "a": ResolvedParameter(
                fixed_value=None, search_domain=DiscreteSearch(("x", "y"))
            ),
            "b": ResolvedParameter(
                fixed_value=None, search_domain=DiscreteSearch((1, 2))
            ),
        }
    )
    tracker = FiniteSpaceTracker(group)

    assert tracker.is_finite
    assert tracker.capacity == 4
    tracker.reserve({"seed": 7, "a": "x", "b": 1})
    with pytest.raises(DuplicateConfigurationError):
        tracker.reserve({"b": 1, "a": "x", "seed": 7})
    tracker.reserve({"a": "x", "b": 2})
    tracker.reserve({"a": "y", "b": 1})
    assert not tracker.exhausted
    tracker.reserve({"a": "y", "b": 2})
    assert tracker.exhausted
    with pytest.raises(SpaceExhaustedError):
        tracker.reserve({"a": "new", "b": 3})


@pytest.mark.parametrize(
    "domain",
    [
        ContinuousSearch(low=1, high=3, log=False, integer=True, step=1),
        ContinuousSearch(low=0.1, high=1.0, log=False, integer=False, step=None),
    ],
)
def test_space_with_any_continuous_domain_never_claims_finite_exhaustion(
    domain: ContinuousSearch,
) -> None:
    tracker = FiniteSpaceTracker(
        make_group(
            {
                "choice": ResolvedParameter(
                    fixed_value=None, search_domain=DiscreteSearch(("x", "y"))
                ),
                "range": ResolvedParameter(fixed_value=None, search_domain=domain),
            }
        )
    )

    assert not tracker.is_finite
    assert tracker.capacity is None
    tracker.reserve({"choice": "x", "range": domain.low})
    assert not tracker.exhausted
    with pytest.raises(DuplicateConfigurationError):
        tracker.reserve({"range": domain.low, "choice": "x"})
