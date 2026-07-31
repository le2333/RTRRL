from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import yaml
from training_sdk.contract import RunConfig

FailureMode = Literal["none", "before_training", "after_training", "after_checkpoint"]
type FrozenValue = (
    str | int | float | bool | None | tuple[FrozenValue, ...] | Mapping[str, FrozenValue]
)

_FAILURE_MODES = frozenset({"none", "before_training", "after_training", "after_checkpoint"})


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        # All document-schema violations deliberately share the load() ValueError contract.
        raise ValueError(f"{location} must be a mapping")  # noqa: TRY004
    if not all(type(key) is str for key in value):
        raise ValueError(f"{location} keys must be strings")
    return cast(Mapping[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{location} keys mismatch: missing={missing}, extra={extra}")


def _integer(value: object, location: str, *, positive: bool) -> int:
    if type(value) is not int:
        raise ValueError(f"{location} must be an integer")
    result = cast(int, value)
    if positive and result <= 0:
        raise ValueError(f"{location} must be positive")
    if not positive and result < 0:
        raise ValueError(f"{location} must be non-negative")
    return result


def _finite_number(value: object, location: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{location} must be a finite number")
    try:
        result = float(cast(int | float, value))
    except OverflowError as error:
        raise ValueError(f"{location} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{location} must be a finite number")
    return result


def _enabled(environ: Mapping[str, str], name: str) -> bool:
    value = environ.get(name)
    if value is None:
        return False
    if value != "1":
        raise ValueError(f"{name} must be exactly '1' when set")
    return True


def _freeze(value: Any) -> FrozenValue:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return cast(str | int | float | bool | None, value)


@dataclass(frozen=True, slots=True, init=False)
class AcceptanceConfig:
    protocol_version: Literal["1"]
    environment: Mapping[str, FrozenValue]
    logging: Mapping[str, FrozenValue]
    parameters: Mapping[str, FrozenValue]
    training_budget: Mapping[str, FrozenValue]
    fast_mode: bool = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("AcceptanceConfig instances must be created with AcceptanceConfig.load()")

    @classmethod
    def from_run_config(
        cls,
        config: RunConfig,
        *,
        environ: Mapping[str, str] = os.environ,
    ) -> AcceptanceConfig:
        params = config.params
        if not isinstance(params, Mapping):
            raise TypeError("params must be a mapping")
        if not all(type(key) is str for key in params):
            raise ValueError("params keys must be strings")

        namespace, separator, environment_name = config.environment.id.partition("::")
        if separator != "::" or namespace != "brax" or not environment_name:
            raise ValueError("environment.id must be a qualified Brax environment")
        if config.environment.observed is not None:
            raise ValueError("environment.observed is not supported")
        environment = {
            "name": environment_name,
            "options": {"backend": config.environment.backend},
        }
        logging = {"aim_every_env_steps": 1, "rerun_every_episodes": 1}
        algorithm = {
            "learning_rate": params["learning_rate"],
            "num_envs": config.environment.num_envs,
            "episode_length": params.get("episode_length", 32),
            "failure_mode": params.get("failure_mode", "none"),
        }
        parameters = {
            "runtime": {"seed": params["seed"]},
            "algorithm": algorithm,
        }
        training_budget = {"env_steps": config.budget.total_steps}

        _exact_keys(
            {"environment": environment, "logging": logging, "parameters": parameters, "training_budget": training_budget},
            {"environment", "logging", "parameters", "training_budget"},
            "params",
        )
        _exact_keys(environment, {"name", "options"}, "environment")
        if environment["name"] != "inverted_pendulum":
            raise ValueError("environment.name must be exactly 'inverted_pendulum'")
        options = _mapping(environment["options"], "environment.options")
        _exact_keys(options, {"backend"}, "environment.options")
        if options["backend"] != "generalized":
            raise ValueError("environment.options.backend must be exactly 'generalized'")
        _exact_keys(logging, {"aim_every_env_steps", "rerun_every_episodes"}, "logging")
        _integer(logging["aim_every_env_steps"], "logging.aim_every_env_steps", positive=True)
        _integer(logging["rerun_every_episodes"], "logging.rerun_every_episodes", positive=True)
        _exact_keys(parameters, {"runtime", "algorithm"}, "parameters")
        runtime = _mapping(parameters["runtime"], "parameters.runtime")
        _exact_keys(runtime, {"seed"}, "parameters.runtime")
        _integer(runtime["seed"], "parameters.runtime.seed", positive=False)
        algorithm_mapping = _mapping(parameters["algorithm"], "parameters.algorithm")
        _exact_keys(
            algorithm_mapping,
            {"learning_rate", "num_envs", "episode_length", "failure_mode"},
            "parameters.algorithm",
        )
        learning_rate = _finite_number(
            algorithm_mapping["learning_rate"], "parameters.algorithm.learning_rate"
        )
        if learning_rate <= 0:
            raise ValueError("parameters.algorithm.learning_rate must be positive")
        _integer(algorithm_mapping["num_envs"], "parameters.algorithm.num_envs", positive=True)
        _integer(
            algorithm_mapping["episode_length"],
            "parameters.algorithm.episode_length",
            positive=True,
        )
        failure_mode = algorithm_mapping["failure_mode"]
        if type(failure_mode) is not str or failure_mode not in _FAILURE_MODES:
            raise ValueError(
                f"parameters.algorithm.failure_mode must be one of {sorted(_FAILURE_MODES)}"
            )
        budget = _mapping(training_budget, "training_budget")
        _exact_keys(budget, {"env_steps"}, "training_budget")
        _integer(budget["env_steps"], "training_budget.env_steps", positive=True)

        test_mode = _enabled(environ, "BRAX_ACCEPTANCE_TEST_MODE")
        fast_mode = _enabled(environ, "BRAX_ACCEPTANCE_E2E_FAST")
        if failure_mode != "none" and not test_mode:
            raise ValueError("non-none failure_mode requires BRAX_ACCEPTANCE_TEST_MODE=1")
        if fast_mode and not test_mode:
            raise ValueError("BRAX_ACCEPTANCE_E2E_FAST=1 requires BRAX_ACCEPTANCE_TEST_MODE=1")

        return cls._create(
            environment=environment,
            logging=logging,
            parameters=parameters,
            training_budget=training_budget,
            fast_mode=fast_mode,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        environ: Mapping[str, str] = os.environ,
    ) -> AcceptanceConfig:
        if not isinstance(environ, Mapping):
            raise TypeError("environ must be a mapping")
        try:
            loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"could not load acceptance config: {error}") from error

        root = _mapping(loaded, "config")
        _exact_keys(
            root,
            {"protocol_version", "environment", "logging", "parameters", "training_budget"},
            "config",
        )

        if root["protocol_version"] != "1":
            raise ValueError("protocol_version must be exactly '1'")

        environment = _mapping(root["environment"], "environment")
        _exact_keys(environment, {"name", "options"}, "environment")
        if environment["name"] != "inverted_pendulum":
            raise ValueError("environment.name must be exactly 'inverted_pendulum'")

        options = _mapping(environment["options"], "environment.options")
        _exact_keys(options, {"backend"}, "environment.options")
        if options["backend"] != "generalized":
            raise ValueError("environment.options.backend must be exactly 'generalized'")

        logging = _mapping(root["logging"], "logging")
        _exact_keys(logging, {"aim_every_env_steps", "rerun_every_episodes"}, "logging")
        _integer(logging["aim_every_env_steps"], "logging.aim_every_env_steps", positive=True)
        _integer(logging["rerun_every_episodes"], "logging.rerun_every_episodes", positive=True)

        parameters = _mapping(root["parameters"], "parameters")
        _exact_keys(parameters, {"runtime", "algorithm"}, "parameters")

        runtime = _mapping(parameters["runtime"], "parameters.runtime")
        _exact_keys(runtime, {"seed"}, "parameters.runtime")
        _integer(runtime["seed"], "parameters.runtime.seed", positive=False)

        algorithm = _mapping(parameters["algorithm"], "parameters.algorithm")
        _exact_keys(
            algorithm,
            {"learning_rate", "num_envs", "episode_length", "failure_mode"},
            "parameters.algorithm",
        )
        learning_rate = _finite_number(
            algorithm["learning_rate"], "parameters.algorithm.learning_rate"
        )
        if learning_rate <= 0:
            raise ValueError("parameters.algorithm.learning_rate must be positive")
        _integer(algorithm["num_envs"], "parameters.algorithm.num_envs", positive=True)
        _integer(algorithm["episode_length"], "parameters.algorithm.episode_length", positive=True)
        failure_mode = algorithm["failure_mode"]
        if type(failure_mode) is not str or failure_mode not in _FAILURE_MODES:
            raise ValueError(f"parameters.algorithm.failure_mode must be one of {sorted(_FAILURE_MODES)}")

        training_budget = _mapping(root["training_budget"], "training_budget")
        _exact_keys(training_budget, {"env_steps"}, "training_budget")
        _integer(training_budget["env_steps"], "training_budget.env_steps", positive=True)

        test_mode = _enabled(environ, "BRAX_ACCEPTANCE_TEST_MODE")
        fast_mode = _enabled(environ, "BRAX_ACCEPTANCE_E2E_FAST")
        if failure_mode != "none" and not test_mode:
            raise ValueError("non-none failure_mode requires BRAX_ACCEPTANCE_TEST_MODE=1")
        if fast_mode and not test_mode:
            raise ValueError("BRAX_ACCEPTANCE_E2E_FAST=1 requires BRAX_ACCEPTANCE_TEST_MODE=1")

        return cls._create(
            environment=environment,
            logging=logging,
            parameters=parameters,
            training_budget=training_budget,
            fast_mode=fast_mode,
        )

    @classmethod
    def _create(
        cls,
        *,
        environment: Mapping[str, Any],
        logging: Mapping[str, Any],
        parameters: Mapping[str, Any],
        training_budget: Mapping[str, Any],
        fast_mode: bool,
    ) -> AcceptanceConfig:
        instance = object.__new__(cls)
        object.__setattr__(instance, "protocol_version", "1")
        object.__setattr__(
            instance, "environment", cast(Mapping[str, FrozenValue], _freeze(environment))
        )
        object.__setattr__(instance, "logging", cast(Mapping[str, FrozenValue], _freeze(logging)))
        object.__setattr__(
            instance, "parameters", cast(Mapping[str, FrozenValue], _freeze(parameters))
        )
        object.__setattr__(
            instance,
            "training_budget",
            cast(Mapping[str, FrozenValue], _freeze(training_budget)),
        )
        object.__setattr__(instance, "fast_mode", fast_mode)
        return instance

    @property
    def environment_name(self) -> str:
        return cast(str, self.environment["name"])

    @property
    def backend(self) -> str:
        options = cast(Mapping[str, FrozenValue], self.environment["options"])
        return cast(str, options["backend"])

    @property
    def aim_every_env_steps(self) -> int:
        return cast(int, self.logging["aim_every_env_steps"])

    @property
    def rerun_every_episodes(self) -> int:
        return cast(int, self.logging["rerun_every_episodes"])

    @property
    def seed(self) -> int:
        runtime = cast(Mapping[str, FrozenValue], self.parameters["runtime"])
        return cast(int, runtime["seed"])

    @property
    def learning_rate(self) -> float:
        algorithm = cast(Mapping[str, FrozenValue], self.parameters["algorithm"])
        return float(cast(int | float, algorithm["learning_rate"]))

    @property
    def num_envs(self) -> int:
        algorithm = cast(Mapping[str, FrozenValue], self.parameters["algorithm"])
        return cast(int, algorithm["num_envs"])

    @property
    def episode_length(self) -> int:
        algorithm = cast(Mapping[str, FrozenValue], self.parameters["algorithm"])
        return cast(int, algorithm["episode_length"])

    @property
    def num_timesteps(self) -> int:
        return cast(int, self.training_budget["env_steps"])

    @property
    def failure_mode(self) -> FailureMode:
        algorithm = cast(Mapping[str, FrozenValue], self.parameters["algorithm"])
        return cast(FailureMode, algorithm["failure_mode"])
