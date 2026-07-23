from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import yaml

FailureMode = Literal["none", "before_training", "after_training", "after_checkpoint"]
type FrozenValue = (
    str | int | float | bool | None | tuple[FrozenValue, ...] | Mapping[str, FrozenValue]
)

_FAILURE_MODES = frozenset({"none", "before_training", "after_training", "after_checkpoint"})
_DEFAULT_ENVIRON: Mapping[str, str] = MappingProxyType(dict(os.environ))


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
    result = float(cast(int | float, value))
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


@dataclass(frozen=True, slots=True)
class AcceptanceConfig:
    protocol_version: Literal["1"]
    environment: Mapping[str, FrozenValue]
    logging: Mapping[str, FrozenValue]
    parameters: Mapping[str, FrozenValue]
    training_budget: Mapping[str, FrozenValue]
    fast_mode: bool = False

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        environ: Mapping[str, str] = _DEFAULT_ENVIRON,
    ) -> AcceptanceConfig:
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
        num_envs = _integer(
            algorithm["num_envs"], "parameters.algorithm.num_envs", positive=True
        )
        episode_length = _integer(
            algorithm["episode_length"], "parameters.algorithm.episode_length", positive=True
        )
        failure_mode = algorithm["failure_mode"]
        if type(failure_mode) is not str or failure_mode not in _FAILURE_MODES:
            raise ValueError(f"parameters.algorithm.failure_mode must be one of {sorted(_FAILURE_MODES)}")

        training_budget = _mapping(root["training_budget"], "training_budget")
        _exact_keys(training_budget, {"env_steps"}, "training_budget")
        env_steps = _integer(
            training_budget["env_steps"], "training_budget.env_steps", positive=True
        )
        if env_steps != num_envs * episode_length:
            raise ValueError("training_budget.env_steps must equal num_envs * episode_length")

        test_mode = _enabled(environ, "BRAX_ACCEPTANCE_TEST_MODE")
        fast_mode = _enabled(environ, "BRAX_ACCEPTANCE_E2E_FAST")
        if failure_mode != "none" and not test_mode:
            raise ValueError("non-none failure_mode requires BRAX_ACCEPTANCE_TEST_MODE=1")
        if fast_mode and not test_mode:
            raise ValueError("BRAX_ACCEPTANCE_E2E_FAST=1 requires BRAX_ACCEPTANCE_TEST_MODE=1")

        return cls(
            protocol_version="1",
            environment=cast(Mapping[str, FrozenValue], _freeze(environment)),
            logging=cast(Mapping[str, FrozenValue], _freeze(logging)),
            parameters=cast(Mapping[str, FrozenValue], _freeze(parameters)),
            training_budget=cast(Mapping[str, FrozenValue], _freeze(training_budget)),
            fast_mode=fast_mode,
        )

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
