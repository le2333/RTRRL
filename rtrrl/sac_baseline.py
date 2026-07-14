"""Brax SAC baseline with rtrrl.py-style configuration and Aim logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pprint import pprint
from typing import Any

import numpy as np
import simple_parsing

import brax
from brax.envs.base import Env as BraxEnv
from brax.training.agents.sac import train as sac

from envs.environments import EnvironmentParams, get_env_specs, print_env_info
from envs.wrappers import FlatObs_BraxWrapper, PO_BraxWrapper
from logging_util import DummyLogger, with_logger


RESERVED_SAC_OVERRIDE_KEYS = frozenset(
    {
        "environment",
        "eval_env",
        "num_timesteps",
        "episode_length",
        "progress_fn",
        "checkpoint_logdir",
        "randomization_fn",
        "seed",
    }
)
EVALS_PER_5M_STEPS = 2
STEPS_PER_EVAL_DENSITY_UNIT = 5_000_000
HOPPER_STEPS_PER_EVAL = 1_000_000


class BraxEnvAdapter(BraxEnv):
    """Expose repository observation wrappers as a Brax v2 Env for SAC."""

    def __init__(self, env, observation_size: int):
        self.env = env
        self._observation_size = observation_size

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, state, action):
        return self.env.step(state, action)

    @property
    def observation_size(self) -> int:
        return self._observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def backend(self) -> str:
        return self.env.backend


@dataclass(unsafe_hash=True)
class SACParams:
    """Parameters for the Brax SAC baseline."""

    debug: int | bool = 0
    seed: int | None = None

    # Logging
    logging: str | None = None
    log_repo: str | None = None
    log_code: bool = False

    # Environment
    env_params: EnvironmentParams = EnvironmentParams(
        render=False,
        env_name="brax-hopper",
        max_ep_length=1000,
        batch_size=None,
    )

    # SAC training
    num_timesteps: int | None = None
    sac_overrides: dict[str, Any] = field(default_factory=dict, hash=False)


def _to_scalar(value):
    """Convert common JAX/NumPy scalar metrics to Python scalars."""
    try:
        array = np.asarray(value)
        if array.shape == ():
            return float(array)
    except TypeError:
        pass
    return value


def _jsonable(value):
    """Convert common JAX/NumPy values into logger-friendly Python objects."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _env_backend(params: EnvironmentParams) -> str | None:
    return params.init_kwargs.get("backend") or params.env_kwargs.get("backend")


def make_sac_env(params: EnvironmentParams):
    """Create an unbatched Brax env for Brax SAC.

    Brax SAC applies its own vectorization, episode wrapper, and auto-reset
    wrapper. This function only applies the repository's flattened observation
    and partial-observation mask conventions.
    """
    env_name = params.env_name
    if not env_name.startswith("brax"):
        raise ValueError("sac_baseline.py currently supports only brax-* environments")

    brax_name = "-".join(env_name.split("-")[1:])
    init_kwargs = {**params.env_kwargs, **params.init_kwargs}
    env = brax.envs.get_environment(env_name=brax_name, **init_kwargs)

    obs_size, discrete, act_size, obs_mask, act_clip = get_env_specs(env, params.obs_mask)
    if discrete:
        raise ValueError("Brax SAC baseline expects continuous-action Brax environments")

    env = FlatObs_BraxWrapper(env)
    env = PO_BraxWrapper(env, obs_mask)
    env = BraxEnvAdapter(env, obs_size)
    env_info = dict(
        obs_size=obs_size,
        discrete=discrete,
        act_size=act_size,
        obs_mask=obs_mask,
        act_clip=act_clip,
    )
    return env, env_info


def _shared_params(args: SACParams, env_info: dict, env: BraxEnv) -> dict[str, Any]:
    return {
        "algorithm": "sac",
        "env_name": args.env_params.env_name,
        "obs_mask": _jsonable(env_info["obs_mask"]),
        "backend": getattr(env, "backend", _env_backend(args.env_params)),
        "seed": args.seed,
        "requested_steps": args.num_timesteps,
        "env/max_ep_length": args.env_params.max_ep_length,
        "env/obs_size": env_info["obs_size"],
        "env/act_size": env_info["act_size"],
        "eval/hopper_steps_per_eval": HOPPER_STEPS_PER_EVAL,
        "eval/evals_per_5m_steps": EVALS_PER_5M_STEPS,
        "sac/overrides": {k: _jsonable(v) for k, v in args.sac_overrides.items()},
    }


def _validate_sac_overrides(overrides: dict[str, Any]) -> None:
    """Reject overrides that would desynchronize config, training, and logs."""
    reserved = sorted(RESERVED_SAC_OVERRIDE_KEYS.intersection(overrides))
    if reserved:
        raise ValueError(
            "sac_overrides may not override entrypoint-controlled arguments: "
            + ", ".join(reserved)
        )


def _default_num_evals(num_timesteps: int, env_name: str) -> int:
    """Use evaluation as a recording probe, not a training hyperparameter."""
    if env_name == "brax-hopper":
        return max(1, int(np.ceil(num_timesteps / HOPPER_STEPS_PER_EVAL)))

    return max(
        1,
        int(np.ceil(num_timesteps * EVALS_PER_5M_STEPS / STEPS_PER_EVAL_DENSITY_UNIT)),
    )


def train_sac(args: SACParams, logger=DummyLogger()):
    """Train the Brax SAC baseline."""
    if args.num_timesteps is None:
        raise ValueError("num_timesteps must be set in the config or CLI")
    _validate_sac_overrides(args.sac_overrides)

    env, env_info = make_sac_env(args.env_params)
    pprint(args, width=1)
    print_env_info(env_info)

    shared_params = _shared_params(args, env_info, env)
    logger.log_params({"sac": asdict(args), "shared": shared_params})
    logger["best_eval_reward"] = -float("inf")
    last_progress_step = {"value": 0}

    def progress(num_steps, metrics):
        env_steps = int(num_steps)
        last_progress_step["value"] = env_steps
        log_metrics = {k: _to_scalar(v) for k, v in metrics.items()}
        log_metrics.update(
            {
                "train/env_steps": env_steps,
                "train/native_step": env_steps,
            }
        )

        eval_reward = log_metrics.get("eval/episode_reward")
        if eval_reward is not None and eval_reward > logger["best_eval_reward"]:
            logger["best_eval_reward"] = eval_reward
            log_metrics["eval/best_eval_reward"] = eval_reward

        logger.log(log_metrics, step=env_steps)

    train_kwargs = {
        "environment": env,
        "num_timesteps": args.num_timesteps,
        "episode_length": args.env_params.max_ep_length,
        "progress_fn": progress,
        "num_evals": _default_num_evals(args.num_timesteps, args.env_params.env_name),
        **args.sac_overrides,
    }
    if args.seed is not None:
        train_kwargs["seed"] = args.seed

    make_policy, params, metrics = sac.train(**train_kwargs)

    final_step = last_progress_step["value"] or args.num_timesteps
    final_metrics = {f"final/{k}": _to_scalar(v) for k, v in metrics.items()}
    final_metrics.update(
        {
            "actual_final_step": final_step,
            "final/best_eval_reward": logger["best_eval_reward"],
        }
    )
    logger.log(final_metrics, step=final_step)
    logger["actual_final_step"] = final_step
    logger["final/best_eval_reward"] = logger["best_eval_reward"]
    logger.finalize()

    # Keep references alive for callers that import train_sac programmatically.
    train_sac.make_policy = make_policy
    train_sac.params = params
    return logger["best_eval_reward"]


if __name__ == "__main__":
    hparams: SACParams = simple_parsing.parse(SACParams, add_config_path_arg=True)
    run_name = f"{hparams.env_params.env_name}-sac"

    with_logger(
        train_sac,
        hparams,
        logger_name=hparams.logging,
        project_name="RTRRL-PPO",
        aim_repo=hparams.log_repo,
        run_name=run_name,
        hparams_type=SACParams,
    )
