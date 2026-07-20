"""Shortest real strict-LRU Brax smoke used by the Task 12 report."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
from typing import Any

import jax
import numpy as np

from experiments.rtrrl_hopper.run import train_legacy
from memorax.algorithms.rtrrl.entrypoint import normalize_legacy_invocation


CONFIG_PAYLOAD: dict[str, Any] = {
    "profile": "aaai25_strict_lru",
    "seed": 7,
    "episodes": 1,
    "steps": 1,
    "patience": 0,
    "eval_every": 1,
    "eval_steps": 1000,
    "render_every_evals": 0,
    "render_steps": 0,
    "logging": None,
    "log_every": 1,
    "hidden_size": 2,
    "env_params": {
        "env_name": "brax-hopper",
        "init_kwargs": {"backend": "spring"},
        "max_ep_length": 2,
        "batch_size": 1,
        "render": False,
    },
}


class RecordingLogger(dict):
    """Capture the exact historical host metrics emitted by the public API."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []
        self.finalized = False

    def __bool__(self) -> bool:
        return True

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if step is None:
            raise AssertionError("historical smoke logger step must be explicit")
        converted = {
            key: (
                np.asarray(value).item()
                if np.asarray(value).ndim == 0
                else np.asarray(value).tolist()
            )
            for key, value in metrics.items()
        }
        self.records.append({"step": step, "metrics": converted})

    def log_params(self, params: dict[str, Any]) -> None:
        del params

    def log_video(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("rendering is disabled in the shortest smoke")

    def finalize(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.finalized = True


def main() -> None:
    logger = RecordingLogger()
    config = normalize_legacy_invocation(CONFIG_PAYLOAD)
    best_reward = train_legacy(config, logger)
    if not logger.finalized:
        raise AssertionError("logger did not finalize")
    if len(logger.records) != 1:
        raise AssertionError(f"expected one historical record: {logger.records}")
    record = logger.records[0]
    metrics = record["metrics"]
    required_training = {
        "steps",
        "mean_reward",
        "num_episodes",
        "mean_delta",
        "mean_r_bar",
        "mean_v",
        "total_td_loss",
        "actor_loss",
        "critic_loss",
        "entropy",
        "v_targ",
    }
    missing = required_training - metrics.keys()
    if missing:
        raise AssertionError(f"missing historical training metrics: {sorted(missing)}")
    if metrics["steps"] != 1:
        raise AssertionError(f"expected one real update, got {metrics['steps']}")
    for name in ("eval/rewards", "eval/best_eval_reward"):
        if name not in metrics:
            raise AssertionError(f"missing historical evaluation metric: {name}")
    if not np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all():
        raise AssertionError(f"non-finite smoke metrics: {metrics}")
    print(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoint": (
                    "experiments.rtrrl_hopper.run.train_legacy"
                ),
                "config_payload": CONFIG_PAYLOAD,
                "runtime": {
                    "python": platform.python_version(),
                    "jax": jax.__version__,
                    "jaxlib": metadata.version("jaxlib"),
                    "flax": metadata.version("flax"),
                    "brax": metadata.version("brax"),
                    "jax_backend": jax.default_backend(),
                    "devices": [str(device) for device in jax.devices()],
                    "brax_backend": "spring",
                },
                "training_transitions": metrics["steps"],
                "evaluation_transitions": CONFIG_PAYLOAD["eval_steps"],
                "record": record,
                "best_reward": float(best_reward),
                "logger_finalized": logger.finalized,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
