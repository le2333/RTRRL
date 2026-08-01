"""RTRRL as the paper's authors wrote it, driven by our control plane.

This is the fifth arm of the Hopper comparison and the only one that is not our
code. The other four are built out of `memo`, where the cell, the traces and the
update are ours to read and to repair; this one is the authors' repository as
forked, at its HEAD, and the entire point of it is that the arithmetic between
`train_rtrrl` and the environment is untouched. What that costs is a translation
layer, which is this file: their training function takes one `RTRRLParams`
dataclass and a logger, and the control plane offers a flat mapping of sampled
parameters and a `Reporter`.

The parameter names in `SPACE` are `memo`'s, not theirs. Five arms searched over
`td_lr` and one over `optimizer_params_td.learning_rate` would be five arms that
cannot be diffed against each other, and the six searched dimensions have to be
recognisably the same six in all five files or the comparison is over the
searches rather than over the algorithms. Where a knob of theirs has no name in
that vocabulary -- `gradient_mode`, `f_align` -- it keeps theirs. The run shape
belongs to the control plane, which injects it separately from the sampler.

Two of their defaults would quietly make this arm a different experiment and are
worth naming here rather than only in the experiment file. `gradient_mode`
defaults to `rflo`, which the LRU path does not implement: it raises
`ValueError("Unknown plasticity for LRU: rflo")` before the first step, so a run
left on the default does not start at all. The control plane sets `patience`
from its early-stop policy, so an arm that stops early does so by the same rule
as the other four arms.

Their code is imported inside the functions that need it rather than at the top
of this file. What is above `SPACE` is what the catalog is built from, and the
catalog is built twice: once in CI inside the image, where jax and brax and a
git checkout of popjym are all installed, and once on a laptop or a micro
instance to produce the `catalog.json` an experiment is validated against
offline. Importing `rtrrl` to read a dictionary of parameter names would make
the second of those impossible on any machine that cannot hold a training
framework, which is the machine this control plane runs on.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rtrrl import RTRRLParams

# Their episode limit, matching the `EpisodeWrapper` length `memo` fixes at 1000
# for every Brax task. It is not offered as a parameter for the same reason it
# is not one there: a truncation length is part of what the task is.
MAX_EPISODE_LENGTH = 1000

_UNIT = {"type": "float", "low": 0.0, "high": 1.0}
_RATE = {"type": "float", "low": 1e-9, "high": 10.0, "log": True}

SPACE: dict[str, Any] = {
    # Their two cells, under the name `memo` gives the published one. `rflo` is
    # absent from `gradient_mode` on purpose: it is their default and the LRU
    # rejects it, so offering it would let a sampler suggest a run that cannot
    # start. The CTRNN accepts all three, and loses one of them here; that is
    # the cheaper of the two mistakes.
    "backbone": ["lru", "ctrnn"],
    "gradient_mode": ["rtrl", "bptt"],
    "hidden_dim": {"type": "int", "low": 1, "high": 512},
    "meta_rl": [False, True],
    "normalize_observation": [False, True],
    "normalize_reward": [False, True],
    # Feedback alignment, an MLP actor head and a layer-normed cell: three
    # variations of theirs that the paper's Brax results do not use. They are
    # declared so that an experiment file has to say it is not using them.
    "f_align": [False, True],
    "mlp_actor": [False, True],
    "layer_norm": [False, True],
    "gamma": {"type": "float", "low": 0.5, "high": 0.9999},
    "lambda_pi": _UNIT,
    "lambda_v": _UNIT,
    "lambda_rnn": _UNIT,
    "td_lr": _RATE,
    "rnn_lr": _RATE,
    "eta_pi": {"type": "float", "low": 0.0, "high": 10.0},
    "eta_f": {"type": "float", "low": 0.0, "high": 10.0},
    "entropy_rate": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
    "update_period": _UNIT,
    "update_trace_before_td": [False, True],
    "rnn_grad_clip": {"type": "float", "low": 0.0, "high": 1e4},
    "trace_mode": ["accumulate", "dutch"],
}

# One name, because one is all their evaluation produces. `eval_model` returns a
# single scalar, the mean over the evaluation environments of the reward each
# accumulated up to its first termination, and the loop records it as
# `eval/rewards`; `entries/reporting.py` renames it to the name below. There is
# no length beside it -- the index of that first termination is computed and
# then spent on the reward sum, never reported -- so `eval/episode/length` and
# `eval/episode/return_per_step`, which the four `memo` arms do report, are
# absent here rather than invented.
METRICS: tuple[str, ...] = ("eval/episode/return",)


def iterations(*, total_steps: int, scan_steps: int, num_envs: int) -> int:
    """How many of their outer iterations a budget of environment steps buys.

    Their loop has no notion of a total: it runs `episodes` iterations of
    `steps` scanned transitions over `batch_size` environments and stops. The
    budget the experiment states has to be divided into that shape exactly,
    because either way of rounding is a lie -- down reports a step count the run
    never reached, up spends money nobody asked for. This is `memo`'s
    `runner.loop.whole_epochs` refusing the same thing for the same reason.
    """

    per_iteration = scan_steps * num_envs
    if total_steps % per_iteration:
        raise ValueError(
            f"total_steps {total_steps} is not whole iterations of "
            f"{scan_steps} steps over {num_envs} environments"
        )
    return total_steps // per_iteration


def settings(
    params: Mapping[str, Any], environment, training, evaluation
) -> dict[str, Any]:
    """Every field of theirs this entry sets, as plain values."""

    chunk_steps = int(training.chunk_steps or training.epoch_steps)
    total = iterations(
        total_steps=training.total_steps,
        scan_steps=chunk_steps,
        num_envs=training.num_envs,
    )
    per_epoch = iterations(
        total_steps=training.epoch_steps,
        scan_steps=chunk_steps,
        num_envs=training.num_envs,
    )
    return {
        "seed": environment.seed,
        "episodes": total,
        "steps": chunk_steps,
        "patience": (
            0
            if training.early_stop_patience is None
            else training.early_stop_patience
        ),
        # Their evaluation cadence is counted in outer iterations, so an epoch
        # of environment steps becomes the number of iterations it takes.
        "eval_every": per_epoch,
        "eval_steps": evaluation.steps,
        "eval_batch_size": evaluation.num_envs,
        "rnn_model": str(params["backbone"]),
        "gradient_mode": str(params["gradient_mode"]),
        "hidden_size": int(params["hidden_dim"]),
        "meta_rl": bool(params["meta_rl"]),
        "f_align": bool(params["f_align"]),
        "mlp_actor": bool(params["mlp_actor"]),
        "layer_norm": bool(params["layer_norm"]),
        "normalize_obs": bool(params["normalize_observation"]),
        "normalize_reward": bool(params["normalize_reward"]),
        "trace_mode": str(params["trace_mode"]),
        "gamma": float(params["gamma"]),
        "lambda_pi": float(params["lambda_pi"]),
        "lambda_v": float(params["lambda_v"]),
        "lambda_rnn": float(params["lambda_rnn"]),
        "eta_pi": float(params["eta_pi"]),
        "eta_f": float(params["eta_f"]),
        "entropy_rate": float(params["entropy_rate"]),
        "update_period": float(params["update_period"]),
        "update_trace_before_td": bool(params["update_trace_before_td"]),
        "environment": {
            "env_name": environment.id.replace("::", "-"),
            "batch_size": training.num_envs,
            "max_ep_length": MAX_EPISODE_LENGTH,
            "render": False,
            "obs_mask": tuple(environment.observed) if environment.observed else None,
            "env_kwargs": {"backend": environment.backend},
        },
        # No gradient clip on the TD optimiser. Their comment beside the default
        # says clipping the TD update makes the eligibility traces explode, and
        # this arm exists to run their choices.
        "td": {"opt_name": "adam", "learning_rate": float(params["td_lr"])},
        "rnn": {
            "opt_name": "adam",
            "learning_rate": float(params["rnn_lr"]),
            "gradient_clip": float(params["rnn_grad_clip"]),
        },
    }


def parameters(
    params: Mapping[str, Any], environment, training, evaluation
) -> RTRRLParams:
    """Assemble the dataclass their training function takes."""

    from envs.environments import EnvironmentParams
    from optimizers import OptimizerConfig
    from rtrrl import RTRRLParams

    chosen = dict(settings(params, environment, training, evaluation))
    return RTRRLParams(
        env_params=EnvironmentParams(**chosen.pop("environment")),
        optimizer_params_td=OptimizerConfig(**chosen.pop("td")),
        optimizer_params_rnn=OptimizerConfig(**chosen.pop("rnn")),
        **chosen,
    )


def run(reporter, config) -> None:
    from entries.reporting import ReporterLogger
    from rtrrl import train_rtrrl

    # Their signature names `DummyLogger` because that is what the argument
    # defaults to, not because the loop needs one: everything it asks of the
    # logger is either a `dict` method or one of the five it calls by name, and
    # the adapter has all of them. Subclassing theirs to say so in the type
    # system would pull PIL, plotly and flax into a module the catalog imports
    # on a machine that has none of them.
    logger: Any = ReporterLogger(reporter)
    train_rtrrl(
        parameters(
            config.params, config.environment, config.training, config.evaluation
        ),
        logger,
    )


def main(argv: list[str] | None = None) -> int:
    del argv
    from training_sdk.reporter import Reporter

    with Reporter.from_env() as reporter:
        run(reporter, reporter.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
