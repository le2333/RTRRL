"""Historical RTRRL entry point backed exclusively by Memorax."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Iterable, Literal, Optional, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MEMO_ROOT = _REPOSITORY_ROOT / "memo"
if str(_MEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMO_ROOT))

from memorax.algorithms.rtrrl.entrypoint import (  # noqa: E402
    audit_repository_configs,
    describe_legacy_build,
    emit_json,
    load_legacy_mapping,
    normalize_legacy_invocation,
    parse_compatibility_cli,
    run_mock_epoch,
)


@dataclass(frozen=True, eq=True)
class EnvironmentParams:
    """Compatibility-only copy of the historical environment data carrier."""

    env_name: str = "CartPole-v1"
    reward_scaling: int = 1
    obs_mask: Optional[
        Iterable[int] | Literal["odd", "even", "first_half"]
    ] = None
    init_kwargs: dict = field(default_factory=dict, hash=False)
    env_kwargs: dict = field(default_factory=dict, hash=False)
    max_ep_length: int = 500
    batch_size: int | None = None
    render: bool = True


@dataclass(frozen=True)
class OptimizerConfig:
    """Compatibility-only copy of the historical optimizer data carrier."""

    opt_name: str = "adam"
    learning_rate: float = 1e-3
    kwargs: dict = field(default_factory=dict, hash=False)
    decay_type: str | None = None
    lr_kwargs: dict = field(default_factory=dict, hash=False)
    weight_decay: float = 0.0
    gradient_clip: float | None = None
    multi_step: int | None = None


@dataclass(unsafe_hash=True)
class RTRRLParams:
    """Historical mutable parameter API retained only for compatibility."""

    debug: int | bool = 0
    seed: int | None = None
    episodes: int = 150_000
    steps: int = 1000
    patience: int = 100
    eval_every: int = 100
    eval_steps: int = 1000
    eval_batch_size: int = 10
    render_every_evals: int = 10
    render_start: int = 0
    render_steps: int = 200
    logging: str | None = None
    log_repo: str | None = None
    run_name: str | None = None
    save_model: bool = False
    log_norms: bool = False
    log_code: bool = False
    log_every: int = 1
    env_params: EnvironmentParams = EnvironmentParams(
        render=False,
        env_name="StatelessCartPoleEasy",
        max_ep_length=1000,
        batch_size=1,
    )
    optimizer_params_td: OptimizerConfig = OptimizerConfig(
        opt_name="adam",
        learning_rate=1e-4,
    )
    optimizer_params_rnn: OptimizerConfig = OptimizerConfig(
        opt_name="adam",
        learning_rate=1e-4,
        gradient_clip=1.0,
    )
    rnn_model: str | None = "lru"
    gradient_mode: str = "rflo"
    hidden_size: int = 32
    wiring: str = "fully_connected"
    trace_mode: str = "accumulate"
    gamma: float = 0.99
    lambda_v: float = 0.9
    lambda_pi: float = 0.9
    lambda_rnn: float = 0.9
    eta_pi: float = 1
    eta_f: float = 1
    entropy_rate: float = 1e-5
    eta: float | None = None
    meta_rl: bool = True
    f_align: bool = False
    normalize_reward: bool = False
    normalize_obs: bool = False
    var_scaling: bool = False
    layer_norm: bool = False
    mlp_actor: bool = False
    pass_obs: bool = False
    align_action_logprob: bool = False
    update_trace_before_td: bool = False
    update_period: float = 1
    dropout_rate: float = 0
    act_magnitude_factor: float = 0
    slow_rnn_factor: float = 0e-2


def train_rtrrl(args: Any, logger=None):
    """Delegate the historical callable API to the Memorax experiment runner."""

    config = normalize_legacy_invocation(args)
    experiments = _MEMO_ROOT / "experiments"
    base = experiments / "base"
    for path in (experiments, base):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    runner = import_module("rtrrl_hopper.run")
    return runner.train_legacy(config, logger)


def main(argv: Sequence[str] | None = None) -> int:
    options, overrides = parse_compatibility_cli(argv)
    if options.compat_action == "mock-epoch":
        emit_json(run_mock_epoch())
        return 0
    if options.compat_action == "audit":
        emit_json(audit_repository_configs(_REPOSITORY_ROOT))
        return 0

    raw = load_legacy_mapping(options.config_path, overrides)
    config = normalize_legacy_invocation(raw)
    if options.compat_action == "build":
        emit_json(describe_legacy_build(config))
        return 0

    experiments = _MEMO_ROOT / "experiments"
    base = experiments / "base"
    for path in (experiments, base):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    runner = import_module("rtrrl_hopper.run")
    runner.run_legacy_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
