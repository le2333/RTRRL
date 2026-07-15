"""Shared experiment facilities for memorax-rtrl (reproduces arXiv 2605.24709).

Reuses streaming-rtrrl's training infrastructure (simple_parsing CLI,
logging_util.with_logger / Aim) wrapped around Memorax algorithms
(StreamAC / StreamACRtrl) and Memorax environments. Metrics produced by the
algorithm via `lox.log` are surfaced with `lox.spool` and forwarded to the Aim
logger — bridging Memorax's lox logging and streaming-rtrrl's Aim logger.

Each experiment directory under experiments/ fixes ONE algorithm-environment
binding (see its run.py); only the backbone variant (agent_type) and
hyperparameters are configurable. Algorithm and environment are never mixed
across experiments.
"""
from dataclasses import asdict, dataclass
from pprint import pprint
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import numpy as np
import optax
from tqdm import trange

from memorax.algorithms import (
    QRC,
    QRCConfig,
    QRCRtrl,
    RTRRL,
    RTRRLConfig,
    StreamAC,
    StreamACConfig,
    StreamACRtrl,
)
from memorax.networks import (
    FeatureExtractor,
    LRUCell,
    LRUConfig,
    Memoroid,
    Network,
    RNN,
    RTUCell,
    RTUConfig,
    heads,
)
from memorax.utils.typing import Discrete

from logging_util import DummyLogger, with_logger

# Persistent JAX compilation cache (matches streaming-rtrrl/rtrrl_lru.py).
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 1000000)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@dataclass(unsafe_hash=True)
class ExperimentConfig:
    """Base config shared by all memorax-rtrl experiments.

    Subclasses fix `experiment` (the algorithm-environment binding name) and
    add environment-specific fields. The algorithm is always Stream AC(λ);
    `agent_type` selects the recurrent backbone variant.
    """

    experiment: str = "base"

    # Training
    seed: int | None = None
    total_timesteps: int = 500_000
    num_epochs: int = 50
    num_envs: int = 16

    # Validation
    eval_every: int = 10
    eval_steps: int = 1000
    patience: int = 100

    # Logging (streaming-rtrrl conventions)
    logging: str | None = None
    log_repo: str | None = None
    run_name: str | None = None
    save_model: bool = False
    log_every: int = 1

    # Backbone variant: rtu_rtrl (StreamACRtrl) | rtu_tbptt (StreamAC + RTU).
    # gru_tbptt / ffn baselines are not implemented in this fork yet.
    agent_type: str = "rtu_rtrl"

    # Stream AC(λ) hyperparameters (shared by StreamAC / StreamACRtrl).
    gamma: float = 0.99
    trace_lambda: float = 0.9
    actor_lr: float = 1.0
    critic_lr: float = 1.0
    actor_kappa: float = 3.0
    critic_kappa: float = 2.0
    entropy_coefficient: float = 0.01
    adaptive: bool = False
    beta2: float = 0.999
    eps: float = 1e-8

    # Network
    hidden_dim: int = 32
    encoder_dim: int = 32
    meta_rl: bool = True

    # QRC(λ) hyperparameters (used by build_qrc_agent; discrete envs only).
    # `trace_lambda` above is reused as QRC's λ. QRC uses optax SGD (not ObGD).
    gradient_correction: bool = True
    reg_coeff: float = 1.0
    q_lr: float = 1e-4
    h_lr: float = 1e-5
    epsilon_end: float = 0.01
    epsilon_fraction: float = 0.2


def build_networks(cfg: ExperimentConfig, action_space) -> tuple[Network, Network]:
    """Build actor/critic networks from cfg.agent_type and the action space.

    Layout: FeatureExtractor([o,(a),(r)]) -> RNN(RTUCell) -> head.
    Head is Categorical for discrete envs, Gaussian for continuous envs.
    """
    feat = cfg.encoder_dim
    observation_extractor = nn.Sequential((nn.Dense(feat), nn.relu))
    action_extractor = (
        nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
    )
    reward_extractor = (
        nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
    )
    feature_extractor = FeatureExtractor(
        observation_extractor=observation_extractor,
        action_extractor=action_extractor,
        reward_extractor=reward_extractor,
    )

    streams = 3 if cfg.meta_rl else 1
    in_dim = feat * streams

    if cfg.agent_type in ("rtu_rtrl", "rtu_tbptt"):
        cell = RTUCell(config=RTUConfig(features=in_dim, hidden_dim=cfg.hidden_dim))
        torso = RNN(cell=cell)
    elif cfg.agent_type == "lru_rtrl":
        # LRU backbone for the RTU-vs-LRU memory comparison. Memoroid(LRU) exposes
        # the same local_jacobian RTRL interface as RNN(RTU); StreamACRtrl calls it
        # by method name so it is backbone-agnostic.
        cell = LRUCell(config=LRUConfig(features=in_dim, hidden_dim=cfg.hidden_dim))
        torso = Memoroid(cell=cell)
    else:
        raise ValueError(
            f"agent_type '{cfg.agent_type}' not implemented; "
            "use 'rtu_rtrl', 'lru_rtrl' or 'rtu_tbptt' (gru_tbptt/ffn TBD)."
        )

    if isinstance(action_space, Discrete):
        actor_head = heads.Categorical(action_dim=action_space.n)
    else:
        actor_head = heads.Gaussian(action_dim=action_space.shape[0])

    actor_network = Network(
        feature_extractor=feature_extractor, torso=torso, head=actor_head
    )
    critic_network = Network(
        feature_extractor=feature_extractor, torso=torso, head=heads.VNetwork()
    )
    return actor_network, critic_network


def build_stream_ac_agent(cfg: ExperimentConfig, env, env_params):
    """Build a StreamAC or StreamACRtrl agent from cfg.agent_type."""
    action_space = env.action_space(env_params)
    actor_network, critic_network = build_networks(cfg, action_space)

    ac_cfg = StreamACConfig(
        num_envs=cfg.num_envs,
        gamma=cfg.gamma,
        trace_lambda=cfg.trace_lambda,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        actor_kappa=cfg.actor_kappa,
        critic_kappa=cfg.critic_kappa,
        entropy_coefficient=cfg.entropy_coefficient,
        adaptive=cfg.adaptive,
        beta2=cfg.beta2,
        eps=cfg.eps,
    )

    if cfg.agent_type in ("rtu_rtrl", "lru_rtrl"):
        return StreamACRtrl(ac_cfg, env, env_params, actor_network, critic_network)
    return StreamAC(ac_cfg, env, env_params, actor_network, critic_network)


def build_rtrrl_agent(cfg: ExperimentConfig, env, env_params):
    """Build an RTRRL agent (shared LRU-RTRL torso -> Gaussian actor + V critic).

    Layout: FeatureExtractor([o,(a),(r)]) -> Memoroid(LRUCell) -> silu -> heads.
    Continuous control only (Gaussian actor); reproduces streaming-rtrrl's
    RNNActorCritic with a shared recurrent core feeding a linear actor + critic.
    """
    action_space = env.action_space(env_params)
    if isinstance(action_space, Discrete):
        raise ValueError("RTRRL currently targets continuous-action envs (Gaussian actor).")

    feat = cfg.encoder_dim
    observation_extractor = nn.Sequential((nn.Dense(feat), nn.relu))
    action_extractor = (
        nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
    )
    reward_extractor = (
        nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
    )
    feature_extractor = FeatureExtractor(
        observation_extractor=observation_extractor,
        action_extractor=action_extractor,
        reward_extractor=reward_extractor,
    )

    streams = 3 if cfg.meta_rl else 1
    in_dim = feat * streams
    # Recurrent backbone: "lru" (Memoroid, linear SSM, the RTRL-HOP-533 baseline)
    # or "rtu" (RNN, complex rotation-decay + tanh => bounded state, gain tied to
    # nu_log with NO free gamma). RTU structurally lacks both divergence drivers
    # (unbounded state + self-aligned free gain), so it's the backbone-stability
    # probe. Both expose the same local_jacobian RTRL interface.
    backbone = getattr(cfg, "backbone", "lru")
    if backbone == "rtu":
        torso = RNN(cell=RTUCell(config=RTUConfig(features=in_dim, hidden_dim=cfg.hidden_dim)))
    elif backbone == "lru":
        torso = Memoroid(
            cell=LRUCell(config=LRUConfig(features=in_dim, hidden_dim=cfg.hidden_dim))
        )
    else:
        raise ValueError(f"backbone '{backbone}' not supported; use 'lru' or 'rtu'.")

    # Diagnostic/faithfulness switches default off (identical to the repro
    # baseline); rtrrl_hopper's config may toggle them for ablations.
    actor_head = heads.Gaussian(
        action_dim=action_space.shape[0], bound=getattr(cfg, "bound_actor", False)
    )
    critic_head = heads.VNetwork()

    # Optional auxiliary self-prediction head (predict next obs + reward): an
    # external fixed-scale target that anchors the representation scale.
    pred_head = None
    if getattr(cfg, "pred_obs", False):
        obs_dim = env.observation_space(env_params).shape[0]
        pred_head = heads.Regressor(out_dim=obs_dim + 1)

    rtrrl_cfg = RTRRLConfig(
        num_envs=cfg.num_envs,
        gamma=cfg.gamma,
        lambda_pi=cfg.lambda_pi,
        lambda_v=cfg.lambda_v,
        lambda_rnn=cfg.lambda_rnn,
        td_lr=cfg.td_lr,
        rnn_lr=cfg.rnn_lr,
        eta_pi=cfg.eta_pi,
        eta_f=cfg.eta_f,
        entropy_rate=cfg.entropy_rate,
        update_period=cfg.update_period,
        b1=cfg.b1,
        b2=cfg.b2,
        eps=cfg.eps,
        rnn_grad_clip=cfg.rnn_grad_clip,
        act_clip=getattr(cfg, "act_clip", 0.0),
        freeze_gamma=getattr(cfg, "freeze_gamma", False),
        pred_obs=getattr(cfg, "pred_obs", False),
        pred_coeff=getattr(cfg, "pred_coeff", 1.0),
    )
    return RTRRL(
        rtrrl_cfg,
        env,
        env_params,
        feature_extractor,
        torso,
        actor_head,
        critic_head,
        pred_head=pred_head,
    )


def build_qrc_networks(cfg: ExperimentConfig, num_actions: int) -> tuple[Network, Network]:
    """Build Q/H networks for QRC: FeatureExtractor -> RNN(RTUCell) -> DiscreteQNetwork."""
    feat = cfg.encoder_dim
    observation_extractor = nn.Sequential((nn.Dense(feat), nn.relu))
    action_extractor = (
        nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
    )
    reward_extractor = (
        nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
    )
    feature_extractor = FeatureExtractor(
        observation_extractor=observation_extractor,
        action_extractor=action_extractor,
        reward_extractor=reward_extractor,
    )

    streams = 3 if cfg.meta_rl else 1
    in_dim = feat * streams
    cell = RTUCell(config=RTUConfig(features=in_dim, hidden_dim=cfg.hidden_dim))
    torso = RNN(cell=cell)

    head = heads.DiscreteQNetwork(action_dim=num_actions)
    q_network = Network(
        feature_extractor=feature_extractor, torso=torso, head=head
    )
    h_network = Network(
        feature_extractor=feature_extractor, torso=torso, head=head
    )
    return q_network, h_network


def build_qrc_agent(cfg: ExperimentConfig, env, env_params):
    """Build a QRC or QRCRtrl agent from cfg.agent_type (discrete envs only)."""
    action_space = env.action_space(env_params)
    if not isinstance(action_space, Discrete):
        raise ValueError("QRC requires a discrete action space.")
    q_network, h_network = build_qrc_networks(cfg, action_space.n)

    qrc_cfg = QRCConfig(
        num_envs=cfg.num_envs,
        gamma=cfg.gamma,
        lamda=cfg.trace_lambda,
        gradient_correction=cfg.gradient_correction,
        reg_coeff=cfg.reg_coeff,
    )
    q_optimizer = optax.sgd(cfg.q_lr)
    h_optimizer = optax.sgd(cfg.h_lr)
    epsilon_schedule = optax.linear_schedule(
        1.0, cfg.epsilon_end, int(cfg.total_timesteps * cfg.epsilon_fraction)
    )

    if cfg.agent_type == "rtu_rtrl":
        return QRCRtrl(
            qrc_cfg, env, env_params, q_network, h_network,
            q_optimizer, h_optimizer, epsilon_schedule,
        )
    return QRC(
        qrc_cfg, env, env_params, q_network, h_network,
        q_optimizer, h_optimizer, epsilon_schedule,
    )


def _episode_stats(info: dict) -> tuple[float, float]:
    """Mean episode return / length over completed episodes from lox info."""
    mask = info.get("returned_episode")
    if mask is None or not jnp.any(mask):
        return float("nan"), float("nan")
    returns = float(
        jnp.mean(info["returned_episode_returns"], where=mask)
    )
    lengths = float(
        jnp.mean(info["returned_episode_lengths"], where=mask)
    )
    return returns, lengths


def train_loop(agent, cfg: ExperimentConfig, logger=DummyLogger()):
    """Epoch loop: train -> eval -> Aim logger, with early stopping.

    Uses lox.spool to collect metrics that the Memorax algorithm emits via
    lox.log, then forwards scalar metrics + episode stats to the streaming-rtrrl
    Aim logger. Returns the best eval reward seen.
    """
    pprint(cfg, width=1)

    cfg.seed = cfg.seed or int(np.random.randint(1e6))
    logger.log_params(asdict(cfg))
    key = jax.random.key(cfg.seed)

    # lox.spool wraps train/evaluate so their internal lox.log calls are
    # returned as a `logs` dict instead of being emitted to a lox sink.
    #
    # IMPORTANT: jit the spooled fns. lox.spool runs make_jaxpr + eval_jaxpr on
    # EVERY call (spooling.py), so an un-jitted spooled fn re-lowers/re-compiles
    # its (large) lax.scan every epoch. On GPU that's merely wasteful; on the CPU
    # backend the LLVM ORC-JIT reserves a FIXED contiguous executable-code arena,
    # so the per-epoch executables accumulate and overflow it ("Unable to
    # allocate section memory", exit 139) after ~18 epochs for the big brax+RTRL
    # program. jitting compiles train/eval ONCE (num_steps is static) and reuses
    # it, which also removes the per-epoch re-trace cost on GPU. lox's own docs
    # prescribe jit(spool(fn)).
    init = jax.jit(agent.init)
    train = jax.jit(lox.spool(agent.train), static_argnums=(2,))
    eval_fn = jax.jit(lox.spool(agent.evaluate), static_argnums=(2,))

    key, init_key = jax.random.split(key)
    state = init(init_key)

    num_steps = cfg.total_timesteps // cfg.num_epochs

    logger["best_eval_reward"] = -jnp.inf
    steps_since_best = 0
    pbar = trange(cfg.num_epochs, mininterval=1)

    try:
        for i in pbar:
            key, train_key, eval_key = jax.random.split(key, 3)
            start = time.perf_counter()
            state, logs = train(train_key, state, num_steps)
            jax.block_until_ready(state)
            sps = num_steps / (time.perf_counter() - start)

            info = logs.pop("info", {})
            avg_r, avg_len = _episode_stats(info)

            metrics = {
                "train/SPS": sps,
                "train/episode_returns": avg_r,
                "train/episode_lengths": avg_len,
            }
            # Forward remaining logs (critic/td_error, actor/entropy, diag/*, ...).
            # lox stacks per-step scalars over the scan -> shape (num_steps,); we
            # summarize each epoch by its mean AND its max|.| (the max is the key
            # leading indicator for exponential blow-up: it catches the first
            # component whose per-step magnitude spikes).
            for k, v in logs.items():
                if not isinstance(v, jnp.ndarray):
                    continue
                if v.ndim == 0:
                    metrics[f"train/{k}"] = float(v)
                else:
                    metrics[f"train/{k}"] = float(jnp.mean(v))
                    metrics[f"train/{k}_max"] = float(jnp.max(jnp.abs(v)))

            # Aim step: use the loop-derived env-step count, NOT state.step.
            # state.step has been observed to read back as int64 garbage
            # (~±2^63) once training diverges, which scrambles metric ordering
            # in Aim. The loop counter is always a clean multiple of num_steps.
            global_step = (i + 1) * num_steps
            # Keep the raw counter as a metric so we can still see WHEN it goes
            # bad (a canary for state corruption during divergence).
            metrics["debug/state_step"] = float(state.step)

            pbar.set_description(f"ep{i} R={avg_r:.2f}", refresh=False)

            # EVAL ------------------------------------------------------------
            if cfg.eval_every and (
                i % cfg.eval_every == 0 or i == cfg.num_epochs - 1
            ):
                _, eval_logs = eval_fn(eval_key, state, cfg.eval_steps)
                eval_info = eval_logs.get("info", {})
                eval_avg, _ = _episode_stats(eval_info)
                if eval_avg == eval_avg:  # not NaN
                    metrics["eval/rewards"] = eval_avg
                    pbar.write(f"Eval reward: {eval_avg:.2f}")

                    if eval_avg > logger["best_eval_reward"]:
                        steps_since_best = 0
                        logger["best_eval_reward"] = eval_avg
                        metrics["eval/best_eval_reward"] = eval_avg
                    else:
                        steps_since_best += 1

            logger.log(metrics, step=global_step)

            if cfg.patience and steps_since_best >= cfg.patience:
                print(f"Early stopping patience {cfg.patience}")
                break
    except Exception as e:
        print("Exception in training loop!")
        raise e
    finally:
        logger.finalize()

    return logger["best_eval_reward"]


def run_experiment(train_fn, cfg: ExperimentConfig, project_name: str = "memorax-rtrl"):
    """Wrap a train callable with streaming-rtrrl's with_logger (Aim/W&B).

    with_logger calls `func(hparams)` when no backend is selected and
    `func(hparams, logger=logger)` otherwise, so we wrap train_fn in a function
    whose `logger` defaults to DummyLogger — matching that contract without
    forcing every experiment's train() to carry a default itself.
    """
    def wrapped(hparams, logger=DummyLogger()):
        return train_fn(hparams, logger)

    run_name = cfg.run_name or cfg.experiment
    with_logger(
        wrapped,
        cfg,
        logger_name=cfg.logging,
        project_name=project_name,
        aim_repo=cfg.log_repo,
        run_name=run_name,
        hparams_type=type(cfg),
    )
