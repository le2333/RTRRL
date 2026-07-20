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
import time
from dataclasses import asdict, dataclass, replace
from pprint import pprint
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import numpy as np
import optax
from logging_util import DummyLogger, with_logger
from memorax.algorithms import (
    QRC,
    RTRRL,
    IndependentRTRRL,
    IndependentRTRRLConfig,
    QRCConfig,
    QRCRtrl,
    RTRRLConfig,
    StreamAC,
    StreamACConfig,
    StreamACRtrl,
)
from memorax.algorithms.rtrrl.compatibility import normalize_legacy_config
from memorax.algorithms.rtrrl.compatibility import to_component_config
from memorax.algorithms.rtrrl.heads import RTRRLTDHead
from memorax.algorithms.rtrrl.lru import AAAI25LRU
from memorax.algorithms.rtrrl.program import (
    LegacyRTRRLEnvironmentAdapter,
    build_rtrrl_program,
)
from memorax.algorithms.rtrrl.types import RTRRLComponents
from memorax.networks import (
    RNN,
    FeatureExtractor,
    LRUCell,
    LRUConfig,
    Memoroid,
    Network,
    RTUCell,
    RTUConfig,
    heads,
)
from memorax.online_ac import (
    LegacyProgram,
    StandardProgramConfig,
    build_standard_program,
    legacy_env_adapter,
    legacy_normalization_config,
)
from memorax.utils.typing import Discrete
from tqdm import trange

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


def _identity(x):
    """Pass-through feature extractor (no params): returns x unchanged.

    obs/action/reward all already carry a trailing feature axis in the RTRRL
    pipeline (reward is [B, T, 1]), so identity concatenation yields the raw
    [obs | action | reward] vector without any learned encoding.
    """
    return x


def _reject_legacy_new_conflict(cfg):
    """Prevent silently mixing legacy scalar fields with a new program recipe."""

    for field in (
        "online_ac_config",
        "program_config",
        "meta_program_config",
        "standard_program_config",
    ):
        if getattr(cfg, field, None) is not None:
            raise ValueError(
                f"legacy/new config conflict: legacy builder received '{field}'"
            )


def _find_pipeline_state(value):
    if hasattr(value, "pipeline_state"):
        return value.pipeline_state
    for name in ("inner_state", "env_state"):
        if hasattr(value, name):
            found = _find_pipeline_state(getattr(value, name))
            if found is not None:
                return found
    return None


def _make_rtrrl_renderer(env):
    render = getattr(env, "render", None)
    if render is None:
        return None

    def render_evaluation(environment_states):
        pipeline_state = _find_pipeline_state(environment_states)
        if pipeline_state is None:
            return None
        first_environment = jax.tree.map(
            lambda value: value[:, 0],
            pipeline_state,
        )
        frames = render(first_environment)
        return np.asarray(frames) if frames is not None else None

    return render_evaluation


def build_networks(cfg: ExperimentConfig, action_space) -> tuple[Network, Network]:
    """Build actor/critic networks from cfg.agent_type and the action space.

    Layout: FeatureExtractor([o,(a),(r)]) -> RNN(RTUCell) -> head.
    Head is Categorical for discrete envs, Gaussian for continuous envs.
    """
    feat = cfg.encoder_dim
    observation_extractor = nn.Sequential((nn.Dense(feat), nn.relu))
    action_extractor = nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
    reward_extractor = nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
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
    """Build StreamAC-RTRL through the composable program lifecycle."""
    _reject_legacy_new_conflict(cfg)
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

    if cfg.agent_type == "rtu_tbptt":
        return StreamAC(ac_cfg, env, env_params, actor_network, critic_network)
    if cfg.agent_type not in ("rtu_rtrl", "lru_rtrl"):
        raise ValueError(
            "build_stream_ac_agent requires exact-RTRL agent_type "
            "'rtu_rtrl' or 'lru_rtrl'."
        )
    parts = StreamACRtrl(ac_cfg, env, env_params, actor_network, critic_network)
    adapter = legacy_env_adapter(env, env_params)
    program_config = StandardProgramConfig.from_legacy_parts(
        parts,
        normalization=legacy_normalization_config(env, cfg),
    )
    return LegacyProgram(
        build_standard_program(program_config, adapter),
        program_config,
    )


def build_rtrrl_agent(cfg: Any, env, env_params):
    """Build an RTRRL agent (shared LRU-RTRL torso -> Gaussian actor + V critic).

    Layout: FeatureExtractor([o,(a),(r)]) -> Memoroid(LRUCell) -> silu -> heads.
    Continuous control only (Gaussian actor); reproduces streaming-rtrrl's
    RNNActorCritic with a shared recurrent core feeding a linear actor + critic.

    use_encoder=False removes the parallel Dense encoders and feeds the RAW
    [obs, action, reward] into the LRU, whose read-out decouples the output width
    via lru_output_dim (through C, + D skip, then silu in RTRRL._forward) — the
    faithful streaming-rtrrl OnlineLRULayer(d_output, d_hidden) topology.
    """
    _reject_legacy_new_conflict(cfg)
    cfg = normalize_legacy_config(cfg)
    action_space = env.action_space(env_params)
    if isinstance(action_space, Discrete):
        raise ValueError(
            "RTRRL currently targets continuous-action envs (Gaussian actor)."
        )

    if cfg.profile == "aaai25_strict_lru":
        observation_dim = env.observation_space(env_params).shape[0]
        action_dim = action_space.shape[0]
        component_config = replace(
            to_component_config(cfg),
            observation_dim=observation_dim,
            action_dim=action_dim,
            discrete=False,
        )
        input_dim = observation_dim + (
            action_dim + 1 if component_config.meta_rl else 0
        )
        components = RTRRLComponents(
            recurrent=AAAI25LRU(
                input_dim=input_dim,
                hidden_dim=component_config.hidden_dim,
                output_dim=component_config.hidden_dim,
            ),
            head=RTRRLTDHead(
                action_dim=action_dim,
                discrete=False,
                f_align=False,
            ),
        )
        concrete_environment = legacy_env_adapter(
            env,
            env_params,
            strip_normalization=True,
        ).build_context["env"]
        environment = LegacyRTRRLEnvironmentAdapter(
            concrete_environment,
            env_params,
            component_config.num_envs,
            normalize_observation=component_config.normalize_observation,
            normalize_reward=component_config.normalize_reward,
        )
        program = build_rtrrl_program(
            component_config,
            components,
            environment,
        )
        return RTRRL.from_program(
            program,
            profile=cfg.profile,
            num_envs=cfg.num_envs,
            runtime_config=cfg,
            render_evaluation=_make_rtrrl_renderer(env),
        )

    use_encoder = getattr(cfg, "use_encoder", True)
    if use_encoder:
        feat = cfg.encoder_dim
        observation_extractor = nn.Sequential((nn.Dense(feat), nn.relu))
        action_extractor = (
            nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
        )
        reward_extractor = (
            nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
        )
        in_dim = feat * (3 if cfg.meta_rl else 1)
        lru_output_dim = None  # coupled to features (legacy behaviour)
    else:
        # Identity front-end: raw [obs, (action), (reward)] fed straight to the LRU.
        observation_extractor = _identity
        action_extractor = _identity if cfg.meta_rl else None
        reward_extractor = _identity if cfg.meta_rl else None
        obs_dim = env.observation_space(env_params).shape[0]
        act_dim = action_space.shape[0]
        in_dim = obs_dim + (act_dim + 1 if cfg.meta_rl else 0)
        # Decoupled read-out width (through C); default to hidden_dim like the
        # original d_output=d_hidden=32.
        lru_output_dim = getattr(cfg, "lru_output_dim", None) or cfg.hidden_dim

    feature_extractor = FeatureExtractor(
        observation_extractor=observation_extractor,
        action_extractor=action_extractor,
        reward_extractor=reward_extractor,
    )

    # Recurrent backbone: "lru" (Memoroid, linear SSM, the RTRL-HOP-533 baseline)
    # or "rtu" (RNN, complex rotation-decay + tanh => bounded state, gain tied to
    # nu_log with NO free gamma). RTU structurally lacks both divergence drivers
    # (unbounded state + self-aligned free gain), so it's the backbone-stability
    # probe. Both expose the same local_jacobian RTRL interface.
    backbone = getattr(cfg, "backbone", "lru")
    if backbone == "rtu":
        torso = RNN(
            cell=RTUCell(config=RTUConfig(features=in_dim, hidden_dim=cfg.hidden_dim))
        )
    elif backbone == "lru":
        torso = Memoroid(
            cell=LRUCell(
                config=LRUConfig(
                    features=in_dim,
                    hidden_dim=cfg.hidden_dim,
                    output_dim=lru_output_dim,
                )
            )
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
        update_trace_before_td=getattr(cfg, "update_trace_before_td", True),
        normalize_obs=getattr(cfg, "normalize_obs", False),
        normalize_reward=getattr(cfg, "normalize_reward", False),
        act_magnitude_factor=getattr(cfg, "act_magnitude_factor", 0.0),
        profile="memo_experimental",
        # sum->mean ablation: original averages log_prob/entropy over action dims.
        logprob_scale=(
            1.0 / action_space.shape[0]
            if getattr(cfg, "logprob_reduction", "sum") == "mean"
            else 1.0
        ),
    )
    parts = RTRRL(
        rtrrl_cfg,
        env,
        env_params,
        feature_extractor,
        torso,
        actor_head,
        critic_head,
        pred_head=pred_head,
        program_normalization=legacy_normalization_config(env, cfg),
        strip_environment_normalization=True,
    )
    return parts.as_legacy_program()


def build_independent_rtrrl_agent(cfg: ExperimentConfig, env, env_params):
    """Build strict two-path legacy RTRRL with no actor/critic state sharing."""
    cfg = normalize_legacy_config(cfg)
    if getattr(cfg, "pred_obs", False):
        raise ValueError("Independent RTRRL does not support pred_obs.")
    action_space = env.action_space(env_params)
    if isinstance(action_space, Discrete):
        raise ValueError("Independent RTRRL currently targets continuous-action envs.")

    use_encoder = getattr(cfg, "use_encoder", True)
    if use_encoder:
        feat = cfg.encoder_dim

        def make_feature_extractor():
            return FeatureExtractor(
                observation_extractor=nn.Sequential((nn.Dense(feat), nn.relu)),
                action_extractor=(
                    nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
                ),
                reward_extractor=(
                    nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
                ),
            )

        in_dim = feat * (3 if cfg.meta_rl else 1)
        lru_output_dim = None
    else:
        # Construct two distinct module objects even though these front-ends have
        # no trainable leaves.
        def make_feature_extractor():
            return FeatureExtractor(
                observation_extractor=_identity,
                action_extractor=_identity if cfg.meta_rl else None,
                reward_extractor=_identity if cfg.meta_rl else None,
            )

        obs_dim = env.observation_space(env_params).shape[0]
        act_dim = action_space.shape[0]
        in_dim = obs_dim + (act_dim + 1 if cfg.meta_rl else 0)
        lru_output_dim = getattr(cfg, "lru_output_dim", None) or cfg.hidden_dim

    backbone = getattr(cfg, "backbone", "lru")

    def make_torso():
        if backbone == "rtu":
            return RNN(
                cell=RTUCell(
                    config=RTUConfig(features=in_dim, hidden_dim=cfg.hidden_dim)
                )
            )
        if backbone == "lru":
            return Memoroid(
                cell=LRUCell(
                    config=LRUConfig(
                        features=in_dim,
                        hidden_dim=cfg.hidden_dim,
                        output_dim=lru_output_dim,
                    )
                )
            )
        raise ValueError(f"backbone '{backbone}' not supported; use 'lru' or 'rtu'.")

    independent_cfg = IndependentRTRRLConfig(
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
        pred_obs=False,
        pred_coeff=getattr(cfg, "pred_coeff", 1.0),
        update_trace_before_td=getattr(cfg, "update_trace_before_td", True),
        logprob_scale=(
            1.0 / action_space.shape[0]
            if getattr(cfg, "logprob_reduction", "sum") == "mean"
            else 1.0
        ),
    )
    return IndependentRTRRL(
        independent_cfg,
        env,
        env_params,
        make_feature_extractor(),
        make_torso(),
        heads.Gaussian(
            action_dim=action_space.shape[0],
            bound=getattr(cfg, "bound_actor", False),
        ),
        make_feature_extractor(),
        make_torso(),
        heads.VNetwork(),
    )


def build_qrc_networks(
    cfg: ExperimentConfig, num_actions: int
) -> tuple[Network, Network]:
    """Build Q/H networks for QRC: FeatureExtractor -> RNN(RTUCell) -> DiscreteQNetwork."""
    feat = cfg.encoder_dim
    observation_extractor = nn.Sequential((nn.Dense(feat), nn.relu))
    action_extractor = nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
    reward_extractor = nn.Sequential((nn.Dense(feat), nn.relu)) if cfg.meta_rl else None
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
    q_network = Network(feature_extractor=feature_extractor, torso=torso, head=head)
    h_network = Network(feature_extractor=feature_extractor, torso=torso, head=head)
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
            qrc_cfg,
            env,
            env_params,
            q_network,
            h_network,
            q_optimizer,
            h_optimizer,
            epsilon_schedule,
        )
    return QRC(
        qrc_cfg,
        env,
        env_params,
        q_network,
        h_network,
        q_optimizer,
        h_optimizer,
        epsilon_schedule,
    )


def _episode_stats(info: dict) -> tuple[float, float]:
    """Mean episode return / length over completed episodes from lox info."""
    mask = info.get("returned_episode")
    if mask is None or not jnp.any(mask):
        return float("nan"), float("nan")
    returns = float(jnp.mean(info["returned_episode_returns"], where=mask))
    lengths = float(jnp.mean(info["returned_episode_lengths"], where=mask))
    return returns, lengths


def _historical_rtrrl_metrics(
    summary,
    *,
    log_td_lr: bool,
    log_rnn_lr: bool,
    log_norms: bool,
):
    """Translate a closed-program summary to the pinned AAAI25 metric schema."""

    metrics = {
        "steps": summary.steps,
        "mean_reward": summary.mean_reward,
        "num_episodes": summary.num_episodes,
        "mean_delta": summary.mean_delta,
        "mean_r_bar": summary.mean_r_bar,
        "mean_v": summary.mean_v,
        "total_td_loss": summary.total_td_loss,
        "actor_loss": summary.actor_loss,
        "critic_loss": summary.critic_loss,
        "entropy": summary.entropy,
        "v_targ": summary.v_targ,
    }
    if summary.magnitude_loss is not None:
        metrics["magnitude_loss"] = summary.magnitude_loss
    if log_td_lr:
        metrics["lr/td"] = summary.learning_rate_td
    if log_rnn_lr:
        metrics["lr/rnn"] = summary.learning_rate_rnn
    if log_norms:
        metrics.update(
            {f"norms/{key}": value for key, value in summary.norms.items()}
        )
    return metrics


def _log_historical_rtrrl_epoch(
    logger,
    summary,
    *,
    epoch_index: int,
    steps_per_epoch: int,
    log_every: int,
    log_td_lr: bool,
    log_rnn_lr: bool,
    log_norms: bool,
    evaluation_summary=None,
    render_evaluation=None,
):
    """Preserve historical logger cadence, step, best-eval, and video semantics."""

    metrics = (
        _historical_rtrrl_metrics(
            summary,
            log_td_lr=log_td_lr,
            log_rnn_lr=log_rnn_lr,
            log_norms=log_norms,
        )
        if epoch_index % log_every == 0
        else {}
    )
    eval_reward = None
    video_frames = None
    if evaluation_summary is not None:
        info = evaluation_summary.info
        completed = info["returned_episode"]
        if bool(jnp.any(completed)):
            eval_reward = float(
                jnp.mean(
                    info["returned_episode_returns"],
                    where=completed,
                )
            )
            if render_evaluation is not None:
                video_frames = render_evaluation(info["environment_state"])
    if eval_reward is not None:
        metrics["eval/rewards"] = eval_reward
        if video_frames is not None:
            logger.log_video(
                "env/video",
                video_frames,
                fps=30,
                caption=f"Reward: {eval_reward:.2f}",
            )
        if eval_reward > logger["best_eval_reward"]:
            logger["best_eval_reward"] = eval_reward
            metrics["eval/best_eval_reward"] = eval_reward
    logger.log(metrics, step=(epoch_index + 1) * steps_per_epoch)


def _train_strict_rtrrl_loop(agent: RTRRL, cfg, logger):
    """Run the closed program and translate its summaries only on the host."""

    key = jax.random.key(cfg.seed)
    init = jax.jit(agent.program.init_fn)
    train = jax.jit(agent.program.train_epoch_fn, static_argnums=(2,))
    evaluate = jax.jit(agent.program.evaluate_fn, static_argnums=(2,))

    key, init_key = jax.random.split(key)
    state = init(init_key)
    steps_per_epoch = (
        cfg.total_timesteps // cfg.num_epochs // agent.num_envs
    )
    if steps_per_epoch < 1:
        raise ValueError("strict RTRRL epoch must contain at least one transition")

    logger["best_eval_reward"] = -jnp.inf
    steps_since_best = 0
    runtime_config = agent.runtime_config
    log_td_lr = bool(
        getattr(
            getattr(runtime_config, "optimizer_params_td", None),
            "decay_type",
            None,
        )
    )
    log_rnn_lr = bool(
        getattr(
            getattr(runtime_config, "optimizer_params_rnn", None),
            "decay_type",
            None,
        )
    )
    log_norms = getattr(runtime_config, "log_norms", False)

    try:
        for epoch_index in trange(cfg.num_epochs, mininterval=1):
            key, train_key, eval_key = jax.random.split(key, 3)
            state, summary = train(
                train_key,
                state,
                steps_per_epoch,
            )
            jax.block_until_ready((state, summary))

            evaluation_summary = None
            if cfg.eval_every and (
                epoch_index % cfg.eval_every == 0
                or epoch_index == cfg.num_epochs - 1
            ):
                _, evaluation_summary = evaluate(
                    eval_key,
                    state,
                    cfg.eval_steps,
                )
                jax.block_until_ready(evaluation_summary)

            environment_config = getattr(
                runtime_config, "env_params", None
            )
            render_every = (
                cfg.eval_every
                * getattr(runtime_config, "render_every_evals", 10)
            )
            should_render = bool(
                evaluation_summary is not None
                and getattr(environment_config, "render", False)
                and (
                    (
                        render_every
                        and epoch_index % render_every == 0
                        and epoch_index > 0
                    )
                    or epoch_index == cfg.num_epochs - 1
                )
            )
            previous_best = logger["best_eval_reward"]
            _log_historical_rtrrl_epoch(
                logger,
                summary,
                epoch_index=epoch_index,
                steps_per_epoch=steps_per_epoch,
                log_every=cfg.log_every,
                log_td_lr=log_td_lr,
                log_rnn_lr=log_rnn_lr,
                log_norms=log_norms,
                evaluation_summary=evaluation_summary,
                render_evaluation=(
                    agent.render_evaluation if should_render else None
                ),
            )
            if logger["best_eval_reward"] > previous_best:
                steps_since_best = 0
            elif evaluation_summary is not None:
                steps_since_best += 1
            if cfg.patience and steps_since_best >= cfg.patience:
                break
    finally:
        logger.finalize()

    return logger["best_eval_reward"]


def train_loop(agent, cfg: ExperimentConfig, logger=DummyLogger()):
    """Epoch loop: train -> eval -> Aim logger, with early stopping.

    Uses lox.spool to collect metrics that the Memorax algorithm emits via
    lox.log, then forwards scalar metrics + episode stats to the streaming-rtrrl
    Aim logger. Returns the best eval reward seen.
    """
    pprint(cfg, width=1)

    cfg.seed = cfg.seed or int(np.random.randint(1e6))
    logger.log_params(asdict(cfg))
    if isinstance(agent, RTRRL) and agent.profile == "aaai25_strict_lru":
        return _train_strict_rtrrl_loop(agent, cfg, logger)

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
            if cfg.eval_every and (i % cfg.eval_every == 0 or i == cfg.num_epochs - 1):
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
