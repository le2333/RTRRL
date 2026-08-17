"""RTRRL: one recurrent torso shared by an actor and a critic.

    RTRRL             the order things happen in, and the scan
    Environment       where every stream is
    Normalization     the scales the environment's numbers are read through
    Core              a torso, two heads, and everything that couples them
      Torso           the shared recurrent representation and following copy
      Actor / Critic  a readout, and the objectives it names

Three blocks and two rule groups: the torso is clipped alone and the two
readouts step together. Its private recurrent subgraph selects an LRU or RTU
and one differentiation method supported by that kernel.

Rebuilt from ``../RTRRL-AAAI25/rtrrl.py``. Driven against it by
``tests/test_rtrrl_parity.py``; behaviour in ``tests/test_rtrrl.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from memorax.building import BuildContext, ComponentBuilder, ComponentFamily
from memorax.networks.backbones import backbone
from memorax.networks.components import LayerNorm
from memorax.networks.readouts import ACTOR_HEAD_FAMILY, CRITIC_HEAD_FAMILY
from memorax.networks.sequence import PLACES, Sequence
from memorax.networks.sequence_models.lru import LRU_DIFFERENTIATION_FAMILY
from memorax.networks.sequence_models.rtu import RTU_DIFFERENTIATION_FAMILY
from memorax.observability.metrics import metric_names
from memorax.parameters import describe_parameters, group, param, structure
from memorax.readings import reading, readings, taken
from memorax.rl import (
    EnvironmentStreams,
    InteractionNormalization,
    NormalizationState,
    action_classes,
    action_dim,
    broadcast_stream,
    encode_feedback,
    make_optax_rule,
    make_td0,
    select_ended,
)
from memorax.rl.normalization import (
    DISCOUNTED_NORMALIZATION_FAMILY,
    NORMALIZATION_FAMILY,
)
from memorax.rl.updates import DRTRRL, STEP_FAMILY, Adam, make_d_rtrrl_rule
from memorax.runtime import ObservationSchema
from memorax.utils import Timestep
from memorax.utils.axes import (
    add_feature_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils.trees import subtree_norms

from .contract import (
    ActionDecision,
    EvaluationConfig,
    InteractionMetrics,
    StepMetrics,
)

TORSO_GROUP = "torso"
HEAD_GROUP = "heads"


# --------------------------------------------------------------- configuration
@dataclass(frozen=True)
class RTRRLConfig:
    """Everything the kernel reads that does not change during a run."""

    num_envs: int
    gamma: float = 0.99

    lambda_pi: float = 0.9
    lambda_v: float = 0.9
    lambda_rnn: float = 0.9

    eta_pi: float = 1.0
    eta_f: float = 1.0
    entropy_rate: float = 1e-5

    torso_optimizer: Adam | DRTRRL = field(default_factory=lambda: Adam(lr=1e-4))
    heads_optimizer: Adam | DRTRRL = field(default_factory=lambda: Adam(lr=1e-4))
    torso_grad_clip: float = 1.0
    torso_follow: float = 1.0

    meta_rl: bool = True
    # How many actions the environment names, or None when it measures them.
    # Only the meta-RL feedback input reads it, and only to widen an integer
    # action into something a concatenation can carry.
    action_classes: int | None = None


RTRRL_OPTIMIZERS = STEP_FAMILY.restricted("adam", "d_rtrrl")


@dataclass(frozen=True)
class LruTorso:
    """RTRRL's projection width and LRU-specific recurrent choices."""

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512))
    feature_dim: int = param(valid=(1, 4096), search=(16, 256))
    differentiation: str = structure(branches=LRU_DIFFERENTIATION_FAMILY.branches)


@dataclass(frozen=True)
class RtuTorso:
    """RTRRL's RTU-specific recurrent choices.

    No readout width, because the RTU has none to set: its output is its two
    carries concatenated, so ``hidden_dim`` fixes it.
    """

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512))
    differentiation: str = structure(branches=RTU_DIFFERENTIATION_FAMILY.branches)


def _construct_torso(selection, builder, *, features: int):
    """Build RTRRL's differentiated recurrent subgraph, and nothing before it.

    The cell reads the observation, and it reads it directly. There used to be
    a ``FFN -> LayerNorm -> Tanh`` in front, which existed only because
    ``LRUConfig.features`` was being used as both the input width and the
    readout width: the cell was configured to emit ``feature_dim``, so
    something had to widen the observation to ``feature_dim`` first. Setting
    ``output_dim`` separates the two, which is what it was added for, and the
    projection has nothing left to do.

    That projection was never the published algorithm's. Its cell takes the
    observation as it arrives and normalises afterwards, and the parity suite
    could not have said so -- it re-hosts the published *wiring* on these same
    networks precisely so a mismatch cannot come from the networks, which
    leaves what is in front of the cell unexamined by construction.
    """

    settings = selection.parameters
    recurrent = backbone(
        selection.kind,
        features=features,
        hidden_dim=settings.hidden_dim,
        # Only the LRU has a readout to size; see RtuTorso.
        output_dim=getattr(settings, "feature_dim", None),
    )
    # After the cell, and affine-free, which is where and what the published
    # implementation's is.
    network = Sequence(
        components=(*recurrent, LayerNorm(use_scale=False, use_bias=False))
    )
    family = {
        "lru": LRU_DIFFERENTIATION_FAMILY,
        "rtu": RTU_DIFFERENTIATION_FAMILY,
    }[selection.kind]
    differentiation = builder.build(
        family,
        f"{selection.path}.differentiation",
        core=network.core,
    )
    return network, differentiation


RTRRL_TORSO_FAMILY = ComponentFamily(
    branches={"lru": LruTorso, "rtu": RtuTorso},
    construct=_construct_torso,
)


@dataclass(frozen=True)
class TorsoParameters:
    backbone: str = structure(branches=RTRRL_TORSO_FAMILY.branches)
    optimizer: str = structure(branches=RTRRL_OPTIMIZERS.branches)
    grad_clip: float = param(valid=(0.0, 100.0), search=(0.0, 10.0))
    follow: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))


@dataclass(frozen=True)
class HeadsParameters:
    optimizer: str = structure(branches=RTRRL_OPTIMIZERS.branches)


@dataclass(frozen=True)
class ActorParameters:
    head: str = structure(branches=ACTOR_HEAD_FAMILY.branches)


@dataclass(frozen=True)
class CriticParameters:
    head: str = structure(branches=CRITIC_HEAD_FAMILY.branches)


@dataclass(frozen=True)
class NormalizationParameters:
    observation: str = structure(branches=NORMALIZATION_FAMILY.branches)
    reward: str = structure(branches=DISCOUNTED_NORMALIZATION_FAMILY.branches)


@dataclass(frozen=True)
class RTRRLParameters:
    torso: TorsoParameters = group(of=TorsoParameters)
    heads: HeadsParameters = group(of=HeadsParameters)
    actor: ActorParameters = group(of=ActorParameters)
    critic: CriticParameters = group(of=CriticParameters)
    normalization: NormalizationParameters = group(of=NormalizationParameters)
    gamma: float = param(valid=(0.5, 0.9999), search=(0.9, 0.9999))
    lambda_pi: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    lambda_v: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    lambda_rnn: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    eta_pi: float = param(valid=(0.0, 100.0), search=(0.0, 10.0))
    eta_f: float = param(valid=(0.0, 100.0), search=(0.0, 10.0))
    entropy_rate: float = param(valid=(0.0, 1.0), search=(1e-8, 1e-2), log=True)
    meta_rl: bool = param(valid=[False, True], search=[True])


PARAMETERS = describe_parameters(RTRRLParameters)


# ----------------------------------------------------------------------- state
class Recurrence(struct.PyTreeNode):
    """Where the sequence is, and what it owes the past."""

    carry: Any
    differentiation_state: Any


class BlockState(struct.PyTreeNode):
    """A readout's parameters and the trace that decides how far they move."""

    params: Any
    traces: Any


class TorsoState(struct.PyTreeNode):
    """The shared block: the same two things, plus what only sharing needs."""

    params: Any
    traces: Any
    slow_params: Any
    recurrence: Recurrence


class CoreState(struct.PyTreeNode):
    """What the algorithm carries, one field per thing that owns one."""

    torso: TorsoState
    actor: BlockState
    critic: BlockState
    rule: Any
    value: Any
    emphasis: Any


# -------------------------------------------------------------------- readings
@dataclass(frozen=True)
class BlockReports:
    """Which position-split torso readings to take."""

    grad_norm: bool = reading(at="grad_norm", split=True)
    trace_norm: bool = reading(at="trace_norm", split=True)


@dataclass(frozen=True)
class HeadReports:
    """Which whole-readout readings to take."""

    grad_norm: bool = reading(at="grad_norm")
    trace_norm: bool = reading(at="trace_norm")


@dataclass(frozen=True)
class GroupReports:
    """Which of one rule group's readings to take."""

    step_size: bool = reading(at="step_size")


@dataclass(frozen=True)
class Reports:
    """Which readings to take, in the shape of what produces them.

    Two levels, because three blocks step in two groups.
    """

    log_prob: bool = reading(at="forward.actor.log_prob")
    entropy: bool = reading(at="forward.actor.entropy")
    value: bool = reading(at="forward.critic.value")
    td_error: bool = reading(at="update.td_error")
    emphasis: bool = reading(at="update.emphasis")
    torso: BlockReports = readings(of=BlockReports, at="update.torso")
    actor: HeadReports = readings(of=HeadReports, at="update.actor")
    critic: HeadReports = readings(of=HeadReports, at="update.critic")
    torso_step: GroupReports = readings(of=GroupReports, at="update.torso_step")
    heads_step: GroupReports = readings(of=GroupReports, at="update.heads_step")


PARTS: tuple[str, ...] = PLACES
REPORTS = Reports()
TRAINING_METRICS: tuple[str, ...] = taken(REPORTS, parts=PARTS)
METRICS: tuple[str, ...] = metric_names("train", TRAINING_METRICS) + metric_names(
    "eval"
)
OBSERVATIONS = ObservationSchema(
    reward="interaction.reward",
    done="interaction.done",
    terminal="interaction.terminal",
    observation="interaction.observation",
    next_observation="interaction.next_observation",
    action="interaction.action",
    series=TRAINING_METRICS,
)


class ActorForward(struct.PyTreeNode):
    """What the policy answered on the pass that chose."""

    log_prob: Any = None
    entropy: Any = None


class CriticForward(struct.PyTreeNode):
    """What the critic answered."""

    value: Any = None


class ForwardMetrics(struct.PyTreeNode):
    """One field per head."""

    actor: ActorForward = ActorForward()
    critic: CriticForward = CriticForward()


class BlockUpdate(struct.PyTreeNode):
    """How big what went into one block's step was."""

    grad_norm: Any = None
    trace_norm: Any = None


class GroupUpdate(struct.PyTreeNode):
    """What one rule group's step did."""

    step_size: Any = None


class UpdateMetrics(struct.PyTreeNode):
    """Three blocks, two groups, and two quantities belonging to neither."""

    td_error: Any = None
    emphasis: Any = None
    torso: BlockUpdate = BlockUpdate()
    actor: BlockUpdate = BlockUpdate()
    critic: BlockUpdate = BlockUpdate()
    torso_step: GroupUpdate = GroupUpdate()
    heads_step: GroupUpdate = GroupUpdate()


class RTRRLState(struct.PyTreeNode):
    """Everything the kernel carries, one field per layer that writes one."""

    step: Any
    update_step: Any
    timestep: Timestep
    terminal: Any

    env_state: Any
    scales: NormalizationState
    core: CoreState


# ---------------------------------------------------------- shapes and streams
def _as_batch(tree):
    """Put the stream axis back as a length of one."""

    return jax.tree.map(lambda leaf: leaf[None], tree)


def _from_batch(tree):
    """Drop the stream axis again, so ``vmap`` stacks one per stream."""

    return jax.tree.map(lambda leaf: leaf[0], tree)


# --------------------------------------------------------- the two rule groups
def _adam_rule(base: Adam, *, clip: float):
    """Adam over the combined ascent, which is what the paper's RTRRL steps."""

    chain: list[Any] = []
    if clip:
        chain.append(optax.clip_by_global_norm(clip))
    chain.extend(
        (
            optax.scale_by_adam(b1=base.b1, b2=base.b2, eps=base.eps),
            optax.scale(base.lr),
        )
    )
    return make_optax_rule(optax.chain(*chain), rate=base.lr)


def _group_rule(step: Adam | DRTRRL, *, clip: float):
    """Whichever rule a group's optimizer names, over that group's blocks.

    A group's entries are its blocks, which is the grouping the D-RTRRL rule
    normalizes over: the torso is its own unit, and the two readouts step
    together but are two units, so a large actor trace cannot spend the critic's
    step. What they share under one selection is ``eta``, not a unit ball.
    """

    if isinstance(step, DRTRRL):
        return make_d_rtrrl_rule(step, clip=clip)
    return _adam_rule(step, clip=clip)


def make_rules(cfg: RTRRLConfig):
    """One rule per group, both answering the same contract."""

    return {
        TORSO_GROUP: _group_rule(cfg.torso_optimizer, clip=cfg.torso_grad_clip),
        HEAD_GROUP: _group_rule(cfg.heads_optimizer, clip=0.0),
    }


# ------------------------------------------------------------ shared algebra
def _initial_traces(params, num_envs):
    """One trace per parameter per stream, which is what online means."""

    return jax.tree.map(
        lambda parameter: jnp.zeros((num_envs, *parameter.shape)), params
    )


def _advance_trace(incoming, gradient, *, decay, reset_before, emphasis):
    """RTRRL's post-transition trace recurrence for one semantic block."""

    return jax.tree.map(
        lambda old, grad: (
            decay * (1 - broadcast_stream(reset_before, old)) * old
            + broadcast_stream(emphasis, grad) * grad
        ),
        incoming,
        gradient,
    )


def _gradient_norms(module, tree):
    """One norm per algorithmic position, per stream."""

    grouped = module.split(tree) if hasattr(module, "split") else {"readout": tree}
    return subtree_norms(grouped, streams=True)


def _head_gradient_norm(tree):
    """One per-stream norm for a readout that has no sequence positions."""

    return subtree_norms({"head": tree}, streams=True)["head"]


# ------------------------------------------------------------ the shared block
class Torso:
    """The recurrent representation shared by both heads."""

    def __init__(self, cfg: RTRRLConfig, network: Any, differentiation: Any) -> None:
        self.cfg = cfg
        self._network = network
        self._differentiation = differentiation
        self._decay = cfg.gamma * cfg.lambda_rnn

    @property
    def carry_shape(self):
        return (self.cfg.num_envs, None)

    def _input(self, timestep: Timestep):
        """The one vector the sequence sees."""

        obs, done, action, reward = timestep
        if not self.cfg.meta_rl:
            return obs
        # Widened first, so what an ended stream carries is the zero vector
        # rather than the one-hot of whichever action happens to be numbered 0.
        action = encode_feedback(action, classes=self.cfg.action_classes)
        ended = add_feature_axis(done)
        return jnp.concatenate(
            [
                obs,
                jnp.where(ended, jnp.zeros_like(action), action),
                jnp.where(ended, jnp.zeros_like(reward), reward),
            ],
            axis=-1,
        )

    def apply(self, params, timestep: Timestep, recurrence: Recurrence):
        """One forward pass over one sequence-shaped step."""

        _, done, _, _ = timestep
        (carry, differentiation_state), output = self._network.walk(
            params,
            self._input(timestep),
            done=done,
            carries=recurrence.carry,
            differentiation_state=recurrence.differentiation_state,
            differentiation=self._differentiation,
        )
        return (
            Recurrence(carry=carry, differentiation_state=differentiation_state),
            output,
        )

    def init(self, keys, timestep: Timestep) -> TorsoState:
        """Fresh online state for the shared block, and a copy for it to follow."""

        param_key, torso_key, dropout_key = keys
        _, done, _, _ = timestep
        carry = self._network.initialize_carry(jax.random.key(0), self.carry_shape)
        differentiation_state = self._differentiation.initialize(
            param_key, self.carry_shape
        )
        with self._differentiation.initialization():
            variables = self._network.init(
                {"params": param_key, "torso": torso_key, "dropout": dropout_key},
                self._input(timestep),
                done=done,
                initial_carry=carry,
            )
        params = variables["params"]
        return TorsoState(
            params=params,
            traces=_initial_traces(params, self.cfg.num_envs),
            slow_params=params,
            recurrence=Recurrence(
                carry=carry, differentiation_state=differentiation_state
            ),
        )

    def reset(self, key, state: TorsoState) -> TorsoState:
        """The same parameters with the sequence begun again."""

        return state.replace(
            recurrence=Recurrence(
                carry=self._network.initialize_carry(key, self.carry_shape),
                differentiation_state=self._differentiation.initialize(
                    key, self.carry_shape
                ),
            )
        )

    def advance_trace(self, incoming, gradient, *, reset_before, emphasis):
        return _advance_trace(
            incoming,
            gradient,
            decay=self._decay,
            reset_before=reset_before,
            emphasis=emphasis,
        )

    def gradient_norms(self, tree):
        return _gradient_norms(self._network, tree)

    def followed(self, params, slow_params):
        """The reading copy takes one step toward the copy that was updated."""

        if self.cfg.torso_follow == 1.0:
            return params
        return optax.incremental_update(params, slow_params, self.cfg.torso_follow)


# ------------------------------------------------------------ the two readouts
class Actor:
    """The policy. It chooses, and it names the two directions it ascends."""

    def __init__(self, cfg: RTRRLConfig, head: Any) -> None:
        self.cfg = cfg
        self._head = head
        self._decay = cfg.gamma * cfg.lambda_pi

    def init(self, key, hidden, timestep: Timestep) -> BlockState:
        _, done, action, reward = timestep
        params = self._head.init(
            {"params": key}, hidden, action=action, reward=reward, done=done
        )["params"]
        return BlockState(
            params=params, traces=_initial_traces(params, self.cfg.num_envs)
        )

    def apply(self, params, hidden, timestep: Timestep):
        _, done, action, reward = timestep
        dist, _ = self._head.apply(
            {"params": params}, hidden, action=action, reward=reward, done=done
        )
        return dist

    def advance_trace(self, incoming, gradient, *, reset_before, emphasis):
        return _advance_trace(
            incoming,
            gradient,
            decay=self._decay,
            reset_before=reset_before,
            emphasis=emphasis,
        )

    def gradient_norms(self, tree):
        return _head_gradient_norm(tree)

    def traced_objective(self, params, hidden, timestep: Timestep, action):
        """What ascends through the trace: the log-probability of what was done."""

        dist = self.apply(params, hidden, timestep)
        return self.cfg.eta_pi * remove_time_axis(dist.log_prob(action))[0]

    def immediate_objective(self, params, hidden, timestep: Timestep):
        """What ascends immediately: entropy, on the step it arises."""

        dist = self.apply(params, hidden, timestep)
        return self.cfg.entropy_rate * remove_time_axis(dist.entropy())[0]


class Critic:
    """The value. It reads, and it ascends its own reading."""

    def __init__(self, cfg: RTRRLConfig, head: Any) -> None:
        self.cfg = cfg
        self._head = head
        self._decay = cfg.gamma * cfg.lambda_v

    def init(self, key, hidden, timestep: Timestep) -> BlockState:
        _, done, action, reward = timestep
        params = self._head.init(
            {"params": key}, hidden, action=action, reward=reward, done=done
        )["params"]
        return BlockState(
            params=params, traces=_initial_traces(params, self.cfg.num_envs)
        )

    def apply(self, params, hidden, timestep: Timestep):
        _, done, action, reward = timestep
        value, _ = self._head.apply(
            {"params": params}, hidden, action=action, reward=reward, done=done
        )
        return remove_feature_axis(remove_time_axis(value))

    def advance_trace(self, incoming, gradient, *, reset_before, emphasis):
        return _advance_trace(
            incoming,
            gradient,
            decay=self._decay,
            reset_before=reset_before,
            emphasis=emphasis,
        )

    def gradient_norms(self, tree):
        return _head_gradient_norm(tree)

    def traced_objective(self, params, hidden, timestep: Timestep):
        """What ascends through the trace: the value itself, with no error in it."""

        return self.apply(params, hidden, timestep)[0]


# --------------------------------------------------------------- the algorithm
class Core:
    """A torso, two heads, and everything that couples them."""

    def __init__(
        self,
        cfg: RTRRLConfig,
        torso_network: Any,
        torso_differentiation: Any,
        actor_head: Any,
        critic_head: Any,
        reports: Reports = Reports(),
    ) -> None:
        self.cfg = cfg
        self.reports = reports
        self.torso = Torso(cfg, torso_network, torso_differentiation)
        self.actor = Actor(cfg, actor_head)
        self.critic = Critic(cfg, critic_head)
        self.td0 = make_td0()
        self.rules = make_rules(cfg)
        self.group_of = {
            "torso": TORSO_GROUP,
            "actor": HEAD_GROUP,
            "critic": HEAD_GROUP,
        }

    def _grouped(self, by_name):
        """Gather the per-block trees into the groups a rule steps over."""

        groups: dict[str, dict] = {group: {} for group in self.rules}
        for name, tree in by_name.items():
            groups[self.group_of[name]][name] = tree
        return groups

    def init(self, keys, timestep: Timestep) -> CoreState:
        """Fresh online state for all three blocks and both rules."""

        torso_keys, actor_key, critic_key = keys
        torso = self.torso.init(torso_keys, timestep)
        _, hidden = self.torso.apply(torso.params, timestep, torso.recurrence)
        actor = self.actor.init(actor_key, hidden, timestep)
        critic = self.critic.init(critic_key, hidden, timestep)

        params = self._grouped(
            {"torso": torso.params, "actor": actor.params, "critic": critic.params}
        )
        traces = self._grouped(
            {"torso": torso.traces, "actor": actor.traces, "critic": critic.traces}
        )
        return CoreState(
            torso=torso,
            actor=actor,
            critic=critic,
            rule={
                group: rule.init(params=params[group], traces=traces[group])
                for group, rule in self.rules.items()
            },
            value=jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32),
            emphasis=jnp.ones((self.cfg.num_envs,), dtype=jnp.float32),
        )

    def reset(self, key, state: CoreState) -> CoreState:
        return state.replace(torso=self.torso.reset(key, state.torso))

    def act(
        self, key, state: CoreState, timestep: Timestep, *, deterministic: bool
    ) -> tuple[Recurrence, Any, ActorForward]:
        """Run forward once and choose, learning nothing."""

        recurrence, hidden = self.torso.apply(
            state.torso.slow_params, timestep.to_sequence(), state.torso.recurrence
        )
        dist = self.actor.apply(state.actor.params, hidden, timestep.to_sequence())
        if deterministic:
            return recurrence, remove_time_axis(dist.mode()), ActorForward()
        action, log_prob = dist.sample_and_log_prob(seed=key)
        return (
            recurrence,
            remove_time_axis(action),
            ActorForward(
                log_prob=remove_time_axis(log_prob),
                entropy=remove_time_axis(dist.entropy()),
            ),
        )

    def _per_stream(self, key, state: CoreState, timestep: Timestep):
        """One forward and two backward passes per stream, routed to the blocks."""

        def one(torso_params, actor_params, critic_params, timestep, recurrence, key):
            timestep = _as_batch(timestep)
            recurrence = _as_batch(jax.lax.stop_gradient(recurrence))

            def forward(params):
                advanced, hidden = self.torso.apply(params, timestep, recurrence)
                return hidden, advanced

            hidden, upstream, advanced = jax.vjp(forward, torso_params, has_aux=True)

            dist = self.actor.apply(actor_params, hidden, timestep)
            action, log_prob = dist.sample_and_log_prob(seed=key)

            actor_traced, actor_upward = jax.grad(
                self.actor.traced_objective, argnums=(0, 1)
            )(actor_params, hidden, timestep, action)
            critic_traced, critic_upward = jax.grad(
                self.critic.traced_objective, argnums=(0, 1)
            )(critic_params, hidden, timestep)
            # A gate would be a factor on one of these two terms.
            torso_traced = upstream(actor_upward + critic_upward)[0]

            actor_direct, direct_upward = jax.grad(
                self.actor.immediate_objective, argnums=(0, 1)
            )(actor_params, hidden, timestep)
            torso_direct = upstream(direct_upward)[0]

            value = self.critic.apply(critic_params, hidden, timestep)[0]
            return (
                _from_batch(advanced),
                remove_time_axis(action)[0],
                value,
                {
                    "torso": torso_traced,
                    "actor": actor_traced,
                    "critic": critic_traced,
                },
                {"torso": torso_direct, "actor": actor_direct},
                ActorForward(
                    log_prob=(
                        remove_time_axis(log_prob)[0] if self.reports.log_prob else None
                    ),
                    entropy=(
                        remove_time_axis(dist.entropy())[0]
                        if self.reports.entropy
                        else None
                    ),
                ),
            )

        return jax.vmap(one, in_axes=(None, None, None, 0, 0, 0))(
            state.torso.slow_params,
            state.actor.params,
            state.critic.params,
            timestep.to_sequence(),
            state.torso.recurrence,
            jax.random.split(key, self.cfg.num_envs),
        )

    def _td_error(self, state, timestep, value, terminal):
        return self.td0(
            reward=timestep.reward,
            value=state.value,
            next_value=value,
            terminal=terminal,
            gamma=self.cfg.gamma,
        )

    def _take_updates(self, state, traced, direct, delta, step):
        """Apply the torso rule once and the joint readout rule once."""

        direct = {**direct, "critic": jax.tree.map(jnp.zeros_like, traced["critic"])}
        params = {
            "torso": state.torso.params,
            "actor": state.actor.params,
            "critic": state.critic.params,
        }
        grouped_params = self._grouped(params)
        grouped_traces = self._grouped(
            {
                "torso": state.torso.traces,
                "actor": state.actor.traces,
                "critic": state.critic.traces,
            }
        )
        grouped_direct = self._grouped(direct)
        deltas = {TORSO_GROUP: delta * self.cfg.eta_f, HEAD_GROUP: delta}
        taken = {
            group: rule.apply(
                grouped_traces[group],
                grouped_direct[group],
                state.rule[group],
                delta=deltas[group],
                step=step,
                params=grouped_params[group],
            )
            for group, rule in self.rules.items()
        }
        stepped = {
            name: jax.tree.map(
                lambda parameter, update: parameter + update,
                params[name],
                taken[self.group_of[name]].updates[name],
            )
            for name in params
        }
        return stepped, taken, grouped_traces

    def _advance_traces(self, state, grouped_traces, traced, reset_before):
        """Advance each block's trace after the current update has used it."""

        emphasis = (
            self.cfg.gamma * state.emphasis * (1 - reset_before) + reset_before
        ).astype(jnp.float32)
        advanced = {
            name: block.advance_trace(
                grouped_traces[self.group_of[name]][name],
                traced[name],
                reset_before=reset_before,
                emphasis=emphasis,
            )
            for name, block in (
                ("torso", self.torso),
                ("actor", self.actor),
                ("critic", self.critic),
            )
        }
        return emphasis, advanced

    def _update_reading(self, delta, emphasis, traced, advanced, taken):
        """Project update products onto the readings this graph declares."""

        blocks = {
            "torso": self.torso,
            "actor": self.actor,
            "critic": self.critic,
        }

        def block_reading(name):
            reports = getattr(self.reports, name)
            block = blocks[name]
            return BlockUpdate(
                grad_norm=(
                    block.gradient_norms(traced[name]) if reports.grad_norm else None
                ),
                trace_norm=(
                    block.gradient_norms(advanced[name]) if reports.trace_norm else None
                ),
            )

        return UpdateMetrics(
            td_error=delta if self.reports.td_error else None,
            emphasis=emphasis if self.reports.emphasis else None,
            torso=block_reading("torso"),
            actor=block_reading("actor"),
            critic=block_reading("critic"),
            torso_step=GroupUpdate(
                step_size=(
                    taken[TORSO_GROUP].metrics.get("step_size")
                    if self.reports.torso_step.step_size
                    else None
                )
            ),
            heads_step=GroupUpdate(
                step_size=(
                    taken[HEAD_GROUP].metrics.get("step_size")
                    if self.reports.heads_step.step_size
                    else None
                )
            ),
        )

    def update_parameters(
        self,
        key,
        state: CoreState,
        timestep: Timestep,
        *,
        terminal,
        reset_before,
        step,
    ) -> tuple[CoreState, Any, ForwardMetrics, UpdateMetrics]:
        """One transition's worth of learning, and the action to take next."""

        recurrence, action, value, traced, direct, actor_reading = self._per_stream(
            key, state, timestep
        )

        delta = self._td_error(state, timestep, value, terminal)
        stepped, taken, grouped_traces = self._take_updates(
            state, traced, direct, delta, step
        )
        emphasis, advanced = self._advance_traces(
            state, grouped_traces, traced, reset_before
        )

        return (
            state.replace(
                torso=state.torso.replace(
                    params=stepped["torso"],
                    traces=advanced["torso"],
                    slow_params=self.torso.followed(
                        stepped["torso"], state.torso.slow_params
                    ),
                    recurrence=recurrence,
                ),
                actor=BlockState(params=stepped["actor"], traces=advanced["actor"]),
                critic=BlockState(params=stepped["critic"], traces=advanced["critic"]),
                rule={group: result.state for group, result in taken.items()},
                value=value,
                emphasis=emphasis,
            ),
            action,
            ForwardMetrics(
                actor=actor_reading,
                critic=CriticForward(value=value if self.reports.value else None),
            ),
            self._update_reading(delta, emphasis, traced, advanced, taken),
        )


# -------------------------------------------------------------------- the flow
class RTRRL:
    """One-invocation train/evaluation flow around the three layers."""

    observations = OBSERVATIONS

    def __init__(
        self,
        cfg: RTRRLConfig,
        env: Any,
        env_params: Any,
        torso_network: Any,
        torso_differentiation: Any,
        actor_head: Any,
        critic_head: Any,
        *,
        observation_normalization: Any = None,
        reward_normalization: Any = None,
        evaluation: EvaluationConfig | None = None,
        record: Iterable[str] = (),
        reports: Reports = Reports(),
    ) -> None:
        self.cfg = cfg
        evaluation = evaluation or EvaluationConfig()
        self.environment = EnvironmentStreams(cfg.num_envs, env, env_params)
        self.normalization = InteractionNormalization(
            cfg.num_envs,
            env,
            observation=observation_normalization,
            reward=reward_normalization,
            reset_on_start=evaluation.reset_on_start,
            update_during_eval=evaluation.update_during_eval,
        )
        self.core = Core(
            cfg,
            torso_network,
            torso_differentiation,
            actor_head,
            critic_head,
            reports,
        )
        self.record = frozenset(record)

    @classmethod
    def graph(
        cls,
        parameters: dict[str, Any],
        components: ComponentBuilder,
        context: BuildContext,
        *,
        record: Iterable[str] = (),
    ) -> RTRRL:
        """Declare the shared torso, two readouts, and their two rule groups."""

        gamma = float(parameters["gamma"])
        meta_rl = bool(parameters["meta_rl"])
        classes = action_classes(context.action_space)
        # What the cell reads, now that nothing widens it first: the observation,
        # and under meta-RL the previous action and reward beside it.
        features = int(context.observation_space.shape[0])
        if meta_rl:
            features += action_dim(context.action_space) + 1
        torso_optimizer = components.build(RTRRL_OPTIMIZERS, "torso.optimizer")
        heads_optimizer = components.build(RTRRL_OPTIMIZERS, "heads.optimizer")
        torso_network, torso_differentiation = components.build(
            RTRRL_TORSO_FAMILY, "torso.backbone", features=features
        )
        return cls(
            RTRRLConfig(
                num_envs=context.num_envs,
                gamma=gamma,
                lambda_pi=float(parameters["lambda_pi"]),
                lambda_v=float(parameters["lambda_v"]),
                lambda_rnn=float(parameters["lambda_rnn"]),
                eta_pi=float(parameters["eta_pi"]),
                eta_f=float(parameters["eta_f"]),
                entropy_rate=float(parameters["entropy_rate"]),
                torso_grad_clip=float(parameters["torso.grad_clip"]),
                torso_follow=float(parameters["torso.follow"]),
                meta_rl=meta_rl,
                action_classes=classes,
                torso_optimizer=torso_optimizer,
                heads_optimizer=heads_optimizer,
            ),
            context.environment,
            context.environment_parameters,
            torso_network,
            torso_differentiation,
            components.build(
                ACTOR_HEAD_FAMILY,
                "actor.head",
                action_dim=action_dim(context.action_space),
                discrete=classes is not None,
            ),
            components.build(CRITIC_HEAD_FAMILY, "critic.head"),
            observation_normalization=components.build(
                NORMALIZATION_FAMILY, "normalization.observation"
            ),
            reward_normalization=components.build(
                DISCOUNTED_NORMALIZATION_FAMILY,
                "normalization.reward",
                discount=gamma,
            ),
            record=record,
            reports=REPORTS,
        )

    def init(self, key: Any) -> RTRRLState:
        (
            env_key,
            torso_key,
            torso_torso_key,
            torso_dropout_key,
            actor_key,
            critic_key,
        ) = jax.random.split(key, 6)
        obs, env_state = self.environment.init(env_key)
        obs, scales = self.normalization.init(obs)
        timestep = self.environment.blank_timestep(obs).to_sequence()
        core = self.core.init(
            ((torso_key, torso_torso_key, torso_dropout_key), actor_key, critic_key),
            timestep,
        )
        return RTRRLState(
            step=jnp.asarray(0, dtype=jnp.int32),
            update_step=jnp.asarray(0, dtype=jnp.int32),
            timestep=timestep.from_sequence(),
            terminal=jnp.zeros((self.cfg.num_envs,), dtype=jnp.bool_),
            env_state=env_state,
            scales=scales,
            core=core,
        )

    def _reset(self, key, state: RTRRLState, *, update=True) -> RTRRLState:
        """Both components begun again for the streams that ended, in order."""

        done = state.timestep.done
        obs, env_state = self.environment.reset(key, state.env_state, done)
        obs, scales = self.normalization.reset(obs, state.scales, done, update=update)
        return state.replace(
            timestep=state.timestep.replace(
                obs=select_ended(done, obs, state.timestep.obs)
            ),
            env_state=env_state,
            scales=scales,
        )

    def _interaction(
        self,
        *,
        observation,
        next_observation,
        action,
        reward,
        done,
        terminal,
        info,
        action_decision=None,
    ) -> InteractionMetrics:
        """One transition, with the trajectory kept only if something reads it."""

        walked = "interaction.observation" in self.record
        return InteractionMetrics(
            observation=observation if walked else None,
            next_observation=next_observation if walked else None,
            action=action if walked else None,
            action_decision=action_decision,
            reward=reward,
            done=done,
            terminal=terminal,
            info=info,
        )

    def train_step(self, state: RTRRLState, key: Any) -> tuple[RTRRLState, StepMetrics]:
        """Learn from the transition that got here, then take the next one."""

        reset_key, action_key, env_key = jax.random.split(key, 3)
        state = self._reset(reset_key, state)
        observation = state.timestep.obs
        current_step = state.update_step + 1

        core, action, forward_reading, update_reading = self.core.update_parameters(
            action_key,
            state.core,
            state.timestep,
            terminal=state.terminal,
            reset_before=state.timestep.done,
            step=current_step,
        )
        obs, env_state, environment_reward, done, terminal, info = (
            self.environment.step(env_key, state.env_state, action)
        )
        obs, reward, scales = self.normalization.apply(
            state.scales, obs, environment_reward, done
        )
        next_timestep = Timestep(obs=obs, action=action, reward=reward, done=done)

        # The transition is carried whole. What an ending drops from the network
        # input is dropped where the input is built, because the reward this
        # timestep holds is also what the next step's TD error is measured on.
        return state.replace(
            step=state.step + self.cfg.num_envs,
            update_step=current_step,
            timestep=next_timestep,
            terminal=terminal,
            env_state=env_state,
            scales=scales,
            core=core,
        ), StepMetrics(
            interaction=self._interaction(
                observation=observation,
                next_observation=next_timestep.obs,
                action=action,
                action_decision=ActionDecision(
                    sampled_action=action,
                    logprob_action=action,
                    env_action=action,
                ),
                reward=environment_reward,
                done=next_timestep.done,
                terminal=terminal,
                info=info,
            ),
            forward=forward_reading,
            update=update_reading,
        )

    def interact(self, key: Any, state: RTRRLState) -> tuple[RTRRLState, StepMetrics]:
        """One behavior-policy transition that learns nothing and costs no budget.

        The stochastic policy and the recurrent sequence continue exactly where
        training left them, so a sampled episode can be finished after the
        training budget without the parameters, traces, rules, normalization
        statistics, or either step counter moving.
        """

        reset_key, action_key, env_key = jax.random.split(key, 3)
        state = self._reset(reset_key, state, update=False)
        observation = state.timestep.obs

        recurrence, action, _ = self.core.act(
            action_key, state.core, state.timestep, deterministic=False
        )
        obs, env_state, environment_reward, done, terminal, info = (
            self.environment.step(env_key, state.env_state, action)
        )
        obs, reward, _ = self.normalization.apply(
            state.scales, obs, environment_reward, done, update=False
        )
        next_timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        return state.replace(
            timestep=next_timestep,
            terminal=terminal,
            env_state=env_state,
            core=state.core.replace(
                torso=state.core.torso.replace(recurrence=recurrence)
            ),
        ), StepMetrics(
            interaction=self._interaction(
                observation=observation,
                next_observation=next_timestep.obs,
                action=action,
                reward=environment_reward,
                done=next_timestep.done,
                terminal=terminal,
                info=info,
            ),
        )

    def evaluate_step(
        self, state: RTRRLState, key: Any
    ) -> tuple[RTRRLState, StepMetrics]:
        """The same interaction with the greedy action and no update at all."""

        reset_key, action_key, env_key = jax.random.split(key, 3)
        update = self.normalization.updates_during_eval
        state = self._reset(reset_key, state, update=update)
        observation = state.timestep.obs

        recurrence, action, _ = self.core.act(
            action_key, state.core, state.timestep, deterministic=True
        )
        obs, env_state, environment_reward, done, terminal, info = (
            self.environment.step(env_key, state.env_state, action)
        )
        obs, reward, scales = self.normalization.apply(
            state.scales, obs, environment_reward, done, update=update
        )
        next_timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        return state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=next_timestep,
            terminal=terminal,
            env_state=env_state,
            scales=scales,
            core=state.core.replace(
                torso=state.core.torso.replace(recurrence=recurrence)
            ),
        ), StepMetrics(
            interaction=self._interaction(
                observation=observation,
                next_observation=next_timestep.obs,
                action=action,
                reward=environment_reward,
                done=next_timestep.done,
                terminal=terminal,
                info=info,
            ),
        )

    @staticmethod
    def _num_scan_steps(num_steps: int, num_envs: int) -> int:
        """How many rounds of every stream a step budget buys."""

        return num_steps // num_envs

    def _evaluation_state(self, key: Any, state: RTRRLState) -> RTRRLState:
        """The trained parameters, opened on a fresh environment and sequence."""

        obs, env_state = self.environment.init(key)
        fresh = self.normalization.resets_on_start
        obs, scales = self.normalization.init(
            obs,
            None if fresh else state.scales,
            update=fresh or self.normalization.updates_during_eval,
        )
        return state.replace(
            timestep=self.environment.blank_timestep(obs),
            terminal=jnp.zeros((self.cfg.num_envs,), dtype=jnp.bool_),
            env_state=env_state,
            scales=scales,
            core=self.core.reset(jax.random.key(0), state.core),
        )

    def train(
        self, key: Any, state: RTRRLState, num_steps: int
    ) -> tuple[RTRRLState, StepMetrics]:
        """Run one fixed-size online-training invocation."""

        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(key, scan_steps)
        return jax.lax.scan(self.train_step, state, keys)

    def evaluate(self, key: Any, state: RTRRLState, num_steps: int) -> StepMetrics:
        """A rollout that leaves nothing behind."""

        reset_key, rollout_key = jax.random.split(key)
        eval_state = self._evaluation_state(reset_key, state)
        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(rollout_key, scan_steps)
        _, metrics = jax.lax.scan(self.evaluate_step, eval_state, keys)
        return metrics
