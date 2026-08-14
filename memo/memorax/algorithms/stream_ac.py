"""StreamAC: an actor and a critic with independent differentiated networks.

    StreamAC          the order things happen in, and the scan
    Environment       where every stream is
    Normalization     the scales the environment's numbers are read through
    Core              a policy, a value function, and the one thing they share
      Actor / Critic  a task: what it produces, and the scalar it ascends
        Network       a block of parameters nothing else owns

Driven against ``stream_ac.py`` by ``tests/test_layered_parity.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from memorax.building import BuildContext, ComponentBuilder
from memorax.networks import Readout, Sequence
from memorax.networks.backbones import BACKBONE_FAMILY
from memorax.networks.readouts import ACTOR_HEAD_FAMILY, CRITIC_HEAD_FAMILY
from memorax.networks.sequence import PLACES
from memorax.observability.metrics import metric_names
from memorax.parameters import describe_parameters, group, param, structure
from memorax.readings import reading, readings, taken
from memorax.rl import (
    EnvironmentStreams,
    InteractionNormalization,
    NormalizationState,
    broadcast_stream,
    make_bounded_rule,
    make_td0,
    select_ended,
)
from memorax.rl.normalization import (
    DISCOUNTED_NORMALIZATION_FAMILY,
    NORMALIZATION_FAMILY,
)
from memorax.rl.updates import BASE_FAMILY, BOUND_FAMILY
from memorax.runtime import ObservationSchema
from memorax.utils import Timestep
from memorax.utils.axes import add_time_axis, remove_feature_axis, remove_time_axis
from memorax.utils.trees import subtree_norms

from .contract import ActionDecision, EvaluationConfig, InteractionMetrics, StepMetrics


# --------------------------------------------------------------- configuration
@dataclass(frozen=True)
class StreamACConfig:
    """Everything the kernel reads that does not change during a run."""

    num_envs: int
    gamma: float
    trace_lambda: float
    actor_bound: Any = None
    actor_base: Any = None
    critic_bound: Any = None
    critic_base: Any = None
    entropy_coefficient: float = 0.01
    meta_rl: bool = False
    normalization_statistics: str = "ours"


STREAM_AC_BACKBONES = BACKBONE_FAMILY.restricted("rtu", "mlp")


@dataclass(frozen=True)
class OptimizerParameters:
    bound: str = structure(branches=BOUND_FAMILY.branches)
    base: str = structure(branches=BASE_FAMILY.branches)


@dataclass(frozen=True)
class ActorParameters:
    head: str = structure(branches=ACTOR_HEAD_FAMILY.branches)
    optimizer: OptimizerParameters = group(of=OptimizerParameters)


@dataclass(frozen=True)
class CriticParameters:
    head: str = structure(branches=CRITIC_HEAD_FAMILY.branches)
    optimizer: OptimizerParameters = group(of=OptimizerParameters)


@dataclass(frozen=True)
class NormalizationParameters:
    observation: str = structure(branches=NORMALIZATION_FAMILY.branches)
    reward: str = structure(branches=DISCOUNTED_NORMALIZATION_FAMILY.branches)


@dataclass(frozen=True)
class StreamACParameters:
    actor: ActorParameters = group(of=ActorParameters)
    critic: CriticParameters = group(of=CriticParameters)
    normalization: NormalizationParameters = group(of=NormalizationParameters)
    backbone: str = structure(branches=STREAM_AC_BACKBONES.branches)
    meta_rl: bool = param(valid=[False, True], search=[False, True])
    gamma: float = param(valid=(0.5, 0.9999), search=(0.9, 0.9999))
    trace_lambda: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    entropy_coefficient: float = param(valid=(1e-8, 1.0), search=(1e-8, 1e-2), log=True)


PARAMETERS = describe_parameters(StreamACParameters)


# ----------------------------------------------------------------------- state
@struct.dataclass(frozen=True)
class Recurrence:
    """Where the sequence is, and what it owes the past."""

    carry: Any
    differentiation_state: Any


@struct.dataclass(frozen=True)
class RuleState:
    """What the update carries between steps, which a forward pass never sees."""

    traces: Any
    v: Any


@struct.dataclass(frozen=True)
class NetworkState:
    """Independent online state for one recurrent actor or critic network."""

    params: Any
    rule: RuleState
    recurrence: Recurrence


# -------------------------------------------------------------------- readings
@dataclass(frozen=True)
class BlockReports:
    """Which of a block's readings to take."""

    step_size: bool = reading(at="step_size")
    grad_norm: bool = reading(at="grad_norm", split=True)
    trace_norm: bool = reading(at="trace_norm", split=True)


@dataclass(frozen=True)
class Reports:
    """Which readings to take, in the shape of what produces them."""

    log_prob: bool = reading(at="forward.actor.log_prob")
    entropy: bool = reading(at="forward.actor.entropy")
    value: bool = reading(at="forward.critic.value")
    next_value: bool = reading(at="forward.critic.next_value")
    td_error: bool = reading(at="update.td_error")
    actor: BlockReports = readings(of=BlockReports, at="update.actor")
    critic: BlockReports = readings(of=BlockReports, at="update.critic")


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
RECORD = OBSERVATIONS.required_fields


@struct.dataclass(frozen=True)
class ActorForward:
    """What the policy answered on the pass that chose."""

    log_prob: Any = None
    entropy: Any = None


@struct.dataclass(frozen=True)
class CriticForward:
    """Both of the critic's readings, which is what a TD error is made of."""

    value: Any = None
    next_value: Any = None


@struct.dataclass(frozen=True)
class ForwardMetrics:
    """One field per head, so a declared name is a path through the components."""

    actor: ActorForward = ActorForward()
    critic: CriticForward = CriticForward()


@struct.dataclass(frozen=True)
class BlockUpdate:
    """What one block's step cost, and how big what went into it was."""

    step_size: Any = None
    grad_norm: Any = None
    trace_norm: Any = None


@struct.dataclass(frozen=True)
class UpdateMetrics:
    """One field per block, plus the TD error, which neither role owns."""

    td_error: Any = None
    actor: BlockUpdate = BlockUpdate()
    critic: BlockUpdate = BlockUpdate()


@struct.dataclass(frozen=True)
class StreamACState:
    """Everything the kernel carries, one field per component that owns one."""

    step: Any
    update_step: Any
    timestep: Timestep

    env_state: Any
    scales: NormalizationState
    actor: NetworkState
    critic: NetworkState


# ---------------------------------------------------------- shapes and streams
def _per_stream(objective, params, *streamed):
    """Differentiate each stream's own objective, and only its own."""

    def one(params, *stream):
        batched = jax.tree.map(lambda leaf: leaf[None], stream)
        return objective(params, *batched)[0]

    return jax.vmap(jax.grad(one), in_axes=(None, *(0,) * len(streamed)))(
        params, *streamed
    )


# ------------------------------------------------------- a block of parameters
class Network:
    """One block of parameters that nothing else owns."""

    def __init__(
        self,
        cfg: StreamACConfig,
        network: Any,
        differentiation: Any,
        *,
        bound,
        base,
        reports: BlockReports = BlockReports(),
    ) -> None:
        self.cfg = cfg
        self.network = network
        self.reports = reports
        self.differentiation = differentiation
        self.rule = make_bounded_rule(bound=bound, base=base)
        self.trace_decay = cfg.gamma * cfg.trace_lambda

    @property
    def carry_shape(self):
        return (self.cfg.num_envs, None)

    def _input(self, obs, action, reward):
        """The one vector a sequence sees."""

        if not self.cfg.meta_rl:
            return obs
        return jnp.concatenate([obs, action, reward], axis=-1)

    def apply(self, params, timestep, recurrence: Recurrence):
        """One forward pass, handing back the advance rather than writing it."""

        obs, done, action, reward = timestep
        (carry, differentiation_state), output = self.network.walk(
            params,
            self._input(obs, action, reward),
            done=done,
            carries=recurrence.carry,
            differentiation_state=recurrence.differentiation_state,
            differentiation=self.differentiation,
        )
        return Recurrence(
            carry=carry, differentiation_state=differentiation_state
        ), output

    def init(self, keys, timestep: Timestep) -> NetworkState:
        """Fresh online state for this block."""

        param_key, torso_key, dropout_key = keys
        obs, done, action, reward = timestep
        carry = self.network.initialize_carry(jax.random.key(0), self.carry_shape)
        differentiation_state = self.differentiation.initialize(
            param_key, self.carry_shape
        )
        with self.differentiation.initialization():
            params = self.network.init(
                {"params": param_key, "torso": torso_key, "dropout": dropout_key},
                self._input(obs, action, reward),
                done=done,
                initial_carry=carry,
            )
        traces = jax.tree.map(
            lambda param: jnp.zeros((self.cfg.num_envs, *param.shape)), params
        )
        return NetworkState(
            params=params,
            rule=RuleState(
                traces=traces,
                v=self.rule.init(params=params, traces=traces),
            ),
            recurrence=Recurrence(
                carry=carry, differentiation_state=differentiation_state
            ),
        )

    def reset(self, key, state: NetworkState) -> NetworkState:
        """The same parameters with the recurrence begun again."""

        return state.replace(
            recurrence=Recurrence(
                carry=self.network.initialize_carry(key, self.carry_shape),
                differentiation_state=self.differentiation.initialize(
                    key, self.carry_shape
                ),
            )
        )

    def trace(self, incoming, gradient, *, reset_before):
        """StreamAC's pre-forward reset, always-fresh trace recurrence."""

        return jax.tree.map(
            lambda old, grad: (
                self.trace_decay * (1 - broadcast_stream(reset_before, old)) * old
                + grad
            ),
            incoming,
            gradient,
        )

    def _norms(self, tree):
        """One norm per position group, per stream."""

        return subtree_norms(self.network.split(tree), streams=True)

    def step(self, state: NetworkState, gradient, delta, *, reset_before, step):
        """Trace the gradient, take the bounded step, and say what it did."""

        traces = self.trace(state.rule.traces, gradient, reset_before=reset_before)
        taken = self.rule.apply(
            traces,
            None,
            state.rule.v,
            delta=delta,
            step=step,
            params=state.params,
        )
        return state.replace(
            params=jax.tree.map(
                lambda param, update: param + update, state.params, taken.updates
            ),
            rule=RuleState(traces=traces, v=taken.state),
        ), BlockUpdate(
            step_size=(
                taken.metrics.get("step_size") if self.reports.step_size else None
            ),
            grad_norm=self._norms(gradient) if self.reports.grad_norm else None,
            trace_norm=self._norms(traces) if self.reports.trace_norm else None,
        )


# --------------------------------------------------------------- the two tasks
class Actor:
    """The policy. It chooses, and it names the scalar its block ascends."""

    def __init__(
        self,
        cfg: StreamACConfig,
        network: Any,
        differentiation: Any,
        reports: Reports = Reports(),
    ) -> None:
        self.cfg = cfg
        self.reports = reports
        self.block = Network(
            cfg,
            network,
            differentiation,
            bound=cfg.actor_bound,
            base=cfg.actor_base,
            reports=reports.actor,
        )

    def init(self, keys, timestep: Timestep) -> NetworkState:
        return self.block.init(keys, timestep)

    def reset(self, key, state: NetworkState) -> NetworkState:
        return self.block.reset(key, state)

    def act(
        self, key, state: NetworkState, timestep: Timestep, *, deterministic: bool
    ) -> tuple[Recurrence, Any, ActorForward]:
        """Run forward once and choose, touching nothing else."""

        recurrence, (dist, _) = self.block.apply(
            state.params, timestep.to_sequence(), state.recurrence
        )
        if deterministic:
            action = (
                jnp.argmax(dist.logits, axis=-1)
                if hasattr(dist, "logits")
                else dist.mode()
            )
            return recurrence, remove_time_axis(action), ActorForward()
        action, log_prob = dist.sample_and_log_prob(seed=key)
        return (
            recurrence,
            remove_time_axis(action),
            ActorForward(
                log_prob=(
                    remove_time_axis(log_prob) if self.reports.log_prob else None
                ),
                entropy=(
                    remove_time_axis(dist.entropy()) if self.reports.entropy else None
                ),
            ),
        )

    def objective(self, output, action, delta):
        """What this head ascends: log pi(a) with entropy riding on it."""

        dist, _ = output
        return remove_time_axis(
            dist.log_prob(add_time_axis(action))
        ) + self.cfg.entropy_coefficient * jnp.sign(
            jax.lax.stop_gradient(delta)
        ) * remove_time_axis(dist.entropy())

    def gradient(self, state: NetworkState, timestep: Timestep, action, delta):
        """This head's ascent, one stream at a time."""

        def ascent(params, timestep, recurrence, action, delta):
            _, output = self.block.apply(params, timestep, recurrence)
            return self.objective(output, action, delta)

        return _per_stream(
            ascent,
            state.params,
            timestep.to_sequence(),
            jax.lax.stop_gradient(state.recurrence),
            action,
            delta,
        )

    def update(
        self,
        state: NetworkState,
        timestep: Timestep,
        next_timestep: Timestep,
        delta,
        *,
        reset_before,
        step,
    ) -> tuple[NetworkState, BlockUpdate]:
        """One transition's worth of learning, from where the acting pass began."""

        gradient = self.gradient(state, timestep, next_timestep.action, delta)
        return self.block.step(
            state, gradient, delta, reset_before=reset_before, step=step
        )


class Critic:
    """The value. It reads, and it ascends its own reading."""

    def __init__(
        self,
        cfg: StreamACConfig,
        network: Any,
        differentiation: Any,
        reports: Reports = Reports(),
    ) -> None:
        self.cfg = cfg
        self.reports = reports
        self.block = Network(
            cfg,
            network,
            differentiation,
            bound=cfg.critic_bound,
            base=cfg.critic_base,
            reports=reports.critic,
        )

    def init(self, keys, timestep: Timestep) -> NetworkState:
        return self.block.init(keys, timestep)

    def reset(self, key, state: NetworkState) -> NetworkState:
        return self.block.reset(key, state)

    def objective(self, output):
        """What this head ascends: the value itself, with no error in it."""

        value, _ = output
        return remove_feature_axis(remove_time_axis(value))

    def apply(
        self, params, timestep: Timestep, recurrence: Recurrence
    ) -> tuple[Recurrence, Any]:
        """What a state is worth, and the recurrence that reading advanced."""

        recurrence, output = self.block.apply(
            params, timestep.to_sequence(), recurrence
        )
        return recurrence, self.objective(output)

    def gradient(self, state: NetworkState, timestep: Timestep):
        """This head's ascent, one stream at a time. See ``Actor.gradient``."""

        def ascent(params, timestep, recurrence):
            _, output = self.block.apply(params, timestep, recurrence)
            return self.objective(output)

        return _per_stream(
            ascent,
            state.params,
            timestep.to_sequence(),
            jax.lax.stop_gradient(state.recurrence),
        )

    def update(
        self,
        state: NetworkState,
        timestep: Timestep,
        delta,
        *,
        recurrence: Recurrence,
        reset_before,
        step,
    ) -> tuple[NetworkState, BlockUpdate]:
        """Learning, plus the recurrence the valuing pass advanced."""

        gradient = self.gradient(state, timestep)
        stepped, reading = self.block.step(
            state, gradient, delta, reset_before=reset_before, step=step
        )
        return stepped.replace(recurrence=recurrence), reading


# --------------------------------------------------------------- the algorithm
class Core:
    """A policy, a value function, and the one thing they are coupled by."""

    def __init__(
        self,
        cfg: StreamACConfig,
        actor_network: Any,
        critic_network: Any,
        actor_differentiation: Any,
        critic_differentiation: Any,
        reports: Reports = Reports(),
    ) -> None:
        self.cfg = cfg
        self.reports = reports
        self.actor = Actor(cfg, actor_network, actor_differentiation, reports)
        self.critic = Critic(cfg, critic_network, critic_differentiation, reports)
        self.td0 = make_td0()

    def init(
        self, actor_keys, critic_keys, timestep: Timestep
    ) -> tuple[NetworkState, NetworkState]:
        return (
            self.actor.init(actor_keys, timestep),
            self.critic.init(critic_keys, timestep),
        )

    def reset(
        self, key, actor_state: NetworkState, critic_state: NetworkState
    ) -> tuple[NetworkState, NetworkState]:
        return (
            self.actor.reset(key, actor_state),
            self.critic.reset(key, critic_state),
        )

    def sample_action(
        self,
        key: Any,
        timestep: Timestep,
        actor_state: NetworkState,
        deterministic: bool,
    ) -> tuple[Recurrence, Any, ActorForward]:
        return self.actor.act(key, actor_state, timestep, deterministic=deterministic)

    def update_parameters(
        self,
        state: StreamACState,
        next_timestep: Timestep,
        *,
        terminal: Any = None,
    ) -> tuple[StreamACState, CriticForward, UpdateMetrics]:
        """One transition's worth of learning for both roles, and what it read."""

        actor = state.actor
        critic = state.critic
        reset_before = state.timestep.done
        terminal = next_timestep.done if terminal is None else terminal
        current_step = state.update_step + 1

        recurrence, value = self.critic.apply(
            critic.params, state.timestep, critic.recurrence
        )
        _, next_value = self.critic.apply(
            jax.lax.stop_gradient(critic.params),
            next_timestep,
            jax.lax.stop_gradient(recurrence),
        )
        td_error = self.td0(
            reward=next_timestep.reward,
            value=value,
            next_value=next_value,
            terminal=terminal,
            gamma=self.cfg.gamma,
        )

        actor_state, actor_reading = self.actor.update(
            actor,
            state.timestep,
            next_timestep,
            td_error,
            reset_before=reset_before,
            step=current_step,
        )
        critic_state, critic_reading = self.critic.update(
            critic,
            state.timestep,
            td_error,
            recurrence=recurrence,
            reset_before=reset_before,
            step=current_step,
        )
        return (
            state.replace(
                update_step=current_step, actor=actor_state, critic=critic_state
            ),
            CriticForward(
                value=value if self.reports.value else None,
                next_value=next_value if self.reports.next_value else None,
            ),
            UpdateMetrics(
                td_error=td_error if self.reports.td_error else None,
                actor=actor_reading,
                critic=critic_reading,
            ),
        )


# -------------------------------------------------------------------- the flow
class StreamAC:
    """One-invocation train/evaluation flow around the three layers."""

    observations = OBSERVATIONS

    def __init__(
        self,
        cfg: StreamACConfig,
        env: Any,
        env_params: Any,
        actor_network: Any,
        critic_network: Any,
        actor_differentiation: Any,
        critic_differentiation: Any,
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
            actor_network,
            critic_network,
            actor_differentiation,
            critic_differentiation,
            reports,
        )
        self.record = frozenset(record)

    @classmethod
    def graph(
        cls,
        parameters: dict[str, Any],
        components: ComponentBuilder,
        context: BuildContext,
    ) -> StreamAC:
        """Declare StreamAC's instances and connections using shared builders."""

        gamma = float(parameters["gamma"])
        meta_rl = bool(parameters["meta_rl"])
        observation_dim = int(context.observation_space.shape[0])
        action_dim = int(context.action_space.shape[0])
        features = observation_dim
        if meta_rl:
            features += action_dim + 1

        def sequence(backbone, head):
            return Sequence(
                components=(*backbone.components, Readout(module=head))
            )

        actor_backbone = components.build(
            STREAM_AC_BACKBONES,
            "backbone",
            features=features,
            output_dim=None,
        )
        critic_backbone = components.build(
            STREAM_AC_BACKBONES,
            "backbone",
            features=features,
            output_dim=None,
        )
        actor_head = components.build(
            ACTOR_HEAD_FAMILY,
            "actor.head",
            action_dim=action_dim,
        )
        critic_head = components.build(CRITIC_HEAD_FAMILY, "critic.head")

        return cls(
            StreamACConfig(
                num_envs=context.num_envs,
                gamma=gamma,
                trace_lambda=float(parameters["trace_lambda"]),
                actor_bound=components.build(BOUND_FAMILY, "actor.optimizer.bound"),
                actor_base=components.build(BASE_FAMILY, "actor.optimizer.base"),
                critic_bound=components.build(BOUND_FAMILY, "critic.optimizer.bound"),
                critic_base=components.build(BASE_FAMILY, "critic.optimizer.base"),
                entropy_coefficient=float(parameters["entropy_coefficient"]),
                meta_rl=meta_rl,
            ),
            context.environment,
            context.environment_parameters,
            sequence(actor_backbone, actor_head),
            sequence(critic_backbone, critic_head),
            actor_backbone.differentiation,
            critic_backbone.differentiation,
            observation_normalization=components.build(
                NORMALIZATION_FAMILY, "normalization.observation"
            ),
            reward_normalization=components.build(
                DISCOUNTED_NORMALIZATION_FAMILY,
                "normalization.reward",
                discount=gamma,
            ),
            record=RECORD,
            reports=REPORTS,
        )

    def init(self, key: Any) -> StreamACState:
        # The published kernel's seven, in its order; two go unused.
        (
            env_key,
            actor_key,
            actor_torso_key,
            actor_dropout_key,
            critic_key,
            critic_torso_key,
            critic_dropout_key,
        ) = jax.random.split(key, 7)
        obs, env_state = self.environment.init(env_key)
        obs, scales = self.normalization.init(obs)
        timestep = self.environment.blank_timestep(obs).to_sequence()
        actor, critic = self.core.init(
            (actor_key, actor_torso_key, actor_dropout_key),
            (critic_key, critic_torso_key, critic_dropout_key),
            timestep,
        )
        return StreamACState(
            step=jnp.asarray(0, dtype=jnp.int32),
            update_step=jnp.asarray(0, dtype=jnp.int32),
            timestep=timestep.from_sequence(),
            env_state=env_state,
            scales=scales,
            actor=actor,
            critic=critic,
        )

    def _reset(self, key, state: StreamACState, *, update=True) -> StreamACState:
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

    def train_step(
        self, state: StreamACState, key: Any
    ) -> tuple[StreamACState, StepMetrics]:
        """Act once, then learn from what the acting produced."""

        reset_key, action_key, env_key = jax.random.split(key, 3)
        state = self._reset(reset_key, state)
        observation = state.timestep.obs

        recurrence, action, actor_reading = self.core.sample_action(
            action_key, state.timestep, state.actor, deterministic=False
        )
        obs, env_state, environment_reward, done, terminal, info = (
            self.environment.step(env_key, state.env_state, action)
        )
        obs, reward, scales = self.normalization.apply(
            state.scales, obs, environment_reward, done
        )
        next_timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        updated, critic_reading, update_reading = self.core.update_parameters(
            state, next_timestep, terminal=terminal
        )

        persisted = self.environment.persisted(next_timestep)
        next_state = updated.replace(
            step=state.step + self.cfg.num_envs,
            timestep=persisted,
            env_state=env_state,
            scales=scales,
            actor=updated.actor.replace(recurrence=recurrence),
        )
        return next_state, StepMetrics(
            interaction=self._interaction(
                observation=observation,
                next_observation=next_timestep.obs,
                action=action,
                action_decision=ActionDecision(
                    sampled_action=action,
                    logprob_action=action,
                    env_action=action,
                    bootstrap_feedback_action=action,
                    persisted_feedback_action=persisted.action,
                ),
                reward=environment_reward,
                done=next_timestep.done,
                terminal=terminal,
                info=info,
            ),
            forward=ForwardMetrics(actor=actor_reading, critic=critic_reading),
            update=update_reading,
        )

    def interact(
        self, key: Any, state: StreamACState
    ) -> tuple[StreamACState, StepMetrics]:
        """One behavior-policy transition that learns nothing and costs no budget.

        The stochastic policy and the actor's recurrence continue exactly where
        training left them, so a sampled episode can be finished after the
        training budget without either step counter, the parameters, the rule
        states, or the normalization statistics moving.
        """

        reset_key, action_key, env_key = jax.random.split(key, 3)
        state = self._reset(reset_key, state, update=False)
        observation = state.timestep.obs

        recurrence, action, _ = self.core.sample_action(
            action_key, state.timestep, state.actor, deterministic=False
        )
        obs, env_state, environment_reward, done, terminal, info = (
            self.environment.step(env_key, state.env_state, action)
        )
        obs, reward, _ = self.normalization.apply(
            state.scales, obs, environment_reward, done, update=False
        )
        next_timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        return state.replace(
            timestep=self.environment.persisted(next_timestep),
            env_state=env_state,
            actor=state.actor.replace(recurrence=recurrence),
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
        self, state: StreamACState, key: Any
    ) -> tuple[StreamACState, StepMetrics]:
        """The same interaction with the greedy action and no update at all."""

        reset_key, action_key, env_key = jax.random.split(key, 3)
        update = self.normalization.updates_during_eval
        state = self._reset(reset_key, state, update=update)
        observation = state.timestep.obs

        recurrence, action, _ = self.core.sample_action(
            action_key, state.timestep, state.actor, deterministic=True
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
            timestep=self.environment.persisted(next_timestep),
            env_state=env_state,
            scales=scales,
            actor=state.actor.replace(recurrence=recurrence),
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

    def _evaluation_state(self, key: Any, state: StreamACState) -> StreamACState:
        """The trained parameters, opened on a fresh environment and recurrence."""

        obs, env_state = self.environment.init(key)
        fresh = self.normalization.resets_on_start
        obs, scales = self.normalization.init(
            obs,
            None if fresh else state.scales,
            update=fresh or self.normalization.updates_during_eval,
        )
        actor, critic = self.core.reset(jax.random.key(0), state.actor, state.critic)
        return state.replace(
            timestep=self.environment.blank_timestep(obs),
            env_state=env_state,
            scales=scales,
            actor=actor,
            critic=critic,
        )

    def train(
        self, key: Any, state: StreamACState, num_steps: int
    ) -> tuple[StreamACState, StepMetrics]:
        """Run one fixed-size online-training invocation."""

        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(key, scan_steps)
        return jax.lax.scan(self.train_step, state, keys)

    def evaluate(self, key: Any, state: StreamACState, num_steps: int) -> StepMetrics:
        """A rollout that leaves nothing behind."""

        reset_key, rollout_key = jax.random.split(key)
        eval_state = self._evaluation_state(reset_key, state)
        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(rollout_key, scan_steps)
        _, metrics = jax.lax.scan(self.evaluate_step, eval_state, keys)
        return metrics
