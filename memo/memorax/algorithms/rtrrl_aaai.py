"""RTRRL: one recurrent torso shared by an actor and a critic.

    RTRRL             the order things happen in, and the scan
    Environment       where every stream is
    Normalization     the scales the environment's numbers are read through
    Core              a torso, two heads, and everything that couples them
      Torso           the shared block: sequence, sensitivity, following copy
      Actor / Critic  a readout, and the objectives it names

Three blocks and two rule groups: the torso is clipped alone, the two readouts
step together. The ``lru-bp-rtrl`` path only -- an LRU credited by exact RTRL,
everything feedforward by backpropagation -- and no gates.

Rebuilt from ``../RTRRL-AAAI25/rtrrl.py``. Driven against it by
``tests/test_rtrrl_parity.py``; behaviour in ``tests/test_rtrrl.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from memorax.readings import reading, readings
from memorax.rl import (
    EnvironmentStreams,
    InteractionNormalization,
    NormalizationState,
    broadcast_stream,
    make_exact_rtrl_credit,
    make_optax_rule,
    make_td0,
    select_ended,
)
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

    torso_lr: float = 1e-4
    head_lr: float = 1e-4
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    torso_grad_clip: float = 1.0
    torso_follow: float = 1.0

    meta_rl: bool = True


# ----------------------------------------------------------------------- state
@struct.dataclass(frozen=True)
class Recurrence:
    """Where the sequence is, and what it owes the past."""

    carry: Any
    sensitivity: Any


@struct.dataclass(frozen=True)
class BlockState:
    """A readout's parameters and the trace that decides how far they move."""

    params: Any
    traces: Any


@struct.dataclass(frozen=True)
class TorsoState:
    """The shared block: the same two things, plus what only sharing needs."""

    params: Any
    traces: Any
    slow_params: Any
    recurrence: Recurrence


@struct.dataclass(frozen=True)
class CoreState:
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
    """Which of one block's readings to take."""

    grad_norm: bool = reading(at="grad_norm", split=True)
    trace_norm: bool = reading(at="trace_norm", split=True)


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
    actor: BlockReports = readings(of=BlockReports, at="update.actor")
    critic: BlockReports = readings(of=BlockReports, at="update.critic")
    torso_step: GroupReports = readings(of=GroupReports, at="update.torso_step")
    heads_step: GroupReports = readings(of=GroupReports, at="update.heads_step")


@struct.dataclass(frozen=True)
class ActorForward:
    """What the policy answered on the pass that chose."""

    log_prob: Any = None
    entropy: Any = None


@struct.dataclass(frozen=True)
class CriticForward:
    """What the critic answered."""

    value: Any = None


@struct.dataclass(frozen=True)
class ForwardMetrics:
    """One field per head."""

    actor: ActorForward = ActorForward()
    critic: CriticForward = CriticForward()


@struct.dataclass(frozen=True)
class BlockUpdate:
    """How big what went into one block's step was."""

    grad_norm: Any = None
    trace_norm: Any = None


@struct.dataclass(frozen=True)
class GroupUpdate:
    """What one rule group's step did."""

    step_size: Any = None


@struct.dataclass(frozen=True)
class UpdateMetrics:
    """Three blocks, two groups, and two quantities belonging to neither."""

    td_error: Any = None
    emphasis: Any = None
    torso: BlockUpdate = BlockUpdate()
    actor: BlockUpdate = BlockUpdate()
    critic: BlockUpdate = BlockUpdate()
    torso_step: GroupUpdate = GroupUpdate()
    heads_step: GroupUpdate = GroupUpdate()


@struct.dataclass(frozen=True)
class RTRRLState:
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
def make_rules(cfg: RTRRLConfig):
    """One rule per group, both answering the same contract."""

    torso: list[Any] = []
    if cfg.torso_grad_clip:
        torso.append(optax.clip_by_global_norm(cfg.torso_grad_clip))
    torso.extend(
        (
            optax.scale_by_adam(b1=cfg.b1, b2=cfg.b2, eps=cfg.eps),
            optax.scale(cfg.torso_lr),
        )
    )
    return {
        TORSO_GROUP: make_optax_rule(optax.chain(*torso), rate=cfg.torso_lr),
        HEAD_GROUP: make_optax_rule(
            optax.chain(
                optax.scale_by_adam(b1=cfg.b1, b2=cfg.b2, eps=cfg.eps),
                optax.scale(cfg.head_lr),
            ),
            rate=cfg.head_lr,
        ),
    }


# ------------------------------------------------------- a block of parameters
class Network:
    """One block of parameters that nothing else owns."""

    def __init__(
        self,
        module: Any,
        *,
        num_envs: int,
        decay: float,
        credit=None,
        reports: BlockReports = BlockReports(),
    ) -> None:
        self.module = module
        self.num_envs = num_envs
        self.decay = decay
        self.credit = credit
        self.reports = reports

    def apply(self, params, x, *, done, action=None, reward=None, recurrence=None):
        if self.credit is None:
            output, _ = self.module.apply(
                {"params": params}, x, action=action, reward=reward, done=done
            )
            return recurrence, output
        (carry, sensitivity), output = self.module.walk(
            params,
            x,
            done=done,
            carries=recurrence.carry,
            sensitivity=recurrence.sensitivity,
            credit=self.credit,
        )
        return Recurrence(carry=carry, sensitivity=sensitivity), output

    def _norms(self, tree):
        """One norm per part, per stream."""

        grouped = (
            self.module.split(tree)
            if hasattr(self.module, "split")
            else {"readout": tree}
        )
        return subtree_norms(grouped, streams=True)

    def initial_traces(self, params):
        """One trace per parameter per stream, which is what online means."""

        return jax.tree.map(
            lambda param: jnp.zeros((self.num_envs, *param.shape)), params
        )

    def trace(self, incoming, gradient, *, reset_before, emphasis):
        """RTRRL's post-transition trace recurrence, at this block's own decay."""

        return jax.tree.map(
            lambda old, grad: (
                self.decay * (1 - broadcast_stream(reset_before, old)) * old
                + broadcast_stream(emphasis, grad) * grad
            ),
            incoming,
            gradient,
        )


# ------------------------------------------------------------ the shared block
class Torso:
    """The one block both heads read, and the only one credited by RTRL."""

    def __init__(
        self, cfg: RTRRLConfig, network: Any, reports: Reports = Reports()
    ) -> None:
        self.cfg = cfg
        self.reports = reports
        self.network = network
        self.block = Network(
            network,
            num_envs=cfg.num_envs,
            decay=cfg.gamma * cfg.lambda_rnn,
            credit=make_exact_rtrl_credit(network.core),
            reports=reports.torso,
        )

    @property
    def carry_shape(self):
        return (self.cfg.num_envs, None)

    def _input(self, timestep: Timestep):
        """The one vector the sequence sees."""

        obs, done, action, reward = timestep
        if not self.cfg.meta_rl:
            return obs
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
        return self.block.apply(
            params, self._input(timestep), done=done, recurrence=recurrence
        )

    def init(self, keys, timestep: Timestep) -> TorsoState:
        """Fresh online state for the shared block, and a copy for it to follow."""

        param_key, torso_key, dropout_key = keys
        _, done, _, _ = timestep
        carry = self.network.initialize_carry(jax.random.key(0), self.carry_shape)
        sensitivity = self.block.credit.initialize(param_key, self.carry_shape)
        with self.block.credit.initialization():
            variables = self.network.init(
                {"params": param_key, "torso": torso_key, "dropout": dropout_key},
                self._input(timestep),
                done=done,
                initial_carry=carry,
            )
        params = variables["params"]
        return TorsoState(
            params=params,
            traces=self.block.initial_traces(params),
            slow_params=params,
            recurrence=Recurrence(carry=carry, sensitivity=sensitivity),
        )

    def reset(self, key, state: TorsoState) -> TorsoState:
        """The same parameters with the sequence begun again."""

        return state.replace(
            recurrence=Recurrence(
                carry=self.network.initialize_carry(key, self.carry_shape),
                sensitivity=self.block.credit.initialize(key, self.carry_shape),
            )
        )

    def followed(self, params, slow_params):
        """The reading copy takes one step toward the copy that was updated."""

        if self.cfg.torso_follow == 1.0:
            return params
        return optax.incremental_update(params, slow_params, self.cfg.torso_follow)


# ------------------------------------------------------------ the two readouts
class Actor:
    """The policy. It chooses, and it names the two directions it ascends."""

    def __init__(
        self, cfg: RTRRLConfig, head: Any, reports: Reports = Reports()
    ) -> None:
        self.cfg = cfg
        self.reports = reports
        self.block = Network(
            head,
            num_envs=cfg.num_envs,
            decay=cfg.gamma * cfg.lambda_pi,
            reports=reports.actor,
        )

    def init(self, key, hidden, timestep: Timestep) -> BlockState:
        _, done, action, reward = timestep
        params = self.block.module.init(
            {"params": key}, hidden, action=action, reward=reward, done=done
        )["params"]
        return BlockState(params=params, traces=self.block.initial_traces(params))

    def apply(self, params, hidden, timestep: Timestep):
        _, done, action, reward = timestep
        _, dist = self.block.apply(
            params, hidden, done=done, action=action, reward=reward
        )
        return dist

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

    def __init__(
        self, cfg: RTRRLConfig, head: Any, reports: Reports = Reports()
    ) -> None:
        self.cfg = cfg
        self.reports = reports
        self.block = Network(
            head,
            num_envs=cfg.num_envs,
            decay=cfg.gamma * cfg.lambda_v,
            reports=reports.critic,
        )

    def init(self, key, hidden, timestep: Timestep) -> BlockState:
        _, done, action, reward = timestep
        params = self.block.module.init(
            {"params": key}, hidden, action=action, reward=reward, done=done
        )["params"]
        return BlockState(params=params, traces=self.block.initial_traces(params))

    def apply(self, params, hidden, timestep: Timestep):
        _, done, action, reward = timestep
        _, value = self.block.apply(
            params, hidden, done=done, action=action, reward=reward
        )
        return remove_feature_axis(remove_time_axis(value))

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
        actor_head: Any,
        critic_head: Any,
        reports: Reports = Reports(),
    ) -> None:
        self.cfg = cfg
        self.reports = reports
        self.torso = Torso(cfg, torso_network, reports)
        self.actor = Actor(cfg, actor_head, reports)
        self.critic = Critic(cfg, critic_head, reports)
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

        delta = self.td0(
            reward=timestep.reward,
            value=state.value,
            next_value=value,
            terminal=terminal,
            gamma=self.cfg.gamma,
        )

        deltas = {TORSO_GROUP: delta * self.cfg.eta_f, HEAD_GROUP: delta}
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
                lambda param, update: param + update,
                params[name],
                taken[self.group_of[name]].updates[name],
            )
            for name in params
        }

        emphasis = (
            self.cfg.gamma * state.emphasis * (1 - reset_before) + reset_before
        ).astype(jnp.float32)
        advanced = {
            name: block.trace(
                grouped_traces[self.group_of[name]][name],
                traced[name],
                reset_before=reset_before,
                emphasis=emphasis,
            )
            for name, block in (
                ("torso", self.torso.block),
                ("actor", self.actor.block),
                ("critic", self.critic.block),
            )
        }

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
            UpdateMetrics(
                td_error=delta if self.reports.td_error else None,
                emphasis=emphasis if self.reports.emphasis else None,
                torso=BlockUpdate(
                    grad_norm=(
                        self.torso.block._norms(traced["torso"])
                        if self.reports.torso.grad_norm
                        else None
                    ),
                    trace_norm=(
                        self.torso.block._norms(advanced["torso"])
                        if self.reports.torso.trace_norm
                        else None
                    ),
                ),
                actor=BlockUpdate(
                    grad_norm=(
                        self.actor.block._norms(traced["actor"])
                        if self.reports.actor.grad_norm
                        else None
                    ),
                    trace_norm=(
                        self.actor.block._norms(advanced["actor"])
                        if self.reports.actor.trace_norm
                        else None
                    ),
                ),
                critic=BlockUpdate(
                    grad_norm=(
                        self.critic.block._norms(traced["critic"])
                        if self.reports.critic.grad_norm
                        else None
                    ),
                    trace_norm=(
                        self.critic.block._norms(advanced["critic"])
                        if self.reports.critic.trace_norm
                        else None
                    ),
                ),
                # Per group, not per block. The two readouts took one step
                # between them, so there is one step size for the pair.
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
            ),
        )


# -------------------------------------------------------------------- the flow
class RTRRL:
    """One-invocation train/evaluation flow around the three layers."""

    def __init__(
        self,
        cfg: RTRRLConfig,
        env: Any,
        env_params: Any,
        torso_network: Any,
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
        self.core = Core(cfg, torso_network, actor_head, critic_head, reports)
        self.record = frozenset(record)

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
