"""StreamAC-RTRL: an actor and a critic with separate recurrent networks.

Neither network sees the other's gradient, so there is no shared backbone to
target and no emphasis to carry. Each keeps its own eligibility trace and steps
under its own overshooting bound, which is what lets the pair learn from a
single transition at a time without a replay buffer.

Four layers, each owning a slice of the state:

    StreamAC          the order things happen in, and the scan
    Environment       where every stream is
    Normalization     the scales the environment's numbers are read through
    Core              a policy, a value function, and the one thing they share
      Actor / Critic  a task: what it produces, and the scalar it ascends
        Network       a block of parameters nothing else owns

``init`` allocates and runs once; ``reset`` begins again, per stream and
selectively, every step. The words follow what Flax, optax and an environment
already mean by them, which is why ``Environment.init`` may call ``env.reset``
without the two colliding.

A head holds as much of a network as belongs to it alone -- here all of one,
because the two roles share nothing. Where a torso were shared it would sit in
``Core`` as a third block and each head would hold only its own output
transform, which is why ``Network`` is composed rather than inherited from.

A reading is not state, so nothing here puts one in the carry: what a layer
measured comes back beside what it carries, in this algorithm's own classes. A
field left ``None`` is an empty pytree and a scan stacks nothing for it.

Everything returned is per step, because an episode ends at a different step in
every stream while a scan emits one fixed shape. Cutting the stream into
episodes is ``memorax/runtime/episode.py``'s, and what this file owes that cut
is ``done`` and ``terminal``.

``tests/test_layered_parity.py`` drives this against ``stream_ac.py``, which is
the same algorithm written flat.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from memorax.rl import (
    environment_owns_normalization,
    make_bounded_rule,
    make_credit,
    make_normalizer,
    make_td0,
)
from memorax.utils import Timestep
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils.trees import subtree_norms

from .contract import (
    ActionDecision,
    EvaluationConfig,
    InteractionMetrics,
    StepMetrics,
    terminal_of,
)


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
    credit: str = "rtrl"
    meta_rl: bool = False
    normalization_statistics: str = "ours"


@struct.dataclass(frozen=True)
class Recurrence:
    """Where the sequence is, and what it owes the past.

    What an ending resets and what a rollout begins again. A forward pass reads
    it and hands back the next one; nothing else writes it.
    """

    carry: Any
    sensitivity: Any


@struct.dataclass(frozen=True)
class RuleState:
    """What the update carries between steps, which a forward pass never sees.

    The eligibility trace and the second moment together, because that is what
    they are: in the published optimiser both are ``self.state[p]`` while the
    parameters are ``p.data``, and here both are allocated together, handed to
    the rule together, and read by nothing else.
    """

    traces: Any
    v: Any


@struct.dataclass(frozen=True)
class NetworkState:
    """Independent online state for one recurrent actor or critic network.

    Grouped by what writes it: the parameter step writes ``params``, the trace
    recurrence and the rule write ``rule``, the forward pass and an ending write
    ``recurrence``. ``params`` stays flat because every forward pass reads it
    and a group of one would only put a hop in the hottest path.
    """

    params: Any
    rule: RuleState
    recurrence: Recurrence


@struct.dataclass(frozen=True)
class Scales:
    """What the two running estimators carry, and nothing else.

    Its own state because normalization is its own layer: the agent never sees a
    raw number, so which scale one was on is not a question anything above it
    answers.
    """

    observation: Any = None
    reward: Any = None


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
    scales: Scales
    actor: NetworkState
    critic: NetworkState


def _broadcast_env(values, leaf):
    return values[(slice(None),) + (None,) * (leaf.ndim - 1)]


def _where_done(done, fresh, carried):
    """Take the fresh one for the streams that ended, the carried one for the rest.

    Not a branch: both are computed and one is selected per stream, which is
    what every reset in this stack already is.
    """

    return jax.tree.map(
        lambda new, old: jnp.where(_broadcast_env(done, old), new, old), fresh, carried
    )


def _per_stream(objective, params, *streamed):
    """Differentiate each stream's own objective, and only its own.

    Streams share parameters but not activations, so the Jacobian of the whole
    batch is zero everywhere but the diagonal, and asking for it costs the
    square of the streams. The stream axis is put back inside as a length of one
    so every layer still sees the batched shapes it was written for.
    """

    def one(params, *stream):
        batched = jax.tree.map(lambda leaf: leaf[None], stream)
        return objective(params, *batched)[0]

    return jax.vmap(jax.grad(one), in_axes=(None, *(0,) * len(streamed)))(
        params, *streamed
    )


class Environment:
    """Where every stream is, and nothing about how its numbers are read."""

    def __init__(self, num_envs: int, env: Any, env_params: Any) -> None:
        self.num_envs = num_envs
        self.env = env
        self.env_params = env_params

    def blank_timestep(self, obs) -> Timestep:
        """The step before the first one: an observation and nothing behind it."""

        action_space = self.env.action_space(self.env_params)
        return Timestep(
            obs=obs,
            action=jnp.zeros(
                (self.num_envs, *action_space.shape), dtype=action_space.dtype
            ),
            reward=jnp.zeros((self.num_envs,), dtype=jnp.float32),
            done=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
        )

    def init(self, key: Any):
        """Every stream opened."""

        keys = jax.random.split(key, self.num_envs)
        return jax.vmap(self.env.reset, in_axes=(0, None))(keys, self.env_params)

    def reset(self, key, env_state, done):
        """Begin again wherever an episode ended, at the top of the act.

        The environment hands back the state its episode ended in, because that
        is what the bootstrap values; starting the next one belongs here.

        The fresh observation comes back unblended: a scale reads it next, and
        the estimator has to be offered the value an episode opens on before
        anything decides which streams keep the one they had.
        """

        obs, opened = self.init(key)
        return obs, _where_done(done, opened, env_state)

    def step(self, key, env_state, action):
        """One transition, before anything has been read through a scale.

        The reward that comes back is what the task paid, which is what an
        episode return and a score are read off. What the agent learns on is
        whatever the scale makes of it, and that is not this component's.
        """

        keys = jax.random.split(key, self.num_envs)
        obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(keys, env_state, action, self.env_params)
        return (
            obs,
            env_state,
            jnp.asarray(reward, dtype=jnp.float32),
            done,
            terminal_of(info, done),
            info,
        )

    def persisted(self, timestep: Timestep) -> Timestep:
        """What the next step is handed back: an ending feeds nothing forward."""

        broadcast_dims = tuple(range(timestep.done.ndim, timestep.action.ndim))
        reward = jnp.asarray(timestep.reward, dtype=jnp.float32)
        return timestep.replace(
            action=jnp.where(
                jnp.expand_dims(timestep.done, axis=broadcast_dims),
                jnp.zeros_like(timestep.action),
                timestep.action,
            ),
            reward=jnp.where(timestep.done, jnp.zeros_like(reward), reward),
        )


class Normalization:
    """The layer between the environment's numbers and the agent's.

    Whether a reading is taken at all, and whether the estimator behind it is
    allowed to move, are this component's questions and nobody else's -- which
    is why ``update_during_eval`` lives here rather than on the environment. It
    was never a statement about the environment.
    """

    def __init__(
        self,
        num_envs: int,
        env: Any,
        *,
        observation: Any = None,
        reward: Any = None,
        evaluation: EvaluationConfig | None = None,
    ) -> None:
        evaluation = evaluation or EvaluationConfig()
        self.num_envs = num_envs

        def estimator(declared):
            if declared is None:
                return None
            return make_normalizer(
                replace(
                    declared,
                    reset_on_start=evaluation.reset_on_start,
                    update_during_eval=evaluation.update_during_eval,
                )
            )

        self.observation = estimator(observation)
        self.reward = estimator(reward)
        self.normalizing = bool(self.observation or self.reward)
        self.resets_on_start = evaluation.reset_on_start
        self.updates_during_eval = evaluation.update_during_eval
        if self.normalizing and environment_owns_normalization(env):
            raise ValueError(
                "normalization owner conflict: wrapper and program normalization "
                "are both enabled"
            )

    def init(self, obs, scales: Scales | None = None, *, update: bool = True):
        """The estimators begun, on nothing or on what another run built.

        One allocator rather than two. An evaluation that inherits training's
        scales differs from a fresh one only in what is handed in, so it is
        handed in; a second constructor said the same thing with a second name.
        """

        observation = None if scales is None else scales.observation
        reward = None if scales is None else scales.reward
        if self.observation is not None:
            obs, observation = self.observation.begin(obs, observation, update=update)
        if self.reward is not None and reward is None:
            reward = self.reward.initial(jnp.zeros((self.num_envs,), dtype=jnp.float32))
        return obs, Scales(observation=observation, reward=reward)

    def reset(self, obs, scales: Scales, done, *, update: bool = True):
        """The value an episode opens on, for the streams that opened one.

        One stream at a time, so the ones still running are not counted twice.
        The reward estimator is not begun: what it sees is an accumulation, and
        dropping that at an ending is ``reset_on_done``'s business.
        """

        observation = scales.observation
        if self.observation is not None:
            obs, observation = self.observation.begin(obs, observation, update=update)
        return obs, scales.replace(
            observation=_where_done(done, observation, scales.observation)
        )

    def apply(self, scales: Scales, obs, reward, done, *, update: bool = True):
        """Both estimators advanced by one transition, or neither if none exists.

        ``Normalizer.observe`` scales by the statistics it has just written, so
        the reading cannot be taken before the advance. That is wiring rather
        than a constraint, and it is only invisible because an accumulator and a
        running estimate are one class there.
        """

        observation = scales.observation
        counted = scales.reward
        if self.observation is not None:
            obs, observation = self.observation.observe(
                observation, obs, done=done, update=update
            )
        if self.reward is not None:
            reward, counted = self.reward.observe(
                counted, reward, done=done, update=update
            )
        return obs, reward, Scales(observation=observation, reward=counted)


class Network:
    """One block of parameters that nothing else owns.

    It carries its own trace and decay, is differentiated as a unit, and steps
    as a unit. How big a block is belongs to the algorithm and not to this
    class: here each role owns a whole sequence, and where a torso is shared a
    head's block is only its own output transform.
    """

    def __init__(self, cfg: StreamACConfig, network: Any, *, bound, base) -> None:
        self.cfg = cfg
        self.network = network
        self.credit = make_credit(cfg.credit, network.core)
        # A rule group is exactly one block here. Where a group spans several,
        # the step moves up to whoever holds them all, because the bound reads a
        # norm over the whole group.
        self.rule = make_bounded_rule(bound=bound, base=base)
        self.trace_decay = cfg.gamma * cfg.trace_lambda

    @property
    def carry_shape(self):
        return (self.cfg.num_envs, None)

    def _input(self, obs, action, reward):
        """The one vector a sequence sees.

        Under ``meta_rl`` the previous action and reward are concatenated onto
        the observation.
        """

        if not self.cfg.meta_rl:
            return obs
        return jnp.concatenate([obs, action, reward], axis=-1)

    def apply(self, params, timestep, recurrence: Recurrence):
        """One forward pass, handing back the advance rather than writing it.

        The gradient a step takes is taken from the carry the pass started on,
        so whoever owns that carry decides when it moves.
        """

        obs, done, action, reward = timestep
        (carry, sensitivity), output = self.network.walk(
            params,
            self._input(obs, action, reward),
            done=done,
            carries=recurrence.carry,
            sensitivity=recurrence.sensitivity,
            credit=self.credit,
        )
        return Recurrence(carry=carry, sensitivity=sensitivity), output

    def init(self, keys, timestep: Timestep) -> NetworkState:
        """Fresh online state for this block.

        Three keys -- parameters, torso, dropout -- in the order the published
        kernel spends them, because a comparison at one seed is only a
        comparison if both sides start from the same draw. ``timestep`` is read
        for its shapes and nothing else.
        """

        param_key, torso_key, dropout_key = keys
        obs, done, action, reward = timestep
        carry = self.network.initialize_carry(jax.random.key(0), self.carry_shape)
        sensitivity = self.credit.initialize(param_key, self.carry_shape)
        with self.credit.initialization():
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
                # Asked for rather than assumed: a bounded rule's second moment
                # follows the traces, an unbounded one follows its transform.
                v=self.rule.init(params=params, traces=traces),
            ),
            recurrence=Recurrence(carry=carry, sensitivity=sensitivity),
        )

    def reset(self, key, state: NetworkState) -> NetworkState:
        """The same parameters with the recurrence begun again.

        Through the credit, not around it: a truncated credit carries no
        sensitivity at all, and asking the recurrence directly hands back a tree
        the evaluation step will not produce, which scan rejects.
        """

        return state.replace(
            recurrence=Recurrence(
                carry=self.network.initialize_carry(key, self.carry_shape),
                sensitivity=self.credit.initialize(key, self.carry_shape),
            )
        )

    def trace(self, incoming, gradient, *, reset_before):
        """StreamAC's pre-forward reset, always-fresh trace recurrence.

        One tree back rather than two, because what this step steps along and
        what the next decays are the same trace here. The decay is this block's
        own; an algorithm with three blocks has three.
        """

        return jax.tree.map(
            lambda old, grad: (
                self.trace_decay * (1 - _broadcast_env(reset_before, old)) * old + grad
            ),
            incoming,
            gradient,
        )

    def _norms(self, tree):
        """One norm per position group, per stream.

        The grouping is the sequence's own -- what came before the recurrence,
        what is the recurrence, what came after -- so a reading says where in
        the network something got large rather than only that something did.
        """

        return subtree_norms(self.network.split(tree), streams=True)

    def step(self, state: NetworkState, gradient, delta, *, reset_before, step):
        """Trace the gradient, take the bounded step, and say what it did.

        ``delta`` is the same for every block: the bound reads it whichever
        objective was differentiated.
        """

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
            step_size=taken.metrics.get("step_size"),
            grad_norm=self._norms(gradient),
            trace_norm=self._norms(traces),
        )


class Actor:
    """The policy. It chooses, and it names the scalar its block ascends.

    It holds a whole network because nothing else reaches those parameters. A
    head whose torso were shared would hold only its own output transform and
    hand a cotangent upward; nothing else here would change.
    """

    def __init__(self, cfg: StreamACConfig, network: Any) -> None:
        self.cfg = cfg
        self.block = Network(cfg, network, bound=cfg.actor_bound, base=cfg.actor_base)

    def init(self, keys, timestep: Timestep) -> NetworkState:
        return self.block.init(keys, timestep)

    def reset(self, key, state: NetworkState) -> NetworkState:
        return self.block.reset(key, state)

    def act(
        self, key, state: NetworkState, timestep: Timestep, *, deterministic: bool
    ) -> tuple[Recurrence, Any, ActorForward]:
        """Run forward once and choose, touching nothing else.

        ``deterministic`` is read at trace time rather than stepped over: the
        greedy rollout and the learning one are two programs, and one carrying
        the branch would also carry the sampler it never uses.
        """

        recurrence, (dist, _) = self.block.apply(
            state.params, timestep.to_sequence(), state.recurrence
        )
        if deterministic:
            action = (
                jnp.argmax(dist.logits, axis=-1)
                if hasattr(dist, "logits")
                else dist.mode()
            )
            # A mode has no draw behind it, so there is no probability of having
            # chosen it and nothing here to report.
            return recurrence, remove_time_axis(action), ActorForward()
        action, log_prob = dist.sample_and_log_prob(seed=key)
        return (
            recurrence,
            remove_time_axis(action),
            ActorForward(
                log_prob=remove_time_axis(log_prob),
                entropy=remove_time_axis(dist.entropy()),
            ),
        )

    def objective(self, output, action, delta):
        """What this head ascends: log pi(a) with entropy riding on it.

        Entropy is signed by the TD error rather than added flat, so it pushes
        toward exploration only where the critic was surprised.
        """

        dist, _ = output
        return remove_time_axis(
            dist.log_prob(add_time_axis(action))
        ) + self.cfg.entropy_coefficient * jnp.sign(
            jax.lax.stop_gradient(delta)
        ) * remove_time_axis(
            dist.entropy()
        )

    def gradient(self, state: NetworkState, timestep: Timestep, action, delta):
        """This head's ascent, one stream at a time.

        From a recurrence already cut out of the graph: what a parameter did to
        the past reaches this step through the sensitivity the credit carries
        and through nothing else.
        """

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
        """One transition's worth of learning, from where the acting pass began.

        ``delta`` arrives rather than being measured: the actor has no value
        function. The recurrence is not advanced, because the acting pass did.
        """

        gradient = self.gradient(state, timestep, next_timestep.action, delta)
        return self.block.step(
            state, gradient, delta, reset_before=reset_before, step=step
        )


class Critic:
    """The value. It reads, and it ascends its own reading.

    The TD error is not measured here. It is what the two roles are coupled by,
    so it belongs to whatever holds both of them; pushed into this class it
    would still be the same coupling, only harder to see.
    """

    def __init__(self, cfg: StreamACConfig, network: Any) -> None:
        self.cfg = cfg
        self.block = Network(cfg, network, bound=cfg.critic_bound, base=cfg.critic_base)

    def init(self, keys, timestep: Timestep) -> NetworkState:
        return self.block.init(keys, timestep)

    def reset(self, key, state: NetworkState) -> NetworkState:
        return self.block.reset(key, state)

    def objective(self, output):
        """What this head ascends: the value itself, with no error in it.

        A scalar, not a direction. The delta reaches the critic in the rule,
        which is where the bound reads it.
        """

        value, _ = output
        return remove_feature_axis(remove_time_axis(value))

    def apply(
        self, params, timestep: Timestep, recurrence: Recurrence
    ) -> tuple[Recurrence, Any]:
        """What a state is worth, and the recurrence that reading advanced.

        One pass, not one per use: the two readings a TD error needs differ only
        in what is handed in, and that is the coupling's business rather than
        this head's.
        """

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
        """Learning, plus the recurrence the valuing pass advanced.

        Unlike the actor's, this block's recurrence *is* written: the pass that
        advanced it was the one the TD error was measured from, and the caller
        has it in hand because that is where the two roles meet.
        """

        gradient = self.gradient(state, timestep)
        stepped, reading = self.block.step(
            state, gradient, delta, reset_before=reset_before, step=step
        )
        return stepped.replace(recurrence=recurrence), reading


class Core:
    """A policy, a value function, and the one thing they are coupled by.

    Everything either role can do alone lives in that role. What is left is the
    algorithm: value the state, value where it ended, take the difference, step
    both roles on it. Nothing here touches an environment.

    Thin, because these two roles share nothing. An algorithm whose heads sat on
    one torso would put that torso here and grow this class by exactly what the
    sharing costs -- routing the cotangents the heads hand up, and gating which
    of them the shared block hears.
    """

    def __init__(
        self,
        cfg: StreamACConfig,
        actor_network: Any,
        critic_network: Any,
    ) -> None:
        self.cfg = cfg
        self.actor = Actor(cfg, actor_network)
        self.critic = Critic(cfg, critic_network)
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
        """One transition's worth of learning for both roles, and what it read.

        ``state`` is the state the transition *began* in -- both carries as they
        were before the acting pass -- and ``next_timestep`` is where it ended.
        ``terminal`` is the ending that says the future is worth nothing;
        without one, an ending is read as a failure, which is the safe reading
        and what a single flag always meant.
        """

        actor = state.actor
        critic = state.critic
        reset_before = state.timestep.done
        terminal = next_timestep.done if terminal is None else terminal
        current_step = state.update_step + 1

        recurrence, value = self.critic.apply(
            critic.params, state.timestep, critic.recurrence
        )
        # The same reading with everything it could learn from cut away, and
        # its advance dropped: the next step repeats the pass from the same
        # carry under updated parameters, and that is the one that is kept.
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
            CriticForward(value=value, next_value=next_value),
            UpdateMetrics(
                td_error=td_error, actor=actor_reading, critic=critic_reading
            ),
        )


class StreamAC:
    """One-invocation train/evaluation flow around the three layers.

    It owns none of the arithmetic and none of the environment: it owns the
    order. Restart what ended, act, step the world, learn from what that
    produced, and report the transition.
    """

    def __init__(
        self,
        cfg: StreamACConfig,
        env: Any,
        env_params: Any,
        actor_network: Any,
        critic_network: Any,
        *,
        observation_normalization: Any = None,
        reward_normalization: Any = None,
        evaluation: EvaluationConfig | None = None,
        record: Iterable[str] = (),
    ) -> None:
        self.cfg = cfg
        self.environment = Environment(cfg.num_envs, env, env_params)
        self.normalization = Normalization(
            cfg.num_envs,
            env,
            observation=observation_normalization,
            reward=reward_normalization,
            evaluation=evaluation,
        )
        self.core = Core(cfg, actor_network, critic_network)
        # Which optional per-step fields to fill, named rather than switched on
        # in bundles, so nothing nobody reduces is ever stacked.
        self.record = frozenset(record)

    def init(self, key: Any) -> StreamACState:
        # Seven keys in the order the published kernel spends them. Two of
        # them feed rng streams these networks never ask for and are drawn
        # anyway: a comparison at one seed is only a comparison if both sides
        # start from the same draw.
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
        """Both components begun again for the streams that ended, in order.

        The environment offers the observation an episode opens on, the scale
        reads it, and only then is it blended with the one the still-running
        streams already had -- so an estimator is offered every fresh value
        exactly once, whichever streams end up keeping it.
        """

        done = state.timestep.done
        obs, env_state = self.environment.reset(key, state.env_state, done)
        obs, scales = self.normalization.reset(obs, state.scales, done, update=update)
        return state.replace(
            timestep=state.timestep.replace(
                obs=_where_done(done, obs, state.timestep.obs)
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
        """One transition, with the trajectory kept only if something reads it.

        The two observations are a vector per stream per step and the only
        expensive thing here, so they are behind the declaration; a name nobody
        declared is never stacked.
        """

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
            # The acting pass advanced this; the update differentiated from
            # where that pass started and left it alone.
            actor=updated.actor.replace(recurrence=recurrence),
        )
        return next_state, StepMetrics(
            interaction=self._interaction(
                observation=observation,
                next_observation=next_timestep.obs,
                action=action,
                # One sample, named every way the step reads it: only the ending
                # tells the bootstrap's action from the one fed forward.
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
            # Only the actor ran, so only the actor's recurrence moved.
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
        """The trained parameters, opened on a fresh environment and recurrence.

        Inheriting training's scales is handing them to the same ``init`` a
        fresh run calls with nothing, so what evaluation does differently is an
        argument and not a second code path.
        """

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
        """A rollout that leaves nothing behind.

        The state an evaluation runs on is built here and dropped here, so a
        caller cannot carry it into training even by accident. ``num_steps`` is
        environment steps, the same as ``train``'s.
        """

        reset_key, rollout_key = jax.random.split(key)
        eval_state = self._evaluation_state(reset_key, state)
        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(rollout_key, scan_steps)
        _, metrics = jax.lax.scan(self.evaluate_step, eval_state, keys)
        return metrics
