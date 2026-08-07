"""StreamAC-RTRL: an actor and a critic with separate recurrent networks.

Neither network sees the other's gradient, so there is no shared backbone to
target and no emphasis to carry. Each keeps its own eligibility trace and each
steps under its own overshooting bound, which is what lets the pair learn from
a single transition at a time without a replay buffer.

Everything static is resolved in the constructor, so no method below asks the
configuration a question: what the methods read are the pieces that answer was
turned into. The point of that, and of these being methods rather than the
closures they used to be, is that a single step can be called on a state you
made up and compared against another implementation. A kernel that can only be
run for a whole epoch can only be judged by whether the curve looked right.

Four layers, each owning a slice of the state:

    StreamAC          the order things happen in, and the scan
    EnvTransformer    the environment and the two scales its numbers pass through
    Core              a policy, a value function, and the one thing they share
      Actor / Critic  a task: what it produces, and the direction it ascends
        Network       a block of parameters nothing else owns

A head holds as much of a network as belongs to it alone. Here that is all of
one, because the two roles share nothing; where a torso were shared it would sit
in ``Core`` as a third block and each head would hold only its own output
transform. ``Network`` is composed rather than inherited from for exactly that
reason -- inheriting would weld the boundary at "all of it".

``Network`` exists because a block is a real thing: it carries its own trace and
decay, it is differentiated as a unit, and it steps as a unit. The two roles'
forward, reset, trace and step were the same code twice and are now the same
code once.

``Core`` exists because one thing is not separable: the TD error is measured by
the critic and ascended toward by the actor, and both bounds read their step
size off it. It is written where both roles are in hand rather than pushed into
either, so the coupling is visible instead of hidden inside the half that
happens to compute it. It is thin here, and would grow by exactly what sharing
costs.

That the interaction is a layer of its own means the update can be handed a
state you made up without an environment being reachable at all, which is the
same line the reference implementation draws between ``sample_action``,
``update_params`` and the loop in its ``main``.

Nothing is observed but the transition. What a step's quantities were -- the TD
error, the step sizes, the norms -- is a reading taken off a graph rather than
part of one, and every one of them is still a local here for whoever comes to
take it. Kept as returned values they would be threaded through every signature
between the arithmetic and the sink, which is how the shape of the update comes
to be argued about in terms of what a dashboard wanted. The transition stays,
because it is not a reading: it is what happened, and it is what an episode
boundary is found in.
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
class EnvironmentState:
    """What the environment layer carries: where each stream is, and the scales.

    The two estimators sit here rather than beside the networks because they
    are what the environment's numbers are read through, and because an ending
    resets them on the same flag it resets the environment on.
    """

    env_state: Any
    observation_statistics: Any = None
    reward_statistics: Any = None


@struct.dataclass(frozen=True)
class StreamACState:
    """Everything the kernel carries, one field per layer that writes one."""

    step: Any
    update_step: Any
    timestep: Timestep

    environment: EnvironmentState
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


def _per_stream(direction, params, *streamed):
    """Differentiate each stream's own direction, and only its own.

    Streams share parameters but not activations, so a stream's direction cannot
    depend on another's hidden state and the Jacobian of the whole batch is zero
    everywhere but the diagonal. Asking for that Jacobian asks the compiler to
    fill the zeros in too, at a cost that grows with the square of the streams;
    one gradient per stream, taken together, costs what the diagonal costs.

    The stream axis is put back inside as a length of one so that every layer
    still sees the batched shapes it was written for, which is also why the
    arithmetic is unchanged and the comparisons against upstream still hold.
    """

    def one(params, *stream):
        batched = jax.tree.map(lambda leaf: leaf[None], stream)
        return direction(params, *batched)[0]

    return jax.vmap(jax.grad(one), in_axes=(None, *(0,) * len(streamed)))(
        params, *streamed
    )


class EnvTransformer:
    """The environment, and the two scales its numbers are read through.

    ``apply`` and ``update`` are one call here rather than two: an estimator
    that answers a value and an estimator that has seen it are the same object a
    moment apart, and asking for the reading without the advance would hand back
    a scale nothing had been told about.
    """

    def __init__(
        self,
        cfg: StreamACConfig,
        env: Any,
        env_params: Any,
        *,
        observation_normalization: Any = None,
        reward_normalization: Any = None,
        evaluation: EvaluationConfig | None = None,
    ) -> None:
        evaluation = evaluation or EvaluationConfig()
        self.cfg = cfg
        self.env = env
        self.env_params = env_params

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

        self.observation_normalizer = estimator(observation_normalization)
        self.reward_normalizer = estimator(reward_normalization)
        self.normalizing = bool(self.observation_normalizer or self.reward_normalizer)
        self.resets_on_start = evaluation.reset_on_start
        self.updates_during_eval = evaluation.update_during_eval
        if self.normalizing and environment_owns_normalization(env):
            raise ValueError(
                "normalization owner conflict: wrapper and program normalization "
                "are both enabled"
            )

    def blank_timestep(self, obs) -> Timestep:
        """The step before the first one: an observation and nothing behind it."""

        action_space = self.env.action_space(self.env_params)
        return Timestep(
            obs=obs,
            action=jnp.zeros(
                (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
            ),
            reward=jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32),
            done=jnp.zeros((self.cfg.num_envs,), dtype=jnp.bool_),
        )

    def _opened(self, key):
        keys = jax.random.split(key, self.cfg.num_envs)
        return jax.vmap(self.env.reset, in_axes=(0, None))(keys, self.env_params)

    def reset(self, key: Any) -> tuple[Any, EnvironmentState]:
        """Every stream opened, and the two estimators begun."""

        obs, env_state = self._opened(key)
        observation_statistics = reward_statistics = None
        if self.observation_normalizer is not None:
            obs, observation_statistics = self.observation_normalizer.begin(obs)
        if self.reward_normalizer is not None:
            reward_statistics = self.reward_normalizer.initial(
                jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
            )
        return obs, EnvironmentState(
            env_state=env_state,
            observation_statistics=observation_statistics,
            reward_statistics=reward_statistics,
        )

    def restart(
        self, key, state: EnvironmentState, timestep: Timestep, *, update: bool = True
    ) -> tuple[Any, EnvironmentState]:
        """Begin again wherever an episode ended, at the top of the act.

        The environment hands back the state its episode ended in, because that
        is what the bootstrap has to value; starting the next one belongs here,
        on the same flag the carries and the traces already read. The statistics
        take the observation the agent is about to act on, one stream at a time,
        so the streams still running are not counted twice.
        """

        obs, env_state = self._opened(key)
        statistics = state.observation_statistics
        if self.observation_normalizer is not None:
            obs, statistics = self.observation_normalizer.begin(
                obs, statistics, update=update
            )
        # The reward estimator is not begun: what it sees is an accumulation,
        # and dropping that at an ending is ``reset_on_done``'s business.
        done = timestep.done
        return _where_done(done, obs, timestep.obs), state.replace(
            env_state=_where_done(done, env_state, state.env_state),
            observation_statistics=_where_done(
                done, statistics, state.observation_statistics
            ),
        )

    def apply(
        self, state: EnvironmentState, obs, reward, done, *, update: bool = True
    ) -> tuple[Any, Any, EnvironmentState]:
        """Both estimators advanced by one transition, or neither if none exists."""

        observation_statistics = state.observation_statistics
        reward_statistics = state.reward_statistics
        if self.observation_normalizer is not None:
            obs, observation_statistics = self.observation_normalizer.observe(
                observation_statistics, obs, done=done, update=update
            )
        if self.reward_normalizer is not None:
            reward, reward_statistics = self.reward_normalizer.observe(
                reward_statistics, reward, done=done, update=update
            )
        return (
            obs,
            reward,
            state.replace(
                observation_statistics=observation_statistics,
                reward_statistics=reward_statistics,
            ),
        )

    def step(
        self, key, state: EnvironmentState, action, *, update: bool = True
    ) -> tuple[Timestep, EnvironmentState, Any, Any, Any]:
        """One transition, as the agent sees it and as the task paid for it.

        Two rewards come back. What the environment paid is what an episode
        return and a score are read off, and those are statements about the task
        rather than about the scale the agent happens to be learning on.
        """

        keys = jax.random.split(key, self.cfg.num_envs)
        obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(keys, state.env_state, action, self.env_params)
        environment_reward = jnp.asarray(reward, dtype=jnp.float32)
        terminal = terminal_of(info, done)
        obs, reward, state = self.apply(
            state.replace(env_state=env_state), obs, reward, done, update=update
        )
        return (
            Timestep(
                obs=obs,
                action=action,
                reward=jnp.asarray(reward, dtype=jnp.float32),
                done=done,
            ),
            state,
            environment_reward,
            terminal,
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

    def opened_for_evaluation(self, key, state: EnvironmentState):
        """A fresh environment, on statistics of its own or the ones training built."""

        obs, env_state = self._opened(key)
        fresh = self.resets_on_start
        observation_statistics = None if fresh else state.observation_statistics
        reward_statistics = None if fresh else state.reward_statistics
        if self.observation_normalizer is not None:
            obs, observation_statistics = self.observation_normalizer.begin(
                obs, observation_statistics, update=fresh or self.updates_during_eval
            )
        if self.reward_normalizer is not None and reward_statistics is None:
            reward_statistics = self.reward_normalizer.initial(
                jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
            )
        return obs, EnvironmentState(
            env_state=env_state,
            observation_statistics=observation_statistics,
            reward_statistics=reward_statistics,
        )


class Network:
    """One block of parameters that nothing else owns.

    A block is what it is because no other component reaches it: it carries its
    own trace and its own decay, it is differentiated as a unit, and it steps as
    a unit. How big a block is depends on the algorithm and not on this class --
    here each role owns a whole sequence, and in an algorithm with a shared
    torso a head's block is only its own output transform.

    ``backward`` takes the cotangent arriving from whatever consumed this
    block's output and hands back both this block's gradient and the cotangent
    to pass further up. Here nothing is further up, so the second is dropped;
    the signature is the same either way, which is what lets a shared block sit
    above these without changing them.
    """

    def __init__(self, cfg: StreamACConfig, network: Any, *, bound, base) -> None:
        self.cfg = cfg
        self.network = network
        # Resolved once. Which of a cell's methods stands for credit, and what
        # shape a step takes, are answered here so that nothing below reads a
        # string out of the configuration.
        self.credit = make_credit(cfg.credit, network.core)
        # The rule lives here because in this algorithm a rule group is exactly
        # one block. Where a group spans several blocks -- two heads bounded
        # together -- the step is the only thing that has to move up to whoever
        # holds them both, because the bound reads a norm over the whole group.
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
        """One forward pass over one sequence-shaped step.

        What comes back is the next recurrence and the readout. The advance is
        handed back rather than written, because the gradient a step takes is
        taken from the carry the pass *started* on.
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

    def backward(self, params, timestep, recurrence: Recurrence, direction):
        """This block's gradient of a scalar its owner named.

        ``direction`` turns the readout into the quantity to ascend, which is
        the head's business and not this class's. Differentiated one stream at a
        time, and from a recurrence already cut out of the graph: what a
        parameter did to the past reaches this step through the sensitivity the
        credit carries, and through nothing else.
        """

        def per_stream(differentiated, timestep, recurrence, *extra):
            _, output = self.apply(differentiated, timestep, recurrence)
            return direction(output, *extra)

        def gradient(*extra):
            return _per_stream(
                per_stream,
                params,
                timestep.to_sequence(),
                jax.lax.stop_gradient(recurrence),
                *extra,
            )

        return gradient

    def reset(self, keys, timestep: Timestep) -> NetworkState:
        """Fresh online state for this block.

        Three keys -- parameters, torso, dropout -- and in that order, because
        that is how the published kernel spends its seed and a comparison at one
        seed is only a comparison if both sides start from the same draw.
        ``timestep`` is one sequence-shaped step, which is all initialisation
        reads: the shapes.
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
                # Asked for rather than assumed: a bounded rule carries a second
                # moment shaped like the traces, an unbounded one carries
                # whatever its base transformation built.
                v=self.rule.init(params=params, traces=traces),
            ),
            recurrence=Recurrence(carry=carry, sensitivity=sensitivity),
        )

    def restarted(self, key, state: NetworkState) -> NetworkState:
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

        One tree back rather than two: what this step steps along and what the
        next step decays are the same trace here, and naming them separately
        only said that some other algorithm's are not. The decay is this
        block's own -- an algorithm with three blocks has three of them.
        """

        return jax.tree.map(
            lambda old, grad: (
                self.trace_decay * (1 - _broadcast_env(reset_before, old)) * old + grad
            ),
            incoming,
            gradient,
        )

    def step(self, state: NetworkState, gradient, delta, *, reset_before, step):
        """Trace the gradient and take the bounded step.

        ``delta`` is the same for every block: the bound reads it whichever
        direction was differentiated.
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
        )


class Actor:
    """The policy. It chooses, and it names the direction its block ascends.

    It holds a whole network because in this algorithm nothing else reaches
    those parameters. A head whose torso were shared would hold only its own
    output transform and hand a cotangent upward instead; nothing else about
    this class would change.
    """

    def __init__(self, cfg: StreamACConfig, network: Any) -> None:
        self.cfg = cfg
        self.block = Network(cfg, network, bound=cfg.actor_bound, base=cfg.actor_base)

    def reset(self, keys, timestep: Timestep) -> NetworkState:
        return self.block.reset(keys, timestep)

    def restarted(self, key, state: NetworkState) -> NetworkState:
        return self.block.restarted(key, state)

    def act(
        self, key, state: NetworkState, timestep: Timestep, *, deterministic: bool
    ) -> tuple[Recurrence, Any]:
        """Run forward once and choose, touching nothing else.

        Back come the advanced recurrence and the action -- a recurrence rather
        than a whole network state, because that is all acting can have changed.

        ``deterministic`` is read at trace time, not stepped over: the greedy
        rollout and the learning one are two programs, and a rollout that had to
        carry a branch would also have to carry the sampler it never uses.
        """

        recurrence, (dist, _) = self.block.apply(
            state.params, timestep.to_sequence(), state.recurrence
        )
        action = (
            (
                jnp.argmax(dist.logits, axis=-1)
                if hasattr(dist, "logits")
                else dist.mode()
            )
            if deterministic
            else dist.sample(seed=key)
        )
        return recurrence, remove_time_axis(action)

    def direction(self, output, action, delta):
        """What this head ascends: log pi(a) with entropy riding on it.

        Entropy is signed by the TD error rather than added flat, so it pushes
        toward exploration only where the critic was surprised. This is the
        whole of what the head contributes to its own update; the block does the
        differentiating.
        """

        dist, _ = output
        return remove_time_axis(
            dist.log_prob(add_time_axis(action))
        ) + self.cfg.entropy_coefficient * jnp.sign(
            jax.lax.stop_gradient(delta)
        ) * remove_time_axis(
            dist.entropy()
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
    ) -> NetworkState:
        """One transition's worth of learning, from where the acting pass began.

        ``delta`` is an argument rather than something measured here: the actor
        has no value function, so the one quantity the two roles are not
        independent in reaches it from whoever holds both.

        The recurrence is not advanced -- the pass that advanced it was the
        acting pass.
        """

        gradient = self.block.backward(
            state.params, timestep, state.recurrence, self.direction
        )(next_timestep.action, delta)
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

    def reset(self, keys, timestep: Timestep) -> NetworkState:
        return self.block.reset(keys, timestep)

    def restarted(self, key, state: NetworkState) -> NetworkState:
        return self.block.restarted(key, state)

    def direction(self, output):
        """What this head ascends: the value itself, with no error in it.

        The delta reaches the critic in the rule, which is where the bound
        reads it.
        """

        value, _ = output
        return remove_feature_axis(remove_time_axis(value))

    def evaluate(
        self, state: NetworkState, timestep: Timestep
    ) -> tuple[Recurrence, Any]:
        """What this state is worth, and the recurrence that reading advanced."""

        recurrence, output = self.block.apply(
            state.params, timestep.to_sequence(), state.recurrence
        )
        return recurrence, self.direction(output)

    def bootstrap(
        self, state: NetworkState, recurrence: Recurrence, timestep: Timestep
    ):
        """What the transition ended on, under the parameters from before the step.

        The environment resets nothing, so this is the state the transition
        actually ended in. The carry this pass produces is thrown away: the next
        step repeats the pass from the same carry but with updated parameters,
        and that is the pass whose recurrent state is kept.
        """

        _, output = self.block.apply(
            jax.lax.stop_gradient(state.params),
            timestep.to_sequence(),
            jax.lax.stop_gradient(recurrence),
        )
        return self.direction(output)

    def update(
        self,
        state: NetworkState,
        timestep: Timestep,
        delta,
        *,
        recurrence: Recurrence,
        reset_before,
        step,
    ) -> NetworkState:
        """Learning, plus the recurrence the valuing pass advanced.

        Unlike the actor's, this block's recurrence *is* written: the pass that
        advanced it was the one the TD error was measured from, and the caller
        has it in hand because that is where the two roles meet.
        """

        gradient = self.block.backward(
            state.params, timestep, state.recurrence, self.direction
        )()
        stepped = self.block.step(
            state, gradient, delta, reset_before=reset_before, step=step
        )
        return stepped.replace(recurrence=recurrence)


class Core:
    """A policy, a value function, and the one thing they are coupled by.

    Everything either role can do alone lives in that role, and each role owns
    the whole block of parameters nothing else reaches. What is left here is the
    coupling, which is what makes this readable as the algorithm: value the
    state, value where it ended, take the difference, step both roles on it.
    Nothing here touches an environment.

    It is thin because these two roles share nothing. An algorithm whose heads
    sat on one torso would put that torso here as a third block, and this class
    would grow by exactly what the sharing costs: routing the cotangents the
    heads hand up, and gating which of them the shared block is allowed to hear.
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

    def reset(
        self, actor_keys, critic_keys, timestep: Timestep
    ) -> tuple[NetworkState, NetworkState]:
        return (
            self.actor.reset(actor_keys, timestep),
            self.critic.reset(critic_keys, timestep),
        )

    def restarted(
        self, key, actor_state: NetworkState, critic_state: NetworkState
    ) -> tuple[NetworkState, NetworkState]:
        return (
            self.actor.restarted(key, actor_state),
            self.critic.restarted(key, critic_state),
        )

    def sample_action(
        self,
        key: Any,
        timestep: Timestep,
        actor_state: NetworkState,
        deterministic: bool,
    ) -> tuple[Recurrence, Any]:
        return self.actor.act(key, actor_state, timestep, deterministic=deterministic)

    def update_parameters(
        self,
        state: StreamACState,
        next_timestep: Timestep,
        *,
        terminal: Any = None,
    ) -> StreamACState:
        """One transition's worth of learning for both roles.

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

        recurrence, value = self.critic.evaluate(critic, state.timestep)
        next_value = self.critic.bootstrap(critic, recurrence, next_timestep)
        td_error = self.td0(
            reward=next_timestep.reward,
            value=value,
            next_value=next_value,
            terminal=terminal,
            gamma=self.cfg.gamma,
        )

        return state.replace(
            update_step=current_step,
            actor=self.actor.update(
                actor,
                state.timestep,
                next_timestep,
                td_error,
                reset_before=reset_before,
                step=current_step,
            ),
            critic=self.critic.update(
                critic,
                state.timestep,
                td_error,
                recurrence=recurrence,
                reset_before=reset_before,
                step=current_step,
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
        self.environment = EnvTransformer(
            cfg,
            env,
            env_params,
            observation_normalization=observation_normalization,
            reward_normalization=reward_normalization,
            evaluation=evaluation,
        )
        self.core = Core(cfg, actor_network, critic_network)
        # Which of the optional per-step fields to fill. A caller names what its
        # metrics need rather than switching a bundle on, so a field nobody
        # reduces is never stacked and a field somebody does is never missing.
        self.record = frozenset(record)

    def reset(self, key: Any) -> StreamACState:
        # Seven keys in this order because that is how the published kernel
        # spends its seed. Two implementations of the same algorithm can only be
        # compared at one seed if the seed buys them the same starting point,
        # and drawing three keys where the reference draws seven makes every
        # such comparison a comparison of two different draws instead. Two of
        # the seven feed rng streams our networks do not ask for; they are drawn
        # anyway, because what matters is which key each stream receives.
        (
            env_key,
            actor_key,
            actor_torso_key,
            actor_dropout_key,
            critic_key,
            critic_torso_key,
            critic_dropout_key,
        ) = jax.random.split(key, 7)
        obs, environment = self.environment.reset(env_key)
        timestep = self.environment.blank_timestep(obs).to_sequence()
        actor, critic = self.core.reset(
            (actor_key, actor_torso_key, actor_dropout_key),
            (critic_key, critic_torso_key, critic_dropout_key),
            timestep,
        )
        return StreamACState(
            step=jnp.asarray(0, dtype=jnp.int32),
            update_step=jnp.asarray(0, dtype=jnp.int32),
            timestep=timestep.from_sequence(),
            environment=environment,
            actor=actor,
            critic=critic,
        )

    def _restarted(self, key, state: StreamACState, *, update=True) -> StreamACState:
        obs, environment = self.environment.restart(
            key, state.environment, state.timestep, update=update
        )
        return state.replace(
            timestep=state.timestep.replace(obs=obs), environment=environment
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

        restart_key, action_key, env_key = jax.random.split(key, 3)
        state = self._restarted(restart_key, state)
        observation = state.timestep.obs

        recurrence, action = self.core.sample_action(
            action_key, state.timestep, state.actor, deterministic=False
        )
        next_timestep, environment, environment_reward, terminal, info = (
            self.environment.step(env_key, state.environment, action)
        )
        updated = self.core.update_parameters(state, next_timestep, terminal=terminal)

        persisted = self.environment.persisted(next_timestep)
        next_state = updated.replace(
            step=state.step + self.cfg.num_envs,
            timestep=persisted,
            environment=environment,
            # The acting pass is what advanced the actor's recurrence; the
            # update differentiated from where that pass started and left it
            # alone. Written here, once, where both are in hand.
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
        )

    def evaluate_step(
        self, state: StreamACState, key: Any
    ) -> tuple[StreamACState, StepMetrics]:
        """The same interaction with the greedy action and no update at all."""

        restart_key, action_key, env_key = jax.random.split(key, 3)
        update = self.environment.updates_during_eval
        state = self._restarted(restart_key, state, update=update)
        observation = state.timestep.obs

        recurrence, action = self.core.sample_action(
            action_key, state.timestep, state.actor, deterministic=True
        )
        next_timestep, environment, environment_reward, terminal, info = (
            self.environment.step(env_key, state.environment, action, update=update)
        )
        return state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=self.environment.persisted(next_timestep),
            environment=environment,
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
        """The trained parameters, opened on a fresh environment and recurrence."""

        obs, environment = self.environment.opened_for_evaluation(
            key, state.environment
        )
        actor, critic = self.core.restarted(
            jax.random.key(0), state.actor, state.critic
        )
        return state.replace(
            timestep=self.environment.blank_timestep(obs),
            environment=environment,
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

    def evaluate(
        self, key: Any, state: StreamACState, num_steps: int
    ) -> tuple[StreamACState, StepMetrics]:
        reset_key, rollout_key = jax.random.split(key)
        eval_state = self._evaluation_state(reset_key, state)
        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(rollout_key, scan_steps)
        return jax.lax.scan(self.evaluate_step, eval_state, keys)
