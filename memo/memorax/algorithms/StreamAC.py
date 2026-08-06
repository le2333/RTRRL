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

The arithmetic and the loop are two objects. :class:`StreamACCore` is the graph
the published kernel writes down -- act, bootstrap, differentiate, trace, step
-- and it never touches an environment; :class:`StreamAC` is the interaction
around it -- reset on an ending, step, normalise, scan, report. That is the same
line the reference implementation draws between ``sample_action`` and
``update_params`` and the loop in its ``main``; drawn here, it means the update
can be handed a state you made up without an environment being reachable at all.

What is a method here is what two callers share: the actor and the critic run
the same forward and the same trace under different parameters, and training and
evaluation interact the same way under different actions. Nothing is a method
for being a phase of the update -- the objective, the target and the step are
written where they are used, once each.

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
class StreamACState:
    """Everything the kernel carries from one transition to the next."""

    step: Any
    update_step: Any
    timestep: Timestep

    env_state: Any
    actor_state: NetworkState
    critic_state: NetworkState

    observation_statistics: Any = None
    reward_statistics: Any = None


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


class StreamACCore:
    """The computation graph, with nothing to interact with.

    Two networks, two credits, two rules, one TD target. Every method here is a
    function of the state and the inputs it was handed: no environment, no
    normalisation, no scan. What it cannot do alone is exactly what the flow
    above it does.
    """

    def __init__(
        self,
        cfg: StreamACConfig,
        actor_network: Any,
        critic_network: Any,
    ) -> None:
        self.cfg = cfg
        self.actor_network = actor_network
        self.critic_network = critic_network

        # Resolved once. Which of a cell's methods stands for credit, and what
        # shape a step takes, are answered here so that nothing below reads a
        # string out of the configuration.
        self.actor_credit = make_credit(cfg.credit, actor_network.core)
        self.critic_credit = make_credit(cfg.credit, critic_network.core)
        self.actor_rule = make_bounded_rule(bound=cfg.actor_bound, base=cfg.actor_base)
        self.critic_rule = make_bounded_rule(
            bound=cfg.critic_bound, base=cfg.critic_base
        )
        self.td0 = make_td0()
        self.trace_decay = cfg.gamma * cfg.trace_lambda

    def _input(self, obs, action, reward):
        """The one vector a sequence sees.

        Under ``meta_rl`` the previous action and reward are concatenated onto
        the observation.
        """

        if not self.cfg.meta_rl:
            return obs
        return jnp.concatenate([obs, action, reward], axis=-1)

    def _walk(self, network, credit, params, timestep, recurrence: Recurrence):
        """One forward pass of one network over one sequence-shaped step.

        The role is an argument rather than a method apiece: the actor and the
        critic differ in which network and which credit they hand over and in
        nothing else, and a per-role wrapper would only be that pair spelled out
        under a name. What comes back is the next recurrence and the readout.
        """

        obs, done, action, reward = timestep
        (carry, sensitivity), output = network.walk(
            params,
            self._input(obs, action, reward),
            done=done,
            carries=recurrence.carry,
            sensitivity=recurrence.sensitivity,
            credit=credit,
        )
        return Recurrence(carry=carry, sensitivity=sensitivity), output

    def _trace(self, incoming, gradient, *, reset_before):
        """StreamAC's pre-forward reset, always-fresh trace recurrence.

        One tree back rather than two: what this step steps along and what the
        next step decays are the same trace here, and naming them separately
        only said that some other algorithm's are not.
        """

        return jax.tree.map(
            lambda old, grad: (
                self.trace_decay * (1 - _broadcast_env(reset_before, old)) * old + grad
            ),
            incoming,
            gradient,
        )

    def _actor_objective(self, dist, action, delta):
        """The actor's ascent direction: log pi(a) with entropy riding on it.

        Entropy is signed by the TD error rather than added flat, so it pushes
        toward exploration only where the critic was surprised. The critic's
        direction is the value itself and is written where it is taken.
        """

        return (
            remove_time_axis(dist.log_prob(add_time_axis(action)))
            + self.cfg.entropy_coefficient
            * jnp.sign(jax.lax.stop_gradient(delta))
            * remove_time_axis(dist.entropy())
        )

    def _actor_gradient(self, params, timestep, recurrence, action, delta):
        """The actor's ascent direction differentiated, one stream at a time.

        A method rather than a closure because this is where the credit setting
        shows: everything else about an update is arithmetic on quantities, and
        this is the one place a parameter's effect on the past enters or does
        not. A test can hand it a state and compare.
        """

        def direction(differentiated, timestep, recurrence, action, delta):
            _, (dist, _) = self._walk(
                self.actor_network,
                self.actor_credit,
                differentiated,
                timestep,
                recurrence,
            )
            return self._actor_objective(dist, action, delta)

        return _per_stream(
            direction, params, timestep.to_sequence(), recurrence, action, delta
        )

    def _critic_gradient(self, params, timestep, recurrence):
        """The critic's ascent direction differentiated, one stream at a time.

        No TD error here. The critic ascends its own value and the delta reaches
        it in the rule, which is where the bound reads it.
        """

        def direction(differentiated, timestep, recurrence):
            _, (value, _) = self._walk(
                self.critic_network,
                self.critic_credit,
                differentiated,
                timestep,
                recurrence,
            )
            return remove_feature_axis(remove_time_axis(value))

        return _per_stream(direction, params, timestep.to_sequence(), recurrence)

    def reset(
        self, actor_keys, critic_keys, timestep: Timestep
    ) -> tuple[NetworkState, NetworkState]:
        """Initialise actor and critic online state for a fresh train state.

        Three keys per role -- parameters, torso, dropout -- and in that order,
        because that is how the published kernel spends its seed and a
        comparison at one seed is only a comparison if both sides start from the
        same draw. ``timestep`` is one sequence-shaped step, which is all
        initialisation reads: the shapes.
        """

        carry_shape = (self.cfg.num_envs, None)
        obs, done, action, reward = timestep

        def role(network, credit, rule, keys) -> NetworkState:
            param_key, torso_key, dropout_key = keys
            carry = network.initialize_carry(jax.random.key(0), carry_shape)
            sensitivity = credit.initialize(param_key, carry_shape)
            with credit.initialization():
                params = network.init(
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
                    # Asked for rather than assumed: a bounded rule carries a
                    # second moment shaped like the traces, an unbounded one
                    # carries whatever its base transformation built.
                    v=rule.init(params=params, traces=traces),
                ),
                recurrence=Recurrence(carry=carry, sensitivity=sensitivity),
            )

        return (
            role(self.actor_network, self.actor_credit, self.actor_rule, actor_keys),
            role(
                self.critic_network,
                self.critic_credit,
                self.critic_rule,
                critic_keys,
            ),
        )

    def restarted(
        self, key, actor_state: NetworkState, critic_state: NetworkState
    ) -> tuple[NetworkState, NetworkState]:
        """The same parameters with both recurrences begun again.

        Through the credit, not around it: a truncated credit carries no
        sensitivity at all, and asking the recurrence directly hands back a tree
        the evaluation step will not produce, which scan rejects.
        """

        carry_shape = (self.cfg.num_envs, None)

        def begun(network, credit) -> Recurrence:
            return Recurrence(
                carry=network.initialize_carry(key, carry_shape),
                sensitivity=credit.initialize(key, carry_shape),
            )

        return (
            actor_state.replace(
                recurrence=begun(self.actor_network, self.actor_credit)
            ),
            critic_state.replace(
                recurrence=begun(self.critic_network, self.critic_credit)
            ),
        )

    def sample_action(
        self,
        key: Any,
        timestep: Timestep,
        actor_state: NetworkState,
        deterministic: bool,
    ) -> tuple[Recurrence, Any]:
        """Run the actor forward once and choose, touching nothing else.

        Back come the advanced recurrence and the action -- a recurrence rather
        than a whole network state, because that is all acting can have changed,
        and returning the rest would leave a reader to check that the parameters
        came back untouched.

        The advance is handed back rather than written, because the gradient
        this step takes is taken from the carry this pass *started* on; who
        holds which of the two is the whole of why acting and updating are
        separate calls.

        ``deterministic`` is read at trace time, not stepped over: the greedy
        rollout and the learning one are two programs, and a rollout that had to
        carry a branch would also have to carry the sampler it never uses.
        """

        recurrence, (dist, _) = self._walk(
            self.actor_network,
            self.actor_credit,
            actor_state.params,
            timestep.to_sequence(),
            actor_state.recurrence,
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

    def update_parameters(
        self,
        state: StreamACState,
        next_timestep: Timestep,
        *,
        terminal: Any = None,
    ) -> StreamACState:
        """One transition's worth of learning for both networks.

        ``state`` is the state the transition *began* in -- both carries as they
        were before the acting pass -- and ``next_timestep`` is where it ended,
        holding the action that was taken, the reward as the agent sees it, and
        the ending. ``terminal`` is the ending that says the future is worth
        nothing; without one, an ending is read as a failure, which is the safe
        reading and what a single flag always meant.

        What comes back is the state with new parameters, traces and second
        moments, and the critic's recurrence advanced. The actor's is not: the
        pass that advanced it was the acting pass, and this one only
        differentiates from where that pass started.
        """

        actor = state.actor_state
        critic = state.critic_state
        reset_before = state.timestep.done
        terminal = next_timestep.done if terminal is None else terminal

        critic_recurrence, (value, _) = self._walk(
            self.critic_network,
            self.critic_credit,
            critic.params,
            state.timestep.to_sequence(),
            critic.recurrence,
        )
        value = remove_feature_axis(remove_time_axis(value))

        # The bootstrap runs the critic forward on the state the transition
        # ended in -- which is the one the environment hands back, because it
        # resets nothing -- under the parameters from before this step's update,
        # and throws away the carry it produced. The next step repeats that
        # forward pass from the same carry but with updated parameters, which is
        # the pass whose recurrent state is kept.
        _, (next_value, _) = self._walk(
            self.critic_network,
            self.critic_credit,
            jax.lax.stop_gradient(critic.params),
            next_timestep.to_sequence(),
            jax.lax.stop_gradient(critic_recurrence),
        )
        next_value = remove_feature_axis(remove_time_axis(next_value))
        td_error = self.td0(
            reward=next_timestep.reward,
            value=value,
            next_value=next_value,
            terminal=terminal,
            gamma=self.cfg.gamma,
        )

        # Both gradients start from the recurrence the acting pass started from,
        # cut out of the graph: what a parameter did to the past reaches this
        # step through the sensitivity the credit carries, and through nothing
        # else.
        actor_grads = self._actor_gradient(
            actor.params,
            state.timestep,
            jax.lax.stop_gradient(actor.recurrence),
            next_timestep.action,
            td_error,
        )
        critic_grads = self._critic_gradient(
            critic.params,
            state.timestep,
            jax.lax.stop_gradient(critic.recurrence),
        )
        actor_traces = self._trace(
            actor.rule.traces, actor_grads, reset_before=reset_before
        )
        critic_traces = self._trace(
            critic.rule.traces, critic_grads, reset_before=reset_before
        )

        current_step = state.update_step + 1
        critic_step = self.critic_rule.apply(
            critic_traces,
            None,
            critic.rule.v,
            delta=td_error,
            step=current_step,
            params=critic.params,
        )
        actor_step = self.actor_rule.apply(
            actor_traces,
            None,
            actor.rule.v,
            delta=td_error,
            step=current_step,
            params=actor.params,
        )

        def stepped(params, updates):
            return jax.tree.map(lambda param, update: param + update, params, updates)

        return state.replace(
            update_step=current_step,
            actor_state=actor.replace(
                params=stepped(actor.params, actor_step.updates),
                rule=RuleState(traces=actor_traces, v=actor_step.state),
            ),
            critic_state=critic.replace(
                params=stepped(critic.params, critic_step.updates),
                rule=RuleState(traces=critic_traces, v=critic_step.state),
                recurrence=critic_recurrence,
            ),
        )


class StreamAC:
    """One-invocation train/evaluation flow around :class:`StreamACCore`."""

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
        evaluation = evaluation or EvaluationConfig()
        self.cfg = cfg
        self.env = env
        self.env_params = env_params
        self.core = StreamACCore(cfg, actor_network, critic_network)
        # Which of the optional per-step fields to fill. A caller names what its
        # metrics need rather than switching a bundle on, so a field nobody
        # reduces is never stacked and a field somebody does is never missing.
        self.record = frozenset(record)

        # One estimator per stream, each knowing nothing about the other. The
        # flow names them because it is the thing that holds them.
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
        self._resets_on_start = evaluation.reset_on_start
        self._updates_during_eval = evaluation.update_during_eval
        if self.normalizing and environment_owns_normalization(env):
            raise ValueError(
                "normalization owner conflict: wrapper and program normalization "
                "are both enabled"
            )

    def _blank_timestep(self, obs) -> Timestep:
        """The step before the first one: an observation and nothing behind it."""

        action_space = self.env.action_space(self.env_params)
        return Timestep(
            obs=obs,
            action=jnp.zeros(
                (self.cfg.num_envs, *action_space.shape),
                dtype=action_space.dtype,
            ),
            reward=jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32),
            done=jnp.zeros((self.cfg.num_envs,), dtype=jnp.bool_),
        )

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
        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys,
            self.env_params,
        )
        observation_statistics = reward_statistics = None
        if self.observation_normalizer is not None:
            obs, observation_statistics = self.observation_normalizer.begin(obs)
        if self.reward_normalizer is not None:
            reward_statistics = self.reward_normalizer.initial(
                jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
            )
        timestep = self._blank_timestep(obs).to_sequence()
        actor_state, critic_state = self.core.reset(
            (actor_key, actor_torso_key, actor_dropout_key),
            (critic_key, critic_torso_key, critic_dropout_key),
            timestep,
        )
        return StreamACState(
            step=jnp.asarray(0, dtype=jnp.int32),
            update_step=jnp.asarray(0, dtype=jnp.int32),
            timestep=timestep.from_sequence(),
            env_state=env_state,
            actor_state=actor_state,
            critic_state=critic_state,
            observation_statistics=observation_statistics,
            reward_statistics=reward_statistics,
        )

    def _restarted(self, key, state: StreamACState, *, update=True) -> StreamACState:
        """Begin again wherever an episode ended, at the top of the act.

        The environment hands back the state its episode ended in, because that
        is what the bootstrap has to value; starting the next one is the act
        phase's business and belongs here, on the same flag the carry and the
        traces already read. The statistics take the observation the agent is
        about to act on, one stream at a time, so the streams still running are
        not counted twice.
        """

        keys = jax.random.split(key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            keys, self.env_params
        )
        statistics = state.observation_statistics
        if self.observation_normalizer is not None:
            obs, statistics = self.observation_normalizer.begin(
                obs, statistics, update=update
            )
        # The reward estimator is not begun: what it sees is an accumulation,
        # and dropping that at an ending is ``reset_on_done``'s business.
        done = state.timestep.done
        return state.replace(
            timestep=state.timestep.replace(
                obs=_where_done(done, obs, state.timestep.obs)
            ),
            env_state=_where_done(done, env_state, state.env_state),
            observation_statistics=_where_done(
                done, statistics, state.observation_statistics
            ),
        )

    def _normalized(self, state: StreamACState, obs, reward, done, *, update=True):
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
        return obs, reward, observation_statistics, reward_statistics

    def _persisted(self, timestep: Timestep) -> Timestep:
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
            action_key, state.timestep, state.actor_state, deterministic=False
        )
        step_keys = jax.random.split(env_key, self.cfg.num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        # What the environment paid, kept before normalisation overwrites it.
        # Episode returns and the score are read off that, and those are
        # statements about the task, not about the scale the agent happens to be
        # learning on.
        environment_reward = jnp.asarray(next_reward, dtype=jnp.float32)
        next_terminal = terminal_of(info, next_done)
        (
            next_obs,
            next_reward,
            observation_statistics,
            reward_statistics,
        ) = self._normalized(state, next_obs, next_reward, next_done)

        # What the update reads: where the transition ended, under the action
        # that was taken rather than the one an ending would zero out.
        next_timestep = Timestep(
            obs=next_obs,
            action=action,
            reward=jnp.asarray(next_reward, dtype=jnp.float32),
            done=next_done,
        )
        updated = self.core.update_parameters(
            state, next_timestep, terminal=next_terminal
        )

        persisted = self._persisted(next_timestep)
        next_state = updated.replace(
            step=state.step + self.cfg.num_envs,
            timestep=persisted,
            env_state=env_state,
            # The acting pass is what advanced the actor's recurrence; the
            # update differentiated from where that pass started and left it
            # alone. Written here, once, where both are in hand.
            actor_state=updated.actor_state.replace(recurrence=recurrence),
            observation_statistics=observation_statistics,
            reward_statistics=reward_statistics,
        )
        return next_state, StepMetrics(
            interaction=self._interaction(
                observation=observation,
                next_observation=next_obs,
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
                done=next_done,
                terminal=next_terminal,
                info=info,
            ),
        )

    def evaluate_step(
        self, state: StreamACState, key: Any
    ) -> tuple[StreamACState, StepMetrics]:
        """The same interaction with the greedy action and no update at all."""

        restart_key, action_key, env_key = jax.random.split(key, 3)
        state = self._restarted(restart_key, state, update=self._updates_during_eval)
        observation = state.timestep.obs

        recurrence, action = self.core.sample_action(
            action_key, state.timestep, state.actor_state, deterministic=True
        )
        step_keys = jax.random.split(env_key, self.cfg.num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, state.env_state, action, self.env_params)
        environment_reward = jnp.asarray(next_reward, dtype=jnp.float32)
        (
            next_obs,
            next_reward,
            observation_statistics,
            reward_statistics,
        ) = self._normalized(
            state,
            next_obs,
            next_reward,
            next_done,
            update=self._updates_during_eval,
        )
        return state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=self._persisted(
                Timestep(
                    obs=next_obs,
                    action=action,
                    reward=jnp.asarray(next_reward, dtype=jnp.float32),
                    done=next_done,
                )
            ),
            env_state=env_state,
            # Only the actor ran, so only the actor's recurrence moved.
            actor_state=state.actor_state.replace(recurrence=recurrence),
            observation_statistics=observation_statistics,
            reward_statistics=reward_statistics,
        ), StepMetrics(
            interaction=self._interaction(
                observation=observation,
                next_observation=next_obs,
                action=action,
                reward=environment_reward,
                done=next_done,
                terminal=terminal_of(info, next_done),
                info=info,
            ),
        )

    @staticmethod
    def _num_scan_steps(num_steps: int, num_envs: int) -> int:
        """How many rounds of every stream a step budget buys."""

        return num_steps // num_envs

    def _evaluation_state(self, key: Any, state: StreamACState) -> StreamACState:
        """The trained parameters, opened on a fresh environment and recurrence."""

        keys = jax.random.split(key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            keys, self.env_params
        )
        # A rollout either opens on statistics of its own or carries the ones
        # training built; either way the observation it opens on goes through.
        fresh = self._resets_on_start
        observation_statistics = None if fresh else state.observation_statistics
        reward_statistics = None if fresh else state.reward_statistics
        if self.observation_normalizer is not None:
            obs, observation_statistics = self.observation_normalizer.begin(
                obs, observation_statistics, update=fresh or self._updates_during_eval
            )
        if self.reward_normalizer is not None and reward_statistics is None:
            reward_statistics = self.reward_normalizer.initial(
                jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
            )
        actor_state, critic_state = self.core.restarted(
            jax.random.key(0), state.actor_state, state.critic_state
        )
        return state.replace(
            timestep=self._blank_timestep(obs),
            env_state=env_state,
            actor_state=actor_state,
            critic_state=critic_state,
            observation_statistics=observation_statistics,
            reward_statistics=reward_statistics,
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
