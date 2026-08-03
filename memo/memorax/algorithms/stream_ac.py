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
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from memorax.rl import (
    ObjectiveDirections,
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
    # One bound and one base per role. The two roles are independent: the shared
    # triple that used to sit here was the only reason asking for different
    # bounds had to be refused, and it was never a limit of the arithmetic.
    actor_bound: Any = None
    actor_base: Any = None
    critic_bound: Any = None
    critic_base: Any = None
    entropy_coefficient: float = 0.01
    credit: str = "rtrl"
    # Concatenate the previous action and reward onto the observation.
    meta_rl: bool = False
    # Whose running statistics to normalise with. Read by ``make_normalizer``
    # when no explicit config is passed, and only meaningful when one of the two
    # normalisation switches is on.
    normalization_statistics: str = "ours"


@struct.dataclass
class TraceDirections:
    """The trace carried to the next step and the trace used now."""

    carried: Any
    update: Any


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


@struct.dataclass(frozen=True)
class NetworkState:
    """Independent online state for one recurrent actor or critic network."""

    params: Any
    traces: Any
    v: Any
    carry: Any
    sensitivity: Any


@struct.dataclass(frozen=True)
class StreamACState:
    """Everything the kernel carries from one transition to the next."""

    step: Any
    update_step: Any
    timestep: Timestep
    env_state: Any
    actor_params: Any
    actor_traces: Any
    actor_v: Any
    actor_carry: Any
    actor_sensitivity: Any
    critic_params: Any
    critic_traces: Any
    critic_v: Any
    critic_carry: Any
    critic_sensitivity: Any
    observation_statistics: Any = None
    reward_statistics: Any = None


@struct.dataclass(frozen=True)
class ForwardMetrics:
    """What the two networks answered about this step."""

    value: Any = None
    next_value: Any = None
    log_prob: Any = None
    entropy: Any = None


@struct.dataclass(frozen=True)
class UpdateMetrics:
    """What the update produced. Absent during evaluation, where none runs.

    Scalars and one step of trajectory only. The kernel runs under ``lax.scan``,
    so whole parameter and trace trees returned from here would be stacked once
    per step.
    """

    td_error: Any = None
    actor_step_size: Any = None
    critic_step_size: Any = None
    actor_grad_norm: Any = None
    critic_grad_norm: Any = None
    actor_trace_norm: Any = None
    critic_trace_norm: Any = None


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


class StreamAC:
    """An actor and a critic learning online from one transition at a time."""

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
        self.actor_network = actor_network
        self.critic_network = critic_network
        # Which of the optional per-step fields to fill. A caller names what its
        # metrics need rather than switching a bundle on, so a field nobody
        # reduces is never stacked and a field somebody does is never missing.
        self.record = frozenset(record)

        # One estimator per stream, each knowing nothing about the other. The
        # kernel names them because it is the thing that holds them.
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

    def _forward(self, network, credit, params, obs, action, reward, done, carry, s):
        return network.walk(
            params,
            self._input(obs, action, reward),
            done=done,
            carries=carry,
            sensitivity=s,
            credit=credit,
        )

    def _actor_forward(self, params, obs, action, reward, done, carry, sensitivity):
        return self._forward(
            self.actor_network,
            self.actor_credit,
            params,
            obs,
            action,
            reward,
            done,
            carry,
            sensitivity,
        )

    def _critic_forward(self, params, obs, action, reward, done, carry, sensitivity):
        return self._forward(
            self.critic_network,
            self.critic_credit,
            params,
            obs,
            action,
            reward,
            done,
            carry,
            sensitivity,
        )

    def _trace(self, incoming, gradient, *, reset_before) -> TraceDirections:
        """StreamAC's pre-forward reset, always-fresh trace recurrence."""

        carried = jax.tree.map(
            lambda old, grad: (
                self.trace_decay * (1 - _broadcast_env(reset_before, old)) * old + grad
            ),
            incoming,
            gradient,
        )
        return TraceDirections(carried=carried, update=carried)

    def _objective(self, *, log_prob, value, entropy, delta) -> ObjectiveDirections:
        """Route the actor and critic ascent directions.

        Entropy rides on the actor objective rather than arriving separately,
        signed by the TD error so it pushes toward exploration only where the
        critic was surprised.
        """

        actor = (
            log_prob
            + self.cfg.entropy_coefficient
            * jnp.sign(jax.lax.stop_gradient(delta))
            * entropy
        )
        return ObjectiveDirections(
            traced_by_domain={"actor": actor, "critic": value},
            direct_by_domain={
                "actor": jnp.zeros_like(actor),
                "critic": jnp.zeros_like(value),
            },
            metrics={"entropy": entropy},
        )

    def _initialize_network(
        self, network, credit, rule, keys, timestep
    ) -> NetworkState:
        carry_shape = (self.cfg.num_envs, None)
        carry = network.initialize_carry(jax.random.key(0), carry_shape)
        param_key, torso_key, dropout_key = keys
        sensitivity = credit.initialize(param_key, carry_shape)
        obs, done, action, reward = timestep
        with credit.initialization():
            params = network.init(
                {"params": param_key, "torso": torso_key, "dropout": dropout_key},
                self._input(obs, action, reward),
                done=done,
                initial_carry=carry,
            )
        traces = jax.tree.map(
            lambda param: jnp.zeros((self.cfg.num_envs, *param.shape)),
            params,
        )
        return NetworkState(
            params=params,
            traces=traces,
            # Asked for rather than assumed: a bounded rule carries a second
            # moment shaped like the traces, an unbounded one carries whatever
            # its base transformation built.
            v=rule.init(params=params, traces=traces),
            carry=carry,
            sensitivity=sensitivity,
        )

    def init(self, key) -> StreamACState:
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
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape),
            dtype=action_space.dtype,
        )
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32),
            done=jnp.zeros((self.cfg.num_envs,), dtype=jnp.bool_),
        ).to_sequence()
        actor = self._initialize_network(
            self.actor_network,
            self.actor_credit,
            self.actor_rule,
            (actor_key, actor_torso_key, actor_dropout_key),
            timestep,
        )
        critic = self._initialize_network(
            self.critic_network,
            self.critic_credit,
            self.critic_rule,
            (critic_key, critic_torso_key, critic_dropout_key),
            timestep,
        )
        return StreamACState(
            step=jnp.asarray(0, dtype=jnp.int32),
            update_step=jnp.asarray(0, dtype=jnp.int32),
            timestep=timestep.from_sequence(),
            env_state=env_state,
            actor_params=actor.params,
            actor_traces=actor.traces,
            actor_v=actor.v,
            actor_carry=actor.carry,
            actor_sensitivity=actor.sensitivity,
            critic_params=critic.params,
            critic_traces=critic.traces,
            critic_v=critic.v,
            critic_carry=critic.carry,
            critic_sensitivity=critic.sensitivity,
            observation_statistics=observation_statistics,
            reward_statistics=reward_statistics,
        )

    def _actor_gradient(self, params, timestep, carry, sensitivity, action, delta):
        """The actor's ascent direction differentiated, one stream at a time.

        A method rather than a closure because this is where the credit setting
        shows: everything else about an update is arithmetic on quantities, and
        this is the one place a parameter's effect on the past enters or does
        not. A test can hand it a state and compare.
        """

        obs, done, previous_action, reward = timestep.to_sequence()

        def direction(
            differentiated,
            obs,
            previous_action,
            reward,
            done,
            carry,
            sensitivity,
            action,
            delta,
        ):
            _, (dist, _) = self._actor_forward(
                differentiated,
                obs,
                previous_action,
                reward,
                done,
                carry,
                sensitivity,
            )
            directions = self._objective(
                log_prob=remove_time_axis(dist.log_prob(add_time_axis(action))),
                value=jnp.zeros_like(delta),
                entropy=remove_time_axis(dist.entropy()),
                delta=delta,
            )
            return directions.traced_by_domain["actor"]

        return _per_stream(
            direction,
            params,
            obs,
            previous_action,
            reward,
            done,
            carry,
            sensitivity,
            action,
            delta,
        )

    def _critic_gradient(self, params, timestep, carry, sensitivity, delta):
        """The critic's ascent direction differentiated, one stream at a time."""

        obs, done, previous_action, reward = timestep.to_sequence()

        def direction(
            differentiated,
            obs,
            previous_action,
            reward,
            done,
            carry,
            sensitivity,
            delta,
        ):
            _, (value_raw, _) = self._critic_forward(
                differentiated,
                obs,
                previous_action,
                reward,
                done,
                carry,
                sensitivity,
            )
            value = remove_feature_axis(remove_time_axis(value_raw))
            directions = self._objective(
                log_prob=jnp.zeros_like(value),
                value=value,
                entropy=jnp.zeros_like(value),
                delta=delta,
            )
            return directions.traced_by_domain["critic"]

        return _per_stream(
            direction,
            params,
            obs,
            previous_action,
            reward,
            done,
            carry,
            sensitivity,
            delta,
        )

    def _readings(self, network, tree):
        """One norm per position group, per stream."""

        return subtree_norms(network.split(tree), streams=True)

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

    def _restarted(self, key, state, *, update=True):
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

    def _step(self, state: Any, key):
        restart_key, action_key, env_key = jax.random.split(key, 3)
        state = self._restarted(restart_key, state)
        obs_before = state.timestep.obs
        obs, done, previous_action, reward = state.timestep.to_sequence()
        reset_before = state.timestep.done
        pre_actor_carry = jax.lax.stop_gradient(state.actor_carry)
        pre_actor_sensitivity = jax.lax.stop_gradient(state.actor_sensitivity)
        pre_critic_carry = jax.lax.stop_gradient(state.critic_carry)
        pre_critic_sensitivity = jax.lax.stop_gradient(state.critic_sensitivity)

        (actor_carry, actor_sensitivity), (dist, _) = self._actor_forward(
            state.actor_params,
            obs,
            previous_action,
            reward,
            done,
            state.actor_carry,
            state.actor_sensitivity,
        )
        sampled_action, log_prob = dist.sample_and_log_prob(seed=action_key)
        entropy = remove_time_axis(dist.entropy()).mean()
        sampled_action = remove_time_axis(sampled_action)
        log_prob = remove_time_axis(log_prob)

        (critic_carry, critic_sensitivity), (value_raw, _) = self._critic_forward(
            state.critic_params,
            obs,
            previous_action,
            reward,
            done,
            state.critic_carry,
            state.critic_sensitivity,
        )
        value = remove_feature_axis(remove_time_axis(value_raw))

        step_keys = jax.random.split(env_key, self.cfg.num_envs)
        next_obs, env_state, next_reward, next_done, info = jax.vmap(
            self.env.step,
            in_axes=(0, 0, 0, None),
        )(step_keys, state.env_state, sampled_action, self.env_params)

        environment_reward = jnp.asarray(next_reward, dtype=jnp.float32)

        next_terminal = terminal_of(info, next_done)
        observation_statistics = state.observation_statistics
        reward_statistics = state.reward_statistics
        if self.observation_normalizer is not None:
            next_obs, observation_statistics = self.observation_normalizer.observe(
                observation_statistics, next_obs, done=next_done
            )
        if self.reward_normalizer is not None:
            next_reward, reward_statistics = self.reward_normalizer.observe(
                reward_statistics, next_reward, done=next_done
            )
        next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
            obs=next_obs,
            action=sampled_action,
            reward=next_reward,
            done=next_done,
        ).to_sequence()

        # The bootstrap runs the critic forward on the state the transition
        # ended in -- which is the one the environment hands back, because it
        # resets nothing -- under the parameters from before this step's update,
        # and throws away the carry it produced. The next step repeats that
        # forward pass from the same carry but with updated parameters, which is
        # the pass whose recurrent state is kept.
        (
            _bootstrap_carry,
            _bootstrap_sensitivity,
        ), (next_value_raw, _) = self._critic_forward(
            jax.lax.stop_gradient(state.critic_params),
            next_obs_s,
            next_action_s,
            next_reward_s,
            next_done_s,
            jax.lax.stop_gradient(critic_carry),
            jax.lax.stop_gradient(critic_sensitivity),
        )
        next_value = remove_feature_axis(remove_time_axis(next_value_raw))
        td_error = self.td0(
            reward=next_reward,
            value=value,
            next_value=next_value,
            terminal=next_terminal,
            gamma=self.cfg.gamma,
        )

        actor_grads = self._actor_gradient(
            state.actor_params,
            state.timestep,
            pre_actor_carry,
            pre_actor_sensitivity,
            sampled_action,
            td_error,
        )
        critic_grads = self._critic_gradient(
            state.critic_params,
            state.timestep,
            pre_critic_carry,
            pre_critic_sensitivity,
            td_error,
        )
        actor_trace_result = self._trace(
            state.actor_traces,
            actor_grads,
            reset_before=reset_before,
        )
        critic_trace_result = self._trace(
            state.critic_traces,
            critic_grads,
            reset_before=reset_before,
        )
        actor_traces = actor_trace_result.carried
        critic_traces = critic_trace_result.carried
        current_step = state.update_step + 1
        critic_step = self.critic_rule.apply(
            critic_trace_result.update,
            None,
            state.critic_v,
            delta=td_error,
            step=current_step,
            params=state.critic_params,
        )
        actor_step = self.actor_rule.apply(
            actor_trace_result.update,
            None,
            state.actor_v,
            delta=td_error,
            step=current_step,
            params=state.actor_params,
        )
        critic_v = critic_step.state
        actor_v = actor_step.state
        critic_params = jax.tree.map(
            lambda param, update: param + update,
            state.critic_params,
            critic_step.updates,
        )
        actor_params = jax.tree.map(
            lambda param, update: param + update,
            state.actor_params,
            actor_step.updates,
        )

        broadcast_dims = tuple(
            range(state.timestep.done.ndim, state.timestep.action.ndim)
        )
        next_reward_f = jnp.asarray(next_reward, dtype=jnp.float32)
        persisted_action = jnp.where(
            jnp.expand_dims(next_done, axis=broadcast_dims),
            jnp.zeros_like(sampled_action),
            sampled_action,
        )
        persisted_reward = jnp.where(
            next_done,
            jnp.zeros_like(next_reward_f),
            next_reward_f,
        )
        action_decision = ActionDecision(
            sampled_action=sampled_action,
            logprob_action=sampled_action,
            env_action=sampled_action,
            bootstrap_feedback_action=sampled_action,
            persisted_feedback_action=persisted_action,
        )
        next_state = state.replace(
            step=state.step + self.cfg.num_envs,
            update_step=current_step,
            timestep=Timestep(
                obs=next_obs,
                action=persisted_action,
                reward=persisted_reward,
                done=next_done,
            ),
            env_state=env_state,
            actor_params=actor_params,
            actor_traces=actor_traces,
            actor_v=actor_v,
            actor_carry=actor_carry,
            actor_sensitivity=actor_sensitivity,
            critic_params=critic_params,
            critic_traces=critic_traces,
            critic_v=critic_v,
            critic_carry=critic_carry,
            critic_sensitivity=critic_sensitivity,
            observation_statistics=observation_statistics,
            reward_statistics=reward_statistics,
        )
        return next_state, StepMetrics(
            interaction=self._interaction(
                observation=obs_before,
                next_observation=next_obs,
                action=sampled_action,
                action_decision=action_decision,
                reward=environment_reward,
                done=next_done,
                terminal=next_terminal,
                info=info,
            ),
            forward=ForwardMetrics(
                value=value,
                next_value=next_value,
                log_prob=log_prob,
                entropy=entropy,
            ),
            update=UpdateMetrics(
                td_error=td_error,
                actor_step_size=actor_step.metrics["step_size"],
                critic_step_size=critic_step.metrics["step_size"],
                actor_grad_norm=self._readings(self.actor_network, actor_grads),
                critic_grad_norm=self._readings(self.critic_network, critic_grads),
                actor_trace_norm=self._readings(self.actor_network, actor_traces),
                critic_trace_norm=self._readings(self.critic_network, critic_traces),
            ),
        )

    def train(self, key, state: Any, num_steps: int):
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        return jax.lax.scan(self._step, state, keys)

    def _evaluate_step(self, current: Any, step_key):
        restart_key, action_key, env_key = jax.random.split(step_key, 3)
        del action_key
        current = self._restarted(
            restart_key, current, update=self._updates_during_eval
        )
        obs_s, done_s, action_s, reward_s = current.timestep.to_sequence()
        (actor_carry, actor_sensitivity), (dist, _) = self._actor_forward(
            current.actor_params,
            obs_s,
            action_s,
            reward_s,
            done_s,
            current.actor_carry,
            current.actor_sensitivity,
        )
        chosen = (
            jnp.argmax(dist.logits, axis=-1) if hasattr(dist, "logits") else dist.mode()
        )
        chosen = remove_time_axis(chosen)
        step_keys = jax.random.split(env_key, self.cfg.num_envs)
        next_obs, next_env_state, next_reward, next_done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, current.env_state, chosen, self.env_params)
        observation_statistics = current.observation_statistics
        reward_statistics = current.reward_statistics
        # What the environment paid, kept before normalisation overwrites
        # it. Episode returns and the score are read off the summary below,
        # and those are statements about the task, not about the scale the
        # agent happens to be learning on.
        environment_reward = jnp.asarray(next_reward, dtype=jnp.float32)
        if self.observation_normalizer is not None:
            next_obs, observation_statistics = self.observation_normalizer.observe(
                observation_statistics,
                next_obs,
                done=next_done,
                update=self._updates_during_eval,
            )
        if self.reward_normalizer is not None:
            next_reward, reward_statistics = self.reward_normalizer.observe(
                reward_statistics,
                next_reward,
                done=next_done,
                update=self._updates_during_eval,
            )
        broadcast_dims = tuple(
            range(current.timestep.done.ndim, current.timestep.action.ndim)
        )
        next_reward = jnp.asarray(next_reward, dtype=jnp.float32)
        return current.replace(
            step=current.step + self.cfg.num_envs,
            timestep=Timestep(
                obs=next_obs,
                action=jnp.where(
                    jnp.expand_dims(next_done, axis=broadcast_dims),
                    jnp.zeros_like(chosen),
                    chosen,
                ),
                reward=jnp.where(next_done, jnp.zeros_like(next_reward), next_reward),
                done=next_done,
            ),
            env_state=next_env_state,
            actor_carry=actor_carry,
            actor_sensitivity=actor_sensitivity,
            observation_statistics=observation_statistics,
            reward_statistics=reward_statistics,
        ), StepMetrics(
            interaction=self._interaction(
                observation=current.timestep.obs,
                next_observation=next_obs,
                action=chosen,
                reward=environment_reward,
                done=next_done,
                terminal=terminal_of(info, next_done),
                info=info,
            ),
            forward=ForwardMetrics(),
        )

    def evaluate(self, key, state: Any, num_steps: int):
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
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
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32),
            done=jnp.zeros((self.cfg.num_envs,), dtype=jnp.bool_),
        )
        carry_shape = (self.cfg.num_envs, None)
        eval_state = state.replace(
            timestep=timestep,
            env_state=env_state,
            actor_carry=self.actor_network.initialize_carry(
                jax.random.key(0), carry_shape
            ),
            critic_carry=self.critic_network.initialize_carry(
                jax.random.key(0), carry_shape
            ),
            # Through the credit, not around it: a truncated credit carries no
            # sensitivity at all, and asking the recurrence directly hands back a tree
            # the evaluation step will not produce, which scan rejects.
            actor_sensitivity=self.actor_credit.initialize(
                jax.random.key(0), carry_shape
            ),
            critic_sensitivity=self.critic_credit.initialize(
                jax.random.key(0), carry_shape
            ),
            observation_statistics=observation_statistics,
            reward_statistics=reward_statistics,
        )
        step_keys = jax.random.split(eval_key, num_steps)
        return jax.lax.scan(self._evaluate_step, eval_state, step_keys)
