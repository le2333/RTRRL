"""StreamAC-RTRL: an actor and a critic with separate recurrent networks.

Neither network sees the other's gradient, so there is no shared torso to
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

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from memorax.rl import (
    ObjectiveDirections,
    environment_owns_normalization,
    make_credit,
    make_normalizer,
    make_obgd_rule,
    make_td0,
    normalization_metrics,
)
from memorax.utils import Timestep
from memorax.utils.axes import (
    add_time_axis,
    remove_feature_axis,
    remove_time_axis,
)
from memorax.utils.trees import subtree_norms

from .contract import ActionDecision, EvalSummary, EvaluationConfig


@dataclass(frozen=True)
class StreamACConfig:
    """Everything the kernel reads that does not change during a run."""

    num_envs: int
    gamma: float
    trace_lambda: float
    actor_lr: float
    critic_lr: float
    actor_kappa: float = 3.0
    critic_kappa: float = 2.0
    entropy_coefficient: float = 0.01
    credit: str = "rtrl"
    bounded_rule: str = "obgd"
    beta2: float = 0.999
    eps: float = 1e-8


@struct.dataclass
class TraceDirections:
    """The trace carried to the next step and the trace used now."""

    carried: Any
    update: Any


def _broadcast_env(values, leaf):
    return values[(slice(None),) + (None,) * (leaf.ndim - 1)]


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
    normalizer_state: Any = None


@struct.dataclass(frozen=True)
class StreamACStepMetrics:
    """Fixed-shape observables from one transition.

    Scalars and one step of trajectory only. The kernel runs under
    ``lax.scan``, so whole parameter and trace trees returned from here would
    be stacked once per step.
    """

    action_decision: ActionDecision | None = None
    log_prob: Any = None
    entropy: Any = None
    value: Any = None
    next_value: Any = None
    td_error: Any = None
    actor_step_size: Any = None
    critic_step_size: Any = None
    actor_grad_norm: Any = None
    critic_grad_norm: Any = None
    actor_trace_norm: Any = None
    critic_trace_norm: Any = None
    observation: Any = None
    reward: Any = None
    done: Any = None
    info: Any = None
    raw_episode_return: Any = None
    normalization: Any = None


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
        normalization: Any = None,
        evaluation: EvaluationConfig | None = None,
        record_trajectory: bool = False,
    ) -> None:
        evaluation = evaluation or EvaluationConfig()
        self.cfg = cfg
        self.env = env
        self.env_params = env_params
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.record_trajectory = record_trajectory

        declared = make_normalizer(normalization or cfg)
        self.normalizer = make_normalizer(
            replace(
                declared.config,
                reset_on_start=evaluation.reset_on_start,
                update_during_eval=evaluation.update_during_eval,
            )
        )
        self.normalizing = (
            self.normalizer.config.normalize_observation
            or self.normalizer.config.normalize_reward
        )
        if self.normalizing and environment_owns_normalization(env):
            raise ValueError(
                "normalization owner conflict: wrapper and program normalization "
                "are both enabled"
            )

        self.actor_credit = make_credit(cfg.credit, actor_network.torso)
        self.critic_credit = make_credit(cfg.credit, critic_network.torso)
        self.actor_rule = make_obgd_rule(
            learning_rate=cfg.actor_lr,
            kappa=cfg.actor_kappa,
            beta2=cfg.beta2,
            eps=cfg.eps,
            rule=cfg.bounded_rule,
        )
        self.critic_rule = make_obgd_rule(
            learning_rate=cfg.critic_lr,
            kappa=cfg.critic_kappa,
            beta2=cfg.beta2,
            eps=cfg.eps,
            rule=cfg.bounded_rule,
        )
        self.td0 = make_td0()
        self.trace_decay = cfg.gamma * cfg.trace_lambda

    def _forward(
        self,
        network,
        credit,
        params,
        obs,
        action,
        reward,
        done,
        carry,
        sensitivity,
    ):
        parameter_tree = params.get("params", params)
        features, _ = network.feature_extractor.apply(
            {"params": parameter_tree["feature_extractor"]},
            observation=obs,
            action=action,
            reward=reward,
            done=done,
        )
        next_carry, hidden, next_sensitivity = credit(
            parameter_tree["torso"],
            features,
            done,
            carry,
            sensitivity,
        )
        output = network.head.apply(
            {"params": parameter_tree["head"]},
            hidden,
            action=action,
            reward=reward,
            done=done,
        )
        return (next_carry, next_sensitivity), output

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

    def _initialize_network(self, network, credit, key, timestep) -> NetworkState:
        carry_shape = (self.cfg.num_envs, None)
        carry = network.initialize_carry(carry_shape)
        sensitivity = credit.initialize(key, carry_shape)
        obs, done, action, reward = timestep
        with credit.initialization():
            params = network.init(
                {"params": key},
                observation=obs,
                action=action,
                reward=reward,
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
            v=jax.tree.map(jnp.zeros_like, traces),
            carry=carry,
            sensitivity=sensitivity,
        )

    def init(self, key) -> StreamACState:
        env_key, actor_key, critic_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys,
            self.env_params,
        )
        normalizer_state = None
        if self.normalizing:
            obs, normalizer_state = self.normalizer.reset(obs)
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape),
            dtype=action_space.dtype,
        )
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32),
            done=jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_),
        ).to_sequence()
        actor = self._initialize_network(
            self.actor_network, self.actor_credit, actor_key, timestep
        )
        critic = self._initialize_network(
            self.critic_network, self.critic_credit, critic_key, timestep
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
            normalizer_state=normalizer_state,
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

    def _step(self, state: Any, key):
        action_key, env_key = jax.random.split(key)
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
        normalizer_state = state.normalizer_state
        raw_episode_return = None
        if self.normalizing:
            normalized = self.normalizer.step(
                normalizer_state,
                observation=next_obs,
                reward=next_reward,
                done=next_done,
            )
            next_obs = normalized.observation
            next_reward = normalized.reward
            normalizer_state = normalized.state
            raw_episode_return = normalized.raw_episode_return
        next_obs_s, next_done_s, next_action_s, next_reward_s = Timestep(
            obs=next_obs,
            action=sampled_action,
            reward=next_reward,
            done=next_done,
        ).to_sequence()

        # The bootstrap runs the critic forward on the observation the agent
        # has not answered yet, under the parameters from before this step's
        # update, and throws away the carry it produced. The next step repeats
        # that forward pass from the same carry but with updated parameters,
        # which is the pass whose recurrent state is kept.
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
            bootstrap_discount=self.cfg.gamma * (1 - next_done),
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
            normalizer_state=normalizer_state,
        )
        return next_state, StreamACStepMetrics(
            action_decision=action_decision,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            next_value=next_value,
            td_error=td_error,
            actor_step_size=actor_step.metrics["step_size"],
            critic_step_size=critic_step.metrics["step_size"],
            actor_grad_norm=subtree_norms(actor_grads, streams=True),
            critic_grad_norm=subtree_norms(critic_grads, streams=True),
            actor_trace_norm=subtree_norms(actor_traces, streams=True),
            critic_trace_norm=subtree_norms(critic_traces, streams=True),
            observation=next_obs if self.record_trajectory else None,
            reward=next_reward_f if self.record_trajectory else None,
            done=next_done if self.record_trajectory else None,
            info=info,
            raw_episode_return=raw_episode_return,
            normalization=normalization_metrics(
                normalizer_state, self.normalizer.config.eps
            ),
        )

    def train(self, key, state: Any, num_steps: int):
        keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        return jax.lax.scan(self._step, state, keys)

    def _evaluate_step(self, current: Any, step_key):
        action_key, env_key = jax.random.split(step_key)
        del action_key
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
        next_normalizer_state = current.normalizer_state
        # What the environment paid, kept before normalisation overwrites
        # it. Episode returns and the score are read off the summary below,
        # and those are statements about the task, not about the scale the
        # agent happens to be learning on.
        environment_reward = jnp.asarray(next_reward, dtype=jnp.float32)
        if self.normalizing:
            normalized = self.normalizer.step(
                next_normalizer_state,
                observation=next_obs,
                reward=next_reward,
                done=next_done,
                update=self.normalizer.config.update_during_eval,
            )
            next_obs = normalized.observation
            next_reward = normalized.reward
            next_normalizer_state = normalized.state
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
            normalizer_state=next_normalizer_state,
        ), EvalSummary(
            info=info,
            normalization=normalization_metrics(
                next_normalizer_state, self.normalizer.config.eps
            ),
            observation=current.timestep.obs,
            next_observation=next_obs,
            action=chosen,
            reward=environment_reward,
            done=next_done,
        )

    def evaluate(self, key, state: Any, num_steps: int):
        reset_key, eval_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_keys, self.env_params
        )
        normalizer_state = state.normalizer_state
        if self.normalizing:
            if self.normalizer.config.reset_on_start:
                obs, normalizer_state = self.normalizer.reset(obs)
            else:
                obs, normalizer_state = self.normalizer.reset(
                    obs,
                    normalizer_state,
                    update=self.normalizer.config.update_during_eval,
                )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        timestep = Timestep(
            obs=obs,
            action=action,
            reward=jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32),
            done=jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_),
        )
        carry_shape = (self.cfg.num_envs, None)
        eval_state = state.replace(
            timestep=timestep,
            env_state=env_state,
            actor_carry=self.actor_network.initialize_carry(carry_shape),
            critic_carry=self.critic_network.initialize_carry(carry_shape),
            # Through the credit, not around it: a truncated credit carries no
            # sensitivity at all, and asking the torso directly hands back a tree
            # the evaluation step will not produce, which scan rejects.
            actor_sensitivity=self.actor_credit.initialize(
                jax.random.key(0), carry_shape
            ),
            critic_sensitivity=self.critic_credit.initialize(
                jax.random.key(0), carry_shape
            ),
            normalizer_state=normalizer_state,
        )
        step_keys = jax.random.split(eval_key, num_steps)
        return jax.lax.scan(self._evaluate_step, eval_state, step_keys)
