from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flashbax.utils import get_tree_shape_prefix
from flax import core, struct

from memorax.buffers import (
    PrioritisedEpisodeBufferSample,
    compute_importance_weights,
)
from memorax.networks import FFN, LayerNorm, Sequence, Tanh, backbone
from memorax.networks.heads import DiscreteQNetwork
from memorax.rl import periodic_incremental_update
from memorax.utils import Timestep, Transition, utils
from memorax.utils.axes import add_feature_axis, remove_feature_axis, remove_time_axis
from memorax.utils.typing import (
    Array,
    Buffer,
    BufferState,
    Environment,
    EnvParams,
    EnvState,
    Key,
    PyTree,
)


@struct.dataclass(frozen=True)
class R2D2Config:
    num_envs: int
    gamma: float
    tau: float
    target_update_frequency: int
    train_frequency: int
    burn_in_length: int = 10
    sequence_length: int = 80
    n_step: int = 5
    priority_exponent: float = 0.9
    importance_sampling_exponent: float = 0.6


@struct.dataclass(frozen=True)
class R2D2State:
    step: int
    update_step: int
    timestep: Timestep
    carry: tuple
    env_state: EnvState
    params: core.FrozenDict[str, Any]
    target_params: core.FrozenDict[str, Any]
    optimizer_state: optax.OptState
    buffer_state: BufferState


@struct.dataclass(frozen=True)
class ReplayTransition:
    observation: Any
    previous_action: Any
    previous_reward: Any
    episode_start: Any
    action: Any
    reward: Any
    next_observation: Any
    done: Any
    terminal: Any
    actor_recurrence: Any


@struct.dataclass(frozen=True)
class RecurrentInputs:
    observation: Any
    previous_action: Any
    previous_reward: Any
    episode_start: Any


@struct.dataclass(frozen=True)
class LearnerSequence:
    inputs: RecurrentInputs
    bootstrap_inputs: RecurrentInputs
    actions: Any
    rewards: Any
    dones: Any
    terminals: Any
    valid: Any
    initial_recurrence: Any
    probabilities: Any
    indices: Any
    buffer_size: Any


def completed_episode_starts(
    experience: ReplayTransition, *, transition_count: int
) -> Array:
    ending_within_window = jnp.zeros_like(experience.done, dtype=jnp.bool_)
    for offset in range(transition_count):
        ending_within_window = ending_within_window | jnp.roll(
            experience.done, -offset, axis=1
        )
    return experience.episode_start & ending_within_window


def tbptt_starts(experience: ReplayTransition, *, burn_in_length: int) -> Array:
    ending_during_burn_in = jnp.zeros_like(experience.done, dtype=jnp.bool_)
    for offset in range(burn_in_length):
        ending_during_burn_in = ending_during_burn_in | jnp.roll(
            experience.done, -offset, axis=1
        )
    return ~ending_during_burn_in


def learner_sequence(
    sample: PrioritisedEpisodeBufferSample,
    *,
    transition_count: int,
    full_episode: bool,
) -> LearnerSequence:
    experience = jax.tree.map(
        lambda value: value[:, :transition_count],
        sample.experience,
    )

    observation = jax.tree.map(
        lambda current, successor: jnp.concatenate(
            (current, successor[:, -1:]), axis=1
        ),
        experience.observation,
        experience.next_observation,
    )
    inputs = RecurrentInputs(
        observation=observation,
        previous_action=jnp.concatenate(
            (experience.previous_action, experience.action[:, -1:]), axis=1
        ),
        previous_reward=jnp.concatenate(
            (experience.previous_reward, experience.reward[:, -1:]), axis=1
        ),
        episode_start=jnp.concatenate(
            (
                experience.episode_start,
                jnp.zeros_like(experience.episode_start[:, -1:]),
            ),
            axis=1,
        ),
    )
    bootstrap_inputs = RecurrentInputs(
        observation=experience.next_observation,
        previous_action=experience.action,
        previous_reward=experience.reward,
        episode_start=jnp.zeros_like(experience.done),
    )
    valid = jnp.cumprod(
        jnp.concatenate(
            (
                jnp.ones_like(experience.done[:, :1], dtype=jnp.int32),
                (~experience.done[:, :-1]).astype(jnp.int32),
            ),
            axis=1,
        ),
        axis=1,
    ).astype(jnp.bool_)
    initial_recurrence = (
        None
        if full_episode
        else jax.tree.map(lambda value: value[:, 0], experience.actor_recurrence)
    )
    return LearnerSequence(
        inputs=inputs,
        bootstrap_inputs=bootstrap_inputs,
        actions=experience.action,
        rewards=experience.reward,
        dones=experience.done,
        terminals=experience.terminal,
        valid=valid,
        initial_recurrence=initial_recurrence,
        probabilities=sample.probabilities,
        indices=sample.indices,
        buffer_size=sample.buffer_size,
    )


def encode_recurrent_inputs(inputs: RecurrentInputs, *, action_dim: int) -> Array:
    prefix = inputs.previous_action.shape
    observation = jnp.asarray(inputs.observation, dtype=jnp.float32).reshape(
        (*prefix, -1)
    )
    previous_action = jax.nn.one_hot(
        inputs.previous_action, action_dim, dtype=observation.dtype
    )
    previous_reward = jnp.asarray(
        inputs.previous_reward, dtype=observation.dtype
    )[..., None]
    episode_start = jnp.asarray(
        inputs.episode_start, dtype=observation.dtype
    )[..., None]
    return jnp.concatenate(
        (observation, previous_action, previous_reward, episode_start), axis=-1
    )


class DuelingQHead(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, hidden: Array) -> Array:
        value = nn.Dense(1, name="value")(hidden)
        advantage = nn.Dense(self.action_dim, name="advantage")(hidden)
        return value + advantage - advantage.mean(axis=-1, keepdims=True)


class _QGraph(nn.Module):
    action_dim: int
    feature_dim: int
    hidden_dim: int
    backbone_kind: str
    head_kind: str

    @nn.nowrap
    def sequence(self) -> Sequence:
        return Sequence(
            (
                FFN(features=self.feature_dim),
                LayerNorm(),
                Tanh(),
                *backbone(
                    self.backbone_kind,
                    features=self.feature_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.feature_dim,
                ),
            )
        )

    @nn.compact
    def __call__(
        self,
        encoded: Array,
        episode_start: Array,
        initial_recurrence: Any,
    ) -> tuple[Any, Array]:
        recurrence, hidden = self.sequence()(
            encoded,
            done=episode_start,
            initial_carry=initial_recurrence,
        )
        head = {
            "linear": DiscreteQNetwork(action_dim=self.action_dim),
            "dueling": DuelingQHead(action_dim=self.action_dim),
        }[self.head_kind]
        output = head(hidden)
        q_values = output[0] if self.head_kind == "linear" else output
        return recurrence, q_values

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Any:
        return self.sequence().initialize_carry(key, input_shape)


@dataclass(frozen=True)
class QFunction:
    action_dim: int
    feature_dim: int
    hidden_dim: int
    backbone_kind: str
    head_kind: str

    @property
    def network(self) -> _QGraph:
        return _QGraph(
            action_dim=self.action_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            backbone_kind=self.backbone_kind,
            head_kind=self.head_kind,
        )

    def init(self, key: Key, timestep: RecurrentInputs) -> tuple[Any, Any]:
        params_key, recurrence_key = jax.random.split(key)
        encoded = encode_recurrent_inputs(timestep, action_dim=self.action_dim)
        recurrence = self.network.initialize_carry(
            recurrence_key, (encoded.shape[0], self.feature_dim)
        )
        params = self.network.init(
            params_key,
            encoded,
            timestep.episode_start,
            recurrence,
        )
        return params, recurrence

    def reset(self, key: Key, recurrence: Any) -> Any:
        batch_size = jax.tree.leaves(recurrence)[0].shape[0]
        return self.network.initialize_carry(key, (batch_size, self.feature_dim))

    def apply(
        self, params: Any, timestep: RecurrentInputs, recurrence: Any
    ) -> tuple[Any, Array]:
        encoded = encode_recurrent_inputs(timestep, action_dim=self.action_dim)
        return self.network.apply(
            params, encoded, timestep.episode_start, recurrence
        )

    def _unroll_with_recurrences(
        self, params: Any, timesteps: RecurrentInputs, recurrence: Any
    ) -> tuple[Any, Array, Any]:
        time_major = jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), timesteps)

        def step(carry, timestep):
            length_one = jax.tree.map(
                lambda value: jnp.expand_dims(value, axis=1), timestep
            )
            next_carry, q_values = self.apply(params, length_one, carry)
            return next_carry, (q_values[:, 0], next_carry)

        final_recurrence, (q_values, post_recurrences) = jax.lax.scan(
            step, recurrence, time_major
        )
        q_values = jnp.swapaxes(q_values, 0, 1)
        post_recurrences = jax.tree.map(
            lambda value: jnp.swapaxes(value, 0, 1), post_recurrences
        )
        return final_recurrence, q_values, post_recurrences

    def unroll(
        self, params: Any, timesteps: RecurrentInputs, recurrence: Any
    ) -> tuple[Any, Array]:
        final_recurrence, q_values, _ = self._unroll_with_recurrences(
            params, timesteps, recurrence
        )
        return final_recurrence, q_values


@struct.dataclass(frozen=True)
class CoreState:
    update_step: Any
    recurrence: Any
    params: Any
    target_params: Any
    optimizer_state: Any


@struct.dataclass(frozen=True)
class ForwardMetrics:
    selected_q: Any
    epsilon: Any


@struct.dataclass(frozen=True)
class UpdateMetrics:
    applied: Any
    loss: Any
    td_error: Any
    q_value: Any
    gradient_norm: Any
    importance_weight: Any
    priority: Any


@struct.dataclass(frozen=True)
class _UpdateReadings:
    td_error: Any
    q_value: Any
    priority: Any
    importance_weight: Any


def _slice_time(tree: Any, start: int, stop: int | None = None) -> Any:
    return jax.tree.map(lambda value: value[:, start:stop], tree)


def _burn_in(
    q_function: Any,
    params: Any,
    target_params: Any,
    inputs: RecurrentInputs,
    initial_recurrence: Any,
    initial_target_recurrence: Any,
    *,
    burn_in_length: int,
) -> tuple[Any, Any, RecurrentInputs]:
    if burn_in_length:
        burn_in_inputs = _slice_time(inputs, 0, burn_in_length)
        warmed, _ = q_function.unroll(
            params, burn_in_inputs, initial_recurrence
        )
        target_warmed, _ = q_function.unroll(
            target_params, burn_in_inputs, initial_target_recurrence
        )
    else:
        warmed = initial_recurrence
        target_warmed = initial_target_recurrence
    warmed = jax.tree.map(jax.lax.stop_gradient, warmed)
    target_warmed = jax.tree.map(jax.lax.stop_gradient, target_warmed)
    return warmed, target_warmed, _slice_time(inputs, burn_in_length)


def _bootstrap_q_values(
    q_function: Any,
    params: Any,
    inputs: RecurrentInputs,
    post_input_recurrences: Any,
) -> Array:
    time_inputs = jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), inputs)
    time_recurrences = jax.tree.map(
        lambda value: jnp.swapaxes(value, 0, 1), post_input_recurrences
    )

    def apply_one(_, values):
        timestep, recurrence = values
        timestep = jax.tree.map(
            lambda value: jnp.expand_dims(value, axis=1), timestep
        )
        _, q_values = q_function.apply(params, timestep, recurrence)
        return None, q_values[:, 0]

    _, q_values = jax.lax.scan(
        apply_one, None, (time_inputs, time_recurrences)
    )
    return jnp.swapaxes(q_values, 0, 1)


@dataclass(frozen=True)
class Core:
    q_function: QFunction
    optimizer: optax.GradientTransformation
    gamma: float
    n_step: int
    burn_in_length: int
    unroll_length: int
    importance_sampling_exponent: float
    max_priority_weight: float
    target_update_period: int
    transform: Callable[[Array], Array]
    inverse_transform: Callable[[Array], Array]

    def init(self, key: Key, timestep: RecurrentInputs) -> CoreState:
        params, recurrence = self.q_function.init(key, timestep)
        return CoreState(
            update_step=jnp.asarray(0, dtype=jnp.int32),
            recurrence=recurrence,
            params=params,
            target_params=params,
            optimizer_state=self.optimizer.init(params),
        )

    def reset(self, key: Key, state: CoreState) -> CoreState:
        return state.replace(
            recurrence=self.q_function.reset(key, state.recurrence)
        )

    def act(
        self,
        key: Key,
        state: CoreState,
        timestep: RecurrentInputs,
        *,
        epsilon: Array,
    ) -> tuple[Any, Array, ForwardMetrics]:
        random_key, epsilon_key = jax.random.split(key)
        recurrence, q_values = self.q_function.apply(
            state.params, timestep, state.recurrence
        )
        q_values = q_values[:, 0]
        greedy_action = jnp.argmax(q_values, axis=-1)
        random_action = jax.random.randint(
            random_key,
            greedy_action.shape,
            minval=0,
            maxval=self.q_function.action_dim,
        )
        action = jnp.where(
            jax.random.uniform(epsilon_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        selected_q = jnp.take_along_axis(
            q_values, action[:, None], axis=-1
        ).squeeze(axis=-1)
        return recurrence, action, ForwardMetrics(
            selected_q=selected_q, epsilon=epsilon
        )

    def _aligned_unroll(
        self,
        params: Any,
        target_params: Any,
        sample: LearnerSequence,
        recurrence: Any,
        target_recurrence: Any,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        learning_inputs = _slice_time(
            sample.inputs,
            self.burn_in_length,
            self.burn_in_length + self.unroll_length + 1,
        )
        _, online_q, online_post = self.q_function._unroll_with_recurrences(
            params, learning_inputs, recurrence
        )
        _, target_q, target_post = self.q_function._unroll_with_recurrences(
            target_params, learning_inputs, target_recurrence
        )
        start = self.burn_in_length
        stop = start + self.unroll_length
        bootstrap_inputs = _slice_time(sample.bootstrap_inputs, start, stop)
        online_bootstrap = _bootstrap_q_values(
            self.q_function,
            params,
            bootstrap_inputs,
            _slice_time(online_post, 0, self.unroll_length),
        )
        target_bootstrap = _bootstrap_q_values(
            self.q_function,
            target_params,
            bootstrap_inputs,
            _slice_time(target_post, 0, self.unroll_length),
        )
        dones = sample.dones[:, start:stop]
        terminals = sample.terminals[:, start:stop]
        truncations = dones & ~terminals
        next_online = jnp.where(
            truncations[..., None], online_bootstrap, online_q[:, 1:]
        )
        next_target = jnp.where(
            truncations[..., None], target_bootstrap, target_q[:, 1:]
        )
        aligned_online = jnp.concatenate((online_q[:, :1], next_online), axis=1)
        aligned_target = jnp.concatenate((target_q[:, :1], next_target), axis=1)
        return (
            online_q[:, :-1],
            aligned_online,
            aligned_target,
            sample.actions[:, start:stop],
            sample.rewards[:, start:stop],
            sample.valid[:, start:stop],
        )

    def _tbptt_loss(
        self,
        params: Any,
        target_params: Any,
        sample: LearnerSequence,
        importance_weights: Array,
    ) -> tuple[Array, _UpdateReadings]:
        recurrence, target_recurrence, _ = _burn_in(
            self.q_function,
            params,
            target_params,
            sample.inputs,
            sample.initial_recurrence,
            sample.initial_recurrence,
            burn_in_length=self.burn_in_length,
        )
        (
            current_online_q,
            aligned_online,
            aligned_target,
            actions,
            rewards,
            valid,
        ) = self._aligned_unroll(
            params, target_params, sample, recurrence, target_recurrence
        )
        start = self.burn_in_length
        stop = start + self.unroll_length
        terminals = sample.terminals[:, start:stop]
        targets = double_q_n_step_targets(
            rewards,
            terminals,
            aligned_online,
            aligned_target,
            valid,
            gamma=self.gamma,
            n_step=self.n_step,
            transform=self.transform,
            inverse_transform=self.inverse_transform,
        )
        q_value = jnp.take_along_axis(
            current_online_q, actions[..., None], axis=-1
        ).squeeze(axis=-1)
        td_error = q_value - targets
        loss = masked_sequence_loss(td_error, valid, importance_weights)
        priority = sequence_priorities(
            td_error, valid, max_weight=self.max_priority_weight
        )
        return loss, _UpdateReadings(
            td_error=td_error,
            q_value=q_value,
            priority=priority,
            importance_weight=importance_weights,
        )

    def _apply_optimizer(
        self, state: CoreState, grads: Any
    ) -> tuple[Any, optax.OptState]:
        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        return optax.apply_updates(state.params, updates), optimizer_state

    def _update_target(
        self, params: Any, target_params: Any, next_update_step: Array
    ) -> Any:
        return periodic_incremental_update(
            params,
            target_params,
            next_update_step,
            self.target_update_period,
            1.0,
        )

    def update_parameters(
        self,
        key: Key,
        state: CoreState,
        sample: LearnerSequence,
        *,
        step: Array,
    ) -> tuple[CoreState, UpdateMetrics, Array]:
        del key, step
        importance_weights = compute_importance_weights(
            sample.probabilities,
            sample.buffer_size,
            self.importance_sampling_exponent,
        )

        def loss_fn(params):
            return self._tbptt_loss(
                params, state.target_params, sample, importance_weights
            )

        (loss, readings), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        params, optimizer_state = self._apply_optimizer(state, grads)
        next_update_step = state.update_step + 1
        target_params = self._update_target(
            params, state.target_params, next_update_step
        )
        next_state = state.replace(
            update_step=next_update_step,
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
        )
        metrics = UpdateMetrics(
            applied=jnp.asarray(True),
            loss=loss,
            td_error=jnp.mean(jnp.abs(readings.td_error)),
            q_value=jnp.mean(readings.q_value),
            gradient_norm=optax.tree.norm(grads),
            importance_weight=jnp.mean(readings.importance_weight),
            priority=jnp.mean(readings.priority),
        )
        return next_state, metrics, readings.priority


def signed_hyperbolic(x: Array, epsilon: float = 1e-3) -> Array:
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1.0) - 1.0) + epsilon * x


def signed_parabolic(x: Array, epsilon: float = 1e-3) -> Array:
    discriminant = jnp.sqrt(1.0 + 4.0 * epsilon * (jnp.abs(x) + 1.0 + epsilon))
    magnitude = (2.0 * (jnp.abs(x) + 1.0 + epsilon) / (discriminant + 1.0)) ** 2 - 1.0
    return jnp.sign(x) * magnitude


def double_q_n_step_targets(
    rewards: Array,
    terminals: Array,
    online_q: Array,
    target_q: Array,
    valid: Array,
    *,
    gamma: float,
    n_step: int,
    transform: Callable[[Array], Array],
    inverse_transform: Callable[[Array], Array],
) -> Array:
    online_actions = jnp.argmax(online_q[:, 1:], axis=-1)
    target_next_q = jnp.take_along_axis(
        target_q[:, 1:], online_actions[..., None], axis=-1
    ).squeeze(axis=-1)
    target_next_q = inverse_transform(target_next_q)
    _, sequence_length = rewards.shape
    valid = valid.astype(jnp.bool_)
    terminals = terminals.astype(jnp.bool_)

    def target_for_start(start: Array) -> Array:
        def accumulate(
            _: int, carry: tuple[Array, Array, Array, Array, Array]
        ) -> tuple[Array, Array, Array, Array, Array]:
            total, discount, accumulating, bootstrap, bootstrap_index = carry
            index = start + _
            in_sequence = index < sequence_length
            index = jnp.minimum(index, sequence_length - 1)
            active = accumulating & in_sequence & valid[:, index]
            total = total + jnp.where(active, discount * rewards[:, index], 0.0)
            return (
                total,
                jnp.where(active, discount * gamma, discount),
                active & ~terminals[:, index],
                jnp.where(active, ~terminals[:, index], bootstrap),
                jnp.where(active, index, bootstrap_index),
            )

        total, discount, _, bootstrap, bootstrap_index = jax.lax.fori_loop(
            0,
            n_step,
            accumulate,
            (
                jnp.zeros(rewards.shape[0], dtype=rewards.dtype),
                jnp.ones(rewards.shape[0], dtype=rewards.dtype),
                jnp.ones(rewards.shape[0], dtype=jnp.bool_),
                jnp.zeros(rewards.shape[0], dtype=jnp.bool_),
                jnp.full((rewards.shape[0],), start, dtype=jnp.int32),
            ),
        )
        bootstrap_q = jnp.take_along_axis(
            target_next_q, bootstrap_index[:, None], axis=1
        ).squeeze(axis=1)
        total = total + jnp.where(
            bootstrap,
            discount * bootstrap_q,
            0.0,
        )
        return transform(total)

    targets = jax.vmap(target_for_start)(jnp.arange(sequence_length))
    return jax.lax.stop_gradient(targets.T)


def masked_sequence_loss(
    td_error: Array, valid: Array, importance_weights: Array
) -> Array:
    mask = valid.astype(td_error.dtype)
    per_sequence = 0.5 * jnp.sum(jnp.square(td_error) * mask, axis=1)
    per_sequence = per_sequence / jnp.maximum(jnp.sum(mask, axis=1), 1.0)
    return jnp.mean(per_sequence * importance_weights)


def sequence_priorities(td_error: Array, valid: Array, *, max_weight: float) -> Array:
    mask = valid.astype(jnp.bool_)
    absolute_error = jnp.abs(td_error)
    maximum = jnp.max(jnp.where(mask, absolute_error, 0.0), axis=1)
    mean = jnp.sum(jnp.where(mask, absolute_error, 0.0), axis=1) / jnp.sum(
        mask, axis=1
    )
    return max_weight * maximum + (1.0 - max_weight) * mean


def compute_n_step_returns(
    rewards: Array,
    dones: Array,
    next_q_values: Array,
    n_step: int,
    gamma: float,
) -> Array:
    batch_size, sequence_length = rewards.shape
    num_targets = sequence_length - n_step + 1

    def compute_target(start_idx: int):
        n_step_return = jnp.zeros(batch_size)
        discount = 1.0
        done = jnp.ones(batch_size)

        for i in range(n_step):
            idx = start_idx + i
            n_step_return = n_step_return + discount * rewards[:, idx] * done
            discount = discount * gamma
            done = done * (1.0 - dones[:, idx])

        bootstrap_idx = start_idx + n_step - 1
        n_step_return = (
            n_step_return + discount * next_q_values[:, bootstrap_idx] * done
        )

        return n_step_return

    targets = jax.vmap(compute_target)(jnp.arange(num_targets))
    targets = targets.T

    return targets


@dataclass
class R2D2:
    cfg: R2D2Config
    env: Environment
    env_params: EnvParams
    q_network: nn.Module
    optimizer: optax.GradientTransformation
    buffer: Buffer
    epsilon_schedule: optax.Schedule
    beta_schedule: optax.Schedule

    def __post_init__(self):
        assert self.cfg.train_frequency >= self.cfg.num_envs, (
            f"train_frequency ({self.cfg.train_frequency}) must be >= num_envs ({self.cfg.num_envs})"
        )
        assert self.cfg.train_frequency % self.cfg.num_envs == 0, (
            f"train_frequency ({self.cfg.train_frequency}) must be divisible by num_envs ({self.cfg.num_envs})"
        )
        assert self.cfg.sequence_length > self.cfg.burn_in_length, (
            f"sequence_length ({self.cfg.sequence_length}) must be > burn_in_length ({self.cfg.burn_in_length})"
        )

    def _greedy_action(
        self, key: Key, state: R2D2State
    ) -> tuple[R2D2State, Array, Array, dict]:
        torso_key = key
        obs, done, action, reward = state.timestep.to_sequence()
        (carry, (q_values, _)), intermediates = self.q_network.apply(
            state.params,
            observation=obs,
            done=done,
            action=action,
            reward=reward,
            initial_carry=state.carry,
            rngs={"torso": torso_key},
            mutable=["intermediates"],
        )
        q_values = remove_time_axis(q_values)
        action = jnp.argmax(q_values, axis=-1)
        state = state.replace(carry=carry)
        return state, action, q_values, intermediates

    def _random_action(
        self, key: Key, state: R2D2State
    ) -> tuple[R2D2State, Array, None, dict]:
        action_key = jax.random.split(key, self.cfg.num_envs)
        action = jax.vmap(self.env.action_space(self.env_params).sample)(action_key)
        return state, action, None, {}

    def _epsilon_greedy_action(
        self, key: Key, state: R2D2State
    ) -> tuple[R2D2State, Array, Array, dict]:
        random_key, greedy_key, sample_key = jax.random.split(key, 3)

        state, random_action, _, _ = self._random_action(random_key, state)
        state, greedy_action, q_values, intermediates = self._greedy_action(greedy_key, state)

        epsilon = self.epsilon_schedule(state.step)
        action = jnp.where(
            jax.random.uniform(sample_key, greedy_action.shape) < epsilon,
            random_action,
            greedy_action,
        )
        return state, action, q_values, intermediates

    def _step(self, state: R2D2State, key: Key, *, policy: Callable) -> tuple[R2D2State, Transition]:
        action_key, step_key = jax.random.split(key)

        initial_carry = state.carry

        state, action, intermediates = policy(action_key, state)
        num_envs, *_ = state.timestep.obs.shape
        step_key = jax.random.split(step_key, num_envs)
        next_obs, env_state, reward, done, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_key, state.env_state, action, self.env_params)

        intermediates = jax.tree.map(
            lambda x: jnp.mean(jnp.stack(x)),
            intermediates.get("intermediates", {}),
            is_leaf=lambda x: isinstance(x, tuple),
        )

        first = Timestep(
            obs=state.timestep.obs,
            action=state.timestep.action,
            reward=state.timestep.reward,
            done=state.timestep.done,
        )
        second = Timestep(
            obs=next_obs,
            action=action,
            reward=reward,
            done=done,
        )
        lox.log({"info": info, "intermediates": intermediates})

        transition = Transition(
            first=first,
            second=second,
            carry=initial_carry,
        )

        buffer_transition = jax.tree.map(lambda x: jnp.expand_dims(x, 1), transition)
        buffer_state = self.buffer.add(state.buffer_state, buffer_transition)

        next_reward = jnp.asarray(reward, dtype=jnp.float32)
        state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=Timestep(
                obs=next_obs,
                action=jnp.where(done, jnp.zeros_like(action), action),
                reward=jnp.where(done, jnp.zeros_like(next_reward), next_reward),
                done=done,
            ),
            env_state=env_state,
            buffer_state=buffer_state,
        )
        return state, transition

    def _update(self, key: Key, state: R2D2State):
        sample_key, torso_key, next_torso_key = jax.random.split(key, 3)
        batch = self.buffer.sample(state.buffer_state, sample_key)

        experience = batch.experience

        initial_carry = None
        initial_target_carry = None
        if experience.carry is not None:
            initial_carry = jax.tree.map(lambda x: x[:, 0], experience.carry)
            initial_target_carry = jax.tree.map(lambda x: x[:, 0], experience.carry)

        initial_carry = utils.burn_in(self.q_network, state.params, experience.first, initial_carry, self.cfg.burn_in_length)
        initial_target_carry = utils.burn_in(self.q_network, state.target_params, experience.second, initial_target_carry, self.cfg.burn_in_length)
        experience = jax.tree.map(lambda x: x[:, self.cfg.burn_in_length:], experience)

        next_obs, next_done, next_action, next_reward = experience.second
        _, (next_target_q_values, _) = self.q_network.apply(
            state.target_params,
            observation=next_obs,
            done=next_done,
            action=next_action,
            reward=next_reward,
            initial_carry=initial_target_carry,
            rngs={"torso": next_torso_key},
        )

        next_target_q_value = jnp.max(next_target_q_values, axis=-1)

        _, sequence_length = experience.second.reward.shape
        if self.cfg.n_step > 1 and sequence_length >= self.cfg.n_step:
            n_step_targets = compute_n_step_returns(
                experience.second.reward,
                experience.second.done,
                next_target_q_value,
                self.cfg.n_step,
                self.cfg.gamma,
            )
            _, num_targets = n_step_targets.shape
            experience = jax.tree.map(lambda x: x[:, :num_targets], experience)
            td_target = n_step_targets
        else:
            td_target = (
                experience.second.reward
                + self.cfg.gamma * (1 - experience.second.done) * next_target_q_value
            )

        beta = self.beta_schedule(state.step)
        add_batch_size, max_length_time_axis = get_tree_shape_prefix(
            state.buffer_state.experience, n_axes=2
        )
        buffer_capacity = add_batch_size * max_length_time_axis
        buffer_size = jnp.where(
            state.buffer_state.is_full,
            buffer_capacity,
            state.buffer_state.current_index * add_batch_size,
        )
        buffer_size = jnp.maximum(buffer_size, 1)
        importance_weights = compute_importance_weights(
            batch.probabilities, buffer_size, beta
        )
        importance_weights = importance_weights[:, None]

        first_obs, first_done, first_action, first_reward = experience.first

        def loss_fn(params: PyTree):
            carry, (q_values, aux) = self.q_network.apply(
                params,
                observation=first_obs,
                done=first_done,
                action=first_action,
                reward=first_reward,
                initial_carry=initial_carry,
                rngs={"torso": torso_key},
            )
            action = add_feature_axis(experience.second.action)
            q_value = jnp.take_along_axis(q_values, action, axis=-1)
            q_value = remove_feature_axis(q_value)
            td_error = q_value - td_target

            loss = (
                importance_weights
                * self.q_network.head.loss(
                    q_value, aux, td_target, transitions=experience
                )
            ).mean()
            return loss, (q_value, td_error, carry)

        (loss, (q_value, td_error, carry)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        lox.log({"q_network/gradient_norm": optax.global_norm(grads)})

        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        params = optax.apply_updates(state.params, updates)
        target_params = periodic_incremental_update(
            params,
            state.target_params,
            state.step,
            self.cfg.target_update_frequency,
            self.cfg.tau,
        )

        mean_td_error = jnp.abs(td_error).mean(axis=1)
        new_priorities = mean_td_error + 1e-6
        buffer_state = self.buffer.set_priorities(
            state.buffer_state, batch.indices, new_priorities
        )

        info = {
            "q_network/loss": loss,
            "q_network/q_value": q_value.mean(),
            "q_network/td_error": mean_td_error.mean(),
            "training/epsilon": self.epsilon_schedule(state.step),
        }

        state = state.replace(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            buffer_state=buffer_state,
        )

        return state, info

    def _update_step(self, state: R2D2State, key: Key) -> tuple[R2D2State, None]:
        step_key, update_key = jax.random.split(key)

        step_keys = jax.random.split(step_key, self.cfg.train_frequency // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._epsilon_greedy_action),
            state,
            step_keys,
        )

        state, info = self._update(update_key, state)

        lox.log(info)

        return state.replace(update_step=state.update_step + 1), None

    def init(self, key: Key) -> R2D2State:
        env_key, q_key, torso_key = jax.random.split(key, 3)
        env_keys = jax.random.split(env_key, self.cfg.num_envs)

        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            env_keys, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        carry = self.q_network.initialize_carry((self.cfg.num_envs, None))

        timestep = Timestep(
            obs=obs, action=action, reward=reward, done=done
        ).to_sequence()
        ts_obs, ts_done, ts_action, ts_reward = timestep
        params = self.q_network.init(
            {"params": q_key, "torso": torso_key},
            observation=ts_obs,
            done=ts_done,
            action=ts_action,
            reward=ts_reward,
            initial_carry=carry,
        )
        target_params = params
        optimizer_state = self.optimizer.init(params)

        timestep = timestep.from_sequence()
        transition = Transition(
            first=timestep,
            second=timestep,
            carry=carry,
        )
        buffer_state = self.buffer.init(jax.tree.map(lambda x: x[0], transition))

        return R2D2State(
            step=0,
            update_step=0,
            timestep=timestep,
            carry=carry,
            env_state=env_state,
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            buffer_state=buffer_state,
        )

    def warmup(self, key: Key, state: R2D2State, num_steps: int) -> R2D2State:
        step_keys = jax.random.split(key, num_steps // self.cfg.num_envs)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._random_action),
            state,
            step_keys,
        )
        return state

    def train(
        self,
        key: Key,
        state: R2D2State,
        num_steps: int,
    ) -> R2D2State:
        num_outer_steps = num_steps // self.cfg.train_frequency
        keys = jax.random.split(key, num_outer_steps)
        state, _ = jax.lax.scan(
            self._update_step,
            state,
            keys,
        )

        return state

    def evaluate(self, key: Key, state: R2D2State, num_steps: int) -> R2D2State:
        reset_key, eval_key = jax.random.split(key)
        reset_key = jax.random.split(reset_key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_key, self.env_params
        )
        action_space = self.env.action_space(self.env_params)
        action = jnp.zeros(
            (self.cfg.num_envs, *action_space.shape), dtype=action_space.dtype
        )
        reward = jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32)
        done = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        carry = self.q_network.initialize_carry((self.cfg.num_envs, None))

        state = state.replace(timestep=timestep, carry=carry, env_state=env_state)

        step_keys = jax.random.split(eval_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self._step, policy=self._greedy_action),
            state,
            step_keys,
        )

        return state
