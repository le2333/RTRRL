from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct

from memorax.buffers import (
    PrioritisedEpisodeBufferSample,
    compute_importance_weights,
    make_prioritised_episode_buffer,
)
from memorax.building import BuildContext, ComponentBuilder, ComponentFamily
from memorax.networks import FFN, LayerNorm, Sequence, Tanh, backbone
from memorax.networks.heads import DiscreteQNetwork
from memorax.observability.metrics import metric_names
from memorax.parameters import describe_parameters, group, param, structure
from memorax.readings import reading, taken
from memorax.rl import EnvironmentStreams, periodic_incremental_update, select_ended
from memorax.rl.updates import BASE_FAMILY, base_transform
from memorax.runtime import ObservationSchema
from memorax.utils import Timestep
from memorax.utils.typing import (
    Array,
    Buffer,
    BufferState,
    Environment,
    EnvParams,
    Key,
)

from .contract import ActionDecision, InteractionMetrics, StepMetrics


@struct.dataclass(frozen=True)
class R2D2Config:
    num_envs: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    evaluation_epsilon: float


# ------------------------------------------------------------- parameter space
@dataclass(frozen=True)
class RecurrentParameters:
    """What either recurrent backbone needs to be built, and nothing else.

    R2D2 declares its own rather than reusing the shared backbone family's,
    which also declares a recurrent differentiation. That is a choice about how
    an online step takes its gradient and has nothing to say about a replay
    sequence, which is differentiated by ordinary reverse mode either way.
    """

    feature_dim: int = param(valid=(1, 4096), search=(16, 256))
    hidden_dim: int = param(valid=(1, 4096), search=(32, 512))


@dataclass(frozen=True)
class TbpttParameters:
    burn_in_length: int = param(valid=(0, 4096), search=(0, 80))
    unroll_length: int = param(valid=(1, 4096), search=(8, 160))


@dataclass(frozen=True)
class ReplayParameters:
    capacity: int = param(valid=(1, 10_000_000), search=(1024, 1_000_000), log=True)
    minimum_size: int = param(valid=(1, 10_000_000), search=(32, 100_000), log=True)
    batch_size: int = param(valid=(1, 4096), search=(4, 256), log=True)
    priority_exponent: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    importance_sampling_exponent: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    max_priority_weight: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))


@dataclass(frozen=True)
class TargetParameters:
    # In learner updates, not environment transitions.
    update_period: int = param(valid=(1, 1_000_000), search=(4, 10_000), log=True)


@dataclass(frozen=True)
class ExplorationParameters:
    epsilon_start: float = param(valid=(0.0, 1.0), search=(0.05, 1.0))
    epsilon_end: float = param(valid=(0.0, 1.0), search=(0.0, 0.2))
    epsilon_decay_steps: int = param(
        valid=(1, 1_000_000_000), search=(1000, 10_000_000), log=True
    )
    evaluation_epsilon: float = param(valid=(0.0, 1.0), search=(0.0, 0.1))


R2D2_BACKBONE_BRANCHES = {"lru": RecurrentParameters, "rtu": RecurrentParameters}
Q_HEAD_BRANCHES = {"dueling": (), "linear": ()}
LEARNING_BRANCHES = {"tbptt": TbpttParameters, "full_bptt": ()}
VALUE_TRANSFORM_BRANCHES = {"signed_hyperbolic": (), "identity": ()}
# Adam alone: the published learner is written over it, and no bounded stream
# rule applies to a replayed sequence.
R2D2_OPTIMIZERS = BASE_FAMILY.restricted("adam")


@dataclass(frozen=True)
class ReturnParameters:
    n_step: int = param(valid=(1, 1000), search=(1, 10))
    value_transform: str = structure(branches=VALUE_TRANSFORM_BRANCHES)


@dataclass(frozen=True)
class R2D2Parameters:
    backbone: str = structure(branches=R2D2_BACKBONE_BRANCHES)
    head: str = structure(branches=Q_HEAD_BRANCHES)
    learning: str = structure(branches=LEARNING_BRANCHES)
    optimizer: str = structure(branches=R2D2_OPTIMIZERS.branches)
    replay: ReplayParameters = group(of=ReplayParameters)
    target: TargetParameters = group(of=TargetParameters)
    returns: ReturnParameters = group(of=ReturnParameters)
    exploration: ExplorationParameters = group(of=ExplorationParameters)
    gamma: float = param(valid=(0.0, 1.0), search=(0.9, 0.9999))


PARAMETERS = describe_parameters(R2D2Parameters)


@dataclass(frozen=True)
class SelectedBackbone:
    kind: str
    feature_dim: int
    hidden_dim: int


@dataclass(frozen=True)
class SelectedLearning:
    """Which gradient the update constructs, and the replay window it reads.

    Full BPTT has no burn-in or unroll horizon of its own -- the episode is the
    horizon -- so both lengths are zero there and its loss never reads them.
    """

    kind: str
    burn_in_length: int
    unroll_length: int

    def window(self, episode_length: int) -> int:
        """Executed transitions per replay item, one fewer than its inputs."""

        if self.kind == "full_bptt":
            return episode_length
        return self.burn_in_length + self.unroll_length

    def start_flags(self, episode_length: int) -> Callable[[Any], Array]:
        """Which stored positions this branch is allowed to sample from."""

        if self.kind == "full_bptt":
            return partial(completed_episode_starts, transition_count=episode_length)
        return partial(tbptt_starts, burn_in_length=self.burn_in_length)


def _identity(values: Array) -> Array:
    return values


def _select_backbone(selection, builder) -> SelectedBackbone:
    del builder
    return SelectedBackbone(
        kind=selection.kind,
        feature_dim=int(selection.parameters.feature_dim),
        hidden_dim=int(selection.parameters.hidden_dim),
    )


def _select_kind(selection, builder) -> str:
    del builder
    return selection.kind


def _select_learning(selection, builder) -> SelectedLearning:
    del builder
    if selection.kind == "full_bptt":
        return SelectedLearning(kind=selection.kind, burn_in_length=0, unroll_length=0)
    return SelectedLearning(
        kind=selection.kind,
        burn_in_length=int(selection.parameters.burn_in_length),
        unroll_length=int(selection.parameters.unroll_length),
    )


def _select_value_transform(
    selection, builder
) -> tuple[Callable[[Array], Array], Callable[[Array], Array]]:
    del builder
    if selection.kind == "signed_hyperbolic":
        return signed_hyperbolic, signed_parabolic
    return _identity, _identity


R2D2_BACKBONES = ComponentFamily(
    branches=R2D2_BACKBONE_BRANCHES,
    construct=_select_backbone,
)
Q_HEAD_FAMILY = ComponentFamily(branches=Q_HEAD_BRANCHES, construct=_select_kind)
LEARNING_FAMILY = ComponentFamily(
    branches=LEARNING_BRANCHES,
    construct=_select_learning,
)
VALUE_TRANSFORM_FAMILY = ComponentFamily(
    branches=VALUE_TRANSFORM_BRANCHES,
    construct=_select_value_transform,
)


@struct.dataclass(frozen=True)
class R2D2State:
    step: Any
    timestep: Timestep
    episode_start: Any
    env_state: Any
    buffer_state: Any
    core: CoreState


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
    previous_reward = jnp.asarray(inputs.previous_reward, dtype=observation.dtype)[
        ..., None
    ]
    episode_start = jnp.asarray(inputs.episode_start, dtype=observation.dtype)[
        ..., None
    ]
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

    def reset(self, key: Key, batch_size: int) -> Any:
        return self.network.initialize_carry(key, (batch_size, self.feature_dim))

    def apply(
        self, params: Any, timestep: RecurrentInputs, recurrence: Any
    ) -> tuple[Any, Array]:
        encoded = encode_recurrent_inputs(timestep, action_dim=self.action_dim)
        return self.network.apply(params, encoded, timestep.episode_start, recurrence)

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


@dataclass(frozen=True)
class Reports:
    selected_q: bool = reading(at="forward.selected_q")
    epsilon: bool = reading(at="forward.epsilon")
    loss: bool = reading(at="update.loss")
    td_error: bool = reading(at="update.td_error")
    q_value: bool = reading(at="update.q_value")
    gradient_norm: bool = reading(at="update.gradient_norm")
    importance_weight: bool = reading(at="update.importance_weight")
    priority: bool = reading(at="update.priority")


REPORTS = Reports()
TRAINING_METRICS: tuple[str, ...] = taken(REPORTS)
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


@struct.dataclass(frozen=True)
class _UpdateReadings:
    td_error: Any
    q_value: Any
    valid: Any
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
        warmed, _ = q_function.unroll(params, burn_in_inputs, initial_recurrence)
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
        timestep = jax.tree.map(lambda value: jnp.expand_dims(value, axis=1), timestep)
        _, q_values = q_function.apply(params, timestep, recurrence)
        return None, q_values[:, 0]

    _, q_values = jax.lax.scan(apply_one, None, (time_inputs, time_recurrences))
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
    learning_kind: str = "tbptt"

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
        batch_size = jax.tree.leaves(state.recurrence)[0].shape[0]
        return state.replace(recurrence=self.q_function.reset(key, batch_size))

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
        selected_q = jnp.take_along_axis(q_values, action[:, None], axis=-1).squeeze(
            axis=-1
        )
        return (
            recurrence,
            action,
            ForwardMetrics(selected_q=selected_q, epsilon=epsilon),
        )

    def _aligned_unroll(
        self,
        params: Any,
        target_params: Any,
        sample: LearnerSequence,
        learning_inputs: RecurrentInputs,
        recurrence: Any,
        target_recurrence: Any,
        *,
        transition_start: int,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        _, online_q, online_post = self.q_function._unroll_with_recurrences(
            params, learning_inputs, recurrence
        )
        _, target_q, target_post = self.q_function._unroll_with_recurrences(
            target_params, learning_inputs, target_recurrence
        )
        transition_count = jax.tree.leaves(learning_inputs)[0].shape[1] - 1
        start = transition_start
        stop = start + transition_count
        bootstrap_inputs = _slice_time(sample.bootstrap_inputs, start, stop)
        online_bootstrap = _bootstrap_q_values(
            self.q_function,
            params,
            bootstrap_inputs,
            _slice_time(online_post, 0, transition_count),
        )
        target_bootstrap = _bootstrap_q_values(
            self.q_function,
            target_params,
            bootstrap_inputs,
            _slice_time(target_post, 0, transition_count),
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

    def _loss_from_unroll(
        self,
        params: Any,
        target_params: Any,
        sample: LearnerSequence,
        importance_weights: Array,
        learning_inputs: RecurrentInputs,
        recurrence: Any,
        target_recurrence: Any,
        *,
        transition_start: int,
    ) -> tuple[Array, _UpdateReadings]:
        (
            current_online_q,
            aligned_online,
            aligned_target,
            actions,
            rewards,
            valid,
        ) = self._aligned_unroll(
            params,
            target_params,
            sample,
            learning_inputs,
            recurrence,
            target_recurrence,
            transition_start=transition_start,
        )
        transition_count = jax.tree.leaves(learning_inputs)[0].shape[1] - 1
        start = transition_start
        stop = start + transition_count
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
            valid=valid,
            priority=priority,
            importance_weight=importance_weights,
        )

    def _tbptt_loss(
        self,
        params: Any,
        target_params: Any,
        sample: LearnerSequence,
        importance_weights: Array,
    ) -> tuple[Array, _UpdateReadings]:
        recurrence, target_recurrence, learning_inputs = _burn_in(
            self.q_function,
            params,
            target_params,
            sample.inputs,
            sample.initial_recurrence,
            sample.initial_recurrence,
            burn_in_length=self.burn_in_length,
        )
        learning_inputs = _slice_time(learning_inputs, 0, self.unroll_length + 1)
        return self._loss_from_unroll(
            params,
            target_params,
            sample,
            importance_weights,
            learning_inputs,
            recurrence,
            target_recurrence,
            transition_start=self.burn_in_length,
        )

    def _full_bptt_loss(
        self,
        params: Any,
        target_params: Any,
        sample: LearnerSequence,
        importance_weights: Array,
    ) -> tuple[Array, _UpdateReadings]:
        batch_size = jax.tree.leaves(sample.inputs)[0].shape[0]
        reset_key = jax.random.key(0)
        recurrence = self.q_function.reset(reset_key, batch_size)
        target_recurrence = self.q_function.reset(reset_key, batch_size)
        return self._loss_from_unroll(
            params,
            target_params,
            sample,
            importance_weights,
            sample.inputs,
            recurrence,
            target_recurrence,
            transition_start=0,
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

        loss_method = {
            "full_bptt": self._full_bptt_loss,
            "tbptt": self._tbptt_loss,
        }[self.learning_kind]

        def loss_fn(params):
            return loss_method(params, state.target_params, sample, importance_weights)

        (loss, readings), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
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
        metric_mask = readings.valid.astype(readings.q_value.dtype)
        metric_count = jnp.maximum(jnp.sum(metric_mask), 1.0)
        metrics = UpdateMetrics(
            applied=jnp.asarray(True),
            loss=loss,
            td_error=jnp.sum(jnp.abs(readings.td_error) * metric_mask) / metric_count,
            q_value=jnp.sum(readings.q_value * metric_mask) / metric_count,
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
    mean = jnp.sum(jnp.where(mask, absolute_error, 0.0), axis=1) / jnp.sum(mask, axis=1)
    return max_weight * maximum + (1.0 - max_weight) * mean


@dataclass(frozen=True)
class R2D2:
    cfg: R2D2Config
    env: Environment
    env_params: EnvParams
    core: Core
    buffer: Buffer
    reports: Reports = Reports()
    record: Iterable[str] = ()

    observations = OBSERVATIONS

    @classmethod
    def graph(
        cls,
        parameters: dict[str, Any],
        components: ComponentBuilder,
        context: BuildContext,
        *,
        record: Iterable[str] = (),
    ) -> R2D2:
        """Declare R2D2's instances and connections using shared builders."""

        selected_backbone = components.build(R2D2_BACKBONES, "backbone")
        learning = components.build(LEARNING_FAMILY, "learning")
        transform, inverse_transform = components.build(
            VALUE_TRANSFORM_FAMILY, "returns.value_transform"
        )

        return cls(
            cfg=R2D2Config(
                num_envs=context.num_envs,
                epsilon_start=float(parameters["exploration.epsilon_start"]),
                epsilon_end=float(parameters["exploration.epsilon_end"]),
                epsilon_decay_steps=int(parameters["exploration.epsilon_decay_steps"]),
                evaluation_epsilon=float(parameters["exploration.evaluation_epsilon"]),
            ),
            env=context.environment,
            env_params=context.environment_parameters,
            core=Core(
                q_function=QFunction(
                    # One Q value per action: a finite discrete action space is
                    # a precondition of this graph rather than a runtime check.
                    action_dim=int(context.action_space.n),
                    feature_dim=selected_backbone.feature_dim,
                    hidden_dim=selected_backbone.hidden_dim,
                    backbone_kind=selected_backbone.kind,
                    head_kind=components.build(Q_HEAD_FAMILY, "head"),
                ),
                optimizer=base_transform(
                    components.build(R2D2_OPTIMIZERS, "optimizer")
                ),
                gamma=float(parameters["gamma"]),
                n_step=int(parameters["returns.n_step"]),
                burn_in_length=learning.burn_in_length,
                unroll_length=learning.unroll_length,
                importance_sampling_exponent=float(
                    parameters["replay.importance_sampling_exponent"]
                ),
                max_priority_weight=float(parameters["replay.max_priority_weight"]),
                target_update_period=int(parameters["target.update_period"]),
                transform=transform,
                inverse_transform=inverse_transform,
                learning_kind=learning.kind,
            ),
            buffer=make_prioritised_episode_buffer(
                max_length=int(parameters["replay.capacity"]),
                min_length=int(parameters["replay.minimum_size"]),
                sample_batch_size=int(parameters["replay.batch_size"]),
                sample_sequence_length=learning.window(context.episode_length),
                get_start_flags=learning.start_flags(context.episode_length),
                add_sequences=False,
                add_batch_size=context.num_envs,
                priority_exponent=float(parameters["replay.priority_exponent"]),
            ),
            reports=REPORTS,
            record=record,
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment",
            EnvironmentStreams(self.cfg.num_envs, self.env, self.env_params),
        )
        object.__setattr__(self, "record", frozenset(self.record))

    @staticmethod
    def _inputs(timestep: Timestep, episode_start: Any) -> RecurrentInputs:
        return RecurrentInputs(
            observation=jax.tree.map(lambda value: value[:, None], timestep.obs),
            previous_action=timestep.action[:, None],
            previous_reward=timestep.reward[:, None],
            episode_start=episode_start[:, None],
        )

    @staticmethod
    def _no_update_metrics() -> UpdateMetrics:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return UpdateMetrics(
            applied=jnp.asarray(False),
            loss=zero,
            td_error=zero,
            q_value=zero,
            gradient_norm=zero,
            importance_weight=zero,
            priority=zero,
        )

    def _epsilon(self, step: Array) -> Array:
        progress = jnp.clip(step / self.cfg.epsilon_decay_steps, 0.0, 1.0)
        return self.cfg.epsilon_start + progress * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    def _reset(self, key: Key, state: R2D2State) -> R2D2State:
        obs, env_state = self.environment.reset(
            key, state.env_state, state.timestep.done
        )
        return state.replace(
            timestep=state.timestep.replace(
                obs=select_ended(state.timestep.done, obs, state.timestep.obs)
            ),
            env_state=env_state,
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
    ) -> InteractionMetrics:
        walked = "interaction.observation" in self.record
        return InteractionMetrics(
            observation=observation if walked else None,
            next_observation=next_observation if walked else None,
            action=action if walked else None,
            action_decision=ActionDecision(
                sampled_action=action,
                logprob_action=action,
                env_action=action,
            ),
            reward=reward,
            done=done,
            terminal=terminal,
            info=info,
        )

    def _forward_metrics(self, metrics: ForwardMetrics) -> ForwardMetrics:
        return ForwardMetrics(
            selected_q=metrics.selected_q if self.reports.selected_q else None,
            epsilon=metrics.epsilon if self.reports.epsilon else None,
        )

    def _update_metrics(self, metrics: UpdateMetrics) -> UpdateMetrics:
        return UpdateMetrics(
            applied=metrics.applied,
            loss=metrics.loss if self.reports.loss else None,
            td_error=metrics.td_error if self.reports.td_error else None,
            q_value=metrics.q_value if self.reports.q_value else None,
            gradient_norm=(
                metrics.gradient_norm if self.reports.gradient_norm else None
            ),
            importance_weight=(
                metrics.importance_weight if self.reports.importance_weight else None
            ),
            priority=metrics.priority if self.reports.priority else None,
        )

    def _maybe_update(
        self,
        key: Key,
        step: Array,
        core_state: CoreState,
        buffer_state: BufferState,
    ) -> tuple[CoreState, BufferState, UpdateMetrics]:
        sample_key, learner_key = jax.random.split(key)

        def update(operand):
            current_core, current_buffer = operand
            sample = self.buffer.sample(current_buffer, sample_key)
            transition_count = jax.tree.leaves(sample.experience)[0].shape[1]
            sequence = learner_sequence(
                sample,
                transition_count=transition_count,
                full_episode=self.core.learning_kind == "full_bptt",
            )
            next_core, metrics, priorities = self.core.update_parameters(
                learner_key, current_core, sequence, step=step
            )
            next_buffer = self.buffer.set_priorities(
                current_buffer, sample.indices, priorities + 1e-6
            )
            return next_core, next_buffer, metrics

        def no_update(operand):
            current_core, current_buffer = operand
            return current_core, current_buffer, self._no_update_metrics()

        return jax.lax.cond(
            self.buffer.can_sample(buffer_state),
            update,
            no_update,
            (core_state, buffer_state),
        )

    def init(self, key: Key) -> R2D2State:
        env_key, core_key = jax.random.split(key)
        obs, env_state = self.environment.init(env_key)
        timestep = self.environment.blank_timestep(obs)
        episode_start = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        core_state = self.core.init(core_key, self._inputs(timestep, episode_start))
        transition = ReplayTransition(
            observation=timestep.obs,
            previous_action=timestep.action,
            previous_reward=timestep.reward,
            episode_start=episode_start,
            action=timestep.action,
            reward=timestep.reward,
            next_observation=timestep.obs,
            done=timestep.done,
            terminal=jnp.zeros_like(timestep.done),
            actor_recurrence=core_state.recurrence,
        )
        buffer_state = self.buffer.init(
            jax.tree.map(lambda value: value[0], transition)
        )

        return R2D2State(
            step=jnp.asarray(0, dtype=jnp.int32),
            timestep=timestep,
            episode_start=episode_start,
            env_state=env_state,
            buffer_state=buffer_state,
            core=core_state,
        )

    def train_step(self, state: R2D2State, key: Key) -> tuple[R2D2State, StepMetrics]:
        reset_key, action_key, env_key, update_key = jax.random.split(key, 4)
        state = self._reset(reset_key, state)
        observation = state.timestep.obs
        actor_recurrence = state.core.recurrence
        epsilon = self._epsilon(state.step)
        recurrence, action, forward = self.core.act(
            action_key,
            state.core,
            self._inputs(state.timestep, state.episode_start),
            epsilon=epsilon,
        )
        next_obs, env_state, reward, done, terminal, info = self.environment.step(
            env_key, state.env_state, action
        )
        transition = ReplayTransition(
            observation=observation,
            previous_action=state.timestep.action,
            previous_reward=state.timestep.reward,
            episode_start=state.episode_start,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
            terminal=terminal,
            actor_recurrence=actor_recurrence,
        )
        buffer_state = self.buffer.add(state.buffer_state, transition)
        next_timestep = self.environment.persisted(
            Timestep(obs=next_obs, action=action, reward=reward, done=done)
        )
        next_step = state.step + self.cfg.num_envs
        core_state, buffer_state, update = self._maybe_update(
            update_key,
            next_step,
            state.core.replace(recurrence=recurrence),
            buffer_state,
        )
        next_state = state.replace(
            step=next_step,
            timestep=next_timestep,
            episode_start=done,
            env_state=env_state,
            buffer_state=buffer_state,
            core=core_state,
        )
        return next_state, StepMetrics(
            interaction=self._interaction(
                observation=observation,
                next_observation=next_obs,
                action=action,
                reward=reward,
                done=done,
                terminal=terminal,
                info=info,
            ),
            forward=self._forward_metrics(forward),
            update=self._update_metrics(update),
        )

    def interact(self, key: Key, state: R2D2State) -> tuple[R2D2State, StepMetrics]:
        """One behavior-policy transition that learns nothing and costs no budget.

        Runtime schedules this only to finish a sampled episode the training
        budget cut short. The epsilon-greedy policy and the actor's recurrence
        continue exactly where training left them, so neither the step counter,
        the parameters, the optimizer state nor replay moves.
        """

        reset_key, action_key, env_key = jax.random.split(key, 3)
        state = self._reset(reset_key, state)
        observation = state.timestep.obs
        recurrence, action, _ = self.core.act(
            action_key,
            state.core,
            self._inputs(state.timestep, state.episode_start),
            epsilon=self._epsilon(state.step),
        )
        next_obs, env_state, reward, done, terminal, info = self.environment.step(
            env_key, state.env_state, action
        )
        return state.replace(
            timestep=self.environment.persisted(
                Timestep(obs=next_obs, action=action, reward=reward, done=done)
            ),
            episode_start=done,
            env_state=env_state,
            core=state.core.replace(recurrence=recurrence),
        ), StepMetrics(
            interaction=self._interaction(
                observation=observation,
                next_observation=next_obs,
                action=action,
                reward=reward,
                done=done,
                terminal=terminal,
                info=info,
            ),
        )

    def _evaluation_state(self, key: Key, state: R2D2State) -> R2D2State:
        env_key, recurrence_key = jax.random.split(key)
        obs, env_state = self.environment.init(env_key)
        return state.replace(
            timestep=self.environment.blank_timestep(obs),
            episode_start=jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_),
            env_state=env_state,
            core=self.core.reset(recurrence_key, state.core),
        )

    def evaluate_step(
        self, state: R2D2State, key: Key
    ) -> tuple[R2D2State, StepMetrics]:
        reset_key, action_key, env_key = jax.random.split(key, 3)
        state = self._reset(reset_key, state)
        observation = state.timestep.obs
        recurrence, action, forward = self.core.act(
            action_key,
            state.core,
            self._inputs(state.timestep, state.episode_start),
            epsilon=jnp.asarray(self.cfg.evaluation_epsilon),
        )
        next_obs, env_state, reward, done, terminal, info = self.environment.step(
            env_key, state.env_state, action
        )
        next_timestep = self.environment.persisted(
            Timestep(obs=next_obs, action=action, reward=reward, done=done)
        )
        return state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=next_timestep,
            episode_start=done,
            env_state=env_state,
            core=state.core.replace(recurrence=recurrence),
        ), StepMetrics(
            interaction=self._interaction(
                observation=observation,
                next_observation=next_obs,
                action=action,
                reward=reward,
                done=done,
                terminal=terminal,
                info=info,
            ),
            forward=self._forward_metrics(forward),
        )

    @staticmethod
    def _num_scan_steps(num_steps: int, num_envs: int) -> int:
        return num_steps // num_envs

    def train(
        self, key: Key, state: R2D2State, num_steps: int
    ) -> tuple[R2D2State, StepMetrics]:
        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(key, scan_steps)
        return jax.lax.scan(self.train_step, state, keys)

    def evaluate(self, key: Key, state: R2D2State, num_steps: int) -> StepMetrics:
        reset_key, rollout_key = jax.random.split(key)
        eval_state = self._evaluation_state(reset_key, state)
        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(rollout_key, scan_steps)
        _, metrics = jax.lax.scan(self.evaluate_step, eval_state, keys)
        return metrics
