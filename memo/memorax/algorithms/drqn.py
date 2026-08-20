"""DRQN as Hausknecht and Stone published it, on a structured diagonal core.

The paper (arXiv:1507.06527) is DQN with the first fully-connected layer
replaced by a recurrent one, and its learner is the 2015 DQN learner unchanged:
uniform replay, one-step targets against a periodically copied target network,
a linear Q head, and epsilon-greedy acting. What recurrence adds is only how a
minibatch is drawn and unrolled -- *bootstrapped random updates*, which pick a
completed episode, then a point inside it far enough from the end to hold a
whole window, zero the hidden state there, and backpropagate through the
window. Drawing the episode first is what keeps a long episode from collecting
probability, and requiring the window to fit is what makes the truncation the
learner declares the truncation its gradient actually crosses.

This is deliberately not R2D2 with pieces switched off. R2D2's additions --
prioritised replay and its importance-sampling correction, n-step returns,
stored actor recurrence with a burn-in, a dueling head, and the invertible value
transform -- are not declared here at all, so no manifest can turn one on and no
tuner can spend a trial discovering that it should not. The two learners share
this repository's replay storage, target-network update and window arithmetic,
and nothing else.

The recurrent core is the one the online arm carries exact recurrent
sensitivity through: the observation enters the structured diagonal cell
directly and is normalised after it, with no projection in front. That is what
makes the comparison a comparison of learners -- replay Q-learning through
backpropagation against online actor-critic with exact recurrent sensitivity --
rather than of representations.

For low-dimensional tasks there is nothing for the paper's convolutional
encoder to do: it existed to turn an 84x84 Atari frame into a feature vector,
and these observations already are one. Its replacement is therefore no
encoder, which is also the only choice that keeps the input topology matched.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct

from memorax.buffers import (
    EpisodeWindowBuffer,
    EpisodeWindowBufferState,
    make_uniform_episode_window_buffer,
)
from memorax.building import BuildContext, ComponentBuilder, ComponentFamily
from memorax.networks import LayerNorm, Sequence, backbone
from memorax.networks.heads import DiscreteQNetwork
from memorax.observability.metrics import metric_names
from memorax.parameters import describe_parameters, group, param, structure
from memorax.readings import reading, taken
from memorax.rl import (
    EnvironmentStreams,
    make_td0,
    masked_sequence_loss,
    periodic_incremental_update,
    select_ended,
)
from memorax.rl.updates import BASE_FAMILY, base_transform
from memorax.runtime import ObservationSchema
from memorax.utils import Timestep
from memorax.utils.typing import Array, Key

from .contract import ActionDecision, InteractionMetrics, StepMetrics

# Every window this learner scores begins with no memory of what came before
# it, so the key that opens one carries no information and is not drawn from
# the run's stream. Named once, because a second literal would look like a
# second decision.
ZERO_MEMORY = jax.random.key(0)

_td0 = make_td0()


class DRQNConfig(struct.PyTreeNode):
    num_envs: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    evaluation_epsilon: float


# ------------------------------------------------------------- parameter space
@dataclass(frozen=True)
class LruCore:
    """The LRU's widths, declared as the online arm declares them.

    ``feature_dim`` is the readout width, which the LRU has and the RTU does
    not. The bounds are the online arm's own, so that a matched pair of runs
    can be pinned to the same numbers without either side's search space
    excluding a value the other allows.
    """

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512))
    feature_dim: int = param(valid=(1, 4096), search=(16, 256))


@dataclass(frozen=True)
class RtuCore:
    """The RTU's width. It has no readout to size: its output is its carries."""

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512))


@dataclass(frozen=True)
class TruncatedParameters:
    """The paper's truncation, which is the whole of what TBPTT(t) declares.

    No burn-in and no stored recurrence: the hidden state at a sampled start is
    zero by construction, which is the published rule and the thing the
    truncation sweep is a sweep over.
    """

    length: int = param(valid=(1, 4096), search=(1, 64))


@dataclass(frozen=True)
class ReplayParameters:
    """Uniform replay: how much is kept, when it may be read, how much at once.

    There is no priority exponent and no importance-sampling exponent because
    there is no priority: every completed episode is equally likely to be drawn
    and every window inside one equally likely to be the one, which is what the
    published algorithm draws. Where the window may begin is decided by the
    episode it was drawn from, not by a parameter.
    """

    capacity: int = param(valid=(1, 10_000_000), search=(1024, 1_000_000), log=True)
    minimum_size: int = param(valid=(1, 10_000_000), search=(32, 100_000), log=True)
    batch_size: int = param(valid=(1, 4096), search=(4, 256), log=True)


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


@dataclass(frozen=True)
class Adadelta:
    """The solver the published DRQN is written over, in its own terms.

    ``recurrent_solver.prototxt`` names ``ADADELTA`` with ``base_lr: 0.1`` and
    ``momentum: 0.95``. The second is not a heavy ball: Caffe's ADADELTA reads
    its ``momentum`` field as the decay of the two running averages the method
    keeps, which is what everyone else calls ``rho``. It is spelled ``rho`` here
    so that the number is not mistaken for the other thing on sight, and the
    published values are the defaults.
    """

    lr: float = param(valid=(1e-9, 10.0), search=(1e-3, 1.0), log=True, default=0.1)
    rho: float = param(valid=(0.0, 1.0), search=(0.8, 0.999), default=0.95)
    eps: float = param(valid=(1e-12, 1e-2), search=[1e-8], default=1e-8, log=True)


DRQN_CORE_BRANCHES = {"lru": LruCore, "rtu": RtuCore}
LEARNING_BRANCHES = {"truncated": TruncatedParameters, "full_bptt": ()}
# Two, and the reproduction is the first. ``adadelta`` is the published solver,
# so an arm claiming to reproduce the paper selects it; ``adam`` is here because
# the matched baseline tunes its optimizer against the online arm and needs a
# step rule that HPO can move. Which one an arm chose is in its run parameters,
# which is what keeps "reproduction" and "matched" from being read off the same
# entry name.
DRQN_STEP_BRANCHES = {"adadelta": Adadelta, "adam": BASE_FAMILY.branches["adam"]}


def _select_step(selection, builder):
    del builder
    return selection.parameters


DRQN_OPTIMIZERS = ComponentFamily(branches=DRQN_STEP_BRANCHES, construct=_select_step)


def step_transform(step: Any, *, grad_clip: float) -> optax.GradientTransformation:
    """The published solver's chain: clip the whole gradient, then step.

    ``clip_gradients: 10`` in the published solver is a bound on the global L2
    norm of the update's gradient, which is what it is here. Zero is no clip
    rather than a clip at zero, so an arm that does not want one says so by
    naming the number the absence corresponds to.
    """

    chain: list[optax.GradientTransformation] = []
    if grad_clip:
        chain.append(optax.clip_by_global_norm(grad_clip))
    if isinstance(step, Adadelta):
        chain.append(optax.adadelta(learning_rate=step.lr, rho=step.rho, eps=step.eps))
    else:
        chain.append(base_transform(step))
    return optax.chain(*chain)


@dataclass(frozen=True)
class DRQNParameters:
    core: str = structure(branches=DRQN_CORE_BRANCHES)
    learning: str = structure(branches=LEARNING_BRANCHES)
    optimizer: str = structure(branches=DRQN_STEP_BRANCHES)
    replay: ReplayParameters = group(of=ReplayParameters)
    target: TargetParameters = group(of=TargetParameters)
    exploration: ExplorationParameters = group(of=ExplorationParameters)
    gamma: float = param(valid=(0.0, 1.0), search=(0.9, 0.9999))
    # The published solver's ``clip_gradients``, which is part of the same step
    # rather than a separate opinion about it. Last only because a declared
    # default has to follow the fields that have none.
    grad_clip: float = param(valid=(0.0, 1000.0), search=(0.0, 10.0), default=10.0)


PARAMETERS = describe_parameters(DRQNParameters)


@dataclass(frozen=True)
class SelectedCore:
    kind: str
    hidden_dim: int
    # ``None`` where the cell has no readout to size, which is how ``backbone``
    # already spells the same absence.
    feature_dim: int | None


@dataclass(frozen=True)
class SelectedLearning:
    """How far back the gradient reaches, and where a window may begin.

    The two branches differ in exactly those two things. Neither differs in
    what the loss computes, because the hidden state at a window's first input
    is zero either way -- full BPTT is the branch whose window is the episode.
    """

    kind: str
    truncation: int

    def window(self, episode_length: int) -> int:
        """Executed transitions per replay item, one fewer than its inputs."""

        if self.kind == "full_bptt":
            return episode_length
        return self.truncation

    def minimum_episode_length(self, episode_length: int) -> int:
        """How long an episode must be before this branch will draw from it.

        A truncated window has to fit whole, or the gradient would reach back
        however far the episode happened to have left rather than the ``t`` the
        learner declares -- and ``t`` is the quantity the sweep sweeps, so it
        may not be a function of where an episode ended. Full BPTT sets no bar:
        every completed episode is one, whatever its length, and its window is
        padded out to the declared limit and masked past the ending.
        """

        if self.kind == "full_bptt":
            return 1
        if self.truncation > episode_length:
            # Refused here rather than at the first sample, because a manifest
            # asking to reach back further than an episode can run is a
            # configuration error and not a run that draws nothing.
            raise ValueError(
                f"truncation {self.truncation} exceeds the declared episode length "
                f"{episode_length}; no window that long fits inside an episode"
            )
        return self.truncation


def _select_core(selection, builder) -> SelectedCore:
    del builder
    return SelectedCore(
        kind=selection.kind,
        hidden_dim=int(selection.parameters.hidden_dim),
        feature_dim=getattr(selection.parameters, "feature_dim", None),
    )


def _select_kind(selection, builder) -> str:
    del builder
    return selection.kind


def _select_learning(selection, builder) -> SelectedLearning:
    del builder
    if selection.kind == "full_bptt":
        return SelectedLearning(kind=selection.kind, truncation=0)
    return SelectedLearning(
        kind=selection.kind, truncation=int(selection.parameters.length)
    )


DRQN_CORES = ComponentFamily(branches=DRQN_CORE_BRANCHES, construct=_select_core)
LEARNING_FAMILY = ComponentFamily(
    branches=LEARNING_BRANCHES, construct=_select_learning
)


class DRQNState(struct.PyTreeNode):
    step: Any
    timestep: Timestep
    episode_start: Any
    env_state: Any
    buffer_state: Any
    core: CoreState


class ReplayTransition(struct.PyTreeNode):
    """One stored transition, and nothing the learner will not read.

    No actor recurrence is kept. A learner that zeroes the hidden state at the
    start of every window has no use for the one the behaviour policy held
    there, and storing it would suggest otherwise.
    """

    observation: Any
    episode_start: Any
    action: Any
    reward: Any
    next_observation: Any
    done: Any
    terminal: Any


class RecurrentInputs(struct.PyTreeNode):
    """What the network reads: the observation, and where an episode began.

    The published network reads the frame alone. The previous action and reward
    are a later addition to recurrent value learning, and feeding them here
    would change what the comparison is between.
    """

    observation: Any
    episode_start: Any


class LearnerSequence(struct.PyTreeNode):
    inputs: RecurrentInputs
    bootstrap_inputs: RecurrentInputs
    actions: Any
    rewards: Any
    dones: Any
    terminals: Any
    valid: Any
    batch_valid: Any


def learner_sequence(sample) -> LearnerSequence:
    """A drawn window as the two sequences the update reads it in.

    ``inputs`` is what the online network is asked about and
    ``bootstrap_inputs`` is what the target network values: the states the
    window's transitions arrived at, as replay stored them. Two sequences of the
    same length rather than one of one more, because the two networks are each
    begun from no memory and neither reads the other's pass.

    Neither mask is rebuilt here. Which steps lie inside the drawn episode, and
    which rows are a drawn episode at all, are answers the sampler already has
    -- it chose the episode -- and rederiving them from the stored ``done``
    flags would be asking the window to testify about a draw it did not make.
    """

    experience = sample.experience
    inputs = RecurrentInputs(
        observation=experience.observation,
        episode_start=experience.episode_start,
    )
    bootstrap_inputs = RecurrentInputs(
        observation=experience.next_observation,
        # A successor sequence has no episode start of its own: it is read as
        # one run, which is what clearing the continuation flag only on its
        # first step amounts to.
        episode_start=jnp.zeros_like(experience.done),
    )
    return LearnerSequence(
        inputs=inputs,
        bootstrap_inputs=bootstrap_inputs,
        actions=experience.action,
        rewards=experience.reward,
        dones=experience.done,
        terminals=experience.terminal,
        valid=sample.valid,
        batch_valid=sample.batch_valid,
    )


def encode_observation(inputs: RecurrentInputs) -> Array:
    """The observation as one vector per step, which is the cell's whole input."""

    prefix = inputs.episode_start.shape
    return jnp.asarray(inputs.observation, dtype=jnp.float32).reshape((*prefix, -1))


class _QGraph(nn.Module):
    action_dim: int
    observation_dim: int
    hidden_dim: int
    feature_dim: int | None
    core_kind: str

    @nn.nowrap
    def sequence(self) -> Sequence:
        """The cell on the observation, normalised after it, and nothing before.

        This is the online arm's representation: no projection in front of the
        cell, and an affine-free normalisation behind it.
        """

        return Sequence(
            (
                *backbone(
                    self.core_kind,
                    features=self.observation_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.feature_dim,
                ),
                LayerNorm(use_scale=False, use_bias=False),
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
        q_values, _ = DiscreteQNetwork(action_dim=self.action_dim)(hidden)
        return recurrence, q_values

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Any:
        return self.sequence().initialize_carry(key, input_shape)


@dataclass(frozen=True)
class QFunction:
    action_dim: int
    observation_dim: int
    hidden_dim: int
    feature_dim: int | None
    core_kind: str

    @property
    def network(self) -> _QGraph:
        return _QGraph(
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            hidden_dim=self.hidden_dim,
            feature_dim=self.feature_dim,
            core_kind=self.core_kind,
        )

    def init(self, key: Key, timestep: RecurrentInputs) -> tuple[Any, Any]:
        params_key, recurrence_key = jax.random.split(key)
        encoded = encode_observation(timestep)
        recurrence = self.network.initialize_carry(
            recurrence_key, (encoded.shape[0], self.observation_dim)
        )
        params = self.network.init(
            params_key, encoded, timestep.episode_start, recurrence
        )
        return params, recurrence

    def reset(self, key: Key, batch_size: int) -> Any:
        return self.network.initialize_carry(key, (batch_size, self.observation_dim))

    def apply(
        self, params: Any, timestep: RecurrentInputs, recurrence: Any
    ) -> tuple[Any, Array]:
        encoded = encode_observation(timestep)
        # ``apply`` also has the shape that returns mutated variables, which
        # nothing here asks for; the network answers with recurrence and value.
        return cast(
            "tuple[Any, Array]",
            self.network.apply(params, encoded, timestep.episode_start, recurrence),
        )

    def unroll(
        self, params: Any, timesteps: RecurrentInputs, recurrence: Any
    ) -> tuple[Any, Array, Any]:
        """Q values over a window, and the recurrence each input left behind.

        The post-step recurrences are what a bootstrap at a cut-off ending has
        to be taken from, so they are handed back rather than recomputed.
        """

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
        return (
            final_recurrence,
            jnp.swapaxes(q_values, 0, 1),
            jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), post_recurrences),
        )


class CoreState(struct.PyTreeNode):
    update_step: Any
    recurrence: Any
    params: Any
    target_params: Any
    optimizer_state: Any


class ForwardMetrics(struct.PyTreeNode):
    selected_q: Any
    epsilon: Any


class UpdateMetrics(struct.PyTreeNode):
    applied: Any
    loss: Any
    td_error: Any
    q_value: Any
    gradient_norm: Any


@dataclass(frozen=True)
class Reports:
    selected_q: bool = reading(at="forward.selected_q")
    epsilon: bool = reading(at="forward.epsilon")
    loss: bool = reading(at="update.loss")
    td_error: bool = reading(at="update.td_error")
    q_value: bool = reading(at="update.q_value")
    gradient_norm: bool = reading(at="update.gradient_norm")


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


class _UpdateReadings(struct.PyTreeNode):
    td_error: Any
    q_value: Any
    valid: Any


@dataclass(frozen=True)
class Core:
    q_function: QFunction
    optimizer: optax.GradientTransformation
    gamma: float
    target_update_period: int

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

    def _loss(
        self, params: Any, target_params: Any, sample: LearnerSequence
    ) -> tuple[Array, _UpdateReadings]:
        """The published update: two networks, two sequences, both from zero.

        The online network reads the window's own states and the target network
        reads the window's *successors*, each opening on no memory of what came
        before it. That is what the published update does -- its target pass is
        begun with the continuation flag cleared on the first successor and set
        thereafter -- and for a recurrent network it is not the same as reading
        the target off the online sequence shifted by one. Doing that would
        give the target at ``s_1`` a hidden state that had also consumed
        ``s_0``, so the two networks would disagree about how much history the
        state they are valuing is supposed to summarise. On a feed-forward Q
        network the distinction does not exist; on this one it is the whole
        subject.

        Reading the successors from what replay stored also means a window
        whose episode ended at its step limit is valued at the state it really
        reached, rather than at whatever the next row of the buffer holds.
        """

        batch_size = jax.tree.leaves(sample.inputs)[0].shape[0]
        opening = self.q_function.reset(ZERO_MEMORY, batch_size)

        _, online_q, _ = self.q_function.unroll(params, sample.inputs, opening)
        _, successor_q, _ = self.q_function.unroll(
            target_params, sample.bootstrap_inputs, opening
        )

        successor_value = jax.lax.stop_gradient(jnp.max(successor_q, axis=-1))
        q_value = jnp.take_along_axis(
            online_q, sample.actions[..., None], axis=-1
        ).squeeze(axis=-1)
        td_error = _td0(
            reward=sample.rewards,
            value=q_value,
            next_value=successor_value,
            terminal=sample.terminals.astype(q_value.dtype),
            gamma=self.gamma,
        )
        loss = masked_sequence_loss(
            td_error, sample.valid, batch_valid=sample.batch_valid
        )
        return loss, _UpdateReadings(
            td_error=td_error, q_value=q_value, valid=sample.valid
        )

    def _apply_optimizer(
        self, state: CoreState, grads: Any
    ) -> tuple[Any, optax.OptState]:
        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        return optax.apply_updates(state.params, updates), optimizer_state

    def _copy_target(
        self, params: Any, target_params: Any, next_update_step: Array
    ) -> Any:
        """A hard copy on the period, which is a Polyak step of size one."""

        return periodic_incremental_update(
            params, target_params, next_update_step, self.target_update_period, 1.0
        )

    def update_parameters(
        self, key: Key, state: CoreState, sample: LearnerSequence
    ) -> tuple[CoreState, UpdateMetrics]:
        del key

        def loss_fn(params):
            return self._loss(params, state.target_params, sample)

        (loss, readings), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        params, optimizer_state = self._apply_optimizer(state, grads)
        next_update_step = state.update_step + 1
        next_state = state.replace(
            update_step=next_update_step,
            params=params,
            target_params=self._copy_target(
                params, state.target_params, next_update_step
            ),
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
        )
        return next_state, metrics


@dataclass(frozen=True)
class DRQN:
    cfg: DRQNConfig
    # A gymnax-shaped environment, which is not the same as one that inherits
    # from gymnax's base class; nothing below asks for more than the shape.
    env: Any
    env_params: Any
    core: Core
    buffer: EpisodeWindowBuffer
    reports: Reports = Reports()
    record: Iterable[str] = ()
    # Derived in __post_init__ from cfg and the environment, so a caller never
    # passes it and never has to keep the two consistent.
    environment: EnvironmentStreams = field(init=False)

    observations = OBSERVATIONS

    @classmethod
    def graph(
        cls,
        parameters: dict[str, Any],
        components: ComponentBuilder,
        context: BuildContext,
        *,
        record: Iterable[str] = (),
    ) -> DRQN:
        """Declare DRQN's instances and connections using shared builders."""

        if context.num_envs != 1:
            # The published agent steps one environment and calls
            # ``UpdateRandom()`` once per transition. This loop adds every
            # stream's transition and then updates once, so with several
            # streams the ratio silently becomes one update per ``num_envs``
            # transitions -- a change to the learner's cadence, which is the
            # one thing an arm reproducing a published learner may not make
            # quietly. Refused rather than corrected, because a vectorised DRQN
            # is not what this arm is for and inventing a cadence for it would
            # be a third algorithm nobody asked for.
            raise ValueError(
                f"drqn runs one environment: num_envs is {context.num_envs}, and "
                f"one learner update per environment transition -- which is what "
                f"the published agent does -- is only true at one"
            )

        selected_core = components.build(DRQN_CORES, "core")
        learning = components.build(LEARNING_FAMILY, "learning")

        return cls(
            cfg=DRQNConfig(
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
                    observation_dim=int(context.observation_space.shape[0]),
                    hidden_dim=selected_core.hidden_dim,
                    feature_dim=selected_core.feature_dim,
                    core_kind=selected_core.kind,
                ),
                optimizer=step_transform(
                    components.build(DRQN_OPTIMIZERS, "optimizer"),
                    grad_clip=float(parameters["grad_clip"]),
                ),
                gamma=float(parameters["gamma"]),
                target_update_period=int(parameters["target.update_period"]),
            ),
            buffer=make_uniform_episode_window_buffer(
                max_length=int(parameters["replay.capacity"]),
                min_length=int(parameters["replay.minimum_size"]),
                sample_batch_size=int(parameters["replay.batch_size"]),
                sample_sequence_length=learning.window(context.episode_length),
                add_batch_size=context.num_envs,
                minimum_episode_length=learning.minimum_episode_length(
                    context.episode_length
                ),
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
        )

    def _epsilon(self, step: Array) -> Array:
        progress = jnp.clip(step / self.cfg.epsilon_decay_steps, 0.0, 1.0)
        return self.cfg.epsilon_start + progress * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    def _reset(self, key: Key, state: DRQNState) -> DRQNState:
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
        )

    def _maybe_update(
        self,
        key: Key,
        core_state: CoreState,
        buffer_state: EpisodeWindowBufferState,
    ) -> tuple[CoreState, UpdateMetrics]:
        sample_key, learner_key = jax.random.split(key)

        def update(current_core):
            sample = self.buffer.sample(buffer_state, sample_key)
            return self.core.update_parameters(
                learner_key, current_core, learner_sequence(sample)
            )

        def no_update(current_core):
            return current_core, self._no_update_metrics()

        return jax.lax.cond(
            self.buffer.can_sample(buffer_state), update, no_update, core_state
        )

    def init(self, key: Key) -> DRQNState:
        env_key, core_key = jax.random.split(key)
        obs, env_state = self.environment.init(env_key)
        timestep = self.environment.blank_timestep(obs)
        episode_start = jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_)
        core_state = self.core.init(core_key, self._inputs(timestep, episode_start))
        transition = ReplayTransition(
            observation=timestep.obs,
            episode_start=episode_start,
            action=timestep.action,
            reward=timestep.reward,
            next_observation=timestep.obs,
            done=timestep.done,
            terminal=jnp.zeros_like(timestep.done),
        )
        buffer_state = self.buffer.init(
            jax.tree.map(lambda value: value[0], transition)
        )

        return DRQNState(
            step=jnp.asarray(0, dtype=jnp.int32),
            timestep=timestep,
            episode_start=episode_start,
            env_state=env_state,
            buffer_state=buffer_state,
            core=core_state,
        )

    def train_step(self, state: DRQNState, key: Key) -> tuple[DRQNState, StepMetrics]:
        reset_key, action_key, env_key, update_key = jax.random.split(key, 4)
        state = self._reset(reset_key, state)
        observation = state.timestep.obs
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
            episode_start=state.episode_start,
            action=action,
            reward=reward,
            next_observation=next_obs,
            done=done,
            terminal=terminal,
        )
        buffer_state = self.buffer.add(state.buffer_state, transition)
        next_timestep = self.environment.persisted(
            Timestep(obs=next_obs, action=action, reward=reward, done=done)
        )
        core_state, update = self._maybe_update(
            update_key,
            state.core.replace(recurrence=recurrence),
            buffer_state,
        )
        next_state = state.replace(
            step=state.step + self.cfg.num_envs,
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

    def interact(self, key: Key, state: DRQNState) -> tuple[DRQNState, StepMetrics]:
        """One behaviour-policy transition that learns nothing and costs no budget.

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

    def open_evaluation(self, key: Key, state: DRQNState) -> DRQNState:
        """The trained parameters, opened on a fresh environment and recurrence."""

        env_key, recurrence_key = jax.random.split(key)
        obs, env_state = self.environment.init(env_key)
        return state.replace(
            timestep=self.environment.blank_timestep(obs),
            episode_start=jnp.ones((self.cfg.num_envs,), dtype=jnp.bool_),
            env_state=env_state,
            core=self.core.reset(recurrence_key, state.core),
        )

    def evaluate_step(
        self, state: DRQNState, key: Key
    ) -> tuple[DRQNState, StepMetrics]:
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
        self, key: Key, state: DRQNState, num_steps: int
    ) -> tuple[DRQNState, StepMetrics]:
        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(key, scan_steps)
        return jax.lax.scan(self.train_step, state, keys)

    def evaluate(
        self, key: Key, state: DRQNState, num_steps: int
    ) -> tuple[DRQNState, StepMetrics]:
        """Advance an opened evaluation rollout, learning nothing from it."""

        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(key, scan_steps)
        return jax.lax.scan(self.evaluate_step, state, keys)
