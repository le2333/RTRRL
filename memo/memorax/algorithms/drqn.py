"""DRQN as Hausknecht and Stone published it, on a structured diagonal core.

The paper (arXiv:1507.06527) is DQN with the first fully-connected layer
replaced by a recurrent one, and its learner is the 2015 DQN learner unchanged:
uniform replay, one-step targets against a periodically copied target network,
a linear Q head, and epsilon-greedy acting. What recurrence adds is only how a
minibatch is drawn and unrolled. The published update is *bootstrapped random
updates* (``truncated``): draw a completed episode, then a point inside it far
enough from the end to hold a whole window, zero the hidden state there, and
backpropagate through the window. Drawing the episode first is what keeps a
long episode from collecting probability; requiring the window to fit is what
makes the ``t`` the learner declares the number of transitions its gradient
actually crosses.

``full_bptt`` is **not** the paper's other scheme and should not be described
as one. The paper's *bootstrapped sequential updates* run an episode as a
succession of fixed-length unrolls, carrying the hidden state from one to the
next and taking an optimizer step at each -- truncated backpropagation with a
stored state, which is nearer to what R2D2 does than to what this branch does.
``full_bptt`` here draws a completed episode and backpropagates through the
whole of it in one step, with no state carried in and no boundary for the
gradient to stop at. That is the untruncated reference this repository wants a
truncation to be compared against; it is a deliberate addition, not a
reproduction, and a write-up should call it full BPTT rather than DRQN's
sequential arm.

The two branches share everything but the window: the same loss over the same
two networks, opened on the same zero hidden state, drawn from the same replay.

This is deliberately not R2D2 with pieces switched off. R2D2's additions --
prioritised replay and its importance-sampling correction, n-step returns,
stored actor recurrence with a burn-in, a dueling head, and the invertible value
transform -- are not declared here at all, so no manifest can turn one on and no
tuner can spend a trial discovering that it should not. The two learners share
this repository's replay storage, target-network update and window arithmetic,
and nothing else.

The recurrent core is a choice between two things a run can be answerable to.
``lru`` and ``rtu`` are the *matched* cores: the ones the online arm carries
exact recurrent sensitivity through, entered directly by the observation and
normalised after, with no projection in front. Selecting one makes the
comparison a comparison of learners -- replay Q-learning through
backpropagation against online actor-critic with exact recurrent sensitivity --
rather than of representations.

``lstm`` is the paper's own cell, and selecting it makes a run answerable to the
paper's network instead. It is a single Flax LSTM layer read directly by the
linear Q head, with nothing before it and no normalisation behind it, which is
what Hausknecht and Stone put after the convolutions. Nothing carries exact
recurrent sensitivity through a dense-gated cell, so no online arm offers it:
the online arm's core family does not list it, and a matched pair of runs cannot
be pinned to it. A learner that differentiates by backpropagation may, and both
of the ones here do -- R2D2 declares the same cell under ``backbone.kind: lstm``,
reached through its own encoder and read by its own head. What the boundary
refuses is a matched pair, not a second learner.

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
from memorax.parameters import (
    describe_parameters,
    group,
    numeric,
    param,
    structure,
)
from memorax.readings import reading, taken
from memorax.rl import (
    EnvironmentStreams,
    make_td0,
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

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), static=True)
    feature_dim: int = param(valid=(1, 4096), search=(16, 256), static=True)


@dataclass(frozen=True)
class RtuCore:
    """The RTU's width. It has no readout to size: its output is its carries."""

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), static=True)


@dataclass(frozen=True)
class LstmCore:
    """The published cell's width, which is all an LSTM has to be told.

    Hausknecht and Stone replace DQN's first fully-connected layer with a single
    LSTM layer that the linear Q head reads directly, so there is no readout
    behind it and no ``feature_dim`` to declare -- the same absence the RTU has,
    for a different reason. The bounds are the other cores' so that a hidden
    size can be pinned to the same number on either side of an architecture
    comparison.
    """

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), static=True)


@dataclass(frozen=True)
class TruncatedParameters:
    """The paper's truncation, which is the whole of what TBPTT(t) declares.

    No burn-in and no stored recurrence: the hidden state at a sampled start is
    zero by construction, which is the published rule. ``length`` is therefore
    the entire description of a random update -- how many transitions it holds,
    and how far back its gradient reaches, which here are the same number.
    """

    length: int = param(valid=(1, 4096), search=(1, 64), static=True)


@dataclass(frozen=True)
class ReplayParameters:
    """Uniform replay: how much is kept, when it may be read, how much at once.

    There is no priority exponent and no importance-sampling exponent because
    there is no priority: every completed episode is equally likely to be drawn
    and every window inside one equally likely to be the one, which is what the
    published algorithm draws. Where the window may begin is decided by the
    episode it was drawn from, not by a parameter.
    """

    capacity: int = param(
        valid=(1, 10_000_000), search=(1024, 1_000_000), log=True, static=True
    )
    minimum_size: int = param(
        valid=(1, 10_000_000), search=(32, 100_000), log=True, static=True
    )
    batch_size: int = param(valid=(1, 4096), search=(4, 256), log=True, static=True)


@dataclass(frozen=True)
class TargetParameters:
    # In learner updates, not environment transitions.
    update_period: int = param(valid=(1, 1_000_000), search=(4, 10_000), log=True)


@dataclass(frozen=True)
class ExplorationParameters:
    """Epsilon-greedy on the published schedule, which is not a step schedule.

    The published agent anneals over solver iterations and reads the value once
    per episode, so exploration holds still inside an episode and stays at
    ``epsilon_start`` for the whole replay warmup, when no update has happened
    yet. That is a different exploration profile from annealing on environment
    steps, not a rescaling of the same one: an agent that has not learned
    anything yet acts uniformly at random rather than at a rate that has
    already begun to decay.
    """

    epsilon_start: float = param(valid=(0.0, 1.0), search=(0.05, 1.0))
    epsilon_end: float = param(valid=(0.0, 1.0), search=(0.0, 0.2))
    # In learner updates, not environment transitions, which is what the
    # published schedule anneals over.
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


DRQN_CORE_BRANCHES = {"lru": LruCore, "rtu": RtuCore, "lstm": LstmCore}
# The cores this learner shares with the online arm, which are the ones an
# affine-free normalisation follows because that is the online arm's own
# topology. ``lstm`` is deliberately not among them: the published network
# normalises nothing between the cell and the head.
MATCHED_CORES = ("lru", "rtu")
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
    """How much of an episode one update sees, and where in it that begins.

    ``truncated`` is the published random update: ``t`` transitions from
    somewhere inside a completed episode. ``full_bptt`` is this repository's
    untruncated reference: a whole completed episode from its first transition,
    differentiated in one piece. They differ in those two things and in nothing
    else -- the loss is the same expression, over the same two networks, opened
    on the same zero hidden state.

    ``full_bptt`` is not the paper's *bootstrapped sequential updates*, which
    carry a hidden state across fixed-length unrolls and step the optimizer at
    each; nothing here carries a state into a window or stops a gradient inside
    one.
    """

    kind: str
    truncation: int

    def window(self, episode_length: int) -> int:
        """Transitions one update unrolls: ``t``, or an episode's declared limit."""

        if self.kind == "full_bptt":
            return episode_length
        return self.truncation

    def minimum_episode_length(self, episode_length: int) -> int:
        """How long an episode must be before this branch will draw from it.

        A random update draws its start from ``U{0 .. L - t}``, which exists
        only for an episode of at least ``t`` transitions. An episode shorter
        than that contributes no start rather than a short window: a window cut
        off at an ending would carry fewer than ``t`` transitions, so the
        gradient would reach back however far that episode happened to have
        left, and the learner would not be performing TBPTT(t).

        Full BPTT sets no bar, because its window is whatever the episode is.
        It begins at the first transition, runs to the last, and the declared
        limit is padding it never reads.
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
    epsilon: Any
    """The exploration rate this episode is being played at.

    Carried rather than recomputed, because the published schedule is read once
    per episode: a value that changed mid-episode would be a different policy
    from the one the episode began under.
    """


class ReplayTransition(struct.PyTreeNode):
    """One stored transition, and nothing the learner will not read.

    No actor recurrence is kept. A learner that zeroes the hidden state at the
    start of every window has no use for the one the behaviour policy held
    there, and storing it would suggest otherwise.

    ``reward`` is the clipped reward, which is what the published agent stores
    and therefore what its Q values are in units of.
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


def published_loss(td_error: Array, valid: Array) -> Array:
    """The Euclidean loss the published net minimises, with its own divisor.

    Caffe's ``EuclideanLoss`` divides the summed squared error by the first
    dimension of its target blob, and for the recurrent net that dimension is
    the unroll length -- not the minibatch size. So the published objective
    sums over the batch and averages over time, where an ordinary mean over
    both would divide by the batch size as well.

    That factor is not absorbed into the learning rate here, because the
    published chain clips the global gradient norm at ten *before* ADADELTA:
    scaling the gradient changes how often the clip binds, so a mean-over-both
    reproduction would clip at a different point in training than the published
    one, on the same data.

    Reproducing the published *scale* additionally needs the published
    ``replay.batch_size`` of 32, since the factor this leaves in is the number
    of windows in the batch. That is a manifest value rather than something
    this function can assert, and an arm choosing a different batch size is
    reproducing the objective rather than the step size.

    Positions past the end of a drawn episode contribute nothing, and neither
    do rows the sampler could not fill -- both arrive already false in
    ``valid``, which is the sampler's answer rather than a re-reading of the
    window.
    """

    return jnp.sum(jnp.where(valid, 0.5 * jnp.square(td_error), 0.0)) / valid.shape[1]


def clipped_reward(reward: Array) -> Array:
    """The reward as replay stores it: its sign, which is DQN's preprocessing.

    The published agent stores ``sign(r)``, so its Q values are in units of
    clipped reward and its gradient magnitudes do not depend on how a task
    happens to scale its payoffs. Applied on the way into replay and nowhere
    else -- what a run is scored on is the reward the environment paid, and
    clipping that would change the number being reported rather than the number
    being learned from.

    This is the published behaviour rather than a choice, so it is not a
    parameter. A task whose reward magnitudes carry information beyond their
    sign is not one this learner can be run on as published; on a task paying
    in units or in +/-1 it is the identity.
    """

    return jnp.sign(reward)


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
        """The cell on the observation, and nothing in front of it either way.

        On a matched core this is the online arm's representation: no projection
        ahead of the cell, and an affine-free normalisation behind it. On
        ``lstm`` there is nothing behind it either -- the published network runs
        its LSTM output straight into the linear head, and a normalisation
        inserted there would be this repository's addition to the paper rather
        than the paper's network.
        """

        cell = backbone(
            self.core_kind,
            features=self.observation_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
        )
        if self.core_kind in MATCHED_CORES:
            return Sequence((*cell, LayerNorm(use_scale=False, use_bias=False)))
        return Sequence(cell)

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
        loss = published_loss(td_error, sample.valid)
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
                epsilon_start=numeric(parameters["exploration.epsilon_start"]),
                epsilon_end=numeric(parameters["exploration.epsilon_end"]),
                epsilon_decay_steps=numeric(
                    parameters["exploration.epsilon_decay_steps"], int
                ),
                evaluation_epsilon=numeric(
                    parameters["exploration.evaluation_epsilon"]
                ),
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
                    grad_clip=numeric(parameters["grad_clip"]),
                ),
                gamma=numeric(parameters["gamma"]),
                target_update_period=numeric(parameters["target.update_period"], int),
            ),
            # Replay's three sizes keep their coercion: each of them declares
            # `static=True`, and a member axis cannot vary what sizes an array.
            buffer=make_uniform_episode_window_buffer(
                max_length=int(parameters["replay.capacity"]),
                min_length=int(parameters["replay.minimum_size"]),
                sample_batch_size=int(parameters["replay.batch_size"]),
                sample_sequence_length=learning.window(context.episode_length),
                add_batch_size=context.num_envs,
                max_episode_length=context.episode_length,
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

    def _epsilon(self, update_step: Array) -> Array:
        """The published anneal, in solver iterations.

        Counting learner updates rather than environment steps is what keeps
        the rate at ``epsilon_start`` through the replay warmup: no update has
        happened, so no progress has been made, so nothing has been learned to
        act on.
        """

        progress = jnp.clip(update_step / self.cfg.epsilon_decay_steps, 0.0, 1.0)
        return self.cfg.epsilon_start + progress * (
            self.cfg.epsilon_end - self.cfg.epsilon_start
        )

    def _episode_epsilon(self, state: DRQNState) -> Array:
        """Re-read at an episode boundary, held everywhere else."""

        return jnp.where(
            state.episode_start,
            self._epsilon(state.core.update_step),
            state.epsilon,
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
            epsilon=jnp.full(
                (self.cfg.num_envs,), self.cfg.epsilon_start, dtype=jnp.float32
            ),
        )

    def train_step(self, state: DRQNState, key: Key) -> tuple[DRQNState, StepMetrics]:
        reset_key, action_key, env_key, update_key = jax.random.split(key, 4)
        state = self._reset(reset_key, state)
        observation = state.timestep.obs
        epsilon = self._episode_epsilon(state)
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
            reward=clipped_reward(reward),
            next_observation=next_obs,
            done=done,
            terminal=terminal,
        )
        # The update reads replay as it stood before this transition, which is
        # what a learner that stores whole episodes does: the published loop
        # accumulates the episode locally, updates once per frame against the
        # episodes it has already remembered, and remembers the current one at
        # its ending. Drawing from replay with this transition already in it
        # would let the update on an episode's last frame draw the episode it
        # has just finished playing.
        #
        # Which is a statement about what the update may *see*, and reading it
        # off the pre-add state is only the most direct way to arrange that --
        # and the most expensive, because a state that is still going to be
        # read cannot be written in place, so the add would copy the whole ring
        # on every step of the run. `as_before` is the same view with none of
        # that: the ring as it is now, under the counters as they were.
        buffer_state = self.buffer.add(state.buffer_state, transition)
        core_state, update = self._maybe_update(
            update_key,
            state.core.replace(recurrence=recurrence),
            self.buffer.as_before(buffer_state, state.buffer_state),
        )
        next_timestep = self.environment.persisted(
            Timestep(obs=next_obs, action=action, reward=reward, done=done)
        )
        next_state = state.replace(
            step=state.step + self.cfg.num_envs,
            timestep=next_timestep,
            episode_start=done,
            env_state=env_state,
            buffer_state=buffer_state,
            core=core_state,
            epsilon=epsilon,
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
        epsilon = self._episode_epsilon(state)
        recurrence, action, _ = self.core.act(
            action_key,
            state.core,
            self._inputs(state.timestep, state.episode_start),
            epsilon=epsilon,
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
            epsilon=epsilon,
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
