"""RTRRL: one recurrent torso shared by an actor and a critic.

Rebuilt from the published algorithm in ``../RTRRL-AAAI25/rtrrl.py`` rather than
from the working copy that grew out of it, because the working copy has since
picked up switches the paper does not describe and a step ordering that is
StreamAC's rather than this one's. Nothing is taken from the older
``rtrrl.py`` in this directory. What is taken from the library is the
*boundaries*: a network component, a recurrent-credit adapter, an update rule
and a TD(0) target are asked for exactly as ``StreamAC.py`` asks for them, so
either side of any of those seams can be swapped or tested against the other
implementation.

Three blocks of parameters, and each is a block because nothing else reaches
it: the torso, the actor's readout, the critic's readout. So there are three
directions to ascend. There are only two rule groups -- the torso is clipped as
a whole, the two readouts step plainly -- and that mismatch is the reason
``Core`` holds a group table rather than letting each block step itself, which
is the one thing StreamAC could not have shown, since there every granularity
coincides.

Four layers, each owning a slice of the state:

    RTRRL             the order things happen in, and the scan
    EnvTransformer    the environment and the two scales its numbers pass through
    Core              a torso, two heads, and everything that couples them
      Torso           the shared block: sequence, sensitivity, following copy
      Actor / Critic  a readout, and the directions it names

Every layer has both an ``init`` and a ``reset``. ``init`` allocates -- opened
streams, fresh estimators, parameters, traces, the sequence and its sensitivity
-- and runs once per invocation. ``reset`` begins again without allocating and
without touching what learning built: per-stream, selective, replacing only the
streams whose episode ended, and it runs on every step. ``init`` is what Flax,
optax and the run contract already call allocation; ``reset`` is what an
environment already calls beginning an episode.

Where StreamAC's ``Core`` was thin because its two roles share nothing, this one
is thick, and it is thick by exactly what sharing costs: the two heads hand up
cotangents that have to be added and pushed back through one torso, and the
step has to be taken over groups that do not line up with the heads.

Three places this algorithm is not StreamAC, all of them load-bearing:

*The step is rotated.* The published loop steps the environment with the action
it decided last time, so the pass that produces a value happens *after* the
transition it will be asked about. That is why ``value`` is carried: the TD
error at step t reads the value computed at step t-1, not a second forward pass.
StreamAC's ``bootstrap`` has no counterpart here and adding one would double the
work and change the target.

*The trace is used before it is advanced.* The update at step t is taken along
the trace as it stood at the end of step t-1 -- which is the trace that holds
the gradient of the value being corrected. Advancing first would weight the TD
error of one transition by the gradient of the next state, and the ending would
lose its credit entirely.

*There are two objectives, and only one of them is traced.* The policy's
log-probability and the value ascend through the eligibility trace and are
weighted by the TD error on arrival. Entropy does not: it applies on the step it
arises, unweighted, because there is nothing temporal about wanting a wider
policy. Both reach the torso, through the same vector-Jacobian product called
twice, which is what the published code does.

Scope. This is the ``lru-bp-rtrl`` path and only that: a linear recurrent unit
credited by exact RTRL, everything feedforward credited by backpropagation. The
published file reaches several other algorithms through flags -- other cells and
wirings, feedback alignment, observation prediction, an MLP policy, discrete
actions, dutch traces, variance scaling, the slow-state and action-magnitude
penalties, and the average-reward formulation. None of them are here, and none
of them are here as a disabled branch either: a switch nobody has turned on is a
claim nobody has checked.

The two gates -- whether the actor and the critic are each allowed to steer the
representation they share -- are also absent, because the published algorithm
has no such thing and this file has to reproduce it first. They go in at exactly
one place, marked below, and putting them there is the first thing to do once a
reproduction runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from memorax.rl import (
    environment_owns_normalization,
    make_exact_rtrl_credit,
    make_normalizer,
    make_optax_rule,
    make_td0,
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
    terminal_of,
)

#: The two rule groups. They are not the three directions: the torso is clipped
#: as a whole and steps under its own rate, while the two readouts step
#: together. Naming them here rather than inline is what lets ``Core`` say which
#: block belongs to which without any block knowing there are groups at all.
TORSO_GROUP = "torso"
HEAD_GROUP = "heads"


@dataclass(frozen=True)
class RTRRLConfig:
    """Everything the kernel reads that does not change during a run.

    The defaults are the published ones. Where a name differs it differs
    because the published one says something untrue: ``torso_follow`` is
    ``update_period`` there, and it is a Polyak coefficient rather than a
    period -- at 1.0 the following copy is simply the current one.
    """

    num_envs: int
    gamma: float = 0.99

    # One decay per block, which is what having three blocks means.
    lambda_pi: float = 0.9
    lambda_v: float = 0.9
    lambda_rnn: float = 0.9

    # How loudly each objective speaks. ``eta_pi`` scales the policy's ascent
    # wherever it lands; ``eta_f`` scales only the TD error the torso's trace is
    # weighted by, which is how the shared block is turned down without turning
    # down either head.
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


@struct.dataclass(frozen=True)
class Recurrence:
    """Where the sequence is, and what it owes the past.

    Only the torso has one. The readouts are functions of the hidden state and
    of nothing else, which is the whole reason they are separate blocks.
    """

    carry: Any
    sensitivity: Any


@struct.dataclass(frozen=True)
class BlockState:
    """A readout's parameters and the trace that decides how far they move."""

    params: Any
    traces: Any


@struct.dataclass(frozen=True)
class TorsoState:
    """The shared block: the same two things, plus what only sharing needs.

    ``slow_params`` is the copy every forward pass actually reads. It follows
    the parameters the updates land on, so the value the TD target is measured
    against does not move with the parameters being corrected by it.
    """

    params: Any
    traces: Any
    slow_params: Any
    recurrence: Recurrence


@struct.dataclass(frozen=True)
class CoreState:
    """What the algorithm carries, one field per thing that owns one.

    ``value`` and ``emphasis`` sit here rather than in a block because neither
    belongs to one: the value is the critic's reading of the state the *torso*
    was in a step ago and it is read by the TD error, and the emphasis is the
    accumulated discount every trace is scaled by.
    """

    torso: TorsoState
    actor: BlockState
    critic: BlockState
    rule: Any
    value: Any
    emphasis: Any


@struct.dataclass(frozen=True)
class Scales:
    """What the two running estimators carry, and nothing else.

    Its own state because normalization is its own layer. It sits between the
    environment's numbers and the agent's, and the agent never sees a raw one,
    so which scale a number was on is not a question anything above it has to
    answer. That an ending resets these on the same flag as the environment is
    a coincidence of lifetime, not of ownership, and packing them into one tree
    would have been the only argument for keeping them there.
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
    """What the critic answered.

    One reading, where StreamAC has two. This algorithm does not value the state
    a transition ended in a second time -- it carries the previous step's
    reading instead -- so there is no ``next_value`` to report. The step
    rotation shows up here as a missing field.
    """

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
    """What one rule group's step did.

    Separate from ``BlockUpdate`` because a step is not per block here: the two
    readouts step together and the torso steps alone, so a step size belongs to
    a group and a gradient norm belongs to a block, and the two do not line up.
    """

    step_size: Any = None


@struct.dataclass(frozen=True)
class UpdateMetrics:
    """Three blocks, two groups, and two quantities belonging to neither.

    This is the shape a metrics schema keyed on components alone could not
    express, and it is why the declaration has to follow the algorithm: StreamAC
    needs no group level, because there a block *is* a group.
    """

    td_error: Any = None
    emphasis: Any = None
    torso: BlockUpdate = BlockUpdate()
    actor: BlockUpdate = BlockUpdate()
    critic: BlockUpdate = BlockUpdate()
    torso_step: GroupUpdate = GroupUpdate()
    heads_step: GroupUpdate = GroupUpdate()


@struct.dataclass(frozen=True)
class RTRRLState:
    """Everything the kernel carries, one field per layer that writes one.

    ``terminal`` rides beside the timestep because in this algorithm the TD
    error is measured a step after the transition it is about, so the ending
    that decides whether there was a future has to survive that long. StreamAC
    hands it straight from the environment to the update inside one step and
    never has to carry it.
    """

    step: Any
    update_step: Any
    timestep: Timestep
    terminal: Any

    env_state: Any
    scales: Scales
    core: CoreState


def _broadcast_env(values, leaf):
    return values[(slice(None),) + (None,) * (leaf.ndim - 1)]


def _where_done(done, fresh, carried):
    """Take the fresh one for the streams that ended, the carried one for the rest."""

    return jax.tree.map(
        lambda new, old: jnp.where(_broadcast_env(done, old), new, old), fresh, carried
    )


def _as_batch(tree):
    """Put the stream axis back as a length of one.

    Every layer below is written against batched shapes, and a stream pulled out
    by ``vmap`` has lost that axis. Putting it back costs nothing and means the
    arithmetic inside a per-stream pass is the same arithmetic a batched pass
    would have done.
    """

    return jax.tree.map(lambda leaf: leaf[None], tree)


def _from_batch(tree):
    """Drop the stream axis again, so ``vmap`` stacks one per stream."""

    return jax.tree.map(lambda leaf: leaf[0], tree)


def make_rules(cfg: RTRRLConfig):
    """One rule per group, both answering the same contract.

    The torso is clipped as a whole before Adam sees it, which is the only
    operation anywhere in this algorithm that spans more than one parameter at
    a time, and therefore the only reason a group has to exist at all. Ascent
    rather than descent: the scale is positive and the update is added, which is
    the published optimiser's negative learning rate written the other way
    round.
    """

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
        is what the bootstrap has to value; starting the next one belongs here,
        on the same flag the carries and the traces already read.

        The fresh observation comes back unblended. What reads it next is a
        scale, and the estimator has to be offered the value an episode opens on
        before anything decides which streams keep the one they had.
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

        One call per estimator, and that is a granularity problem rather than a
        property of the algorithm. ``Normalizer`` is two things welded together:
        an accumulator that turns a reward into the discounted return it belongs
        to -- ``Statistics.trace`` -- and a running estimate of that quantity's
        spread -- ``mean``, ``M2``, ``count``. The observation path has only the
        second; the reward path has both. Three pieces, two objects.

        ``observe`` advances whichever it has and then scales by what it has
        just written, so there is no reading to take before the advance. The
        order is not a constraint, it is the wiring -- and it is invisible only
        because the pieces are not separately expressible.

        Give each piece its own component and this class becomes what it should
        be: an assembler that says accumulate, then update, then read. That is a
        change to ``memorax/rl/normalization.py``, not to this file.
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

    All three blocks here carry a trace and decay at their own rate, allocate
    that trace against their own shapes, and are differentiated as a unit. What
    they do not share is the step: the groups a step is taken over span more
    than one block, so stepping is ``Core``'s and this class has no ``step``.
    That is the difference the shared torso introduced, and it is visible as the
    method StreamAC's version has and this one does not.

    ``credit`` is what makes the block recurrent, and it is the only thing that
    does. Given none, the block is a readout: it hands the recurrence it was
    passed straight back, because it did not advance anything.
    """

    def __init__(
        self, module: Any, *, num_envs: int, decay: float, credit=None
    ) -> None:
        self.module = module
        self.num_envs = num_envs
        self.decay = decay
        self.credit = credit

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
        """One norm per part, per stream.

        A sequence is split into what came before the recurrence, the recurrence
        itself, and what came after, so the reading says *where* something got
        large. A readout has no such split and no stable set of top-level names
        -- one head spells them ``mean``/``pre_std``, another ``Dense_0`` -- so
        it is reported under one name that does not depend on what it is made
        of.
        """

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
        """RTRRL's post-transition trace recurrence, at this block's own decay.

        ``emphasis`` is the accumulated discount: a gradient taken deep into an
        episode enters the trace already discounted by how far in it is, which
        is what makes the whole thing an episodic return rather than a
        continuing one. It is the same tree for every block, and the decay is
        not -- three blocks, three decays.
        """

        return jax.tree.map(
            lambda old, grad: (
                self.decay * (1 - _broadcast_env(reset_before, old)) * old
                + _broadcast_env(emphasis, grad) * grad
            ),
            incoming,
            gradient,
        )


class Torso:
    """The one block both heads read, and the only one credited by RTRL.

    It is a block by the same rule as the others -- nothing else reaches these
    parameters -- and it is *shared* in a different sense entirely: two things
    read its output, and both have an opinion about it. That opinion arrives as
    a cotangent, not as a gradient, which is why this class has no update of its
    own and why ``Core`` is the one that pushes anything through it.
    """

    def __init__(self, cfg: RTRRLConfig, network: Any) -> None:
        self.cfg = cfg
        self.network = network
        self.block = Network(
            network,
            num_envs=cfg.num_envs,
            decay=cfg.gamma * cfg.lambda_rnn,
            credit=make_exact_rtrl_credit(network.core),
        )

    @property
    def carry_shape(self):
        return (self.cfg.num_envs, None)

    def _input(self, timestep: Timestep):
        """The one vector the sequence sees.

        Under ``meta_rl`` the previous action and reward ride along with the
        observation. Both are dropped for the streams that just ended, because
        across an episode boundary neither happened: the action was taken in a
        world this observation is not a continuation of. The reward is only
        blanked *here* -- what the TD error reads is the reward the task
        actually paid, and this algorithm reads it a step later than the input
        does.
        """

        obs, done, action, reward = timestep
        if not self.cfg.meta_rl:
            return obs
        # ``done`` is one flag per stream per step and the other two carry a
        # feature axis on top of that, so the flag grows one axis rather than
        # however many an env-broadcast would give it.
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
        """One forward pass over one sequence-shaped step.

        The advance is handed back rather than written, because the gradient a
        step takes is taken from the recurrence the pass *started* on.
        """

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
        """The same parameters with the sequence begun again.

        Through the credit rather than around it, so that what comes back has
        the shape the credit will produce on every later step.
        """

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


class Actor:
    """The policy. It chooses, and it names the two directions it ascends.

    It holds only its own readout, because that is all it owns alone. The
    hidden state it reads is an argument, and what it wants done about that
    hidden state leaves as a cotangent -- so both of its directions are written
    as functions of ``(its parameters, the hidden state)`` and neither of them
    differentiates anything. Whoever holds the torso does that.
    """

    def __init__(self, cfg: RTRRLConfig, head: Any) -> None:
        self.cfg = cfg
        self.block = Network(
            head, num_envs=cfg.num_envs, decay=cfg.gamma * cfg.lambda_pi
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
        """What ascends through the trace: the log-probability of what was done.

        Scaled here rather than by the rule, because ``eta_pi`` is a statement
        about this objective and it has to reach the torso already applied --
        the cotangent this direction sends up is the only thing the shared block
        ever learns about the policy.
        """

        dist = self.apply(params, hidden, timestep)
        return self.cfg.eta_pi * remove_time_axis(dist.log_prob(action))[0]

    def immediate_objective(self, params, hidden, timestep: Timestep):
        """What ascends immediately: entropy, on the step it arises.

        Not traced and not weighted by the TD error. A wider policy is not
        something the critic was surprised by, so there is no error to weight it
        by and no earlier state to credit it to.
        """

        dist = self.apply(params, hidden, timestep)
        return self.cfg.entropy_rate * remove_time_axis(dist.entropy())[0]


class Critic:
    """The value. It reads, and it ascends its own reading.

    The TD error is not measured here. It reads a value this class produced a
    step ago against one it produces now, so it belongs to whatever is holding
    both -- and holding the earlier one across a step is not something a readout
    does.
    """

    def __init__(self, cfg: RTRRLConfig, head: Any) -> None:
        self.cfg = cfg
        self.block = Network(
            head, num_envs=cfg.num_envs, decay=cfg.gamma * cfg.lambda_v
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


class Core:
    """A torso, two heads, and everything that couples them.

    Four things live here and nothing else does, and each of them is here
    because it is about more than one block:

    1. The TD error. It reads the value carried from the previous step against
       the one this step produced, and it is broadcast to every block.
    2. The cotangents. Each head says what it wants done to the hidden state;
       they are added and pushed back through the torso once per objective. The
       gates go on that addition and nowhere else.
    3. The group table. Which block steps with which is not a property of any
       block, so the step is taken here.
    4. The following copy of the torso, which only exists because a shared block
       is read by a target it is also being corrected against.

    Nothing here touches an environment.
    """

    def __init__(
        self,
        cfg: RTRRLConfig,
        torso_network: Any,
        actor_head: Any,
        critic_head: Any,
    ) -> None:
        self.cfg = cfg
        self.torso = Torso(cfg, torso_network)
        self.actor = Actor(cfg, actor_head)
        self.critic = Critic(cfg, critic_head)
        self.td0 = make_td0()
        self.rules = make_rules(cfg)
        # Three blocks, two groups. Written out rather than derived, because it
        # is a decision about this algorithm and not a fact about its shapes.
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
        """Fresh online state for all three blocks and both rules.

        The carried value starts at zero rather than at a reading taken here.
        Nothing reads it: the first step's trace is empty, so the TD error it is
        weighted by multiplies nothing, and by the second step the value carried
        is one this kernel actually produced.
        """

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
        """Run forward once and choose, learning nothing.

        Reads the following copy, which is what the published implementation
        evaluates. No gradient is taken, so this is one batched pass rather than
        one per stream.
        """

        recurrence, hidden = self.torso.apply(
            state.torso.slow_params, timestep.to_sequence(), state.torso.recurrence
        )
        dist = self.actor.apply(state.actor.params, hidden, timestep.to_sequence())
        if deterministic:
            # A mode has no draw behind it, so there is no probability of having
            # chosen it and nothing here to report.
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
        """One forward and two backward passes per stream, routed to the blocks.

        Streams share parameters but not activations, so a stream's direction
        cannot depend on another's hidden state and the Jacobian of the whole
        batch is zero everywhere but the diagonal. One pass per stream, taken
        together, costs what the diagonal costs.

        The vector-Jacobian product the torso hands back is called twice, once
        per objective, which is what the published implementation does with its
        ``rnn_backwards``. Building it once and calling it twice is not an
        optimisation: the two objectives have to reach the same block through
        the same linearisation of the same forward pass.
        """

        def one(torso_params, actor_params, critic_params, timestep, recurrence, key):
            timestep = _as_batch(timestep)
            recurrence = _as_batch(jax.lax.stop_gradient(recurrence))

            def forward(params):
                advanced, hidden = self.torso.apply(params, timestep, recurrence)
                return hidden, advanced

            hidden, upstream, advanced = jax.vjp(forward, torso_params, has_aux=True)

            # Sampled from the copy the forward pass read, once, and handed to
            # every reader: the environment steps on it and the policy's
            # objective is about it. Drawing it twice would be two actions.
            dist = self.actor.apply(actor_params, hidden, timestep)
            action, log_prob = dist.sample_and_log_prob(seed=key)

            actor_traced, actor_upward = jax.grad(
                self.actor.traced_objective, argnums=(0, 1)
            )(actor_params, hidden, timestep, action)
            critic_traced, critic_upward = jax.grad(
                self.critic.traced_objective, argnums=(0, 1)
            )(critic_params, hidden, timestep)
            # This addition is the whole of what sharing means, and it is the
            # only place either head can be denied a say in the representation:
            # a gate is a factor on one of these two terms. Left open here
            # because the published algorithm has no gate to reproduce.
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
                    log_prob=remove_time_axis(log_prob)[0],
                    entropy=remove_time_axis(dist.entropy())[0],
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
        """One transition's worth of learning, and the action to take next.

        ``timestep`` is where the transition *ended* -- the reward it paid, the
        ending it hit, and the observation it left the agent in. The state it
        began in is not passed, because nothing here needs it: what the critic
        said about it is carried in ``state.value``, and what the torso knew is
        carried in the recurrence.

        The order is the published one and every line of it matters. The step is
        taken along the trace as it stood *before* this pass, because that trace
        is the one holding the gradient of the value being corrected. Only then
        is the trace advanced, and only then does an ending clear it -- clearing
        it first would throw away the credit for the transition that ended the
        episode, which is the one transition that carries the outcome.
        """

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

        # The shared block hears a quieter version of the same error. Nothing
        # else about it is scaled, which is what makes this a dial on how much
        # the representation is allowed to chase the critic.
        deltas = {TORSO_GROUP: delta * self.cfg.eta_f, HEAD_GROUP: delta}
        # The critic has no immediate objective. It is given zeros rather than
        # nothing because it shares a rule with the actor, which does.
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

        # An ending restores full emphasis, because the next gradient begins an
        # episode rather than continuing one.
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
            ForwardMetrics(actor=actor_reading, critic=CriticForward(value=value)),
            UpdateMetrics(
                td_error=delta,
                emphasis=emphasis,
                torso=BlockUpdate(
                    grad_norm=self.torso.block._norms(traced["torso"]),
                    trace_norm=self.torso.block._norms(advanced["torso"]),
                ),
                actor=BlockUpdate(
                    grad_norm=self.actor.block._norms(traced["actor"]),
                    trace_norm=self.actor.block._norms(advanced["actor"]),
                ),
                critic=BlockUpdate(
                    grad_norm=self.critic.block._norms(traced["critic"]),
                    trace_norm=self.critic.block._norms(advanced["critic"]),
                ),
                # Per group, not per block. The two readouts took one step
                # between them, so there is one step size for the pair.
                torso_step=GroupUpdate(
                    step_size=taken[TORSO_GROUP].metrics.get("step_size")
                ),
                heads_step=GroupUpdate(
                    step_size=taken[HEAD_GROUP].metrics.get("step_size")
                ),
            ),
        )


class RTRRL:
    """One-invocation train/evaluation flow around the three layers.

    It owns none of the arithmetic and none of the environment: it owns the
    order. Restart what ended, learn from the transition that got here, decide,
    and step the world -- which is the published loop read from a different
    starting point. There the environment is stepped first, with the action
    decided last time; carrying the action and carrying the transition are the
    same loop, and carrying the transition is the one that lets a step be handed
    a state somebody made up.
    """

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
        self.core = Core(cfg, torso_network, actor_head, critic_head)
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
        """A rollout that leaves nothing behind.

        Only the readings come back: the state an evaluation runs on is built
        here and dropped here, so a caller cannot carry it into training even by
        accident.
        """

        reset_key, rollout_key = jax.random.split(key)
        eval_state = self._evaluation_state(reset_key, state)
        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(rollout_key, scan_steps)
        _, metrics = jax.lax.scan(self.evaluate_step, eval_state, keys)
        return metrics
