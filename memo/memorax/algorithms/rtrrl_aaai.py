"""RTRRL: one recurrent torso shared by an actor and a critic.

    RTRRL             the order things happen in, and the scan
    Environment       where every stream is
    Normalization     the scales the environment's numbers are read through
    Core              a torso, two heads, and everything that couples them
      Torso           the shared recurrent representation and following copy
        Aggregation   where the two heads' credit for it is combined
      Actor / Critic  a readout, and the objectives it names

Three blocks, and each is its own parameter group: it selects its own
optimizer, carries its own rule state, and reports its own step size. The
torso is the one with an outer clip. Its private recurrent subgraph selects an
LRU or RTU and one differentiation method supported by that kernel.

The two readouts used to step together under one selection, which was the
grouping the published implementation has and cost nothing while every
optimizer here was a fixed rate. It costs something now: the intentional
update's ``eta`` is 0.05 for a policy and 0.5 for a value function in the work
it comes from, and one selection for both readouts could not say that.

Each readout also owns an eligibility trace -- a component, in
``memorax.rl.traces``, not a rule's private state -- and which recurrence it
runs is paired with the optimizer that reads it. See ``make_head_traces``.

The torso owns as many as its aggregation has paths. The actor's and the
critic's credit for the one shared block is added either before the trace and
the rule, which is the published topology and one of each, or after them, which
is a trace and a rule state per head summed only where the parameters are
written. Neither rule that can sit there is linear in what it is given, so the
two are different algorithms; ``docs/rtrrl-torso-aggregation.md`` says what each
means and :class:`TorsoAggregation` is where it lives.

Rebuilt from ``../RTRRL-AAAI25/rtrrl.py``. Driven against it by
``tests/test_rtrrl_parity.py``; behaviour in ``tests/test_rtrrl.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct

from memorax.building import BuildContext, ComponentBuilder, ComponentFamily
from memorax.networks.backbones import backbone
from memorax.networks.components import LayerNorm
from memorax.networks.readouts import ACTOR_HEAD_FAMILY, CRITIC_HEAD_FAMILY
from memorax.networks.sequence import PLACES, Sequence
from memorax.networks.sequence_models.lru import LRU_DIFFERENTIATION_FAMILY
from memorax.networks.sequence_models.rtu import RTU_DIFFERENTIATION_FAMILY
from memorax.observability.metrics import metric_names
from memorax.parameters import (
    describe_parameters,
    group,
    numeric,
    param,
    structure,
)
from memorax.readings import reading, readings, taken
from memorax.rl import (
    EnvironmentStreams,
    InteractionNormalization,
    NormalizationState,
    action_classes,
    action_dim,
    broadcast_stream,
    encode_feedback,
    make_optax_rule,
    make_td0,
    select_ended,
)
from memorax.rl.intentional import (
    ADVANTAGE,
    TD,
    IntentionalReading,
    IntentionalUpdate,
)
from memorax.rl.normalization import (
    DISCOUNTED_NORMALIZATION_FAMILY,
    NORMALIZATION_FAMILY,
)
from memorax.rl.traces import CARRIED, CURRENT, Trace
from memorax.rl.updates import (
    DRTRRL,
    OBGD_BOUND_FAMILY,
    STEP_FAMILY,
    Adam,
    AdaptiveObBound,
    ObGD,
    ObGDReading,
    ObGDStep,
    Sgd,
    make_bounded_rule,
    make_d_rtrrl_rule,
    make_intentional_rule,
)
from memorax.runtime import ObservationSchema
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
)

# Three parameter groups, one per block. They used to be two, with the actor
# and the critic stepping together under one selection; the intentional update
# is the first rule for which that is not a grouping but a constraint, since
# the published actor-critic sets a different intended reduction for the policy
# than for the value function. A group of one also makes every rule's step size
# a block's reading rather than a group's.
BLOCKS: tuple[str, ...] = ("torso", "actor", "critic")
#: The two that read the shared representation rather than being it.
HEADS: tuple[str, ...] = ("actor", "critic")

# Where the actor's and the critic's credit for the shared torso is added
# together. Both positions read the same two cotangents out of the same
# recurrent forward; what differs is whether the sum happens before the trace
# and the rule or after them. Neither the intentional update nor ObGD is
# linear in what it is given -- both carry running statistics of it and derive
# a step size from them -- so `Rule(actor + critic)` is not `Rule(actor) +
# Rule(critic)` and the two are different algorithms rather than two spellings
# of one.
INPUT = "input"
OUTPUT = "output"
# The path an input aggregation traces, which is the whole torso's and is named
# for the block rather than for either head: what reaches it is neither head's
# derivative but their sum through the shared recurrence.
JOINT = "torso"
# The paths an output aggregation traces, one per head whose credit it carries.
# They are named for the heads because that is what they are -- the actor's
# route to the torso and the critic's -- and because each runs the trace decay
# and the intentional signal of the head it belongs to.
TORSO_BRANCHES: tuple[str, ...] = ("actor", "critic")


# --------------------------------------------------------------- configuration
@dataclass(frozen=True)
class RTRRLConfig:
    """Everything the kernel reads that does not change during a run."""

    num_envs: int
    gamma: float = 0.99

    lambda_pi: float = 0.9
    lambda_v: float = 0.9
    lambda_rnn: float = 0.9

    eta_pi: float = 1.0
    eta_f: float = 1.0
    entropy_rate: float = 1e-5

    # A bare step aggregates at the input: one joint derivative, one torso
    # trace, one rule. `OutputSteps` aggregates at the output: two of each,
    # summed only where the parameters are written.
    torso_optimizer: (
        Sgd | Adam | DRTRRL | IntentionalUpdate | ObGDStep | OutputSteps
    ) = field(default_factory=lambda: Adam(lr=1e-4))
    actor_optimizer: Sgd | Adam | DRTRRL | IntentionalUpdate = field(
        default_factory=lambda: Adam(lr=1e-4)
    )
    critic_optimizer: Sgd | Adam | DRTRRL | IntentionalUpdate = field(
        default_factory=lambda: Adam(lr=1e-4)
    )
    torso_grad_clip: float = 1.0
    torso_follow: float = 1.0

    meta_rl: bool = True
    # How many actions the environment names, or None when it measures them.
    # Only the meta-RL feedback input reads it, and only to widen an integer
    # action into something a concatenation can carry.
    action_classes: int | None = None


#: What a readout may step under. A head has one derivative and one trace, so
#: there is nothing to aggregate and no position to name.
#:
#: Adam stays first, and the order is not cosmetic: a configuration that names
#: no optimizer is filled from the front of the search domain, so putting a new
#: rule ahead of Adam would change what every unpinned run is.
RTRRL_OPTIMIZERS = STEP_FAMILY.restricted("adam", "sgd", "d_rtrrl", "iu")


@dataclass(frozen=True)
class OutputIntentional:
    """Two intentional torso steps that meet only at the parameter update.

    Each branch declares a whole :class:`IntentionalUpdate` of its own, and
    ``eta`` above all: the published actor-critic intends a different fraction
    for a policy than for a value function, and two branches sharing one
    setting would be the input aggregation with extra state.
    """

    actor: IntentionalUpdate = group(of=IntentionalUpdate)
    critic: IntentionalUpdate = group(of=IntentionalUpdate)


@dataclass(frozen=True)
class OutputObGD:
    """The same split, under ObGD's bound rather than the intentional step.

    Each branch declares its own base rate, its own ``kappa`` and its own
    bound, because each bounds its own contribution and nothing bounds the
    sum. See :class:`OutputSteps`.
    """

    actor: ObGD = group(of=ObGD)
    critic: ObGD = group(of=ObGD)


@dataclass(frozen=True)
class OutputSteps:
    """An output aggregation as it was built: one step per branch.

    The finished torso update is ``Delta_actor + Delta_critic``, added once,
    element by element, and written once. Nothing bounds the sum: each branch's
    rule has already bounded or sized its own contribution, and a third bound
    over the total would be a limit no configuration declared and no branch
    could account for. So the sum may be longer than either part, which is what
    it means for the two paths to be independent all the way to the update.
    """

    actor: Any
    critic: Any


def _obgd_step(settings: ObGD, builder: ComponentBuilder, path: str) -> ObGDStep:
    """One ``obgd`` declaration with the bound it names built rather than named.

    A group nested inside a branch is a second choice, and reading a branch
    back fills in only the first; the builder is what descends. See
    :class:`memorax.rl.updates.ObGDStep`.
    """

    return ObGDStep(
        bound=builder.build(OBGD_BOUND_FAMILY, f"{path}.bound"),
        lr=settings.lr,
    )


def _construct_torso_step(selection, builder: ComponentBuilder):
    """The torso's step, which for two of the six branches is two steps."""

    settings = selection.parameters
    if isinstance(settings, ObGD):
        return _obgd_step(settings, builder, selection.path)
    if isinstance(settings, OutputIntentional):
        return OutputSteps(actor=settings.actor, critic=settings.critic)
    if isinstance(settings, OutputObGD):
        return OutputSteps(
            actor=_obgd_step(settings.actor, builder, f"{selection.path}.actor"),
            critic=_obgd_step(settings.critic, builder, f"{selection.path}.critic"),
        )
    return settings


#: What the shared torso may step under, which is the readouts' choices plus
#: the position the two heads' credit is combined at. The position is in the
#: name of the branch rather than beside it, because it is not a modifier of
#: the rule: ``output_iu`` maintains two intentional optimizers where
#: ``input_iu`` maintains one, and no setting of one is the other.
#:
#: The two plain rates carry no position because they have nothing to gain
#: from one. Both are linear in the direction they are handed, so
#: ``Rate(a + b)`` *is* ``Rate(a) + Rate(b)`` -- to the last bit for SGD, and
#: for Adam up to the moments it would then have to keep two of. Splitting
#: either would be two states for one algorithm, which is why the split is
#: offered exactly where it changes the answer.
#:
#: ``iu`` is deliberately absent. It named what is now ``input_iu`` and means
#: exactly that, but a name that had one meaning and now has a position in it
#: is worth failing on rather than translating silently: a run document that
#: still says ``iu`` is refused at build with the branches it could have named,
#: which is a migration the reader performs rather than one performed for them.
TORSO_STEP_BRANCHES = {
    "adam": Adam,
    "sgd": Sgd,
    "d_rtrrl": DRTRRL,
    "input_iu": IntentionalUpdate,
    "input_obgd": ObGD,
    "output_iu": OutputIntentional,
    "output_obgd": OutputObGD,
}

RTRRL_TORSO_OPTIMIZERS = ComponentFamily(
    branches=TORSO_STEP_BRANCHES,
    construct=_construct_torso_step,
)


@dataclass(frozen=True)
class LruTorso:
    """RTRRL's projection width and LRU-specific recurrent choices."""

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), static=True)
    feature_dim: int = param(valid=(1, 4096), search=(16, 256), static=True)
    differentiation: str = structure(branches=LRU_DIFFERENTIATION_FAMILY.branches)


@dataclass(frozen=True)
class RtuTorso:
    """RTRRL's RTU-specific recurrent choices.

    No readout width, because the RTU has none to set: its output is its two
    carries concatenated, so ``hidden_dim`` fixes it.
    """

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), static=True)
    differentiation: str = structure(branches=RTU_DIFFERENTIATION_FAMILY.branches)


def _construct_torso(selection, builder, *, features: int):
    """Build RTRRL's differentiated recurrent subgraph, and nothing before it.

    The cell reads the observation, and it reads it directly. There used to be
    a ``FFN -> LayerNorm -> Tanh`` in front, which existed only because
    ``LRUConfig.features`` was being used as both the input width and the
    readout width: the cell was configured to emit ``feature_dim``, so
    something had to widen the observation to ``feature_dim`` first. Setting
    ``output_dim`` separates the two, which is what it was added for, and the
    projection has nothing left to do.

    That projection was never the published algorithm's. Its cell takes the
    observation as it arrives and normalises afterwards, and the parity suite
    could not have said so -- it re-hosts the published *wiring* on these same
    networks precisely so a mismatch cannot come from the networks, which
    leaves what is in front of the cell unexamined by construction.
    """

    settings = selection.parameters
    recurrent = backbone(
        selection.kind,
        features=features,
        hidden_dim=settings.hidden_dim,
        # Only the LRU has a readout to size; see RtuTorso.
        output_dim=getattr(settings, "feature_dim", None),
    )
    # After the cell, and affine-free, which is where and what the published
    # implementation's is.
    network = Sequence(
        components=(*recurrent, LayerNorm(use_scale=False, use_bias=False))
    )
    family = {
        "lru": LRU_DIFFERENTIATION_FAMILY,
        "rtu": RTU_DIFFERENTIATION_FAMILY,
    }[selection.kind]
    differentiation = builder.build(
        family,
        f"{selection.path}.differentiation",
        core=network.core,
    )
    return network, differentiation


RTRRL_TORSO_FAMILY = ComponentFamily(
    branches={"lru": LruTorso, "rtu": RtuTorso},
    construct=_construct_torso,
)


@dataclass(frozen=True)
class TorsoParameters:
    """The shared block's own settings.

    ``optimizer`` names both the rule and where the two heads' credit for this
    one shared block is combined; see :data:`TORSO_STEP_BRANCHES`.

    ``grad_clip`` is the outer bound on the finished torso update, and it is
    the one parameter here whose valid range depends on a sibling. The two
    rules that size or bound their own step refuse to have a second, undeclared
    bound placed over it, so the four branches that select one of them --
    ``input_iu``, ``output_iu``, ``input_obgd`` and ``output_obgd`` -- require
    ``grad_clip: 0``. The three that do not are ``adam``, ``sgd`` and
    ``d_rtrrl``: for the two plain rates this clip is the only bound there is,
    and D-RTRRL declares it as its own outer bound on the finished update.

    The declaration cannot say that -- a parameter tree conditions on branches,
    not on another parameter's value -- so the build refuses the pair with the
    reason rather than accepting a run that is not the algorithm it names. A
    search space selecting one of the four has to pin this to zero, or every
    trial fails at construction.
    """

    backbone: str = structure(branches=RTRRL_TORSO_FAMILY.branches)
    optimizer: str = structure(branches=RTRRL_TORSO_OPTIMIZERS.branches)
    grad_clip: float = param(valid=(0.0, 100.0), search=(0.0, 10.0))
    follow: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))


@dataclass(frozen=True)
class ActorParameters:
    head: str = structure(branches=ACTOR_HEAD_FAMILY.branches)
    optimizer: str = structure(branches=RTRRL_OPTIMIZERS.branches)


@dataclass(frozen=True)
class CriticParameters:
    head: str = structure(branches=CRITIC_HEAD_FAMILY.branches)
    optimizer: str = structure(branches=RTRRL_OPTIMIZERS.branches)


@dataclass(frozen=True)
class NormalizationParameters:
    observation: str = structure(branches=NORMALIZATION_FAMILY.branches)
    reward: str = structure(branches=DISCOUNTED_NORMALIZATION_FAMILY.branches)


@dataclass(frozen=True)
class RTRRLParameters:
    torso: TorsoParameters = group(of=TorsoParameters)
    actor: ActorParameters = group(of=ActorParameters)
    critic: CriticParameters = group(of=CriticParameters)
    normalization: NormalizationParameters = group(of=NormalizationParameters)
    gamma: float = param(valid=(0.5, 0.9999), search=(0.9, 0.9999))
    lambda_pi: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    lambda_v: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    lambda_rnn: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    eta_pi: float = param(valid=(0.0, 100.0), search=(0.0, 10.0))
    eta_f: float = param(valid=(0.0, 100.0), search=(0.0, 10.0))
    entropy_rate: float = param(valid=(0.0, 1.0), search=(1e-8, 1e-2), log=True)
    meta_rl: bool = param(valid=[False, True], search=[True])


PARAMETERS = describe_parameters(RTRRLParameters)


# ----------------------------------------------------------------------- state
class Recurrence(struct.PyTreeNode):
    """Where the sequence is, and what it owes the past."""

    carry: Any
    differentiation_state: Any


class BlockState(struct.PyTreeNode):
    """A readout's parameters and the trace that decides how far they move."""

    params: Any
    traces: Any


class TorsoState(struct.PyTreeNode):
    """The shared block: the same two things, plus what only sharing needs."""

    params: Any
    traces: Any
    slow_params: Any
    recurrence: Recurrence


class CoreState(struct.PyTreeNode):
    """What the algorithm carries, one field per thing that owns one."""

    torso: TorsoState
    actor: BlockState
    critic: BlockState
    rule: Any
    value: Any
    emphasis: Any


# -------------------------------------------------------------------- readings
@dataclass(frozen=True)
class IntentionalReports:
    """Which of one intentional block's readings to take.

    Every one of these is a quantity the intentional step passed through and
    that nothing downstream could recover from the update: the step size and
    the signal reach the parameters only as their product, and the statistic
    the step size divides by reaches them not at all. A block whose optimizer
    is not the intentional one produces none of them, which is why the graph
    builds this declaration from what it selected rather than taking the
    module's.
    """

    clipped_delta: bool = reading(at="clipped_delta", default=False)
    signal: bool = reading(at="signal", default=False)
    advantage_scale: bool = reading(at="advantage_scale", default=False)
    rms_scale: bool = reading(at="rms_scale", default=False)
    sigma_bar: bool = reading(at="sigma_bar", default=False)
    trace_quadratic: bool = reading(at="trace_quadratic", default=False)
    denominator: bool = reading(at="denominator", default=False)
    update_norm: bool = reading(at="update_norm", default=False)
    non_finite: bool = reading(at="non_finite", default=False)


@dataclass(frozen=True)
class ObGDReports:
    """Which of one bounded block's readings to take.

    Off unless the block selected ObGD, for the reason
    :class:`IntentionalReports` is off unless it selected the intentional
    update: a name whose value is permanently absent is the failure a reading
    declaration exists to make impossible.

    Every one of these is a term the bounded step size is a quotient of and
    that nothing downstream could recover from it -- see
    :class:`memorax.rl.updates.ObGDReading`, where each is what it is.
    """

    trace_sum: bool = reading(at="trace_sum", default=False)
    delta_bar: bool = reading(at="delta_bar", default=False)
    bound_denominator: bool = reading(at="bound_denominator", default=False)
    bound_scale: bool = reading(at="bound_scale", default=False)
    second_moment_rms: bool = reading(at="second_moment_rms", default=False)


@dataclass(frozen=True)
class BlockReports:
    """Which position-split torso readings to take."""

    grad_norm: bool = reading(at="grad_norm", split=True)
    trace_norm: bool = reading(at="trace_norm", split=True)
    step_size: bool = reading(at="step_size")
    intentional: IntentionalReports = readings(of=IntentionalReports, at="intentional")
    obgd: ObGDReports = readings(of=ObGDReports, at="obgd")


@dataclass(frozen=True)
class BranchReports:
    """Which of one output branch's readings to take.

    The same four a block declares, and off by default for the same reason
    :class:`IntentionalReports` is: they exist only where the torso's credit is
    combined at the output, and a declaration on by default would advertise
    them for every run that does not have them.
    """

    grad_norm: bool = reading(at="grad_norm", split=True, default=False)
    trace_norm: bool = reading(at="trace_norm", split=True, default=False)
    step_size: bool = reading(at="step_size", default=False)
    intentional: IntentionalReports = readings(of=IntentionalReports, at="intentional")
    obgd: ObGDReports = readings(of=ObGDReports, at="obgd")


@dataclass(frozen=True)
class TorsoReports(BlockReports):
    """The shared block's readings, at whichever position it was aggregated.

    The four inherited fields are the *joint* path's: they exist under an input
    aggregation, where there is one derivative, one trace and one step size to
    report, and are absent under an output one, where each of those is two
    things and reporting either alone would be a number nothing produced.

    ``actor`` and ``critic`` are the other way round. They are what an output
    aggregation reports, one whole block's worth per branch, which is what
    makes the two paths readable apart: a run that could only see the summed
    update could not tell a branch that stopped moving from one whose
    contribution the other cancelled.
    """

    actor: BranchReports = readings(of=BranchReports, at="actor")
    critic: BranchReports = readings(of=BranchReports, at="critic")


@dataclass(frozen=True)
class HeadReports:
    """Which whole-readout readings to take."""

    grad_norm: bool = reading(at="grad_norm")
    trace_norm: bool = reading(at="trace_norm")
    step_size: bool = reading(at="step_size")
    intentional: IntentionalReports = readings(of=IntentionalReports, at="intentional")


@dataclass(frozen=True)
class Reports:
    """Which readings to take, in the shape of what produces them.

    Two levels, because a block's readings hang off the block that took them.
    """

    log_prob: bool = reading(at="forward.actor.log_prob")
    entropy: bool = reading(at="forward.actor.entropy")
    value: bool = reading(at="forward.critic.value")
    td_error: bool = reading(at="update.td_error")
    emphasis: bool = reading(at="update.emphasis")
    torso: TorsoReports = readings(of=TorsoReports, at="update.torso")
    actor: HeadReports = readings(of=HeadReports, at="update.actor")
    critic: HeadReports = readings(of=HeadReports, at="update.critic")


def _intentional_reports(*, taken: bool, advantage: bool = False) -> IntentionalReports:
    """One block's intentional readings, on only where they are produced.

    ``advantage_scale`` is the running scale a normalized advantage is divided
    by, so it exists for the block whose signal is an advantage and nowhere
    else. Declaring it for the two TD blocks would advertise a name whose value
    is permanently absent, which is the failure the reading declaration exists
    to make impossible.
    """

    return IntentionalReports(
        **{
            item.name: taken and (advantage or item.name != "advantage_scale")
            for item in fields(IntentionalReports)
        }
    )


def _torso_reports(step) -> TorsoReports:
    """Which of the torso's readings exist, given where its credit is combined.

    Exactly one of the two levels is on. Under an input aggregation the joint
    path is the torso and there are no branches; under an output one there are
    two branches and no joint anything -- not a derivative, not a trace, not a
    step size -- so leaving the inherited fields on would advertise four names
    whose values are permanently absent.
    """

    if not isinstance(step, OutputSteps):
        bounded, adaptive = _bounded(step)
        return TorsoReports(
            intentional=_intentional_reports(taken=isinstance(step, IntentionalUpdate)),
            obgd=_obgd_reports(taken=bounded, adaptive=adaptive),
        )
    return TorsoReports(
        grad_norm=False,
        trace_norm=False,
        step_size=False,
        intentional=_intentional_reports(taken=False),
        actor=_branch_reports(step.actor, advantage=True),
        critic=_branch_reports(step.critic),
    )


def _branch_reports(step, *, advantage: bool = False) -> BranchReports:
    """One output branch's readings, which are on wherever the branch exists."""

    bounded, adaptive = _bounded(step)
    return BranchReports(
        grad_norm=True,
        trace_norm=True,
        step_size=True,
        intentional=_intentional_reports(
            taken=isinstance(step, IntentionalUpdate), advantage=advantage
        ),
        obgd=_obgd_reports(taken=bounded, adaptive=adaptive),
    )


def _obgd_reports(*, taken: bool, adaptive: bool = False) -> ObGDReports:
    """One block's bounded readings, on only where they are produced.

    ``second_moment_rms`` is the mean of the denominator the *adaptive* bounds
    divide by, so it exists for a block that selected one and nowhere else.
    The plain bound normalizes by nothing, and a reading of one there would be
    a normalization that a reader could not tell from an absent one.
    """

    return ObGDReports(
        **{
            item.name: taken and (adaptive or item.name != "second_moment_rms")
            for item in fields(ObGDReports)
        }
    )


def _bounded(step) -> tuple[bool, bool]:
    """Whether a step is ObGD, and whether the bound it named is an adaptive one."""

    if not isinstance(step, ObGDStep):
        return False, False
    return True, isinstance(step.bound, AdaptiveObBound)


def reports_for(*, torso, actor, critic) -> Reports:
    """The readings one configuration takes, given the rules it built.

    The three arguments are the built steps rather than three booleans, because
    what a block reports no longer follows from one fact about it: a torso
    reports at one of two positions, and under the output one it reports twice.

    ``step_size`` is every block's, whichever rule stepped it: Adam reports its
    rate, D-RTRRL its threshold, ObGD the rate its bound left, and the
    intentional update the dynamic step it derived. One name, because a reader
    comparing two runs is asking the same question of both.
    """

    return Reports(
        torso=_torso_reports(torso),
        actor=HeadReports(
            intentional=_intentional_reports(
                taken=isinstance(actor, IntentionalUpdate), advantage=True
            )
        ),
        critic=HeadReports(
            intentional=_intentional_reports(
                taken=isinstance(critic, IntentionalUpdate)
            )
        ),
    )


PARTS: tuple[str, ...] = PLACES
REPORTS = Reports()
# Every reading this kernel offers under some configuration. A catalog
# advertises this, because an experiment selecting the intentional optimizer
# has to be able to name what it will then be able to score on; what any one
# run files is narrower and travels on the built graph, not on the class.
AVAILABLE_REPORTS = Reports(
    torso=TorsoReports(
        intentional=_intentional_reports(taken=True),
        obgd=_obgd_reports(taken=True, adaptive=True),
        actor=BranchReports(
            grad_norm=True,
            trace_norm=True,
            step_size=True,
            intentional=_intentional_reports(taken=True, advantage=True),
            obgd=_obgd_reports(taken=True, adaptive=True),
        ),
        critic=BranchReports(
            grad_norm=True,
            trace_norm=True,
            step_size=True,
            intentional=_intentional_reports(taken=True),
            obgd=_obgd_reports(taken=True, adaptive=True),
        ),
    ),
    actor=HeadReports(intentional=_intentional_reports(taken=True, advantage=True)),
    critic=HeadReports(intentional=_intentional_reports(taken=True)),
)
TRAINING_METRICS: tuple[str, ...] = taken(REPORTS, parts=PARTS)
METRICS: tuple[str, ...] = metric_names(
    "train", taken(AVAILABLE_REPORTS, parts=PARTS)
) + metric_names("eval")
OBSERVATIONS = ObservationSchema(
    reward="interaction.reward",
    done="interaction.done",
    terminal="interaction.terminal",
    observation="interaction.observation",
    next_observation="interaction.next_observation",
    action="interaction.action",
    series=TRAINING_METRICS,
)


class ActorForward(struct.PyTreeNode):
    """What the policy answered on the pass that chose."""

    log_prob: Any = None
    entropy: Any = None


class CriticForward(struct.PyTreeNode):
    """What the critic answered."""

    value: Any = None


class ForwardMetrics(struct.PyTreeNode):
    """One field per head."""

    actor: ActorForward = ActorForward()
    critic: CriticForward = CriticForward()


class BlockUpdate(struct.PyTreeNode):
    """How big what went into one block's step was, and what it passed through.

    ``intentional`` and ``obgd`` are empty unless the block selected that rule.
    They are the two rules here that derive their own step size, and so the two
    with statistics behind that number which nothing downstream could recover
    from it.
    """

    grad_norm: Any = None
    trace_norm: Any = None
    step_size: Any = None
    intentional: IntentionalReading = IntentionalReading()
    obgd: ObGDReading = ObGDReading()


class TorsoUpdate(BlockUpdate):
    """The shared block's, at whichever position its two credits were combined.

    The inherited fields carry the joint path's quantities and the two nested
    blocks carry the branches'. One level is filled and the other is empty; see
    :class:`TorsoReports`.
    """

    actor: BlockUpdate = BlockUpdate()
    critic: BlockUpdate = BlockUpdate()


class UpdateMetrics(struct.PyTreeNode):
    """Three blocks, and two quantities belonging to none of them."""

    td_error: Any = None
    emphasis: Any = None
    torso: TorsoUpdate = TorsoUpdate()
    actor: BlockUpdate = BlockUpdate()
    critic: BlockUpdate = BlockUpdate()


class RTRRLState(struct.PyTreeNode):
    """Everything the kernel carries, one field per layer that writes one."""

    step: Any
    update_step: Any
    timestep: Timestep
    terminal: Any

    env_state: Any
    scales: NormalizationState
    core: CoreState


# ---------------------------------------------------------- shapes and streams
def _as_batch(tree):
    """Put the stream axis back as a length of one."""

    return jax.tree.map(lambda leaf: leaf[None], tree)


def _from_batch(tree):
    """Drop the stream axis again, so ``vmap`` stacks one per stream."""

    return jax.tree.map(lambda leaf: leaf[0], tree)


# ------------------------------------------------------- one rule per block
def _rate_rule(base: Sgd | Adam, *, clip: float):
    """A learning rate over the combined ascent, preconditioned or not.

    Adam is what the paper's RTRRL steps; plain SGD is the same chain with the
    preconditioner left out, so the two share the order the clip is written in.
    ``clip`` bounds the finished ascent direction before either sees it, which
    is where the paper's ``grad_clip`` sits and is therefore where it stays for
    a rate that does nothing else to the direction.

    The scale is written **positive**, and this is the whole reason the chain
    is spelled out rather than delegated to ``optax.sgd``/``optax.adam``: those
    fold a negation into the rate because optax steps ``params + updates`` down
    a gradient, while everything reaching a rule here is an *ascent* direction
    the algorithm adds. Calling the packaged optimizer would silently descend.
    """

    chain: list[Any] = []
    if clip:
        chain.append(optax.clip_by_global_norm(clip))
    if isinstance(base, Adam):
        chain.append(optax.scale_by_adam(b1=base.b1, b2=base.b2, eps=base.eps))
    chain.append(optax.scale(base.lr))
    return make_optax_rule(optax.chain(*chain), rate=base.lr)


# Which of the three lambdas a block's trace forgets at, and which scalar its
# step is proportional to. Both are routing decisions: the intentional-update
# paper names one trace and one signal per learner, and RTRRL has three
# parameter groups because it shares a torso between two of them.
BLOCK_DECAYS = {
    "torso": lambda cfg: cfg.lambda_rnn,
    "actor": lambda cfg: cfg.lambda_pi,
    "critic": lambda cfg: cfg.lambda_v,
}
# The actor and the critic are the paper's two algorithms exactly: Intentional
# TD over the value function's own derivative, and the intentional policy
# gradient over the log-probability's.
#
# **The torso is neither, and should not be described as either.** What reaches
# it is RTRRL's joint derivative, the actor's cotangent and the critic's summed
# through the shared recurrence, which is not `grad V` and not `grad log pi`.
# The step size derived from it is a real quantity -- it is `eta` divided by the
# same denominator, over the same statistics -- but the functional reading of
# `eta` does not survive the sum: it no longer names a fraction of a TD error
# the step sets out to spend, because the sum is not the derivative of any one
# thing the TD error is measured on. Call this what it is, an IU-style step over
# RTRRL's joint torso signal, and do not read the torso's `eta` on the same axis
# as the actor's and the critic's.
BLOCK_SIGNALS = {"torso": TD, "actor": ADVANTAGE, "critic": TD}


def _refuse_a_second_bound(step, *, name: str, clip: float) -> None:
    """A rule that sizes its own step is not stepped under an outer clip too.

    The intentional update derives its step size from the statistics it
    carries, and ObGD shrinks its rate whenever a step would cross the TD
    target. Both are bounds a configuration declared and can account for.
    Clipping the finished update afterwards is a second one it did not, so it
    is refused with the reason rather than applied on top.
    """

    if not clip:
        return
    what = (
        "the intentional update sets its own step size from the statistics it "
        "carries"
        if isinstance(step, IntentionalUpdate)
        else "ObGD bounds its own step so that one update cannot cross the TD " "target"
    )
    raise ValueError(
        f"{what}; clipping the finished {name} step to {clip} would be a "
        "second, undeclared bound on it"
    )


def _block_rule(step, *, cfg: RTRRLConfig, name: str, clip: float):
    """Whichever rule a block's optimizer names, over that one block.

    A rule is handed ``{name: tree}`` rather than the bare tree, because that
    is the shape the D-RTRRL rule's normalization units are cut from and the
    shape the intentional rule keys its per-block optimizers by. With one block
    per rule the two D-RTRRL scopes name the same unit; see :class:`DRTRRL`.

    ``name`` is a block's for a head and for the torso's joint path, and a
    *branch's* for one path of an output aggregation -- which is why the decay
    and the signal are read from the same two tables either way: a torso branch
    runs the trace and steps along the signal of the head whose credit it
    carries.
    """

    if isinstance(step, IntentionalUpdate):
        _refuse_a_second_bound(step, name=name, clip=clip)
        return make_intentional_rule(
            step,
            signals={name: BLOCK_SIGNALS[name]},
            decays={name: cfg.gamma * BLOCK_DECAYS[name](cfg)},
            streams=cfg.num_envs,
        )
    if isinstance(step, ObGDStep):
        _refuse_a_second_bound(step, name=name, clip=clip)
        # The published pair, through the rule StreamAC already answers to:
        # the bound this path declared, written over a plain rate.
        return make_bounded_rule(bound=step.bound, base=Sgd(lr=step.lr))
    if isinstance(step, DRTRRL):
        return make_d_rtrrl_rule(step, clip=clip)
    return _rate_rule(step, clip=clip)


def _trace_for(step, *, decay: float) -> Trace:
    """The eligibility recurrence that goes with what was selected.

    Which recurrence a path runs is not a free setting. An intentional update
    is derived against ``z_t = gamma*lambda*z_{t-1} + p_t`` read *after* this
    step's derivative has joined it, and against no emphasis; RTRRL's own rules
    -- Adam, D-RTRRL and ObGD alike -- are written over the trace as it stood,
    weighted by the followed-trace emphasis. Pairing them the other way round
    would leave the step size dividing by a trace the step was not taken along,
    so the algorithm -- which owns both -- constructs the one that goes with
    what was selected.

    The optimizer never sees this choice. It is handed a trace and told nothing
    about where in the transition it was read.
    """

    if isinstance(step, IntentionalUpdate):
        return Trace(decay=decay, reads=CURRENT, emphasized=False)
    return Trace(decay=decay, reads=CARRIED, emphasized=True)


def make_rules(cfg: RTRRLConfig):
    """Every rule one configuration steps under, by the name it is filed under.

    ``torso``, ``actor`` and ``critic`` when the torso's credit is combined at
    the input, and ``torso.actor`` and ``torso.critic`` in place of the first
    when it is combined at the output -- the same names the readings are filed
    under, because they are the same paths.

    Nothing in the algorithm calls this: :class:`Core` builds the readouts'
    rules and the aggregation separately, because it needs the aggregation
    itself and not a view of it. What this is for is asking what a
    configuration steps under without building a graph to find out, which is a
    question about routing rather than about arithmetic and is worth being
    able to ask directly.
    """

    aggregation = make_torso_aggregation(cfg)
    return {
        JOINT if name == JOINT else f"{JOINT}.{name}": rule
        for name, rule in aggregation.rules.items()
    } | make_head_rules(cfg)


def make_head_rules(cfg: RTRRLConfig):
    """One rule per readout. The torso's are the aggregation's."""

    optimizers = {"actor": cfg.actor_optimizer, "critic": cfg.critic_optimizer}
    return {
        name: _block_rule(optimizers[name], cfg=cfg, name=name, clip=0.0)
        for name in HEADS
    }


def make_head_traces(cfg: RTRRLConfig):
    """One eligibility recurrence per readout, paired with its optimizer."""

    optimizers = {"actor": cfg.actor_optimizer, "critic": cfg.critic_optimizer}
    return {
        name: _trace_for(optimizers[name], decay=cfg.gamma * BLOCK_DECAYS[name](cfg))
        for name in HEADS
    }


def _folded(traced, direct, *, sign, folds: bool):
    """What one path's trace accumulates, and what of it stays untraced.

    For a path folding the entropy in, the traced derivative is the sum the
    paper differentiates as one objective. The sign is the TD error's rather
    than the normalized advantage's, and they are the same sign: the clip
    preserves it and the running scale it is divided by is never negative.
    Taking it here is what lets the optimizer stay ignorant of the entropy
    entirely.
    """

    if direct is None or not folds:
        return traced, direct
    return (
        jax.tree.map(
            lambda leaf, term: leaf + broadcast_stream(sign, term) * term,
            traced,
            direct,
        ),
        None,
    )


def _step_size(result, key):
    """The step one path took, under the one name every rule reports it by.

    A rule serving several blocks reports one per block; a rule serving one
    reports the number. Both are the same reading, and the caller asking for it
    should not have to know which kind of rule answered.
    """

    size = result.metrics.get("step_size")
    if isinstance(size, Mapping):
        return size[key]
    return size


def _obgd_reading(result, reports: ObGDReports):
    """One path's bounded readings, gated by what this graph declares.

    Not keyed by block, unlike the intentional rule's: the bounded rule reads
    the whole tree it was handed as one unit and derives one step size for it,
    so what it reports is one reading rather than one per block.
    """

    produced = result.metrics.get("obgd")
    if produced is None:
        return ObGDReading()
    return ObGDReading(
        **{
            item.name: (
                getattr(produced, item.name) if getattr(reports, item.name) else None
            )
            for item in fields(ObGDReports)
        }
    )


def _intentional_reading(result, key, reports: IntentionalReports):
    """One path's intentional readings, gated by what this graph declares."""

    produced = result.metrics.get("intentional")
    if produced is None:
        return IntentionalReading()
    reading = produced[key]
    return IntentionalReading(
        **{
            item.name: (
                getattr(reading, item.name) if getattr(reports, item.name) else None
            )
            for item in fields(IntentionalReports)
        }
    )


# -------------------------------------------- where the two credits are joined
class Aggregated(struct.PyTreeNode):
    """One transition's worth of the shared torso's learning.

    ``update`` is the finished parameter update, whichever position produced
    it, so the caller writes the parameters once and knows nothing about how
    many rules contributed to what it was handed. The other four are per path,
    and what a path is depends on the position: one thing under an input
    aggregation, one per branch under an output one.
    """

    update: Any
    traces: Any
    state: Any
    derivative: Any
    taken: Any


class _Taken(NamedTuple):
    """What one path produced this transition, named rather than positional."""

    update: Any
    traces: Any
    derivative: Any
    result: Any


class _ContributionPath:
    """One route from a head's objective to the shared torso's parameters.

    A derivative, a trace, a rule and that rule's state -- everything that has
    to be its own for two contributions to be genuinely two. A private helper
    rather than a component: it owns no part of the algorithm the aggregation
    around it does not already own, and its whole content is that these four
    travel together.
    """

    def __init__(self, cfg: RTRRLConfig, name: str, step, *, clip: float) -> None:
        self.name = name
        self.step = step
        self.trace = _trace_for(step, decay=cfg.gamma * BLOCK_DECAYS[name](cfg))
        self.rule = _block_rule(step, cfg=cfg, name=name, clip=clip)
        # The same fact as which trace it runs; see `_folded`.
        self.folds_entropy = self.trace.reads == CURRENT
        self.streams = cfg.num_envs

    def initial_traces(self, params):
        return self.trace.initial(params, self.streams)

    def init(self, params, traces):
        return self.rule.init(params={self.name: params}, traces={self.name: traces})

    def take(
        self,
        *,
        params,
        carried,
        traced,
        direct,
        delta,
        sign,
        step,
        state,
        reset,
        emphasis,
    ):
        """Advance this path's trace, then take this path's step along it."""

        derivative, untraced = _folded(
            traced, direct, sign=sign, folds=self.folds_entropy
        )
        stepped, advanced = self.trace.stepped(
            carried, derivative, reset=reset, emphasis=emphasis
        )
        taken = self.rule.apply(
            {self.name: stepped},
            None if untraced is None else {self.name: untraced},
            state,
            delta=delta,
            derivative={self.name: derivative},
            step=step,
            params={self.name: params},
        )
        return _Taken(
            update=taken.updates[self.name],
            traces=advanced,
            derivative=derivative,
            result=taken,
        )

    def reading(self, norms, reports, derivative, advanced, taken):
        """What this path's step passed through, as far as it was asked for."""

        return BlockUpdate(
            grad_norm=norms(derivative) if reports.grad_norm else None,
            trace_norm=norms(advanced) if reports.trace_norm else None,
            step_size=_step_size(taken, self.name) if reports.step_size else None,
            intentional=_intentional_reading(taken, self.name, reports.intentional),
            obgd=_obgd_reading(taken, reports.obgd),
        )


class TorsoAggregation:
    """Where the actor's and the critic's credit for the shared torso meets.

    Both positions read the same two cotangents out of the same recurrent
    forward, and neither costs a second forward or a second sensitivity. What
    they differ in is where the two are added::

        input    p       = J^T (u_actor + u_critic);  one trace, one rule
        output   p_actor = J^T u_actor               a trace and a rule each,
                 p_critic = J^T u_critic             summed as finished updates

    and that is a difference in the algorithm rather than in its spelling,
    because neither rule that can sit here is linear in what it is given. The
    intentional update preconditions by a second moment of the derivative and
    divides by a statistic of the trace; ObGD shrinks its rate by the trace's
    own norm. So ``Rule(a + b) != Rule(a) + Rule(b)``, and a run has to say
    which of the two it took.

    A parent constructs one implementation or the other and otherwise treats
    them alike: it hands over the pullback of the torso forward and the two
    heads' cotangents, and receives a finished update, the traces the next
    transition carries, the rule state, and the readings.
    """

    position: str

    @property
    def recurrences(self) -> Mapping[str, Trace]:
        """The eligibility recurrence on each path, by the name it is filed under.

        One under ``torso`` at ``gamma * lambda_rnn`` when the contributions are
        combined before the trace, and one under each head's name at that
        head's own ``gamma * lambda`` when they are combined after it. Nothing
        in the algorithm reads this -- each path steps its own -- but which
        recurrence a position declares is a claim worth being able to check.
        """

        raise NotImplementedError

    @property
    def rules(self) -> Mapping[str, Any]:
        """The rule on each path, by the name it is filed under.

        The same keys :attr:`recurrences` uses, and for the same reason: a
        caller asking what a configuration steps under should not have to know
        how many things the answer is. See :func:`make_rules`.
        """

        raise NotImplementedError

    def cotangents(self, upstream, *, actor, critic):
        """What this position traces, pulled back through the shared torso."""

        raise NotImplementedError

    def untraced(self, upstream, entropy):
        """Where the actor's entropy direction reaches the torso, if anywhere."""

        raise NotImplementedError

    def initial_traces(self, params):
        """One trace per path, which is what online means."""

        raise NotImplementedError

    def init(self, params, traces):
        """Fresh rule state, one per path."""

        raise NotImplementedError

    def step(
        self,
        *,
        params,
        carried,
        traced,
        direct,
        delta,
        sign,
        step,
        state,
        reset,
        emphasis,
    ) -> Aggregated:
        """One transition: trace, step, and whatever the parameters are owed.

        ``delta`` arrives already scaled by ``eta_f``, and ``sign`` is the sign
        of the TD error before that scaling, which is what an entropy fold is
        signed by. ``carried``, ``traced``, ``direct`` and ``state`` are per
        path, and what a path is is the implementation's to say.
        """

        raise NotImplementedError

    def reading(
        self, norms, reports: TorsoReports, aggregated: Aggregated
    ) -> TorsoUpdate:
        """Project what the step passed through onto the declared readings."""

        raise NotImplementedError


class InputAggregation(TorsoAggregation):
    """The two cotangents summed before anything reads them.

    RTRRL's own topology and the one every recorded run answers to. One
    instantaneous derivative, one eligibility trace at ``gamma * lambda_rnn``,
    one rule state, one step. The sum happens inside the pullback -- the two
    cotangents are added and pulled back once -- which is where it has always
    happened and is not the same arithmetic, to the last bit, as pulling back
    twice and adding.

    What reaches this path is neither ``grad V`` nor ``grad log pi`` but their
    sum through the shared recurrence, which is why its intentional signal is
    the plain TD error and why the ``eta`` it derives a step from is not on the
    same axis as a head's. See :data:`BLOCK_SIGNALS`.
    """

    position = INPUT

    def __init__(self, cfg: RTRRLConfig, step, *, clip: float) -> None:
        self.path = _ContributionPath(cfg, JOINT, step, clip=clip)

    @property
    def recurrences(self) -> Mapping[str, Trace]:
        return {JOINT: self.path.trace}

    @property
    def rules(self) -> Mapping[str, Any]:
        return {JOINT: self.path.rule}

    def cotangents(self, upstream, *, actor, critic):
        # A gate would be a factor on one of these two terms.
        return upstream(actor + critic)[0]

    def untraced(self, upstream, entropy):
        return upstream(entropy)[0]

    def initial_traces(self, params):
        return self.path.initial_traces(params)

    def init(self, params, traces):
        return self.path.init(params, traces)

    def step(
        self,
        *,
        params,
        carried,
        traced,
        direct,
        delta,
        sign,
        step,
        state,
        reset,
        emphasis,
    ) -> Aggregated:
        taken = self.path.take(
            params=params,
            carried=carried,
            traced=traced,
            direct=direct,
            delta=delta,
            sign=sign,
            step=step,
            state=state,
            reset=reset,
            emphasis=emphasis,
        )
        return Aggregated(
            update=taken.update,
            traces=taken.traces,
            state=taken.result.state,
            derivative=taken.derivative,
            taken=taken.result,
        )

    def reading(
        self, norms, reports: TorsoReports, aggregated: Aggregated
    ) -> TorsoUpdate:
        block = self.path.reading(
            norms, reports, aggregated.derivative, aggregated.traces, aggregated.taken
        )
        # Every field of the block, by name, rather than the four this used to
        # copy. A joint path *is* a block's worth of readings and the two types
        # differ only in what an output aggregation adds, so listing the shared
        # fields here made the reading silently narrower than the declaration
        # each time one was added -- which is how `obgd` came to be advertised
        # under `update.torso.obgd.*` and filed as nothing at all.
        return TorsoUpdate(
            **{item.name: getattr(block, item.name) for item in fields(BlockUpdate)}
        )


class OutputAggregation(TorsoAggregation):
    """The two cotangents kept apart until the parameters are written.

    Two paths, and everything a path is is two things: an instantaneous
    derivative pulled back from one head's cotangent alone, an eligibility
    trace at that head's own ``gamma * lambda`` rather than at ``lambda_rnn``,
    a rule with its own settings, and that rule's whole state -- second
    moments, clipping statistic, advantage scale, bound statistics, dynamic
    step size. Changing one branch's configuration moves nothing on the other.

    The actor's branch carries the entropy direction and the critic's does not,
    which follows from whose objective the term belongs to. Whether it is
    traced or applied on the step it arises is that branch's rule's business,
    the same as anywhere else.

    Both branches read the parameters this transition started with, and their
    updates are added element by element and written once. Applying one and
    then computing the other would make the pair order-dependent, and would not
    be this topology.
    """

    position = OUTPUT

    def __init__(self, cfg: RTRRLConfig, steps: OutputSteps, *, clip: float) -> None:
        self.paths = {
            name: _ContributionPath(cfg, name, getattr(steps, name), clip=clip)
            for name in TORSO_BRANCHES
        }

    @property
    def recurrences(self) -> Mapping[str, Trace]:
        return {name: path.trace for name, path in self.paths.items()}

    @property
    def rules(self) -> Mapping[str, Any]:
        return {name: path.rule for name, path in self.paths.items()}

    def cotangents(self, upstream, *, actor, critic):
        cotangent = {"actor": actor, "critic": critic}
        # One pullback per branch, over the one forward and the one sensitivity
        # both branches share. What is independent is the state each path
        # carries, not the recurrence they read.
        return {name: upstream(cotangent[name])[0] for name in TORSO_BRANCHES}

    def untraced(self, upstream, entropy):
        return {"actor": upstream(entropy)[0]}

    def initial_traces(self, params):
        return {name: path.initial_traces(params) for name, path in self.paths.items()}

    def init(self, params, traces):
        return {
            name: path.init(params, traces[name]) for name, path in self.paths.items()
        }

    def step(
        self,
        *,
        params,
        carried,
        traced,
        direct,
        delta,
        sign,
        step,
        state,
        reset,
        emphasis,
    ) -> Aggregated:
        took = {
            name: path.take(
                params=params,
                carried=carried[name],
                traced=traced[name],
                direct=None if direct is None else direct.get(name),
                delta=delta,
                sign=sign,
                step=step,
                state=state[name],
                reset=reset,
                emphasis=emphasis,
            )
            for name, path in self.paths.items()
        }
        return Aggregated(
            # Added element by element, and nothing else: no clip, no norm, no
            # third bound. See :class:`OutputSteps`.
            update=jax.tree.map(
                lambda *parts: sum(parts),
                *[taken.update for taken in took.values()],
            ),
            traces={name: taken.traces for name, taken in took.items()},
            state={name: taken.result.state for name, taken in took.items()},
            derivative={name: taken.derivative for name, taken in took.items()},
            taken={name: taken.result for name, taken in took.items()},
        )

    def reading(
        self, norms, reports: TorsoReports, aggregated: Aggregated
    ) -> TorsoUpdate:
        def branch(name: str) -> BlockUpdate:
            return self.paths[name].reading(
                norms,
                getattr(reports, name),
                aggregated.derivative[name],
                aggregated.traces[name],
                aggregated.taken[name],
            )

        return TorsoUpdate(actor=branch("actor"), critic=branch("critic"))


def make_torso_aggregation(cfg: RTRRLConfig) -> TorsoAggregation:
    """The aggregation the torso's optimizer selection names.

    Two steps or one is the whole of the selection, so the built type is the
    discriminator: nothing else has to be consulted to know which position a
    configuration asked for.
    """

    optimizer = cfg.torso_optimizer
    if isinstance(optimizer, OutputSteps):
        return OutputAggregation(cfg, optimizer, clip=cfg.torso_grad_clip)
    return InputAggregation(cfg, optimizer, clip=cfg.torso_grad_clip)


class Block:
    """What each readout has: a stream count, a decay and a trace.

    The trace is a component rather than a method here. What it does is one
    recurrence and one reading decision, both of which the two readouts would
    otherwise have to spell identically, and both of which an algorithm has to
    be able to swap without the block noticing.

    Left unsaid it is RTRRL's own -- emphasis-weighted, and read as it stood
    before this transition's derivative joined it. That is the recurrence this
    algorithm has always run and the one every rule but the intentional update
    is written over, so it is the default rather than something each caller
    has to know to ask for.

    The torso is not one of these. It has as many traces as its credit has
    paths, which is one or two, and they belong to the component that decides
    how many there are; see :class:`TorsoAggregation`.
    """

    def __init__(
        self, cfg: RTRRLConfig, decay: float, *, trace: Trace | None = None
    ) -> None:
        self.cfg = cfg
        self.trace = trace or Trace(decay=decay, reads=CARRIED, emphasized=True)

    def initial_traces(self, params):
        return self.trace.initial(params, self.cfg.num_envs)

    def stepped_trace(self, carried, derivative, *, reset_before, emphasis):
        """What this update steps along, and what the next transition carries."""

        return self.trace.stepped(
            carried, derivative, reset=reset_before, emphasis=emphasis
        )


def _gradient_norms(module, tree):
    """One norm per algorithmic position, per stream."""

    grouped = module.split(tree) if hasattr(module, "split") else {"readout": tree}
    return subtree_norms(grouped, streams=True)


def _head_gradient_norm(tree):
    """One per-stream norm for a readout that has no sequence positions."""

    return subtree_norms({"head": tree}, streams=True)["head"]


def kernel_constraint(network: Sequence):
    """Project a torso's parameters back onto the set its kernel allows.

    The kernel states the set; this knows only where in the sequence's tree
    that kernel's parameters are -- the recurrent component's slot, and the
    ``cell`` the ``RNN`` wrapper keeps them under. A kernel naming no
    constraint gets no projection, so this is not something every core has to
    implement to be usable as a torso: of the cores this repository declares,
    the CTRNN bounds ``tau`` from below and the dense state-space core bounds
    the row norms of ``A``, and every other one names nothing.

    It lives here, beside the :class:`Torso` that applies it, because two
    algorithms now need the same traversal of the same tree shape. Before the
    second one it was private to ``rtrrl_ctrnn_rflo``, which was the right
    place for it while the contract had one implementation.
    """

    index = network.recurrent
    if index is None:
        return None
    name = network.names[index]
    cell = network.components[index].cell
    constrain = getattr(cell, "constrain", None)
    if constrain is None:
        return None

    def project(params):
        branch = params[name]
        return {**params, name: {**branch, "cell": constrain(branch["cell"])}}

    return project


# ------------------------------------------------------------ the shared block
class Torso:
    """The recurrent representation shared by both heads.

    Not a :class:`Block`, because a block is one eligibility trace and this
    block has as many as its aggregation has paths -- one or two. What credits
    the shared parameters is :class:`TorsoAggregation`, which owns the traces,
    the rules and the rule states; what is here is the network, its online
    sensitivity, the set its kernel allows and the copy that reads.
    """

    def __init__(
        self,
        cfg: RTRRLConfig,
        network: Any,
        differentiation: Any,
        *,
        constraint: Any = None,
    ) -> None:
        self.cfg = cfg
        self._network = network
        self._differentiation = differentiation
        self._constraint = constraint

    @property
    def carry_shape(self):
        return (self.cfg.num_envs, None)

    def _input(self, timestep: Timestep):
        """The one vector the sequence sees."""

        obs, done, action, reward = timestep
        if not self.cfg.meta_rl:
            return obs
        # Widened first, so what an ended stream carries is the zero vector
        # rather than the one-hot of whichever action happens to be numbered 0.
        action = encode_feedback(action, classes=self.cfg.action_classes)
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
        """One forward pass over one sequence-shaped step."""

        _, done, _, _ = timestep
        (carry, differentiation_state), output = self._network.walk(
            params,
            self._input(timestep),
            done=done,
            carries=recurrence.carry,
            differentiation_state=recurrence.differentiation_state,
            differentiation=self._differentiation,
        )
        return (
            Recurrence(carry=carry, differentiation_state=differentiation_state),
            output,
        )

    def init(self, keys, timestep: Timestep) -> TorsoState:
        """Fresh online state for the shared block, and a copy for it to follow.

        The eligibility comes back empty. How many traces this block carries is
        the aggregation's answer and not the network's, so the algorithm that
        holds both fills the field; see :meth:`Core.init`.
        """

        param_key, torso_key, dropout_key = keys
        _, done, _, _ = timestep
        carry = self._network.initialize_carry(jax.random.key(0), self.carry_shape)
        differentiation_state = self._differentiation.initialize(
            param_key, self.carry_shape
        )
        with self._differentiation.initialization():
            variables = self._network.init(
                {"params": param_key, "torso": torso_key, "dropout": dropout_key},
                self._input(timestep),
                done=done,
                initial_carry=carry,
            )
        # Drawn, then projected: a floor above the initial draw would
        # otherwise leave the first transition acting on a parameter the kernel
        # says is outside its domain.
        params = self.constrain(variables["params"])
        return TorsoState(
            params=params,
            traces=None,
            slow_params=params,
            recurrence=Recurrence(
                carry=carry, differentiation_state=differentiation_state
            ),
        )

    def reset(self, key, state: TorsoState) -> TorsoState:
        """The same parameters with the sequence begun again."""

        return state.replace(
            recurrence=Recurrence(
                carry=self._network.initialize_carry(key, self.carry_shape),
                differentiation_state=self._differentiation.initialize(
                    key, self.carry_shape
                ),
            )
        )

    def gradient_norms(self, tree):
        return _gradient_norms(self._network, tree)

    def constrain(self, params):
        """The stepped parameters put back inside the set the kernel allows.

        Most recurrent kernels allow every real number their parameters can
        hold and hand back what they were given. One that does not -- a time
        constant a division by it needs held away from zero -- says so, and the
        projection is applied here, after the step and before the reading copy
        follows it, so the followed copy is a point of that set too.
        """

        if self._constraint is None:
            return params
        return self._constraint(params)

    def followed(self, params, slow_params):
        """The reading copy takes one step toward the copy that was updated."""

        if self.cfg.torso_follow == 1.0:
            return params
        return optax.incremental_update(params, slow_params, self.cfg.torso_follow)


# ------------------------------------------------------------ the two readouts
class Actor(Block):
    """The policy. It chooses, and it names the two directions it ascends."""

    def __init__(
        self, cfg: RTRRLConfig, head: Any, *, trace: Trace | None = None
    ) -> None:
        super().__init__(cfg, cfg.gamma * cfg.lambda_pi, trace=trace)
        self._head = head

    def init(self, key, hidden, timestep: Timestep) -> BlockState:
        _, done, action, reward = timestep
        params = self._head.init(
            {"params": key}, hidden, action=action, reward=reward, done=done
        )["params"]
        return BlockState(params=params, traces=self.initial_traces(params))

    def apply(self, params, hidden, timestep: Timestep):
        _, done, action, reward = timestep
        dist, _ = self._head.apply(
            {"params": params}, hidden, action=action, reward=reward, done=done
        )
        return dist

    def gradient_norms(self, tree):
        return _head_gradient_norm(tree)

    def traced_objective(self, params, hidden, timestep: Timestep, action):
        """What ascends through the trace: the log-probability of what was done.

        ``eta_pi`` scales it, which is RTRRL's own dial and is not the paper's
        ``eta``. Under an intentional step the two compose in a way worth
        saying out loud: the intentional step size is derived to move this
        objective by a fixed fraction, so scaling the objective by ``eta_pi``
        rescales what "the objective" means, and the entropy coefficient rides
        along at ``entropy_rate / eta_pi`` relative to it. Neither is a defect;
        both make ``eta`` stop being the paper's "about five percent of the
        log-probability". A formal intentional run should pin ``eta_pi`` at 1
        and search ``eta`` alone rather than searching both, which would be two
        dials over one quantity.
        """

        dist = self.apply(params, hidden, timestep)
        return self.cfg.eta_pi * remove_time_axis(dist.log_prob(action))[0]

    def immediate_objective(self, params, hidden, timestep: Timestep):
        """What ascends immediately: entropy, on the step it arises."""

        dist = self.apply(params, hidden, timestep)
        return self.cfg.entropy_rate * remove_time_axis(dist.entropy())[0]


class Critic(Block):
    """The value. It reads, and it ascends its own reading."""

    def __init__(
        self, cfg: RTRRLConfig, head: Any, *, trace: Trace | None = None
    ) -> None:
        super().__init__(cfg, cfg.gamma * cfg.lambda_v, trace=trace)
        self._head = head

    def init(self, key, hidden, timestep: Timestep) -> BlockState:
        _, done, action, reward = timestep
        params = self._head.init(
            {"params": key}, hidden, action=action, reward=reward, done=done
        )["params"]
        return BlockState(params=params, traces=self.initial_traces(params))

    def apply(self, params, hidden, timestep: Timestep):
        _, done, action, reward = timestep
        value, _ = self._head.apply(
            {"params": params}, hidden, action=action, reward=reward, done=done
        )
        return remove_feature_axis(remove_time_axis(value))

    def gradient_norms(self, tree):
        return _head_gradient_norm(tree)

    def traced_objective(self, params, hidden, timestep: Timestep):
        """What ascends through the trace: the value itself, with no error in it."""

        return self.apply(params, hidden, timestep)[0]


# --------------------------------------------------------------- the algorithm
class Core:
    """A torso, two heads, and everything that couples them."""

    def __init__(
        self,
        cfg: RTRRLConfig,
        torso_network: Any,
        torso_differentiation: Any,
        actor_head: Any,
        critic_head: Any,
        reports: Reports = Reports(),
        torso_constraint: Any = None,
    ) -> None:
        self.cfg = cfg
        self.reports = reports
        self.td0 = make_td0()
        self.rules = make_head_rules(cfg)
        traces = make_head_traces(cfg)
        # Which readouts trace the entropy direction instead of adding it on
        # the step it arises. It is the same fact as which trace they run, and
        # it is the algorithm's: the intentional policy gradient is the
        # derivative of the log-probability and the entropy together, signed by
        # the TD error, and it is that sum the trace accumulates. RTRRL's own
        # rules take the entropy untraced, as the published implementation
        # does. A torso path answers the same question for itself.
        self.folds_entropy = {name: traces[name].reads == CURRENT for name in HEADS}
        self.aggregation = make_torso_aggregation(cfg)
        self.torso = Torso(
            cfg,
            torso_network,
            torso_differentiation,
            constraint=torso_constraint,
        )
        self.actor = Actor(cfg, actor_head, trace=traces["actor"])
        self.critic = Critic(cfg, critic_head, trace=traces["critic"])
        self.heads = {"actor": self.actor, "critic": self.critic}

    def init(self, keys, timestep: Timestep) -> CoreState:
        """Fresh online state for all three blocks and every rule they select.

        The torso's is the aggregation's, because how many rules there are
        is the same question as where the two contributions are combined.
        """

        torso_keys, actor_key, critic_key = keys
        torso = self.torso.init(torso_keys, timestep)
        # One eligibility per path the two contributions travel by, which the
        # block itself has no way to count.
        torso = torso.replace(traces=self.aggregation.initial_traces(torso.params))
        _, hidden = self.torso.apply(torso.params, timestep, torso.recurrence)
        actor = self.actor.init(actor_key, hidden, timestep)
        critic = self.critic.init(critic_key, hidden, timestep)

        params = {"actor": actor.params, "critic": critic.params}
        traces = {"actor": actor.traces, "critic": critic.traces}
        return CoreState(
            torso=torso,
            actor=actor,
            critic=critic,
            rule={
                "torso": self.aggregation.init(torso.params, torso.traces),
                **{
                    name: rule.init(
                        params={name: params[name]}, traces={name: traces[name]}
                    )
                    for name, rule in self.rules.items()
                },
            },
            value=jnp.zeros((self.cfg.num_envs,), dtype=jnp.float32),
            emphasis=jnp.ones((self.cfg.num_envs,), dtype=jnp.float32),
        )

    def reset(self, key, state: CoreState) -> CoreState:
        return state.replace(torso=self.torso.reset(key, state.torso))

    def act(
        self, key, state: CoreState, timestep: Timestep, *, deterministic: bool
    ) -> tuple[Recurrence, Any, ActorForward]:
        """Run forward once and choose, learning nothing."""

        recurrence, hidden = self.torso.apply(
            state.torso.slow_params, timestep.to_sequence(), state.torso.recurrence
        )
        dist = self.actor.apply(state.actor.params, hidden, timestep.to_sequence())
        if deterministic:
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
        """One forward and two backward passes per stream, routed to the blocks."""

        def one(torso_params, actor_params, critic_params, timestep, recurrence, key):
            timestep = _as_batch(timestep)
            recurrence = _as_batch(jax.lax.stop_gradient(recurrence))

            def forward(params):
                advanced, hidden = self.torso.apply(params, timestep, recurrence)
                return hidden, advanced

            hidden, upstream, advanced = jax.vjp(forward, torso_params, has_aux=True)

            dist = self.actor.apply(actor_params, hidden, timestep)
            action, log_prob = dist.sample_and_log_prob(seed=key)

            actor_traced, actor_upward = jax.grad(
                self.actor.traced_objective, argnums=(0, 1)
            )(actor_params, hidden, timestep, action)
            critic_traced, critic_upward = jax.grad(
                self.critic.traced_objective, argnums=(0, 1)
            )(critic_params, hidden, timestep)
            torso_traced = self.aggregation.cotangents(
                upstream, actor=actor_upward, critic=critic_upward
            )

            actor_direct, direct_upward = jax.grad(
                self.actor.immediate_objective, argnums=(0, 1)
            )(actor_params, hidden, timestep)
            torso_direct = self.aggregation.untraced(upstream, direct_upward)

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
                    log_prob=(
                        remove_time_axis(log_prob)[0] if self.reports.log_prob else None
                    ),
                    entropy=(
                        remove_time_axis(dist.entropy())[0]
                        if self.reports.entropy
                        else None
                    ),
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

    def _td_error(self, state, timestep, value, terminal):
        return self.td0(
            reward=timestep.reward,
            value=state.value,
            next_value=value,
            terminal=terminal,
            gamma=self.cfg.gamma,
        )

    def _emphasis(self, state, reset_before):
        """RTRRL's followed-trace weight on this transition's derivative.

        Advanced whatever the rules are, because it is the algorithm's own
        quantity and something reads it. Whether it reaches a trace is the
        trace component's to say: the paper's recurrence has no emphasis in it.
        """

        return (
            self.cfg.gamma * state.emphasis * (1 - reset_before) + reset_before
        ).astype(jnp.float32)

    def _derivatives(self, traced, direct, delta):
        """What each readout's trace accumulates, and what stays untraced.

        The torso's paths answer the same question inside the aggregation,
        which is where the number of them is known. See
        :func:`_folded`, which both callers share.
        """

        sign = jnp.sign(delta)
        derivative, untraced = {}, {}
        for name in HEADS:
            derivative[name], untraced[name] = _folded(
                traced[name],
                direct.get(name),
                sign=sign,
                folds=self.folds_entropy[name],
            )
        return derivative, untraced

    def _take_updates(self, state, traced, direct, delta, step, reset_before):
        """Advance every trace, then take the step each rule sizes along it.

        The trace moves first and hands back both readings; which one an update
        steps along is the trace component's decision and not visible here.
        What is visible is that the algorithm keeps every trace it had,
        whichever rule is downstream and however many of them the torso's
        credit is routed through.

        ``eta_f`` scales the TD error the torso is aggregated by, at either
        position and on every path: it is RTRRL's own dial on how far the
        shared representation moves relative to the readouts, and it is a
        property of crediting the torso rather than of where the crediting is
        combined.
        """

        params = {"actor": state.actor.params, "critic": state.critic.params}
        carried = {"actor": state.actor.traces, "critic": state.critic.traces}
        emphasis = self._emphasis(state, reset_before)
        derivative, untraced = self._derivatives(traced, direct, delta)
        aggregated = self.aggregation.step(
            params=state.torso.params,
            carried=state.torso.traces,
            traced=traced["torso"],
            direct=direct["torso"],
            delta=delta * self.cfg.eta_f,
            sign=jnp.sign(delta),
            step=step,
            state=state.rule["torso"],
            reset=reset_before,
            emphasis=emphasis,
        )
        stepped_traces = {
            name: head.stepped_trace(
                carried[name],
                derivative[name],
                reset_before=reset_before,
                emphasis=emphasis,
            )
            for name, head in self.heads.items()
        }
        taken = {
            name: rule.apply(
                {name: stepped_traces[name][0]},
                None if untraced[name] is None else {name: untraced[name]},
                state.rule[name],
                delta=delta,
                derivative={name: derivative[name]},
                step=step,
                params={name: params[name]},
            )
            for name, rule in self.rules.items()
        }
        stepped = {
            "torso": jax.tree.map(
                lambda parameter, update: parameter + update,
                state.torso.params,
                aggregated.update,
            ),
            **{
                name: jax.tree.map(
                    lambda parameter, update: parameter + update,
                    params[name],
                    taken[name].updates[name],
                )
                for name in params
            },
        }
        rule = {
            "torso": aggregated.state,
            **{name: result.state for name, result in taken.items()},
        }
        advanced = {
            "torso": aggregated.traces,
            **{name: pair[1] for name, pair in stepped_traces.items()},
        }
        return stepped, taken, aggregated, rule, emphasis, advanced, derivative

    def _update_reading(self, delta, emphasis, traced, advanced, taken, aggregated):
        """Project update products onto the readings this graph declares."""

        def head_reading(name):
            reports = getattr(self.reports, name)
            head = self.heads[name]
            return BlockUpdate(
                grad_norm=(
                    head.gradient_norms(traced[name]) if reports.grad_norm else None
                ),
                trace_norm=(
                    head.gradient_norms(advanced[name]) if reports.trace_norm else None
                ),
                step_size=(
                    _step_size(taken[name], name) if reports.step_size else None
                ),
                intentional=_intentional_reading(
                    taken[name], name, reports.intentional
                ),
            )

        return UpdateMetrics(
            td_error=delta if self.reports.td_error else None,
            emphasis=emphasis if self.reports.emphasis else None,
            torso=self.aggregation.reading(
                self.torso.gradient_norms, self.reports.torso, aggregated
            ),
            actor=head_reading("actor"),
            critic=head_reading("critic"),
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
        """One transition's worth of learning, and the action to take next."""

        recurrence, action, value, traced, direct, actor_reading = self._per_stream(
            key, state, timestep
        )

        delta = self._td_error(state, timestep, value, terminal)
        (
            stepped,
            taken,
            aggregated,
            rule,
            emphasis,
            advanced,
            derivative,
        ) = self._take_updates(state, traced, direct, delta, step, reset_before)

        torso_params = self.torso.constrain(stepped["torso"])
        return (
            state.replace(
                torso=state.torso.replace(
                    params=torso_params,
                    traces=advanced["torso"],
                    # The reading copy is projected too. It is what acts and
                    # what the sequence is walked with, so a constraint that
                    # held only for the copy being stepped would leave the one
                    # every forward pass reads outside the set whenever the
                    # follow is partial.
                    slow_params=self.torso.constrain(
                        self.torso.followed(torso_params, state.torso.slow_params)
                    ),
                    recurrence=recurrence,
                ),
                actor=BlockState(params=stepped["actor"], traces=advanced["actor"]),
                critic=BlockState(params=stepped["critic"], traces=advanced["critic"]),
                rule=rule,
                value=value,
                emphasis=emphasis,
            ),
            action,
            ForwardMetrics(
                actor=actor_reading,
                critic=CriticForward(value=value if self.reports.value else None),
            ),
            self._update_reading(
                delta, emphasis, derivative, advanced, taken, aggregated
            ),
        )


# -------------------------------------------------------------------- the flow
class RTRRL:
    """One-invocation train/evaluation flow around the three layers."""

    observations = OBSERVATIONS

    def __init__(
        self,
        cfg: RTRRLConfig,
        env: Any,
        env_params: Any,
        torso_network: Any,
        torso_differentiation: Any,
        actor_head: Any,
        critic_head: Any,
        *,
        observation_normalization: Any = None,
        reward_normalization: Any = None,
        evaluation: EvaluationConfig | None = None,
        record: Iterable[str] = (),
        reports: Reports = Reports(),
        torso_constraint: Any = None,
    ) -> None:
        self.cfg = cfg
        evaluation = evaluation or EvaluationConfig()
        self.environment = EnvironmentStreams(cfg.num_envs, env, env_params)
        self.normalization = InteractionNormalization(
            cfg.num_envs,
            env,
            observation=observation_normalization,
            reward=reward_normalization,
            reset_on_start=evaluation.reset_on_start,
            update_during_eval=evaluation.update_during_eval,
        )
        self.core = Core(
            cfg,
            torso_network,
            torso_differentiation,
            actor_head,
            critic_head,
            reports,
            torso_constraint=torso_constraint,
        )
        self.record = frozenset(record)
        # What this configuration files, which is not always what the class
        # names: an optimizer carrying state worth reading adds series that no
        # other configuration produces, and a schema naming them everywhere
        # would fail every other run on a series that was never going to
        # arrive. A build taking exactly what the module names keeps the
        # module's schema, so nothing downstream sees a copy of it.
        series = taken(reports, parts=PARTS)
        self.observations = (
            OBSERVATIONS
            if series == OBSERVATIONS.series
            else replace(OBSERVATIONS, series=series)
        )

    @classmethod
    def graph(
        cls,
        parameters: dict[str, Any],
        components: ComponentBuilder,
        context: BuildContext,
        *,
        record: Iterable[str] = (),
    ) -> RTRRL:
        """Declare the shared torso, the two readouts, and a rule for each."""

        gamma = numeric(parameters["gamma"])
        meta_rl = bool(parameters["meta_rl"])
        classes = action_classes(context.action_space)
        # What the cell reads, now that nothing widens it first: the observation,
        # and under meta-RL the previous action and reward beside it.
        features = int(context.observation_space.shape[0])
        if meta_rl:
            features += action_dim(context.action_space) + 1
        torso_optimizer = components.build(RTRRL_TORSO_OPTIMIZERS, "torso.optimizer")
        actor_optimizer = components.build(RTRRL_OPTIMIZERS, "actor.optimizer")
        critic_optimizer = components.build(RTRRL_OPTIMIZERS, "critic.optimizer")
        torso_network, torso_differentiation = components.build(
            RTRRL_TORSO_FAMILY, "torso.backbone", features=features
        )
        return cls(
            RTRRLConfig(
                num_envs=context.num_envs,
                gamma=gamma,
                lambda_pi=numeric(parameters["lambda_pi"]),
                lambda_v=numeric(parameters["lambda_v"]),
                lambda_rnn=numeric(parameters["lambda_rnn"]),
                eta_pi=numeric(parameters["eta_pi"]),
                eta_f=numeric(parameters["eta_f"]),
                entropy_rate=numeric(parameters["entropy_rate"]),
                torso_grad_clip=numeric(parameters["torso.grad_clip"]),
                torso_follow=numeric(parameters["torso.follow"]),
                meta_rl=meta_rl,
                action_classes=classes,
                torso_optimizer=torso_optimizer,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
            ),
            context.environment,
            context.environment_parameters,
            torso_network,
            torso_differentiation,
            components.build(
                ACTOR_HEAD_FAMILY,
                "actor.head",
                action_dim=action_dim(context.action_space),
                discrete=classes is not None,
            ),
            components.build(CRITIC_HEAD_FAMILY, "critic.head"),
            observation_normalization=components.build(
                NORMALIZATION_FAMILY, "normalization.observation"
            ),
            reward_normalization=components.build(
                DISCOUNTED_NORMALIZATION_FAMILY,
                "normalization.reward",
                discount=gamma,
            ),
            record=record,
            reports=reports_for(
                torso=torso_optimizer,
                actor=actor_optimizer,
                critic=critic_optimizer,
            ),
        )

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
                obs=select_ended(done, obs, state.timestep.obs)
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

    def interact(self, key: Any, state: RTRRLState) -> tuple[RTRRLState, StepMetrics]:
        """One behavior-policy transition that learns nothing and costs no budget.

        The stochastic policy and the recurrent sequence continue exactly where
        training left them, so a sampled episode can be finished after the
        training budget without the parameters, traces, rules, normalization
        statistics, or either step counter moving.
        """

        reset_key, action_key, env_key = jax.random.split(key, 3)
        state = self._reset(reset_key, state, update=False)
        observation = state.timestep.obs

        recurrence, action, _ = self.core.act(
            action_key, state.core, state.timestep, deterministic=False
        )
        obs, env_state, environment_reward, done, terminal, info = (
            self.environment.step(env_key, state.env_state, action)
        )
        obs, reward, _ = self.normalization.apply(
            state.scales, obs, environment_reward, done, update=False
        )
        next_timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
        return state.replace(
            timestep=next_timestep,
            terminal=terminal,
            env_state=env_state,
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

    def open_evaluation(self, key: Any, state: RTRRLState) -> RTRRLState:
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

    def evaluate(
        self, key: Any, state: RTRRLState, num_steps: int
    ) -> tuple[RTRRLState, StepMetrics]:
        """Advance an opened evaluation rollout, learning nothing from it.

        ``state`` is what ``open_evaluation`` returned, or what an earlier call
        to this handed back: an evaluation runs until it has the episodes it
        was asked for, and one call is only as much of it as a scan may hold.
        """

        scan_steps = self._num_scan_steps(num_steps, self.cfg.num_envs)
        keys = jax.random.split(key, scan_steps)
        return jax.lax.scan(self.evaluate_step, state, keys)
