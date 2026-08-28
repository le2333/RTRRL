"""Interchangeable rules that turn ascent directions into parameter updates.

A rule owns one parameter group. Which parameters form a group, and what
scaling an algorithm applies to a trace before handing it over, stays with the
algorithm; a rule only decides how far to step along the direction it is given.

Every rule receives the eligibility trace and the TD error separately rather
than a finished gradient. Adam does not need the split, but OBGD does: its
step-size bound reads |delta| and the trace norm together, so delta cannot be
folded in beforehand. The D-RTRRL rule needs it for the opposite reason: it
reads only the *sign* of delta, and a folded-in delta cannot be unfolded.

One contract, four inputs::

    apply(trace, direct, state, *, delta, derivative, step, params)

``trace`` is the eligibility the algorithm's trace component left; ``direct``
is what ascends untraced; ``delta`` is the scalar the trace is weighted by; and
``derivative`` is the instantaneous derivative that trace was accumulated from,
which most rules ignore. The intentional update is the one that reads it -- its
second moment is a statistic of the derivative while its step size is one of
the trace -- and reading it is not the same as owning it: no rule accumulates a
trace of its own, so which eligibility an algorithm keeps stays readable from
the algorithm.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from memorax.building import ComponentFamily
from memorax.parameters import param, structure

from .intentional import IntentionalOptimizer, IntentionalUpdate


class ObjectiveDirections(struct.PyTreeNode):
    """What an objective asks for, split by how the ascent reaches a parameter.

    Traced directions are accumulated into an eligibility trace and later
    weighted by the TD error; direct ones apply on the step they arise. Which
    domains exist, and which parameter group each maps onto, is the
    algorithm's own routing decision.
    """

    traced_by_domain: Any
    direct_by_domain: Any
    metrics: Any


class RuleOutput(struct.PyTreeNode):
    """Parameter updates, the rule's carried state, and its diagnostics."""

    updates: Any
    state: Any
    metrics: Any


class ObGDReading(struct.PyTreeNode):
    """What one bounded step passed through, per stream.

    The step size alone says a rate was small and not why it was small. It is
    the quotient of four things -- the base rate, the TD error, the trace's own
    size and, under the adaptive bounds, the second moment the trace was
    normalized by -- and none of the four can be recovered from it afterwards.
    A run reporting only the quotient cannot tell a rate held down by a large
    trace from one held down by a large TD error, which is the question anyone
    reading a bounded run is asking.

    ``trace_sum`` is the L1 the bound reads: ``sum |z|`` under the plain bound,
    and ``sum |z| / sqrt(v_hat)`` under the adaptive ones, which is why it is
    not the trace norm an algorithm reports beside it.

    ``delta_bar`` is ``max(|delta|, 1)``, the TD error as the bound reads it.
    Reported rather than left to be derived from whatever TD error a caller
    also files, because an algorithm may scale the error before handing it over
    -- RTRRL multiplies the torso's by ``eta_f`` -- and then the two are
    different numbers.

    ``bound_denominator`` is ``delta_bar * trace_sum * lr * kappa``, *before*
    the maximum with one. That is the product the bound compares against one,
    and the step size is ``lr`` divided by ``max(1, this)``. Before the maximum
    because afterwards a run approaching its bound and a run nowhere near it
    are the same number, and which of those a run is is worth being able to
    see. The bound is active exactly where this exceeds one.

    ``bound_scale`` is what survived: ``step_size / lr``, in ``(0, 1]``.

    ``second_moment_rms`` is the mean over parameters of the denominator the
    adaptive bounds divide by, which is ``sqrt(v_hat) + eps`` or
    ``sqrt(v_hat + eps)`` depending on which was declared. It is None for the
    plain bound, where nothing normalizes and carrying a zero would say there
    was a normalization that happened to be one.
    """

    trace_sum: Any = None
    delta_bar: Any = None
    bound_denominator: Any = None
    bound_scale: Any = None
    second_moment_rms: Any = None


@dataclass(frozen=True)
class UpdateRule:
    """Build-time pairing of a rule's state constructor and its step.

    ``init`` takes both the parameter tree and the trace tree because the two
    rules carry different shapes: Adam's moments follow the parameters, which
    the env axis has already been averaged out of, while OBGD's second moment
    follows the trace and stays per-env, since its bound is measured per-env.

    Every rule answers one contract, and a rule that needs the instantaneous
    derivative reads the ``derivative`` its algorithm passes beside the trace
    rather than accumulating one of its own.
    """

    init: Callable[..., Any]
    apply: Callable[..., RuleOutput]


def _broadcast_env(values, leaf):
    return values[(slice(None),) + (None,) * (leaf.ndim - 1)]


def _combine(traced, direct, delta):
    """Weight the trace by the TD error and add the untraced directions.

    ``direct`` may be None when an objective routes everything through the
    trace, which saves carrying a tree of zeros through every step.
    """

    if direct is None:
        return jax.tree.map(lambda trace: _broadcast_env(delta, trace) * trace, traced)
    return jax.tree.map(
        lambda trace, immediate: _broadcast_env(delta, trace) * trace + immediate,
        traced,
        direct,
    )


def make_optax_rule(transform, *, rate=None) -> UpdateRule:
    """Step along delta * trace through any optax transformation.

    The transformation sees one finished ascent direction per parameter, so
    whatever an algorithm expresses through optax -- Adam, clipping, freezing
    a subtree -- it expresses by composing the transformation it passes in.

    ``rate`` is the scalar the transformation multiplies the ascent by, when the
    caller knows it. Given, the rule reports it as the step size it took, which
    is the same reading a bound reports and lets a caller read one name whether
    or not a bound is in the way.
    """

    def init(*, params, traces):
        del traces
        return transform.init(params)

    def apply(traced, direct, state, *, delta, derivative, step, params):
        del step, derivative
        ascent = jax.tree.map(
            lambda leaf: jnp.mean(leaf, axis=0),
            _combine(traced, direct, delta),
        )
        updates, state = transform.update(ascent, state, params)
        metrics = {} if rate is None else {"step_size": jnp.full_like(delta, rate)}
        return RuleOutput(updates=updates, state=state, metrics=metrics)

    return UpdateRule(init=init, apply=apply)


@dataclass(frozen=True)
class ObBound:
    kappa: float = param(valid=(0.0, 100.0), search=(0.5, 10.0))


@dataclass(frozen=True)
class AdaptiveObBound:
    kappa: float = param(valid=(0.0, 100.0), search=(0.5, 10.0))
    beta2: float = param(valid=(0.0, 1.0), search=(0.9, 0.9999))
    eps: float = param(valid=(1e-12, 1e-2), search=[1e-8], log=True)


@dataclass(frozen=True)
class AdaptiveObBoundFixed(AdaptiveObBound):
    """The published placement of eps: inside the root rather than beside it.

    Identical to declare and identical to read, and a different component all
    the same, because it is a different denominator. A branch that shared a
    class with another could not be told apart by what it hands back.
    """


@dataclass(frozen=True)
class Sgd:
    lr: float = param(valid=(1e-9, 10.0), search=(1e-5, 1.0), log=True)


@dataclass(frozen=True)
class Adam:
    lr: float = param(valid=(1e-9, 10.0), search=(1e-5, 1e-2), log=True)
    # The published constants, so that an Adam asked for in a test about
    # something else can be spelled by the one number that test is about.
    b1: float = param(valid=(0.0, 1.0), search=[0.9], default=0.9)
    b2: float = param(valid=(0.0, 1.0), search=[0.999], default=0.999)
    eps: float = param(valid=(1e-12, 1e-2), search=[1e-8], default=1e-8, log=True)


def base_transform(base):
    """The optax transform a base names, for the paths that have no bound."""

    import optax

    if isinstance(base, Adam):
        return optax.adam(base.lr, b1=base.b1, b2=base.b2, eps=base.eps)
    return optax.sgd(base.lr)


@dataclass(frozen=True)
class DRTRRL:
    """The two update-scale arms written against the original clip's threshold.

    ``c`` is that threshold, and it is not a learning rate. The original rule
    bounds an update by ``c``; these two arms are what is left of it when one
    of the two magnitudes feeding that bound is taken away, so they are only a
    control for it while they are written over the same number. Selecting ``c``
    independently would make them a different optimizer being compared against
    the original rather than a decomposition of it.

    ``magnitude`` is which arm:

    ``sign``
        ``c * sign(delta) * z / (||z|| + eps)`` -- the fixed-step saturated
        limit. Both magnitudes are gone: the trace is exported as a unit
        vector and the TD error is reduced to its sign, so the step is ``c``
        long on every step. This is the original clip with the clip always
        active, which is what "saturated limit" means, and it is the default.
    ``td_out``
        ``delta * z * min(1, c / ||z||)`` -- the trace's magnitude is bounded
        by ``c`` and the TD error's is left in. Note this is a *clip*, not a
        normalization: a trace already shorter than ``c`` passes through
        untouched, where ``sign`` would have stretched it to ``c``.

        **Unsafe by construction.** With the step still proportional to the
        raw TD error the critic closes a positive feedback loop -- an
        overshoot grows the next ``|delta|``, which grows the next step. It is
        a control arm, run under a safety horizon, and its divergence is the
        result rather than a failure.

    Between them the two arms are a decomposition: the original clips the
    product ``delta * z``, ``sign`` removes both magnitudes, and ``td_out``
    removes only the trace's. What is not offered is the fourth cell, a
    normalized trace with ``|delta|`` left in, because no arm names it.

    ``scope`` is what gets normalized or clipped as one unit:

    ``block``
        Each top-level entry of the tree handed in, separately. Which entries
        exist is the caller's grouping decision -- for RTRRL a group's entries
        are its blocks, which is what the name is for.
    ``group``
        The whole tree at once. Two blocks under one norm share a single unit
        ball, so the one with the larger trace takes nearly all of the step;
        that is an ablation, not a default.

        **Inert under RTRRL as it now groups.** RTRRL used to step its two
        readouts together, which was the one place a group held two blocks;
        since the actor and the critic became separate parameter groups every
        group RTRRL hands this rule holds exactly one block, so the two scopes
        cut the same unit. The setting stays because the rule is not RTRRL's --
        a caller that does group several blocks still gets two behaviours --
        and because a recorded run that named it should keep meaning what it
        meant. An RTRRL run selecting ``group`` today selects ``block``.

    ``denominator`` is where ``eps`` sits, which is two rules rather than two
    spellings and is declared rather than chosen at the point of division:

    ``shifted``
        ``z / (||z|| + eps)``, the preregistered form. The default.
    ``clamped``
        ``z / max(||z||, eps)``.

    Both are finite at a trace of exactly zero, and both give exactly zero
    there -- ``0 / (0 + eps)`` is ``0`` just as ``0 / eps`` is. What separates
    them is the approach: at ``||z|| == eps`` the shifted form has already
    halved the step while the clamped form is still taking the whole of it,
    and they agree only once ``||z||`` clears ``eps`` by enough for the shift
    not to matter. So a run answers to one of them and substituting the other
    is a change of algorithm.

    ``denominator`` reaches ``sign`` only. ``td_out`` divides by nothing --
    it scales by ``min(1, c / ||z||)`` -- so there is no denominator for
    ``eps`` to sit in and the setting is inert there.
    """

    c: float = param(valid=(1e-9, 100.0), search=(1e-4, 10.0), log=True)
    magnitude: str = param(valid=["sign", "td_out"], search=["sign"], default="sign")
    scope: str = param(valid=["block", "group"], search=["block"], default="block")
    denominator: str = param(
        valid=["shifted", "clamped"], search=["shifted"], default="shifted"
    )
    eps: float = param(valid=(1e-12, 1e-2), search=[1e-8], default=1e-8, log=True)


BOUND_BRANCHES = {
    "none": (),
    "ob": ObBound,
    "adaptive_ob": AdaptiveObBound,
    "adaptive_ob_fixed": AdaptiveObBoundFixed,
}

BASE_BRANCHES = {"sgd": Sgd, "adam": Adam}

STEP_BRANCHES = {
    "sgd": Sgd,
    "adam": Adam,
    "d_rtrrl": DRTRRL,
    "iu": IntentionalUpdate,
}


def _selected_parameters(selection, builder):
    del builder
    return selection.parameters


BOUND_FAMILY = ComponentFamily(
    branches=BOUND_BRANCHES,
    construct=_selected_parameters,
)
BASE_FAMILY = ComponentFamily(
    branches=BASE_BRANCHES,
    construct=_selected_parameters,
)
STEP_FAMILY = ComponentFamily(
    branches=STEP_BRANCHES,
    construct=_selected_parameters,
)

# The three bounds that are an ObGD, which is every branch of the bound family
# except the one that says there is no bound. ``none`` there names the base
# alone, and a base alone is SGD or Adam rather than a bounded step; a rule
# selected by the name ``obgd`` and then handed no bound would be a rule whose
# name said nothing about what it does.
OBGD_BOUND_FAMILY = BOUND_FAMILY.restricted("ob", "adaptive_ob", "adaptive_ob_fixed")


@dataclass(frozen=True)
class ObGD:
    """The overshooting-bounded step, declared where one rule is selected.

    :func:`make_bounded_rule` already takes a bound and a base, and StreamAC
    selects the two side by side because it wants the whole cross product --
    including ``none``, which is the base alone. An algorithm choosing *this
    rule against others* is asking a narrower question: it wants the published
    ObGD, which is a bound written over a plain rate. So the base is a rate
    here rather than a second choice, and the bound is the one axis left.

    ``lr`` is that rate, and it is the same number ``Sgd.lr`` carries: the
    bound shrinks it whenever a step would cross the TD target, and leaves it
    alone otherwise. Which bound -- ``ob``, ``adaptive_ob`` or the published
    ``adaptive_ob_fixed`` -- stays a branch of its own, because the three are
    different denominators and a run has to be able to say which it answered
    to.
    """

    bound: str = structure(branches=OBGD_BOUND_FAMILY.branches)
    lr: float = param(valid=(1e-9, 10.0), search=(1e-5, 1.0), log=True)


@dataclass(frozen=True)
class ObGDStep:
    """One ``obgd`` selection with its bound built rather than named.

    A declaration nested inside a branch is read back as the name it selected
    and not as what that name declares -- :func:`memorax.parameters.read_branch`
    fills in only the branch a group chose, and a group inside a branch is a
    second such choice. So the bound is resolved through the component builder,
    which does descend, and what an algorithm receives is this: the bound
    itself, and the rate it is written over.
    """

    bound: Any
    lr: float


def make_bounded_rule(*, bound, base) -> UpdateRule:
    """Step along delta * trace under an overshooting bound on the step size.

    Two axes rather than one name. The bound shrinks the step whenever
    ``|delta| * ||z||_1 * lr * kappa`` exceeds one, which keeps a single update
    from crossing the TD target; the base is what it is written over. One name
    for the pair meant nothing could ask for a different bound without also
    changing the base, or the other way round.

    ``bound=None``
        No bound: the base alone, which is what an experiment asks for by
        naming ``optimizer_bound: none``.
    ``ObBound``
        No second moment and no eps at all, as published.
    ``AdaptiveObBound``
        Divides by ``sqrt(v_hat) + eps``, which is what memorax and everything
        forked from it compute. Kept exactly as it is because the recorded runs
        answer to it.
    ``AdaptiveObBoundFixed``
        Divides by ``sqrt(v_hat + eps)``, which is what AdaptiveObGD as
        published computes. The difference is not a rounding. While the second
        moment is near eps the two denominators stand a factor apart, and the
        step with it: measured against the published optimiser, this bound and
        the one above are about a factor of two apart there. They agree again
        once the second moment clears eps, where both denominators are
        ``sqrt(v_hat)``.
    """

    if bound is None:
        return make_optax_rule(base_transform(base), rate=base.lr)
    if not isinstance(base, Sgd):
        raise ValueError(
            "the overshooting bound is written over a plain rate; putting it "
            f"over {type(base).__name__.lower()} is a different rule and none "
            "is published, so it is refused rather than guessed at"
        )
    learning_rate = base.lr
    kappa = bound.kappa
    adaptive = isinstance(bound, AdaptiveObBound)
    beta2 = getattr(bound, "beta2", 0.0)
    eps = getattr(bound, "eps", 1e-8)
    eps_inside_root = isinstance(bound, AdaptiveObBoundFixed)

    def denominator(second):
        if eps_inside_root:
            return jnp.sqrt(second + eps)
        return jnp.sqrt(second) + eps

    def init(*, params, traces):
        del params
        return jax.tree.map(jnp.zeros_like, traces)

    def apply(traced, direct, state, *, delta, derivative, step, params):
        del params, derivative
        # An unbounded second moment is only carried where something reads it.
        # Under the plain bound the step never does, and accumulating one there
        # meant the rule needed a decay rate that changed nothing.
        moment = (
            jax.tree.map(
                lambda old, trace: (
                    beta2 * old
                    + (1 - beta2) * jnp.square(_broadcast_env(delta, trace) * trace)
                ),
                state,
                traced,
            )
            if adaptive
            else state
        )

        if adaptive:
            corrected = jax.tree.map(lambda leaf: leaf / (1.0 - beta2**step), moment)
            normalized = jax.tree.map(
                lambda trace, second: jnp.abs(trace) / denominator(second),
                traced,
                corrected,
            )
        else:
            corrected = None
            normalized = jax.tree.map(jnp.abs, traced)

        trace_sum = sum(
            jnp.sum(leaf, axis=tuple(range(1, leaf.ndim)))
            for leaf in jax.tree.leaves(normalized)
        )
        delta_bar = jnp.maximum(jnp.abs(delta), 1.0)
        pressure = delta_bar * trace_sum * learning_rate * kappa
        step_size = learning_rate / jnp.maximum(1.0, pressure)

        # The bounded step size reaches the TD error first, and their product
        # multiplies the trace. That is the order StreamAC multiplies in, and
        # multiplying in any other order is the same number differently
        # rounded, which would leave this rule permanently a last bit away from
        # the implementation it has to answer to. So the combined direction
        # ``_combine`` builds for optax is not reused here.
        ascent = jax.tree.map(
            lambda trace: (
                (_broadcast_env(step_size, trace) * _broadcast_env(delta, trace))
                * trace
            ),
            traced,
        )
        if direct is not None:
            ascent = jax.tree.map(
                lambda leaf, immediate: (
                    leaf + _broadcast_env(step_size, immediate) * immediate
                ),
                ascent,
                direct,
            )
        if adaptive:
            ascent = jax.tree.map(
                lambda leaf, second: leaf / denominator(second),
                ascent,
                corrected,
            )
        updates = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), ascent)
        return RuleOutput(
            updates=updates,
            state=moment,
            metrics={
                "step_size": step_size,
                # The four the step size is a quotient of, so that a small rate
                # can be attributed rather than only observed. See ObGDReading.
                "obgd": ObGDReading(
                    trace_sum=trace_sum,
                    delta_bar=delta_bar,
                    bound_denominator=pressure,
                    bound_scale=step_size / learning_rate,
                    second_moment_rms=(
                        _stream_mean(jax.tree.map(denominator, corrected))
                        if adaptive
                        else None
                    ),
                ),
            },
        )

    return UpdateRule(init=init, apply=apply)


def _stream_mean(tree):
    """One mean per stream over every element of every leaf.

    The same reduction :mod:`memorax.rl.intentional` takes of its
    preconditioner, and for the same reason: a per-parameter statistic is not
    a series, and what a reader can compare between two runs is its scale.
    """

    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    total = sum(jnp.sum(leaf.reshape(leaf.shape[0], -1), axis=1) for leaf in leaves)
    count = sum(int(leaf.size // leaf.shape[0]) for leaf in leaves)
    return total / max(count, 1)


def _stream_norm(tree):
    """One Euclidean norm per stream over every leaf of one normalization unit.

    The env axis is axis 0 of every trace leaf, so the sum runs over everything
    but it and a unit is a whole subtree rather than a leaf: normalizing leaf by
    leaf would keep only each leaf's sign pattern and throw the direction away,
    which is a different algorithm.
    """

    squares = sum(
        jnp.sum(jnp.square(leaf), axis=tuple(range(1, leaf.ndim)))
        for leaf in jax.tree.leaves(tree)
    )
    return jnp.sqrt(squares)


def _units(tree, *, by_block):
    """The normalization units of one tree, and how to put them back together.

    A block-scoped group goes back together as a plain mapping rather than as
    whatever mapping arrived. What a unit is, and that each is scaled on its
    own, is the whole of this split; the container the parts are handed back in
    is read by ``jax.tree.map`` and by name, and neither asks for more than a
    dict.
    """

    if by_block and isinstance(tree, Mapping):
        return dict(tree), dict
    return {None: tree}, lambda parts: parts[None]


def make_d_rtrrl_rule(step, *, clip=0.0) -> UpdateRule:
    """Take the original clip's threshold and remove a magnitude from it.

    Both arms are written over ``c``, the threshold the original rule clips an
    update to. ``sign`` removes the trace's magnitude and the TD error's, and
    takes a step of exactly ``c``; ``td_out`` removes only the trace's, and
    takes a step of ``c * |delta|`` whenever the trace is long enough for the
    clip to bite. See :class:`DRTRRL` for what each arm is a control for.

    The trace itself is untouched by either. It accumulates at full magnitude
    and the torso's direction is synthesized from the raw upstream vectors;
    the normalizing or the clipping happens here, on the way out, on a value
    nothing carries forward.

    ``clip`` is a different clip: the original bound on the *finished* group
    update, which RTRRL declares as ``torso.grad_clip``. It is worth keeping
    even under ``sign``, where the traced part is bounded by construction,
    because the direct directions do not pass through the trace and are the
    only part of either arm that can be any length at all.

    Setting it equal to ``c`` is a coherent thing to ask for and not the same
    experiment: under ``sign`` the traced part is already ``c`` long, so the
    outer bound binds on essentially every step and the arm stops being the
    saturated limit of anything. Nothing here refuses that -- which of the two
    clips an arm is written against is the experiment's to declare.
    """

    c = step.c
    eps = step.eps
    saturated = step.magnitude == "sign"
    by_block = step.scope == "block"
    shifted = step.denominator == "shifted"
    bound = None
    if clip:
        import optax

        bound = optax.clip_by_global_norm(clip)

    def factor(tree):
        """What one unit's trace is multiplied by on the way out, per stream."""

        length = _stream_norm(tree)
        if saturated:
            # A unit vector times the threshold: every step is `c` long.
            return c / (length + eps if shifted else jnp.maximum(length, eps))
        # A clip, not a normalization. A trace shorter than `c` is passed
        # through untouched rather than stretched up to it, which is the whole
        # difference between this arm and the one above. `maximum` guards the
        # division only; at a trace of zero the ratio is huge, the minimum
        # picks one, and one times zero is zero.
        return jnp.minimum(1.0, c / jnp.maximum(length, eps))

    def init(*, params, traces):
        del params, traces
        return ()

    def apply(traced, direct, state, *, delta, derivative, step, params):
        del step, params, derivative
        # `jnp.sign` takes a TD error of exactly zero to a step of exactly
        # zero, which is what a rule that moves a fixed distance every step
        # otherwise would not do.
        surprise = jnp.sign(delta) if saturated else delta

        units, rebuild = _units(traced, by_block=by_block)
        ascent = rebuild(
            {
                name: jax.tree.map(
                    lambda leaf, scale=factor(part): (
                        _broadcast_env(surprise * scale, leaf) * leaf
                    ),
                    part,
                )
                for name, part in units.items()
            }
        )
        # The direct directions are added as they arrive, in both arms and as
        # the original adds them. They carry their own coefficient already --
        # RTRRL's `entropy_rate` -- and putting `c` on top of it would make
        # the entropy term mean something different in each arm, which is the
        # one thing a control comparison cannot afford.
        if direct is not None:
            ascent = jax.tree.map(
                lambda leaf, immediate: leaf + immediate, ascent, direct
            )
        updates = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), ascent)
        if bound is not None:
            updates, _ = bound.update(updates, ())
        # The threshold, under the name every rule reports its step scale by,
        # rather than the distance this step happened to move: one name means
        # the same thing whichever rule is in the way. Under `sign` the traced
        # part moved exactly this, or zero. Under `td_out` it moved this times
        # `|delta|`, or less when the trace was too short to clip.
        return RuleOutput(
            updates=updates,
            state=state,
            metrics={"step_size": jnp.full_like(delta, c)},
        )

    return UpdateRule(init=init, apply=apply)


def make_intentional_rule(
    step: IntentionalUpdate,
    *,
    signals: Mapping[str, str],
    decays: Mapping[str, float],
    streams: int,
) -> UpdateRule:
    """One intentional optimizer per block of a group, each with its own state.

    A group's blocks share the settings this names and nothing else. Each
    carries its own second moments, clipping statistic and dynamic step size,
    because the whole point of the step size is that it is measured against the
    parameters it moves.

    ``signals`` says which scalar each block's step is proportional to and
    ``decays`` says the ``gamma * lambda`` its trace runs at -- the rate
    ``sigma_bar`` averages at, which must be the rate the algorithm's trace
    component forgets at. Both are the algorithm's to declare.

    ``streams`` is the parallel environment count. The other rules read it off
    the trace tree they are handed; this one allocates per-stream statistics
    before any trace has arrived, so it is told.
    """

    optimizers = {
        name: IntentionalOptimizer(step, decay=decays[name], signal=signal)
        for name, signal in signals.items()
    }

    def init(*, params, traces):
        del traces
        return {
            name: optimizer.init(params[name], streams=streams)
            for name, optimizer in optimizers.items()
        }

    def apply(traced, direct, state, *, delta, derivative, step, params):
        # ``step`` here is the transition counter every bias correction reads,
        # and not the settings of the same name outside -- the optimizers were
        # built from those already, which is the only reason the shadowing is
        # harmless.
        taken = {
            name: optimizer.update(
                delta=delta,
                trace=traced[name],
                derivative=derivative[name],
                direct=None if direct is None else direct.get(name),
                step=step,
                params=params[name],
                state=state[name],
            )
            for name, optimizer in optimizers.items()
        }
        readings = {name: reading for name, (_, _, reading) in taken.items()}
        return RuleOutput(
            updates={name: updates for name, (updates, _, _) in taken.items()},
            state={name: carried for name, (_, carried, _) in taken.items()},
            metrics={
                "intentional": readings,
                # Under the one name every rule reports its step scale by. This
                # rule's is dynamic and per block, which is why the name is a
                # block's rather than a group's.
                "step_size": {
                    name: reading.step_size for name, reading in readings.items()
                },
            },
        )

    return UpdateRule(init=init, apply=apply)
