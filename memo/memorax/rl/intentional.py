"""The Intentional Update optimizer, written as its paper defines it.

    paper           https://arxiv.org/abs/2604.19033
    implementation  https://github.com/sharifnassab/Intentional_RL

An intentional update sizes a step by what the step is supposed to spend:

    nu      a diagonal second moment of the instantaneous derivative
    rho     its inverse root, the preconditioner every term is read through
    sigma   the statistic <rho p, p>, averaged into sigma_bar at the trace's
            own decay
    m       a first moment of the eligibility trace, at beta_momentum
    alpha   eta / max(sqrt(sigma_bar <rho m, m>), floor)

and one transition's update is ``alpha * signal * rho * m``, where the signal
is a clipped TD error for a value function and an advantage normalized by its
own running scale for a policy.

``m`` is Adam's first moment moved onto the direction this optimizer steps
along, and at ``beta_momentum = 0`` it *is* the trace, to the last bit: the
published update is what this reduces to rather than what it approximates. See
:func:`moment_average` and :class:`IntentionalUpdate`.

**The trace is not this optimizer's.** It arrives already accumulated, from
:mod:`memorax.rl.traces`, and so does the instantaneous derivative it was
accumulated from. Both are needed and they are needed for different things:
``nu`` and ``sigma`` are statistics of the *derivative*, while the step size
and the step itself read the *trace*. What this optimizer carries is only what
no one else could carry for it -- two running second moments, a clipping
statistic and, for a policy, an advantage scale.

The trace's decay still arrives here, because ``sigma_bar`` averages at it.
That is a number the two components have to agree on, not a second trace: an
algorithm constructs the trace and the optimizer from one ``gamma * lambda``.

Every quantity is per stream: the leading axis of a derivative leaf is the
parallel environment axis, and each stream carries its own second moment,
statistic and step size. Only the finished update is averaged across them,
which is where the caller's parameters -- which have no stream axis -- wait.

**One thing here is not the paper's**, and it is at the bottom of the file
behind a heading that says so: :class:`JointIntentionalOptimizer`, the step for
a block that *two* objectives credit. The paper has no such block -- its
actor-critic is two separate networks -- so this is derived from Eq. 12 rather
than quoted from it, and it reduces to Eq. 12 when one of the two objectives
is removed. Everything above the heading is the published optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from memorax.parameters import param

from .interaction import broadcast_stream

TD = "td"
ADVANTAGE = "advantage"
SIGNALS: tuple[str, ...] = (TD, ADVANTAGE)

# The published implementation divides the clipped advantage by its running
# scale under this floor rather than guarding the zero it starts at. It is not
# a setting there and it is not one here: at every scale a run reaches, it is
# the same arithmetic as dividing.
SCALE_FLOOR = 1e-12


@dataclass(frozen=True)
class IntentionalUpdate:
    """What an intentional update reads, and what the published values are.

    ``eta`` is the intended fractional reduction: the share of the outcome one
    step sets out to spend. It is the only quantity here an experiment is
    expected to search over, and it is *not* one number for an agent -- the
    published actor-critic uses ``0.05`` for the policy and ``0.5`` for the
    value function, and sweeps them separately. Anything selecting one ``eta``
    for both is not that configuration.

    ``clip`` is the paper's ``C``, the number of root-mean-square TD errors a
    single TD error may be before it is cut back to that many. A multiplier on
    a running scale rather than an absolute threshold, so it means the same
    thing whatever the reward scale is.

    ``denominator_floor`` is the published implementation's ``max(u, 1e-8)``.
    The paper's Eq. 12 writes the step size as a plain quotient; the
    implementation every reported result came from divides by the floored
    denominator, which bounds one step at ``eta / floor`` instead of letting a
    vanishing denominator take an unbounded one. For an optimizer whose whole
    subject is streaming stability that safeguard is part of the algorithm, so
    it is declared rather than dropped -- and declared, so that a run answering
    to the unfloored equation can ask for a smaller one and say it did.

    ``eps`` sits beside the root -- ``(sqrt(nu) + eps)`` -- which is where both
    the paper and the implementation put it.

    ``beta_momentum`` is not the paper's. It is Adam's ``beta1``, written over
    the eligibility trace: what this optimizer steps along is the trace, which
    is the place a gradient method keeps its gradient, so smoothing it is the
    same operation Adam performs and takes the same coefficient. The smoothed
    trace replaces the raw one in *both* places the raw one is read -- the
    direction the step is taken along, and the quadratic its step size divides
    by -- because a step size derived against one direction and then spent on
    another is not the size of anything.

    Zero is the default and it is an exact zero, not a small one: the moment is
    ``0 * m + 1 * z`` and its bias correction divides by ``1 - 0``, so at zero
    every operation returns the trace itself and this is the published
    optimizer bit for bit. A run that leaves it there is the paper's; a run
    that raises it is a different optimizer, and says so by the number it
    carries.
    """

    eta: float = param(valid=(1e-9, 100.0), search=(1e-4, 1.0), log=True)
    clip: float = param(valid=(0.0, 1e4), search=[20.0], default=20.0)
    beta_rms: float = param(valid=(0.0, 1.0), search=[0.999], default=0.999)
    beta_clip: float = param(valid=(0.0, 1.0), search=[0.9998], default=0.9998)
    beta_advantage: float = param(valid=(0.0, 1.0), search=[0.9998], default=0.9998)
    beta_momentum: float = param(valid=(0.0, 1.0), search=[0.0], default=0.0)
    denominator_floor: float = param(
        valid=(0.0, 1.0), search=[1e-8], default=1e-8, log=True
    )
    eps: float = param(valid=(1e-12, 1e-2), search=[1e-8], default=1e-8, log=True)


class IntentionalState(struct.PyTreeNode):
    """One parameter group's intentional state, all of it per stream.

    No eligibility trace: that is the algorithm's, and it arrives at each step
    from the trace component that owns it. ``momentum`` is not one either -- it
    is a running average *of* the trace that arrives, kept here because nothing
    outside reads it and because at ``beta_momentum = 0`` it is that trace
    unchanged.

    ``rho`` and ``alpha`` are not here either, being functions of what is --
    the preconditioner of ``nu``, the step size of ``sigma_bar`` and this
    step's trace. Carrying them would be a second copy of a derived number that
    a resumption could disagree with; they are computed where they are used and
    reported as readings.

    ``advantage_scale`` is None for a group whose signal is a TD error: there
    is no advantage to normalize there, and carrying a zero would say there was
    one that happened to be zero.
    """

    nu: Any
    momentum: Any
    sigma_bar: Any
    delta_square: Any
    advantage_scale: Any = None


class IntentionalReading(struct.PyTreeNode):
    """Every quantity one intentional step passed through, per stream.

    They are reported rather than derived afterwards because most of them
    cannot be recovered from the update: the step size and the signal reach the
    parameters only as their product, and ``sigma_bar`` not at all.
    """

    clipped_delta: Any = None
    signal: Any = None
    advantage_scale: Any = None
    rms_scale: Any = None
    sigma_bar: Any = None
    # `<rho m, m>`, measured on the direction the step was actually taken
    # along. At `beta_momentum = 0` that direction is the trace and this is the
    # quantity the name has always meant; above zero it is the smoothed trace,
    # which is what the step size divided by and so what is worth reading.
    trace_quadratic: Any = None
    denominator: Any = None
    step_size: Any = None
    update_norm: Any = None
    non_finite: Any = None


def corrected_average(previous, sample, *, beta, step):
    """A ``beta``-weighted running mean with its bias correction folded in.

    The paper writes every one of its averages this way::

        x_t = x_{t-1} + (1 - beta) / (1 - beta^t) * (sample - x_{t-1})

    which is the same sequence as the published implementation's -- a plain
    exponential average kept, and divided by ``1 - beta^t`` where it is read.
    The two are equal term by term, not approximately: this recursion is what
    the corrected value satisfies. Keeping the corrected one is what makes the
    first sample arrive whole rather than at ``(1 - beta)`` of itself, with
    hundreds of steps spent climbing out of a zero nobody measured.

    A ``beta`` of exactly one names an average that never moves. The rate is
    ``0/0`` there, and the limit is the reading taken: nothing is averaged in.
    """

    remaining = 1.0 - beta**step
    moving = remaining > 0
    rate = jnp.where(moving, (1.0 - beta) / jnp.where(moving, remaining, 1.0), 0.0)
    return previous + rate * (sample - previous)


def moment_average(previous, sample, *, beta, step):
    """Adam's first moment over a whole tree, corrected where it is read.

    The other averages here are kept in their corrected form, which is what the
    paper writes and what makes a first sample arrive whole. This one is kept
    the way Adam keeps it -- ``beta * previous + (1 - beta) * sample``, divided
    by ``1 - beta**step`` on the way out -- for one reason: at ``beta = 0`` that
    recursion is ``0 * previous + 1 * sample`` and the correction is a division
    by exactly one, so what comes back is the sample with no rounding anywhere.
    The corrected recursion would compute ``previous + (sample - previous)``,
    which is the same number in exact arithmetic and is not the same float, and
    the whole claim about ``beta_momentum = 0`` is that there is no difference
    to find.

    Both the moment to carry and the moment to read are returned, because the
    correction belongs to the reading and carrying a corrected moment would
    make the next step's ``beta * previous`` a factor off.

    A ``beta`` of exactly one names a moment that never moves. Its correction is
    zero, the division is guarded, and what comes back is the moment as it
    stands -- the zeros it was allocated with.
    """

    moving = jax.tree.map(
        lambda old, leaf: beta * old + (1.0 - beta) * leaf, previous, sample
    )
    correction = 1.0 - beta**step
    divisor = jnp.where(correction > 0, correction, 1.0)
    return moving, jax.tree.map(lambda leaf: leaf / divisor, moving)


def clipped_td_error(delta, mean_square, *, beta, clip, step):
    """The TD error cut back to ``clip`` times its own running RMS.

    The scale is measured on the same quantity being clipped, so this is a
    dimensionless bound: doubling every reward doubles ``delta``, doubles
    ``sqrt(mean_square)``, and clips at exactly the same place. The sign
    survives, which is what makes the clipped error still a direction.
    """

    mean_square = corrected_average(
        mean_square, jnp.square(delta), beta=beta, step=step
    )
    bound = clip * jnp.sqrt(mean_square)
    return jnp.sign(delta) * jnp.minimum(jnp.abs(delta), bound), mean_square


def normalized_advantage(advantage, scale, *, beta, step):
    """The advantage divided by the running mean of its own magnitude.

    The mean is of ``|A|`` and is therefore never negative, so the normalized
    advantage keeps the sign of the advantage exactly. Before anything has been
    averaged the scale is zero, and the division is floored rather than
    guarded, which is what the published implementation does and gives the same
    zero at the only place the two could differ.
    """

    scale = corrected_average(scale, jnp.abs(advantage), beta=beta, step=step)
    return advantage / jnp.maximum(scale, SCALE_FLOOR), scale


def _stream_sum(tree):
    """One sum per stream over every element of every leaf."""

    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return sum(jnp.sum(leaf.reshape(leaf.shape[0], -1), axis=1) for leaf in leaves)


def _stream_scaled(tree, values):
    """One value per stream multiplied through a streamed tree."""

    return jax.tree.map(lambda leaf: broadcast_stream(values, leaf) * leaf, tree)


def _stream_count(tree) -> int:
    """How many parameters one stream of a streamed tree holds."""

    return sum(int(leaf.size // leaf.shape[0]) for leaf in jax.tree.leaves(tree))


def _stream_non_finite(tree):
    """One indicator per stream: was anything in this stream not a number.

    Reported rather than repaired. An intentional step divides by two running
    statistics, and the run that has to be read afterwards is the one where
    they went to zero -- a quietly zeroed update would leave the parameters
    where they were and say nothing about why.
    """

    finite = jnp.stack(
        [
            jnp.all(jnp.isfinite(leaf.reshape(leaf.shape[0], -1)), axis=1)
            for leaf in jax.tree.leaves(tree)
        ]
    ).all(axis=0)
    return 1.0 - finite.astype(jnp.float32)


class IntentionalOptimizer:
    """One parameter group's intentional update, and the state it carries.

    ``decay`` is ``gamma * lambda`` for the group: the rate ``sigma_bar``
    averages at, which has to be the rate the group's trace forgets at.
    ``signal`` is which scalar the step is proportional to -- the clipped TD
    error of Intentional TD, or the normalized advantage of the intentional
    policy gradient -- and it is passed in rather than declared because it
    follows from what the group's objective is, which only the algorithm knows.
    """

    def __init__(
        self, settings: IntentionalUpdate, *, decay: float, signal: str = TD
    ) -> None:
        if signal not in SIGNALS:
            raise ValueError(
                f"{signal!r} is not one of the signals an intentional update "
                f"steps along ({', '.join(SIGNALS)})"
            )
        self.settings = settings
        self.decay = decay
        self.signal = signal

    def init(self, params, *, streams: int) -> IntentionalState:
        """Fresh state: empty averages, one set per stream."""

        empty = jax.tree.map(
            lambda leaf: jnp.zeros((streams, *leaf.shape), dtype=jnp.float32), params
        )
        zero = jnp.zeros((streams,), dtype=jnp.float32)
        return IntentionalState(
            nu=empty,
            momentum=jax.tree.map(jnp.zeros_like, empty),
            sigma_bar=zero,
            delta_square=zero,
            advantage_scale=None if self.signal == TD else zero,
        )

    def _signal(self, delta, state, *, step):
        """What the step is proportional to, and the statistics that shaped it."""

        settings = self.settings
        clipped, delta_square = clipped_td_error(
            delta,
            state.delta_square,
            beta=settings.beta_clip,
            clip=settings.clip,
            step=step,
        )
        if self.signal == TD:
            return clipped, clipped, delta_square, None
        signal, scale = normalized_advantage(
            clipped, state.advantage_scale, beta=settings.beta_advantage, step=step
        )
        return signal, clipped, delta_square, scale

    def update(
        self,
        *,
        delta,
        trace,
        derivative,
        direct,
        step,
        params,
        state: IntentionalState,
    ) -> tuple[Any, IntentionalState, IntentionalReading]:
        """One transition: precondition, size the step, and take it.

        ``trace`` is the group's eligibility trace as its own component left
        it, and ``derivative`` is the instantaneous derivative that trace was
        accumulated from. Both are read and neither is written.

        ``direct`` is refused. The paper's update has no untraced term: its
        policy gradient is the derivative of the log-probability *and* the
        entropy together, signed by the signal, and an algorithm pairing this
        optimizer with an entropy term applied outside the trace would be
        stepping a rule the paper does not define. Where that folding happens
        is the algorithm's business -- it is the thing that owns the objective
        -- but handing an intentional group a direction it never traced is a
        wiring mistake rather than a configuration, so it is refused here.

        ``params`` is accepted and unused. An intentional step reads the
        derivative's scale and never the parameters', which is a property worth
        being able to see in the signature rather than in the prose.
        """

        del params
        if direct is not None:
            raise ValueError(
                "an intentional update has no untraced term; the entropy "
                "direction belongs in the derivative the trace accumulates, "
                "signed by the signal, before either reaches this optimizer"
            )
        settings = self.settings
        signal, clipped, delta_square, scale = self._signal(delta, state, step=step)

        nu = jax.tree.map(
            lambda old, leaf: corrected_average(
                old, jnp.square(leaf), beta=settings.beta_rms, step=step
            ),
            state.nu,
            derivative,
        )
        rho = jax.tree.map(lambda leaf: 1.0 / (jnp.sqrt(leaf) + settings.eps), nu)
        sigma = _stream_sum(
            jax.tree.map(
                lambda inverse, leaf: inverse * jnp.square(leaf), rho, derivative
            )
        )
        sigma_bar = corrected_average(
            state.sigma_bar, sigma, beta=self.decay, step=step
        )

        # The direction this step is taken along. At `beta_momentum = 0` it is
        # `trace` and every line below reads the trace exactly as it always
        # did; above zero it is the trace smoothed, and the step size is
        # measured on it rather than on the raw trace, because a step size is
        # how far to go along *this* direction.
        carried, momentum = moment_average(
            state.momentum, trace, beta=settings.beta_momentum, step=step
        )

        quadratic = _stream_sum(
            jax.tree.map(
                lambda inverse, leaf: inverse * jnp.square(leaf), rho, momentum
            )
        )
        # `sqrt(a) * sqrt(b)` rather than `sqrt(a * b)`, which is the same
        # number for two non-negative factors and is representable over a far
        # wider range of them. Both are sums of squares over every parameter in
        # a stream, so in float32 their product leaves the range from either
        # end well before either factor does: it overflows above 3.4e38 and
        # flushes to zero below 1e-38, and the second is the dangerous one,
        # since a zero denominator hands the step the whole of `eta / floor`.
        denominator = jnp.sqrt(sigma_bar) * jnp.sqrt(quadratic)
        alpha = settings.eta / jnp.maximum(denominator, settings.denominator_floor)

        ascent = _stream_scaled(
            jax.tree.map(lambda inverse, leaf: inverse * leaf, rho, momentum),
            alpha * signal,
        )
        updates = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), ascent)
        return (
            updates,
            IntentionalState(
                nu=nu,
                momentum=carried,
                sigma_bar=sigma_bar,
                delta_square=delta_square,
                advantage_scale=scale,
            ),
            IntentionalReading(
                clipped_delta=clipped,
                signal=signal,
                advantage_scale=scale,
                rms_scale=_stream_sum(rho) / max(_stream_count(rho), 1),
                sigma_bar=sigma_bar,
                trace_quadratic=quadratic,
                denominator=denominator,
                step_size=alpha,
                update_norm=jnp.sqrt(_stream_sum(jax.tree.map(jnp.square, ascent))),
                non_finite=_stream_non_finite(ascent),
            ),
        )


# ------------------------------------------- two objectives, one shared block
#
# What follows is not in the paper, and the reason it is not is that the paper
# never needs it: its actor-critic is two separate networks, so every parameter
# is credited by exactly one objective and Eq. 12 answers for it whole. A
# shared recurrent torso is credited by two, and one transition can write only
# one `Delta_theta`, so *something* has to be decided that the paper does not
# decide. This is that decision, derived from Eq. 12 rather than bolted beside
# it.


@dataclass(frozen=True)
class JointIntentional:
    """One intentional step over a block that two objectives credit.

    ``eta_actor`` and ``eta_critic`` are the paper's ``eta`` twice, and they
    keep the paper's meaning exactly: the fraction of its own objective's
    outcome that objective's credit sets out to spend. What changes is only
    that one step now has to honour both at once, which it does by taking the
    largest step that violates neither -- see
    :class:`JointIntentionalOptimizer`. The published pair, ``0.05`` for a
    policy and ``0.5`` for a value function, means here what it means there.

    Everything else is one setting rather than two, and each for a reason
    rather than for tidiness:

    ``clip`` and ``beta_clip``
        There is one TD error on a transition. Both branches' signals are cut
        from it -- the critic's is the clipped error and the actor's is that
        error normalized -- so there is one clipping statistic to keep.

    ``beta_advantage``
        The normalized advantage exists on the actor's branch alone.

    ``beta_rms``, ``eps``
        There is one ``nu``, and so one ``rho``. It is a second moment of the
        *summed* derivative, because ``rho`` is the only quantity in an
        intentional step that belongs to the parameters rather than to an
        objective: Eq. 12's Cauchy-Schwarz holds for any positive diagonal
        common to both of its factors, and what the paper asks of ``rho`` is
        that it normalize the magnitude of an *entry*. Split per objective it
        stops doing that -- an entry one head never touches has a second
        moment that decays at ``beta_rms`` with nothing to refresh it, and
        ``1 / sqrt(nu)`` reads "this objective does not use this entry" as
        "this entry is finely scaled". See issue 87.

    ``beta_momentum``, ``denominator_floor``
        Properties of the one direction and the one step size.
    """

    eta_actor: float = param(valid=(1e-9, 100.0), search=(1e-4, 1.0), log=True)
    eta_critic: float = param(valid=(1e-9, 100.0), search=(1e-4, 1.0), log=True)
    clip: float = param(valid=(0.0, 1e4), search=[20.0], default=20.0)
    beta_rms: float = param(valid=(0.0, 1.0), search=[0.999], default=0.999)
    beta_clip: float = param(valid=(0.0, 1.0), search=[0.9998], default=0.9998)
    beta_advantage: float = param(valid=(0.0, 1.0), search=[0.9998], default=0.9998)
    beta_momentum: float = param(valid=(0.0, 1.0), search=[0.0], default=0.0)
    denominator_floor: float = param(
        valid=(0.0, 1.0), search=[1e-8], default=1e-8, log=True
    )
    eps: float = param(valid=(1e-12, 1e-2), search=[1e-8], default=1e-8, log=True)


class JointIntentionalState(struct.PyTreeNode):
    """One shared block's joint intentional state, all of it per stream.

    ``nu`` is one tree because ``rho`` is one metric. ``momentum`` and
    ``sigma_bar`` are one per branch, because a trace and the statistic of the
    gradient it accumulated are an objective's. ``delta_square`` and
    ``advantage_scale`` are one each, because there is one TD error on a
    transition and one advantage cut from it.

    Which is the whole shape of the position: what belongs to the parameters is
    shared, what belongs to an objective is not.
    """

    nu: Any
    momentum: Any
    sigma_bar: Any
    delta_square: Any
    advantage_scale: Any


class JointIntentionalOptimizer:
    """Two objectives' intentional step sizes, honoured by one step.

    Each branch states the paper's requirement over its own objective::

        alpha_b = eta_b / sqrt(sigma_bar_b * <rho m_b, m_b>)        (Eq. 12)

    and on separate parameters those two are simply two updates. Here they are
    two requirements on one ``Delta_theta``, and one number cannot satisfy two
    equations. It can satisfy two *inequalities*, and that is what the paper's
    ``eta`` has always been -- an intended spend, not an achieved one.

    Write the block's outcome as the two functions' movements, and ask that the
    step stay inside the ellipsoid the two allowances describe::

        sum_b (Delta J_b / eta_b)^2 <= 1

    With ``Delta_theta = alpha * rho * u`` and ``u = sum_b signal_b * m_b``,
    each ``Delta J_b`` is ``alpha * <g_b, rho u>``, and Eq. 12's own
    Cauchy-Schwarz bounds that by ``alpha * sqrt(sigma_b * <rho m_b, m_b>)``.
    Substituting leaves

        1 / alpha^2 = sum_b 1 / alpha_b^2

    so the step taken here is the two published step sizes combined as the
    reciprocal of the root of the sum of their reciprocals' squares. Four
    things follow, and they are the reason this exists:

    **It is Eq. 12 on one objective.** Delete a branch -- ``sigma_bar`` of
    zero, nothing in ``u`` -- and ``alpha`` is
    ``eta / sqrt(sigma_bar * <rho m, m>)``, the floor is where the published
    implementation puts it, and the update is ``alpha * signal * rho * m``.
    Not an approximation of the published step: the same expression, computed
    as one reciprocal rather than one quotient, which costs the one rounding a
    test comparing the two has to allow and nothing else.

    **Both allowances survive.** ``alpha * sqrt(sigma_bar_b * q_b) <= eta_b``
    holds for *each* branch simultaneously, because each term of the sum stands
    on its own. So ``eta_actor`` and ``eta_critic`` mean here what they mean in
    the paper, rather than becoming shares of some third budget.

    **No cross term.** ``sigma`` is never summed across branches before it is
    squared, so ``<p_actor, rho p_critic>`` appears nowhere. A coordinate the
    two heads' credit opposes on cannot shrink the denominator.

    **The tighter branch caps the looser.** ``alpha <= min_b alpha_b``. When
    one head's credit for an entry is naturally tiny -- because that head does
    not need it -- that branch's own ``alpha_b`` grows without bound and the
    other holds the step. Divergence needs *both* denominators to collapse,
    which is the single-objective case the paper already answers for.

    ``decays`` is each branch's ``gamma * lambda``: the rate its ``sigma_bar``
    averages at, which has to be the rate its trace forgets at. ``signals`` is
    which scalar each branch's contribution to the direction is proportional
    to. Both are the algorithm's to declare.
    """

    def __init__(self, settings: JointIntentional, *, decays, signals) -> None:
        if set(decays) != set(signals):
            raise ValueError(
                "a joint intentional step needs one decay and one signal per "
                f"branch; got decays for {sorted(decays)} and signals for "
                f"{sorted(signals)}"
            )
        etas = {}
        for name, signal in signals.items():
            if signal not in SIGNALS:
                raise ValueError(
                    f"{signal!r} is not one of the signals an intentional "
                    f"update steps along ({', '.join(SIGNALS)})"
                )
            allowance = f"eta_{name}"
            if not hasattr(settings, allowance):
                raise ValueError(
                    f"a joint intentional step has no allowance for a branch "
                    f"named {name!r}: {type(settings).__name__} declares no "
                    f"{allowance}"
                )
            etas[name] = getattr(settings, allowance)
        self.settings = settings
        self.decays = dict(decays)
        self.signals = dict(signals)
        self.etas = etas

    @property
    def branches(self) -> tuple[str, ...]:
        """The branch names in a fixed order, so two runs sum them alike."""

        return tuple(sorted(self.signals))

    def init(self, params, *, streams: int) -> JointIntentionalState:
        """Fresh state: empty averages, one set per stream."""

        empty = jax.tree.map(
            lambda leaf: jnp.zeros((streams, *leaf.shape), dtype=jnp.float32), params
        )
        zero = jnp.zeros((streams,), dtype=jnp.float32)
        return JointIntentionalState(
            nu=empty,
            momentum={
                name: jax.tree.map(jnp.zeros_like, empty) for name in self.branches
            },
            sigma_bar={name: zero for name in self.branches},
            delta_square=zero,
            advantage_scale=zero,
        )

    def _cut(self, delta, state, *, step):
        """Both branches' signals, cut from the one TD error the block saw."""

        settings = self.settings
        clipped, delta_square = clipped_td_error(
            delta,
            state.delta_square,
            beta=settings.beta_clip,
            clip=settings.clip,
            step=step,
        )
        advantage, scale = normalized_advantage(
            clipped, state.advantage_scale, beta=settings.beta_advantage, step=step
        )
        cut = {TD: clipped, ADVANTAGE: advantage}
        return (
            {name: cut[signal] for name, signal in self.signals.items()},
            clipped,
            delta_square,
            scale,
        )

    def update(
        self,
        *,
        delta,
        traces,
        derivatives,
        direct,
        step,
        params,
        state: JointIntentionalState,
    ):
        """One transition: one metric, two budgets, one step.

        ``traces`` and ``derivatives`` are one per branch, from the trace
        components that own them, and neither is written. ``direct`` is refused
        for the reason :meth:`IntentionalOptimizer.update` refuses it -- the
        entropy direction belongs inside the derivative the actor's branch
        traces, not beside the step. ``params`` is accepted and unused, which
        is the same property held here as there: an intentional step reads the
        derivative's scale and never the parameters'.

        Returns the update, the state, the readings belonging to the block, and
        the readings belonging to each branch -- separately, because that split
        is what this position *is* and a caller filing them under one name
        would have to undo it.
        """

        del params
        if direct is not None:
            raise ValueError(
                "a joint intentional update has no untraced term; the entropy "
                "direction belongs in the derivative the actor's branch "
                "accumulates, signed by the signal, before it reaches this "
                "optimizer"
            )
        settings = self.settings
        branches = self.branches
        signals, clipped, delta_square, scale = self._cut(delta, state, step=step)

        # One metric, off the summed derivative. See `JointIntentional`.
        summed = jax.tree.map(
            lambda *parts: sum(parts), *[derivatives[name] for name in branches]
        )
        nu = jax.tree.map(
            lambda old, leaf: corrected_average(
                old, jnp.square(leaf), beta=settings.beta_rms, step=step
            ),
            state.nu,
            summed,
        )
        rho = jax.tree.map(lambda leaf: 1.0 / (jnp.sqrt(leaf) + settings.eps), nu)

        def quadratic(tree):
            return _stream_sum(
                jax.tree.map(
                    lambda inverse, leaf: inverse * jnp.square(leaf), rho, tree
                )
            )

        sigma_bar = {
            name: corrected_average(
                state.sigma_bar[name],
                quadratic(derivatives[name]),
                beta=self.decays[name],
                step=step,
            )
            for name in branches
        }
        moments = {
            name: moment_average(
                state.momentum[name],
                traces[name],
                beta=settings.beta_momentum,
                step=step,
            )
            for name in branches
        }
        momentum = {name: moments[name][1] for name in branches}
        trace_quadratic = {name: quadratic(momentum[name]) for name in branches}

        # Each branch's own Eq. 12 denominator, as `sqrt(a) * sqrt(b)` rather
        # than `sqrt(a * b)`: both factors are sums of squares over every
        # parameter in a stream, so their product leaves float32's range from
        # either end well before either factor does.
        denominators = {
            name: jnp.sqrt(sigma_bar[name]) * jnp.sqrt(trace_quadratic[name])
            for name in branches
        }
        # The floor is per branch, where the published implementation puts
        # it: it bounds one branch's own step size at `eta_b / floor`, and
        # since `alpha <= min_b alpha_b` it bounds the block's by the same
        # number. Flooring the combined denominator instead would be a bound
        # on a quantity no branch declared.
        floored = {
            name: jnp.maximum(denominators[name], settings.denominator_floor)
            for name in branches
        }
        step_sizes = {name: self.etas[name] / floored[name] for name in branches}
        # `1 / alpha^2 = sum_b 1 / alpha_b^2`, written as the Euclidean norm of
        # the reciprocals through the overflow-safe primitive for it. Squaring
        # `d_b / eta_b` and adding would reintroduce the range problem the
        # branch denominators were just written to avoid.
        reciprocals = [floored[name] / self.etas[name] for name in branches]
        denominator = reciprocals[0]
        for term in reciprocals[1:]:
            denominator = jnp.hypot(denominator, term)
        alpha = 1.0 / denominator

        # One step: each branch's smoothed trace, read through the one metric
        # and scaled by `alpha * signal_b`, added once. Written per branch as
        # `alpha * signal` rather than as one `alpha` over a summed direction,
        # because per branch each term is the published `alpha * signal * rho *
        # m` exactly, and on a single branch that is what this has to be.
        ascent = jax.tree.map(
            lambda *parts: sum(parts),
            *[
                _stream_scaled(
                    jax.tree.map(
                        lambda inverse, leaf: inverse * leaf, rho, momentum[name]
                    ),
                    alpha * signals[name],
                )
                for name in branches
            ],
        )
        updates = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), ascent)

        block = IntentionalReading(
            clipped_delta=clipped,
            rms_scale=_stream_sum(rho) / max(_stream_count(rho), 1),
            # `sqrt(sum_b (1 / alpha_b)^2)`, which is `1 / alpha`. Not the
            # quantity a single-objective block files under this name -- there
            # it is what `eta` is divided by, and here `eta` is already inside
            # each term. The two are the same number only when one branch is
            # left, which is the case where this reduces to Eq. 12.
            denominator=denominator,
            step_size=alpha,
            update_norm=jnp.sqrt(_stream_sum(jax.tree.map(jnp.square, ascent))),
            non_finite=_stream_non_finite(ascent),
        )
        per_branch = {
            name: IntentionalReading(
                signal=signals[name],
                advantage_scale=None if self.signals[name] == TD else scale,
                sigma_bar=sigma_bar[name],
                # `<rho m_b, m_b>` on *this branch's* trace, which is what
                # Eq. 12 divides by for this branch. Not the quadratic of the
                # direction the block finally stepped along -- that direction
                # is the sum of both branches', and no branch's step size was
                # measured on it.
                trace_quadratic=trace_quadratic[name],
                denominator=denominators[name],
                # What Eq. 12 alone would have taken along this branch. Read
                # beside the block's `alpha`, it says which branch is holding
                # the step -- which is the question this position exists to
                # answer and the one no summed update can be asked.
                step_size=step_sizes[name],
            )
            for name in branches
        }
        return (
            updates,
            JointIntentionalState(
                nu=nu,
                momentum={name: moments[name][0] for name in branches},
                sigma_bar=sigma_bar,
                delta_square=delta_square,
                advantage_scale=scale,
            ),
            block,
            per_branch,
        )
