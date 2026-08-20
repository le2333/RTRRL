"""The Intentional Update optimizer, written as its paper defines it.

    paper           https://arxiv.org/abs/2604.19033
    implementation  https://github.com/sharifnassab/Intentional_RL

An intentional update sizes a step by what the step is supposed to spend:

    nu      a diagonal second moment of the instantaneous derivative
    rho     its inverse root, the preconditioner every term is read through
    sigma   the statistic <rho p, p>, averaged into sigma_bar at the trace's
            own decay
    alpha   eta / max(sqrt(sigma_bar <rho z, z>), floor)

and one transition's update is ``alpha * signal * rho * z``, where the signal
is a clipped TD error for a value function and an advantage normalized by its
own running scale for a policy.

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
    """

    eta: float = param(valid=(1e-9, 100.0), search=(1e-4, 1.0), log=True)
    clip: float = param(valid=(0.0, 1e4), search=[20.0], default=20.0)
    beta_rms: float = param(valid=(0.0, 1.0), search=[0.999], default=0.999)
    beta_clip: float = param(valid=(0.0, 1.0), search=[0.9998], default=0.9998)
    beta_advantage: float = param(valid=(0.0, 1.0), search=[0.9998], default=0.9998)
    denominator_floor: float = param(
        valid=(0.0, 1.0), search=[1e-8], default=1e-8, log=True
    )
    eps: float = param(valid=(1e-12, 1e-2), search=[1e-8], default=1e-8, log=True)


class IntentionalState(struct.PyTreeNode):
    """One parameter group's intentional state, all of it per stream.

    No eligibility trace: that is the algorithm's, and it arrives at each step
    from the trace component that owns it.

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

        quadratic = _stream_sum(
            jax.tree.map(lambda inverse, leaf: inverse * jnp.square(leaf), rho, trace)
        )
        denominator = jnp.sqrt(sigma_bar * quadratic)
        alpha = settings.eta / jnp.maximum(denominator, settings.denominator_floor)

        ascent = _stream_scaled(
            jax.tree.map(lambda inverse, leaf: inverse * leaf, rho, trace),
            alpha * signal,
        )
        updates = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), ascent)
        return (
            updates,
            IntentionalState(
                nu=nu,
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
