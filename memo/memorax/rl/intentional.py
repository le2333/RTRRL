"""The Intentional Update optimizer, written as its paper defines it.

    https://arxiv.org/abs/2604.19033

An intentional update asks how far a step may go before it has spent the
outcome it was aiming at, and answers with a step size rather than with a clip
placed after the fact. The pieces are:

    nu      a diagonal second moment of the instantaneous derivative
    rho     its inverse root, the preconditioner every term is read through
    z       the eligibility trace, accumulated *here* rather than by the caller
    sigma   the trace-weighted statistic <rho g, g>, averaged into sigma_bar
    alpha   eta / sqrt(sigma_bar <rho z, z>), the intentional step size

and one transition's update is ``alpha * signal * rho * z``, where the signal
is a clipped TD error for a value function and a normalized advantage for a
policy.

**The trace belongs to the optimizer.** Every other rule in this package is
handed a finished trace and decides only how far to step along it; this one is
handed the instantaneous derivative and accumulates ``z = decay * z + p``
itself. That is not a packaging choice: ``sigma_bar`` averages a statistic of
the *derivative* at the same decay the trace runs at, and ``alpha`` divides one
by the other, so a rule that only saw the trace could not compute its own step
size. The decay is the algorithm's ``gamma * lambda`` and arrives at
construction, because which lambda a parameter group answers to is the
algorithm's routing decision and not a setting of this optimizer.

Every quantity here is per stream: the leading axis of a derivative leaf is the
parallel environment axis, and each stream carries its own trace, second
moment, statistic and step size. Only the finished update is averaged across
them, which is where the caller's parameters -- which have no stream axis --
are waiting.
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


@dataclass(frozen=True)
class IntentionalUpdate:
    """What an intentional update reads, and what the paper's defaults are.

    ``eta`` is the intended fractional reduction: the share of the outcome one
    step sets out to spend. It is the only quantity here an experiment is
    expected to search over; the rest are the published constants and are
    declared with their published values so that a run asking for one of them
    by name says so, and a run that does not ask still records what it used.

    ``clip`` is the paper's ``C``, the number of root-mean-square TD errors a
    single TD error may be before it is cut back to that many. It is a
    multiplier on a running scale rather than an absolute threshold, so it
    means the same thing whatever the reward scale is.

    ``eps`` sits beside the root -- ``(sqrt(nu) + eps)`` -- which is where the
    paper puts it. It is not the same rule as ``sqrt(nu + eps)``; see
    :class:`memorax.rl.updates.AdaptiveObBoundFixed`, where the same two
    spellings are two components because a recorded run answers to one of them.
    """

    eta: float = param(valid=(1e-9, 100.0), search=(1e-4, 1.0), log=True)
    clip: float = param(valid=(0.0, 1e4), search=[20.0], default=20.0)
    beta_rms: float = param(valid=(0.0, 1.0), search=[0.999], default=0.999)
    beta_clip: float = param(valid=(0.0, 1.0), search=[0.9998], default=0.9998)
    beta_advantage: float = param(valid=(0.0, 1.0), search=[0.9998], default=0.9998)
    eps: float = param(valid=(1e-12, 1e-2), search=[1e-8], default=1e-8, log=True)


class IntentionalState(struct.PyTreeNode):
    """One parameter group's intentional state, all of it per stream.

    ``advantage_scale`` is None for a group whose signal is a TD error: there
    is no advantage to normalize there, and carrying a zero would say there was
    one that happened to be zero.
    """

    z: Any
    nu: Any
    sigma_bar: Any
    delta_square: Any
    advantage_scale: Any = None


class IntentionalReading(struct.PyTreeNode):
    """Every quantity one intentional step passed through, per stream.

    All ten are produced whether or not anything reads them; which are filed is
    the caller's declaration. They are here rather than derived afterwards
    because most of them cannot be recovered from the update: ``alpha`` and the
    signal reach it only as their product, and ``sigma_bar`` not at all.
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

    which is the same sequence as an exponential average divided by
    ``1 - beta^t`` afterwards, and differs from correcting at the end in that
    the corrected value is what the next step averages into. At ``t = 1`` the
    rate is exactly one, so the first sample is taken whole and a zero
    initialization is never something a later step is still shrinking away
    from.

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
    averaged the scale is zero and the ratio is ``0/0``; the reading taken
    there is zero, which is the one value that leaves the parameters where they
    were on a step that had no scale to measure against.
    """

    scale = corrected_average(scale, jnp.abs(advantage), beta=beta, step=step)
    positive = scale > 0
    return jnp.where(positive, advantage / jnp.where(positive, scale, 1.0), 0.0), scale


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

    ``decay`` is ``gamma * lambda`` for the group: the rate its trace forgets
    at, and the rate ``sigma_bar`` averages at. ``signal`` is which scalar the
    step is proportional to -- the clipped TD error of Intentional TD, or the
    normalized advantage of the intentional policy gradient -- and it is passed
    in rather than declared because it follows from what the group's objective
    is, which only the algorithm knows.
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
        """Fresh state: an empty trace and empty averages, one set per stream."""

        empty = jax.tree.map(
            lambda leaf: jnp.zeros((streams, *leaf.shape), dtype=jnp.float32), params
        )
        zero = jnp.zeros((streams,), dtype=jnp.float32)
        return IntentionalState(
            z=empty,
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
        derivative,
        direct,
        reset,
        step,
        params,
        state: IntentionalState,
    ) -> tuple[Any, IntentionalState, IntentionalReading]:
        """One transition: accumulate, precondition, and step as far as intended.

        ``derivative`` is the instantaneous derivative of the group's traced
        objective, per stream. ``direct`` is what ascends on the step it arises
        -- the policy's entropy term, already carrying its own coefficient --
        and it is folded into the derivative before the trace sees it, because
        the paper's policy gradient is the derivative of the log-probability
        *and* the entropy together. It enters multiplied by the sign of the
        signal, since the whole trace is later multiplied by the signal and an
        entropy term that flipped with it would descend entropy half the time.

        ``params`` is accepted and unused. An intentional step reads the
        derivative's scale and never the parameters', which is a property worth
        being able to see in the signature rather than in the prose.
        """

        del params
        settings = self.settings
        signal, clipped, delta_square, scale = self._signal(delta, state, step=step)

        gradient = derivative
        if direct is not None:
            gradient = jax.tree.map(
                lambda traced, immediate: (
                    traced + broadcast_stream(jnp.sign(signal), immediate) * immediate
                ),
                derivative,
                direct,
            )

        nu = jax.tree.map(
            lambda old, leaf: corrected_average(
                old, jnp.square(leaf), beta=settings.beta_rms, step=step
            ),
            state.nu,
            gradient,
        )
        rho = jax.tree.map(lambda leaf: 1.0 / (jnp.sqrt(leaf) + settings.eps), nu)
        sigma = _stream_sum(
            jax.tree.map(
                lambda inverse, leaf: inverse * jnp.square(leaf), rho, gradient
            )
        )
        sigma_bar = corrected_average(
            state.sigma_bar, sigma, beta=self.decay, step=step
        )

        # The trace takes this step's derivative before the step is taken, so
        # the update reads `z_t` and not `z_{t-1}`. RTRRL's own rules read the
        # trace as it stood and advance it afterwards; that is a difference
        # between two algorithms rather than an ordering detail, and it is the
        # reason the first transition of an intentional run already moves.
        kept = self.decay * (1.0 - reset)
        z = jax.tree.map(
            lambda old, leaf: broadcast_stream(kept, old) * old + leaf,
            state.z,
            gradient,
        )

        quadratic = _stream_sum(
            jax.tree.map(lambda inverse, leaf: inverse * jnp.square(leaf), rho, z)
        )
        denominator = jnp.sqrt(sigma_bar * quadratic)
        intended = denominator > 0
        alpha = jnp.where(
            intended, settings.eta / jnp.where(intended, denominator, 1.0), 0.0
        )

        ascent = _stream_scaled(
            jax.tree.map(lambda inverse, leaf: inverse * leaf, rho, z), alpha * signal
        )
        updates = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), ascent)
        return (
            updates,
            IntentionalState(
                z=z,
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
