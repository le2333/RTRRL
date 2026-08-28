"""The intentional update, against the equations it is a transcription of.

Three oracles, because one is never enough for a transcription:

``tests/test_intentional_parity.py``
    the published implementation itself, driven side by side with this one.
    That file is the authority and it is marked ``external``, so it runs where
    the clone exists and skips where it does not.
``paper_step``, below
    Algorithm 1 and Algorithm 3 in NumPy, line for line and in the paper's
    order. It catches what a reordering breaks without needing the clone.
the properties
    what the equations are *defined* by, which a second transcription of the
    same misreading would still fail:

        one step spends exactly ``eta`` of the TD error  (what "intentional" is)
        the first sample of every average is taken whole  (bias correction)
        the clip is a multiple of the error's own RMS     (scale invariance)
        the floored denominator bounds one step           (the safeguard)

The trace is not the optimizer's, so these drive the pair the algorithm drives:
a :class:`Trace` reading ``current`` and unemphasized, which is the recurrence
the intentional update is derived against, and the optimizer that steps along
what it hands back.

Streams are the env axis, axis 0 of every derivative leaf. Most cases use one,
where the finished update is one stream's update; the case about independence
uses three and is the only place their averaging matters.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.rl.intentional import (
    ADVANTAGE,
    TD,
    IntentionalOptimizer,
    IntentionalUpdate,
    clipped_td_error,
)
from memorax.rl.traces import CURRENT, Trace

ETA = 0.25
DECAY = 0.8
EPS = 1e-8
FLOOR = 1e-8
SETTINGS = IntentionalUpdate(
    eta=ETA,
    clip=20.0,
    beta_rms=0.9,
    beta_clip=0.95,
    beta_advantage=0.9,
    denominator_floor=FLOOR,
    eps=EPS,
)

# Adam's `beta1` over the eligibility trace. The published optimizer is the
# zero of this, which is why `SETTINGS` leaves it at its default and why every
# case here that is about something else runs at zero.
MOMENTUM = 0.6
SMOOTHED = replace(SETTINGS, beta_momentum=MOMENTUM)

WIDTH = 4
PARAMS = {"w": jnp.zeros((WIDTH,), dtype=jnp.float32)}


def pair(*, decay=DECAY, signal=TD, settings=SETTINGS):
    """The trace and the optimizer, wired the way RTRRL wires them."""

    return (
        Trace(decay=decay, reads=CURRENT, emphasized=False),
        IntentionalOptimizer(settings, decay=decay, signal=signal),
    )


def streamed(values, streams=1):
    """One row per stream, which is the shape every derivative arrives in."""

    return jnp.asarray(np.tile(np.asarray(values, dtype=np.float32), (streams, 1)))


def drive(trace, rule, sequence, *, streams=1):
    """A whole sequence of transitions, and everything each one produced.

    The entropy direction is folded into the derivative here rather than
    inside the optimizer, because that is where RTRRL folds it: the paper's
    policy gradient is one derivative, and the trace has to accumulate the sum.
    """

    carried = trace.initial(PARAMS, streams)
    state = rule.init(PARAMS, streams=streams)
    ones = jnp.ones((streams,), dtype=jnp.float32)
    taken = []
    for step, transition in enumerate(sequence, start=1):
        delta = jnp.asarray([transition["delta"]] * streams, dtype=jnp.float32)
        derivative = {"w": streamed(transition["derivative"], streams)}
        if transition.get("direct") is not None:
            entropy = {"w": streamed(transition["direct"], streams)}
            derivative = jax.tree.map(
                lambda leaf, term: leaf + jnp.sign(delta)[:, None] * term,
                derivative,
                entropy,
            )
        used, carried = trace.stepped(
            carried,
            derivative,
            reset=jnp.asarray([transition.get("reset", 0.0)] * streams),
            emphasis=ones,
        )
        updates, state, reading = rule.update(
            delta=delta,
            trace=used,
            derivative=derivative,
            direct=None,
            step=step,
            params=PARAMS,
            state=state,
        )
        taken.append((updates, state, reading, used))
    return taken


# ------------------------------------------------------- the paper, in NumPy


def paper_step(carried, transition, *, settings, decay, signal, step):
    """Algorithm 1 (``td``) and Algorithm 3 (``advantage``), in their order.

    Everything is one stream and one flat vector, because that is what the
    paper's is. The only liberty taken is the reset, which the paper does not
    write down: the trace is what an episode boundary drops, and this is the
    algorithm's convention for dropping it.
    """

    z, m, nu, sigma_bar, mean_square, scale = carried
    delta = float(transition["delta"])
    gradient = np.asarray(transition["derivative"], dtype=np.float64)
    direct = np.asarray(
        transition.get("direct", np.zeros_like(gradient)), dtype=np.float64
    )
    reset = float(transition.get("reset", 0.0))

    def averaged(previous, sample, beta):
        return previous + (1 - beta) / (1 - beta**step) * (sample - previous)

    mean_square = averaged(mean_square, delta**2, settings.beta_clip)
    clipped = np.sign(delta) * min(abs(delta), settings.clip * np.sqrt(mean_square))
    if signal == ADVANTAGE:
        scale = averaged(scale, abs(clipped), settings.beta_advantage)
        chosen = clipped / max(scale, 1e-12)
    else:
        chosen = clipped

    gradient = gradient + np.sign(chosen) * direct
    nu = averaged(nu, gradient**2, settings.beta_rms)
    rho = 1.0 / (np.sqrt(nu) + settings.eps)
    sigma = float(rho @ (gradient**2))
    sigma_bar = averaged(sigma_bar, sigma, decay)
    z = decay * (1 - reset) * z + gradient
    # Adam's first moment, kept plain and corrected on the way out, over the
    # trace rather than over a gradient. At `beta_momentum = 0` it is `z`.
    beta1 = settings.beta_momentum
    m = beta1 * m + (1 - beta1) * z
    direction = m / (1 - beta1**step)
    quadratic = float(rho @ (direction**2))
    denominator = np.sqrt(sigma_bar * quadratic)
    alpha = settings.eta / max(denominator, settings.denominator_floor)

    update = alpha * chosen * rho * direction
    return (z, m, nu, sigma_bar, mean_square, scale), update


def paper(sequence, *, settings=SETTINGS, decay=DECAY, signal=TD):
    """Every update the paper's equations produce over one sequence."""

    carried = (np.zeros(WIDTH), np.zeros(WIDTH), np.zeros(WIDTH), 0.0, 0.0, 0.0)
    updates = []
    for step, transition in enumerate(sequence, start=1):
        carried, update = paper_step(
            carried,
            transition,
            settings=settings,
            decay=decay,
            signal=signal,
            step=step,
        )
        updates.append(update)
    return updates


SEQUENCE = [
    {"delta": 0.5, "derivative": [1.0, -2.0, 0.5, 0.25]},
    {"delta": -1.5, "derivative": [0.5, 0.5, -1.0, 2.0]},
    {"delta": 0.0, "derivative": [-1.0, 0.25, 0.75, -0.5]},
    {"delta": 40.0, "derivative": [2.0, 1.0, -0.5, 1.5]},
    {"delta": -0.25, "derivative": [0.1, -0.1, 0.2, -0.2], "reset": 1.0},
    {"delta": 0.75, "derivative": [1.5, 0.5, -1.5, 0.5]},
]

WITH_ENTROPY = [
    {**transition, "direct": [0.05, -0.05, 0.1, 0.0]} for transition in SEQUENCE
]


@pytest.mark.parametrize(
    "settings", [SETTINGS, SMOOTHED], ids=["published", "smoothed"]
)
@pytest.mark.parametrize("signal", [TD, ADVANTAGE])
@pytest.mark.parametrize("sequence", [SEQUENCE, WITH_ENTROPY], ids=["plain", "entropy"])
def test_every_update_is_the_one_the_paper_s_equations_produce(
    signal, sequence, settings
):
    """Both algorithms, over a sequence with a reset and an outlier in it.

    The outlier is what makes the clip do something, the zero TD error is where
    a sign function has to be exactly zero, and the reset is the one line the
    paper does not write. Six steps is enough for every average to have left
    its first sample behind.
    """

    taken = drive(*pair(signal=signal, settings=settings), sequence)
    expected = paper(sequence, signal=signal, settings=settings)

    for step, ((updates, _, _, _), reference) in enumerate(
        zip(taken, expected), start=1
    ):
        np.testing.assert_allclose(
            np.asarray(updates["w"]),
            reference,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"step {step}",
        )


# ----------------------------------------------------- what "intentional" is


def test_a_momentum_of_zero_carries_the_trace_itself_and_nothing_rounder():
    """What makes ``beta_momentum: 0`` the published optimizer rather than near it.

    The moment is ``0 * m + 1 * z`` corrected by ``1 - 0**t``, which is a
    multiplication by one and a division by one, so the direction the step is
    taken along is the trace the algorithm handed over -- the same floats, not
    the same value recomputed. Written as ``m + (z - m)``, the form every other
    average here is kept in, it would not be, and the equality the case above
    asserts at ``rtol=1e-5`` would be hiding a difference rather than there
    being none to hide.
    """

    for updates, state, _, used in drive(*pair(), SEQUENCE):
        del updates
        assert np.array_equal(
            np.asarray(state.momentum["w"]), np.asarray(used["w"])
        ), "the moment at beta_momentum = 0 is not the trace it was handed"


def test_a_momentum_above_zero_steps_along_a_direction_the_trace_is_not():
    """And the difference is the smoothing, not a rescaling of the same vector.

    A first moment of a sequence points where the sequence has been, which is
    not where its latest term points. So the two are checked for being
    non-parallel rather than merely unequal: a step that had only been scaled
    would be the published algorithm with a different ``eta``.
    """

    smoothed = drive(*pair(settings=SMOOTHED), SEQUENCE)
    plain = drive(*pair(), SEQUENCE)

    angles = []
    for (_, state, _, used), _ in zip(smoothed, plain):
        moment = np.asarray(state.momentum["w"])[0]
        trace = np.asarray(used["w"])[0]
        lengths = np.linalg.norm(moment) * np.linalg.norm(trace)
        if lengths > 0:
            angles.append(abs(float(moment @ trace) / lengths))
    assert angles, "the sequence never produced a trace to compare against"
    assert min(angles) < 0.999, "the moment stayed parallel to the trace throughout"


def test_one_step_spends_exactly_the_intended_fraction_of_the_td_error():
    """The property the step size is derived from, at the horizon it holds at.

    With ``lambda`` at zero the trace is this step's derivative and nothing
    else, so the value the critic reads moves by ``<g, dtheta>`` -- and that is
    ``alpha * delta * <rho g, g>``, where ``alpha`` is ``eta`` divided by
    exactly that quantity. A linear critic's TD error therefore comes back
    ``eta`` smaller, whatever the derivative was and whatever ``eps`` is: the
    preconditioner cancels between the step size and the step.

    This is the whole claim of an intentional update, and no rearrangement of
    the arithmetic that got the denominator wrong could pass it.
    """

    delta = 0.5
    derivative = [1.0, -2.0, 0.5, 0.25]
    ((updates, _, reading, _),) = drive(
        *pair(decay=0.0), [{"delta": delta, "derivative": derivative}]
    )

    spent = float(np.dot(np.asarray(derivative), np.asarray(updates["w"])))
    np.testing.assert_allclose(spent, ETA * delta, rtol=1e-5)
    # And it is the clipped error that is spent, which at one step is the
    # error itself: the first sample of the clipping average is taken whole,
    # so the bound is exactly |delta| and the minimum picks either.
    np.testing.assert_allclose(float(reading.clipped_delta[0]), delta, rtol=1e-6)


def test_the_step_is_along_the_trace_the_optimizer_was_handed():
    """The trace is an input. Nothing here accumulates one.

    A call whose trace is zero takes no step at all, however large the
    derivative it is told about -- which is what an optimizer that had quietly
    kept a trace of its own could not do.
    """

    _, rule = pair()
    derivative = {"w": streamed([1.0, -2.0, 0.5, 0.25])}

    updates, _, _ = rule.update(
        delta=jnp.asarray([1.0]),
        trace={"w": jnp.zeros((1, WIDTH))},
        derivative=derivative,
        direct=None,
        step=1,
        params=PARAMS,
        state=rule.init(PARAMS, streams=1),
    )

    assert float(jnp.max(jnp.abs(updates["w"]))) == 0.0


def test_an_untraced_direction_is_refused():
    """The paper's update has no term the trace did not accumulate.

    An algorithm pairing this optimizer with an entropy term applied outside
    the trace would be stepping a rule nobody published, and would do it
    silently. Refusing at the wiring is what makes that unreachable by
    accident.
    """

    _, rule = pair()
    derivative = {"w": streamed([1.0, 0.0, 0.0, 0.0])}

    with pytest.raises(ValueError, match="no untraced term"):
        rule.update(
            delta=jnp.asarray([1.0]),
            trace=derivative,
            derivative=derivative,
            direct=derivative,
            step=1,
            params=PARAMS,
            state=rule.init(PARAMS, streams=1),
        )


# -------------------------------------------------- the averages and the clip


def test_the_first_sample_of_every_average_is_taken_whole():
    """What folding the bias correction into the rate buys.

    An uncorrected exponential average would start at ``(1 - beta)`` of its
    first sample and spend hundreds of steps climbing out of a zero nobody
    measured. Each of these is exactly its own first sample instead, which is
    also why the first step's clip bound is the error itself and the first
    normalized advantage is exactly a sign.
    """

    derivative = [1.0, -2.0, 0.5, 0.25]
    ((_, state, reading, _),) = drive(
        *pair(signal=ADVANTAGE), [{"delta": -0.5, "derivative": derivative}]
    )

    np.testing.assert_allclose(
        np.asarray(state.nu["w"][0]), np.square(derivative), rtol=1e-6
    )
    np.testing.assert_allclose(float(state.delta_square[0]), 0.25, rtol=1e-6)
    np.testing.assert_allclose(float(state.advantage_scale[0]), 0.5, rtol=1e-6)
    np.testing.assert_allclose(float(reading.signal[0]), -1.0, rtol=1e-6)
    # sigma_bar is the same average at the trace's own decay, and its first
    # sample is <rho g, g> whole.
    np.testing.assert_allclose(
        float(state.sigma_bar[0]), float(reading.trace_quadratic[0]), rtol=1e-6
    )


def test_the_clip_is_a_multiple_of_the_error_s_own_running_rms():
    """A bound in units of the signal, which is why it survives a rescaling.

    Doubling every TD error doubles the outlier and doubles the scale it is
    measured against, so the same outlier is cut back to twice as much: the
    clip is dimensionless, and a run whose rewards are scaled differently
    answers to the same rule.

    Read on the statistic directly, because the bound cannot bite early. The
    outlier joins the average before it is clipped against it, and while the
    corrected rate is still around ``1/t`` an outlier raises the scale enough
    to clear its own bound. It takes a few hundred transitions at the published
    decay before twenty root-mean-square errors is a bound anything reaches --
    which is a property of the rule, not of this test.
    """

    calm, outlier = 0.01, 40.0
    clipped, mean_square = clipped_td_error(
        jnp.asarray(outlier), jnp.asarray(calm), beta=0.9998, clip=20.0, step=1000
    )
    twice, _ = clipped_td_error(
        jnp.asarray(2 * outlier),
        jnp.asarray(4 * calm),
        beta=0.9998,
        clip=20.0,
        step=1000,
    )

    assert float(clipped) < outlier
    np.testing.assert_allclose(
        float(clipped), 20.0 * float(jnp.sqrt(mean_square)), rtol=1e-6
    )
    np.testing.assert_allclose(float(twice), 2 * float(clipped), rtol=1e-6)


def test_the_clip_keeps_an_error_it_does_not_reach_exactly_as_it_was():
    """A bound, not a normalization: what is under it passes through whole."""

    clipped, _ = clipped_td_error(
        jnp.asarray(0.3), jnp.asarray(1.0), beta=0.9998, clip=20.0, step=1000
    )

    np.testing.assert_allclose(float(clipped), 0.3, rtol=1e-6)


def test_an_advantage_is_divided_by_the_running_mean_of_its_own_magnitude():
    """The actor's second statistic, and the sign it is guaranteed to keep."""

    sequence = [
        {"delta": 2.0, "derivative": [1.0, 0.0, 0.0, 0.0]},
        {"delta": -1.0, "derivative": [0.0, 1.0, 0.0, 0.0]},
    ]
    taken = drive(*pair(signal=ADVANTAGE), sequence)

    scale = float(taken[-1][1].advantage_scale[0])
    np.testing.assert_allclose(float(taken[-1][2].signal[0]), -1.0 / scale, rtol=1e-5)
    assert scale > 0


def test_a_td_group_carries_no_advantage_scale_at_all():
    """Absent rather than zero: nothing normalizes a value function's error."""

    ((_, state, reading, _),) = drive(*pair(), SEQUENCE[:1])

    assert state.advantage_scale is None
    assert reading.advantage_scale is None


# ------------------------------------------------------------- the safeguard


def test_the_denominator_is_floored_where_the_implementation_floors_it():
    """``eta / max(u, 1e-8)``, which bounds one step instead of letting it run.

    The paper's Eq. 12 is a plain quotient and the implementation every
    reported number came from is not. It matters only where the denominator
    collapses -- an unfloored step there is unbounded, and this optimizer's
    entire subject is what a streaming run does at its worst moment.

    Driven at a derivative small enough for ``sqrt(sigma_bar <rho z, z>)`` to
    fall under the floor, where the two rules differ by orders of magnitude.
    """

    tiny = [1e-9, 0.0, 0.0, 0.0]
    ((updates, _, reading, _),) = drive(
        *pair(decay=0.0), [{"delta": 1.0, "derivative": tiny}]
    )

    assert float(reading.denominator[0]) < FLOOR
    np.testing.assert_allclose(float(reading.step_size[0]), ETA / FLOOR, rtol=1e-6)
    # And the step is finite, which is the whole of what the floor buys.
    assert np.all(np.isfinite(np.asarray(updates["w"])))
    assert float(reading.non_finite[0]) == 0.0


# ------------------------------------------------------ streams, and repeating


def test_each_stream_carries_its_own_statistics_and_its_own_step_size():
    """Three streams under one optimizer are three intentional updates."""

    trace, rule = pair()
    derivative = {
        "w": jnp.asarray(
            [
                [1.0, -2.0, 0.5, 0.25],
                [0.1, 0.1, 0.1, 0.1],
                [3.0, 0.0, -1.0, 0.5],
            ]
        )
    }
    used, _ = trace.stepped(
        trace.initial(PARAMS, 3),
        derivative,
        reset=jnp.asarray([0.0, 0.0, 1.0]),
        emphasis=jnp.ones((3,)),
    )
    updates, state, reading = rule.update(
        delta=jnp.asarray([0.5, -1.0, 2.0]),
        trace=used,
        derivative=derivative,
        direct=None,
        step=1,
        params=PARAMS,
        state=rule.init(PARAMS, streams=3),
    )

    assert len(set(np.asarray(reading.step_size).tolist())) == 3
    # And the finished update is their mean, which is where the parameters --
    # which have no stream axis -- are waiting.
    np.testing.assert_allclose(
        np.asarray(updates["w"]),
        np.mean(
            np.asarray(reading.step_size)[:, None]
            * np.asarray(reading.signal)[:, None]
            * np.asarray(used["w"])
            / (np.sqrt(np.asarray(state.nu["w"])) + EPS),
            axis=0,
        ),
        rtol=1e-5,
    )


def test_the_same_sequence_gives_the_same_state_and_the_same_updates():
    """Nothing here is sampled, so a repeated drive is bit-identical."""

    once = drive(*pair(signal=ADVANTAGE), WITH_ENTROPY)
    again = drive(*pair(signal=ADVANTAGE), WITH_ENTROPY)

    for (updates, state, _, _), (repeated, carried, _, _) in zip(once, again):
        assert jax.tree.all(
            jax.tree.map(lambda a, b: bool(jnp.array_equal(a, b)), updates, repeated)
        )
        assert jax.tree.all(
            jax.tree.map(lambda a, b: bool(jnp.array_equal(a, b)), state, carried)
        )


def test_a_signal_nothing_names_is_refused():
    """Which scalar a group steps along is a routing decision, not a guess."""

    with pytest.raises(ValueError, match="not one of the signals"):
        IntentionalOptimizer(SETTINGS, decay=DECAY, signal="surprise")


def test_a_trace_reading_nothing_names_is_refused():
    """And when an update reads is the trace's own declaration."""

    with pytest.raises(ValueError, match="not one of the traces"):
        Trace(decay=DECAY, reads="afterwards")
