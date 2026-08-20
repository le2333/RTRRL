"""The intentional update, against the equations it is a transcription of.

There is no reference implementation to drive this against, so the reference
is written here: ``paper_step`` is Algorithm 1 and Algorithm 3 transcribed into
NumPy, line for line and in the paper's order, and the optimizer is driven
against it over a sequence with resets, sign changes and an outlier in it. A
transcription checked against a transcription would only agree about what both
authors read, so the equations are also pinned from the other side -- by the
properties that make them the algorithm they are:

    one step spends exactly ``eta`` of the TD error       (what "intentional" is)
    the trace takes this step's derivative before the step (ordering)
    the first sample of every average is taken whole       (bias correction)
    the clip is a multiple of the error's own RMS          (scale invariance)

Streams are the env axis, axis 0 of every derivative leaf. Most cases use one,
where the finished update is one stream's update; the case about independence
uses three and is the only place their averaging matters.
"""

from __future__ import annotations

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

ETA = 0.25
DECAY = 0.8
EPS = 1e-8
SETTINGS = IntentionalUpdate(
    eta=ETA, clip=20.0, beta_rms=0.9, beta_clip=0.95, beta_advantage=0.9, eps=EPS
)

WIDTH = 4
PARAMS = {"w": jnp.zeros((WIDTH,), dtype=jnp.float32)}


def optimizer(*, decay=DECAY, signal=TD, settings=SETTINGS):
    return IntentionalOptimizer(settings, decay=decay, signal=signal)


def streamed(values, streams=1):
    """One row per stream, which is the shape every derivative arrives in."""

    return jnp.asarray(np.tile(np.asarray(values, dtype=np.float32), (streams, 1)))


def stepped(rule, state, *, delta, derivative, step, direct=None, reset=0.0):
    """One transition through the optimizer, spelled the way RTRRL spells it."""

    return rule.update(
        delta=jnp.asarray([delta], dtype=jnp.float32),
        derivative={"w": streamed(derivative)},
        direct=None if direct is None else {"w": streamed(direct)},
        reset=jnp.asarray([reset], dtype=jnp.float32),
        step=step,
        params=PARAMS,
        state=state,
    )


def drive(rule, sequence, *, streams=1):
    """A whole sequence of transitions, and everything the last one produced."""

    state = rule.init(PARAMS, streams=streams)
    taken = []
    for step, transition in enumerate(sequence, start=1):
        updates, state, reading = stepped(rule, state, step=step, **transition)
        taken.append((updates, state, reading))
    return taken


# ------------------------------------------------------- the paper, in NumPy


def paper_step(carried, transition, *, settings, decay, signal, step):
    """Algorithm 1 (``td``) and Algorithm 3 (``advantage``), in their order.

    Everything is one stream and one flat vector, because that is what the
    paper's is. The only liberty taken is the reset, which the paper does not
    write down: the trace is what an episode boundary drops, and this is the
    algorithm's convention for dropping it.
    """

    z, nu, sigma_bar, mean_square, scale = carried
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
        chosen = clipped / scale if scale > 0 else 0.0
    else:
        chosen = clipped

    gradient = gradient + np.sign(chosen) * direct
    nu = averaged(nu, gradient**2, settings.beta_rms)
    rho = 1.0 / (np.sqrt(nu) + settings.eps)
    sigma = float(rho @ (gradient**2))
    sigma_bar = averaged(sigma_bar, sigma, decay)
    z = decay * (1 - reset) * z + gradient
    quadratic = float(rho @ (z**2))
    denominator = np.sqrt(sigma_bar * quadratic)
    alpha = settings.eta / denominator if denominator > 0 else 0.0

    update = alpha * chosen * rho * z
    return (z, nu, sigma_bar, mean_square, scale), update


def paper(sequence, *, settings=SETTINGS, decay=DECAY, signal=TD):
    """Every update the paper's equations produce over one sequence."""

    carried = (
        np.zeros(WIDTH),
        np.zeros(WIDTH),
        0.0,
        0.0,
        0.0,
    )
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


@pytest.mark.parametrize("signal", [TD, ADVANTAGE])
@pytest.mark.parametrize("sequence", [SEQUENCE, WITH_ENTROPY], ids=["plain", "entropy"])
def test_every_update_is_the_one_the_paper_s_equations_produce(signal, sequence):
    """Both algorithms, over a sequence with a reset and an outlier in it.

    The outlier is what makes the clip do something, the zero TD error is where
    a sign function has to be exactly zero, and the reset is the one line the
    paper does not write. Six steps is enough for every average to have left
    its first sample behind.
    """

    taken = drive(optimizer(signal=signal), sequence)
    expected = paper(sequence, signal=signal)

    for step, ((updates, _, _), reference) in enumerate(zip(taken, expected), start=1):
        np.testing.assert_allclose(
            np.asarray(updates["w"]),
            reference,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"step {step}",
        )


# ----------------------------------------------------- what "intentional" is


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
    ((updates, _, reading),) = drive(
        optimizer(decay=0.0), [{"delta": delta, "derivative": derivative}]
    )

    spent = float(np.dot(np.asarray(derivative), np.asarray(updates["w"])))
    np.testing.assert_allclose(spent, ETA * delta, rtol=1e-5)
    # And it is the clipped error that is spent, which at one step is the
    # error itself: the first sample of the clipping average is taken whole,
    # so the bound is exactly |delta| and the minimum picks either.
    np.testing.assert_allclose(float(reading.clipped_delta[0]), delta, rtol=1e-6)


def test_the_trace_takes_this_step_s_derivative_before_the_step_is_taken():
    """``z_t``, not ``z_{t-1}``: the first transition already moves.

    RTRRL's own rules read the trace as it stood and advance it afterwards, so
    their first step moves nothing at all. This is a different algorithm and
    the difference is visible at the first transition, which is the only place
    the two orderings can be told apart without unwinding a decay.
    """

    ((updates, state, _),) = drive(
        optimizer(), [{"delta": 1.0, "derivative": [1.0, -2.0, 0.5, 0.25]}]
    )

    assert float(jnp.max(jnp.abs(updates["w"]))) > 0
    np.testing.assert_allclose(
        np.asarray(state.z["w"][0]), [1.0, -2.0, 0.5, 0.25], rtol=1e-6
    )


# ------------------------------------------------------------------ the trace


def test_the_trace_decays_at_gamma_lambda_and_adds_the_derivative():
    """Accumulation, read off two steps of a trace that has nothing else in it."""

    first = [1.0, -2.0, 0.5, 0.25]
    second = [0.5, 0.5, -1.0, 2.0]
    taken = drive(
        optimizer(),
        [
            {"delta": 1.0, "derivative": first},
            {"delta": 1.0, "derivative": second},
        ],
    )

    np.testing.assert_allclose(
        np.asarray(taken[-1][1].z["w"][0]),
        DECAY * np.asarray(first) + np.asarray(second),
        rtol=1e-6,
    )


def test_a_reset_drops_what_the_trace_carried_and_keeps_what_arrived():
    """An episode boundary ends the credit, and does not skip a derivative."""

    first = [1.0, -2.0, 0.5, 0.25]
    second = [0.5, 0.5, -1.0, 2.0]
    taken = drive(
        optimizer(),
        [
            {"delta": 1.0, "derivative": first},
            {"delta": 1.0, "derivative": second, "reset": 1.0},
        ],
    )

    np.testing.assert_allclose(np.asarray(taken[-1][1].z["w"][0]), second, rtol=1e-6)


def test_a_lambda_of_zero_leaves_the_trace_at_the_derivative():
    """The degenerate case, and the one the intended fraction is exact at."""

    sequence = SEQUENCE[:3]
    taken = drive(optimizer(decay=0.0), sequence)

    for (_, state, _), transition in zip(taken, sequence):
        np.testing.assert_allclose(
            np.asarray(state.z["w"][0]), transition["derivative"], rtol=1e-6
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
    ((_, state, reading),) = drive(
        optimizer(signal=ADVANTAGE), [{"delta": -0.5, "derivative": derivative}]
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
    taken = drive(optimizer(signal=ADVANTAGE), sequence)

    scale = float(taken[-1][1].advantage_scale[0])
    np.testing.assert_allclose(float(taken[-1][2].signal[0]), -1.0 / scale, rtol=1e-5)
    assert scale > 0


def test_a_td_group_carries_no_advantage_scale_at_all():
    """Absent rather than zero: nothing normalizes a value function's error."""

    ((_, state, reading),) = drive(optimizer(), SEQUENCE[:1])

    assert state.advantage_scale is None
    assert reading.advantage_scale is None


# ---------------------------------------------- entropy, and where it is folded


def test_entropy_is_folded_into_the_derivative_and_signed_by_the_signal():
    """The paper's policy gradient is one derivative, not two directions.

    RTRRL's own rules add the entropy direction on the step it arises and never
    trace it. This one traces it, because the paper's ``g`` is the derivative
    of the log-probability and the entropy together -- and it enters with the
    sign of the signal, since the whole trace is later multiplied by that
    signal and an entropy term that did not flip with it would spend half its
    steps descending entropy.
    """

    derivative = [1.0, -2.0, 0.5, 0.25]
    entropy = [0.05, -0.05, 0.1, 0.0]
    negative = {"delta": -0.5, "derivative": derivative, "direct": entropy}
    positive = {**negative, "delta": 0.5}

    down = drive(optimizer(signal=ADVANTAGE), [negative])[0][1]
    up = drive(optimizer(signal=ADVANTAGE), [positive])[0][1]

    np.testing.assert_allclose(
        np.asarray(down.z["w"][0]),
        np.asarray(derivative) - np.asarray(entropy),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(up.z["w"][0]),
        np.asarray(derivative) + np.asarray(entropy),
        rtol=1e-6,
    )


# ------------------------------------------------------ streams, and repeating


def test_each_stream_carries_its_own_trace_and_its_own_step_size():
    """Three streams under one optimizer are three intentional updates."""

    rule = optimizer()
    state = rule.init(PARAMS, streams=3)
    updates, state, reading = rule.update(
        delta=jnp.asarray([0.5, -1.0, 2.0]),
        derivative={
            "w": jnp.asarray(
                [
                    [1.0, -2.0, 0.5, 0.25],
                    [0.1, 0.1, 0.1, 0.1],
                    [3.0, 0.0, -1.0, 0.5],
                ]
            )
        },
        direct=None,
        reset=jnp.asarray([0.0, 0.0, 1.0]),
        step=1,
        params=PARAMS,
        state=state,
    )

    assert len(set(np.asarray(reading.step_size).tolist())) == 3
    # And the finished update is their mean, which is where the parameters --
    # which have no stream axis -- are waiting.
    per_stream = (
        np.asarray(reading.step_size)[:, None] * np.asarray(reading.signal)[:, None]
    )
    assert per_stream.shape == (3, 1)
    np.testing.assert_allclose(
        np.asarray(updates["w"]),
        np.mean(
            np.asarray(reading.step_size)[:, None]
            * np.asarray(reading.signal)[:, None]
            * np.asarray(state.z["w"])
            / (np.sqrt(np.asarray(state.nu["w"])) + EPS),
            axis=0,
        ),
        rtol=1e-5,
    )


def test_the_same_sequence_gives_the_same_state_and_the_same_updates():
    """Nothing here is sampled, so a repeated drive is bit-identical."""

    once = drive(optimizer(signal=ADVANTAGE), WITH_ENTROPY)
    again = drive(optimizer(signal=ADVANTAGE), WITH_ENTROPY)

    for (updates, state, _), (repeated, carried, _) in zip(once, again):
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
