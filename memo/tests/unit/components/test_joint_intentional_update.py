"""The joint intentional step, against the equation it is derived from.

The paper's actor-critic is two separate networks, so every parameter it
updates is credited by one objective and Eq. 12 answers for it whole. RTRRL's
torso is credited by two, and one transition writes one ``Delta_theta``. What
:class:`JointIntentionalOptimizer` does is take the largest step that stays
inside *both* branches' intended fractions::

    alpha_b = eta_b / sqrt(sigma_bar_b * <rho m_b, m_b>)          (Eq. 12)
    1 / alpha^2 = sum_b 1 / alpha_b^2

There is no published oracle for that, because there is no published shared
block. So what is held here is the four properties the derivation is *defined*
by, each of which a plausible-looking alternative fails:

    one branch alone is the published step        (it generalizes Eq. 12)
    each branch's own fraction still holds        (eta keeps its meaning)
    the block never outsteps either branch        (alpha <= min alpha_b)
    sigma is each branch's own, in a shared rho   (no cross term)

``tests/unit/components/test_intentional_update.py`` holds the published
optimizer against the paper. Nothing here re-derives it; what is tested here is
what happens when a second objective arrives.
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
    JointIntentional,
    JointIntentionalOptimizer,
)
from memorax.rl.traces import CARRIED, Trace

# The published actor-critic's two intended reductions.
ETA_ACTOR = 0.05
ETA_CRITIC = 0.5
DECAY_ACTOR = 0.8
DECAY_CRITIC = 0.6
EPS = 1e-8
FLOOR = 1e-8

SETTINGS = JointIntentional(
    eta_actor=ETA_ACTOR,
    eta_critic=ETA_CRITIC,
    clip=20.0,
    beta_rms=0.9,
    beta_clip=0.95,
    beta_advantage=0.9,
    beta_momentum=0.0,
    denominator_floor=FLOOR,
    eps=EPS,
)

BRANCHES = ("actor", "critic")
DECAYS = {"actor": DECAY_ACTOR, "critic": DECAY_CRITIC}
SIGNALS = {"actor": ADVANTAGE, "critic": TD}

WIDTH = 4
PARAMS = {"w": jnp.zeros((WIDTH,), dtype=jnp.float32)}


def streamed(values, streams=1):
    """One row per stream, which is the shape every derivative arrives in."""

    return jnp.asarray(np.tile(np.asarray(values, dtype=np.float32), (streams, 1)))


def optimizer(settings=SETTINGS, *, decays=DECAYS, signals=SIGNALS):
    return JointIntentionalOptimizer(settings, decays=decays, signals=signals)


def traces(*, decays=DECAYS):
    """One recurrence per branch, wired the way the aggregation wires them."""

    return {
        name: Trace(decay=decay, reads=CARRIED, emphasized=False)
        for name, decay in decays.items()
    }


def drive(rule, sequence, *, streams=1, decays=DECAYS):
    """A whole sequence of transitions, and everything each one produced.

    The traces are advanced outside the optimizer, because they are the
    algorithm's: the optimizer is handed the trace as the aggregation left it
    and the derivative that trace was accumulated from, and reads both.
    """

    recurrences = traces(decays=decays)
    carried = {
        name: recurrence.initial(PARAMS, streams)
        for name, recurrence in recurrences.items()
    }
    state = rule.init(PARAMS, streams=streams)
    ones = jnp.ones((streams,), dtype=jnp.float32)
    zeros = jnp.zeros((streams,), dtype=jnp.float32)
    taken = []
    for step, transition in enumerate(sequence, start=1):
        delta = jnp.asarray([transition["delta"]] * streams, dtype=jnp.float32)
        derivatives = {
            name: {"w": streamed(transition[name], streams)} for name in recurrences
        }
        stepped = {}
        for name, recurrence in recurrences.items():
            stepped[name], carried[name] = recurrence.stepped(
                carried[name], derivatives[name], reset=zeros, emphasis=ones
            )
        updates, state, block, branch = rule.update(
            delta=delta,
            traces=stepped,
            derivatives=derivatives,
            direct=None,
            step=step,
            params=PARAMS,
            state=state,
        )
        taken.append((updates, state, block, branch))
    return taken


SEQUENCE = [
    {"delta": 0.7, "actor": [1.0, -0.5, 0.25, 0.0], "critic": [0.2, 0.4, -1.0, 0.5]},
    {"delta": -1.3, "actor": [0.3, 0.9, -0.2, 0.6], "critic": [-0.7, 0.1, 0.8, -0.4]},
    {"delta": 0.4, "actor": [-0.6, 0.2, 1.1, -0.3], "critic": [0.5, -0.9, 0.3, 0.7]},
    {"delta": 2.1, "actor": [0.8, 0.4, -0.7, 0.2], "critic": [0.1, 0.6, 0.2, -0.8]},
]


# ------------------------------------------------- it generalizes the paper's
def test_one_branch_alone_is_the_published_intentional_step():
    """Delete a branch and what is left is Eq. 12, over the paper's optimizer.

    The published rule is the case this has to contain, and containing it is
    what makes the second branch an addition rather than a replacement. The
    comparison is `allclose` rather than exact because the two arrange the same
    quotient differently -- ``eta / d`` there, ``1 / (d / eta)`` here -- which
    costs one rounding and nothing else.
    """

    single = JointIntentionalOptimizer(
        SETTINGS, decays={"critic": DECAY_CRITIC}, signals={"critic": TD}
    )
    published = IntentionalOptimizer(
        IntentionalUpdate(
            eta=ETA_CRITIC,
            clip=SETTINGS.clip,
            beta_rms=SETTINGS.beta_rms,
            beta_clip=SETTINGS.beta_clip,
            beta_advantage=SETTINGS.beta_advantage,
            beta_momentum=SETTINGS.beta_momentum,
            denominator_floor=FLOOR,
            eps=EPS,
        ),
        decay=DECAY_CRITIC,
        signal=TD,
    )

    recurrence = Trace(decay=DECAY_CRITIC, reads=CARRIED, emphasized=False)
    joint_carried = recurrence.initial(PARAMS, 1)
    published_carried = recurrence.initial(PARAMS, 1)
    joint_state = single.init(PARAMS, streams=1)
    published_state = published.init(PARAMS, streams=1)
    ones = jnp.ones((1,), dtype=jnp.float32)
    zeros = jnp.zeros((1,), dtype=jnp.float32)

    for step, transition in enumerate(SEQUENCE, start=1):
        delta = jnp.asarray([transition["delta"]], dtype=jnp.float32)
        derivative = {"w": streamed(transition["critic"])}
        joint_used, joint_carried = recurrence.stepped(
            joint_carried, derivative, reset=zeros, emphasis=ones
        )
        published_used, published_carried = recurrence.stepped(
            published_carried, derivative, reset=zeros, emphasis=ones
        )
        joint_update, joint_state, _, _ = single.update(
            delta=delta,
            traces={"critic": joint_used},
            derivatives={"critic": derivative},
            direct=None,
            step=step,
            params=PARAMS,
            state=joint_state,
        )
        published_update, published_state, _ = published.update(
            delta=delta,
            trace=published_used,
            derivative=derivative,
            direct=None,
            step=step,
            params=PARAMS,
            state=published_state,
        )
        np.testing.assert_allclose(
            np.asarray(joint_update["w"]),
            np.asarray(published_update["w"]),
            rtol=1e-6,
            atol=0.0,
            err_msg=f"step {step}",
        )


# ------------------------------------------------------ eta keeps its meaning
def test_every_branch_keeps_its_own_intended_fraction():
    """``alpha * sqrt(sigma_bar_b * q_b) <= eta_b``, for each branch at once.

    This is the property the output position does not have. There each branch
    honours its own ``eta`` over its own update and nothing honours anything
    over the sum; here one step honours both, which is what makes ``eta_actor``
    and ``eta_critic`` mean what the paper means by them.
    """

    etas = {"actor": ETA_ACTOR, "critic": ETA_CRITIC}
    for step, (_, _, block, branch) in enumerate(drive(optimizer(), SEQUENCE), start=1):
        alpha = float(block.step_size[0])
        for name in BRANCHES:
            spent = alpha * float(branch[name].denominator[0])
            assert spent <= etas[name] * (1 + 1e-5), (step, name, spent)


def test_the_block_never_steps_further_than_either_branch_alone_would():
    """``alpha <= min_b alpha_b``, which is what caps a collapsing branch."""

    for step, (_, _, block, branch) in enumerate(drive(optimizer(), SEQUENCE), start=1):
        alpha = float(block.step_size[0])
        each = [float(branch[name].step_size[0]) for name in BRANCHES]
        assert alpha <= min(each) * (1 + 1e-5), (step, alpha, each)


def test_a_branch_whose_credit_vanishes_cannot_take_the_block_with_it():
    """The failure this position exists to close.

    An entry one head does not need has a naturally tiny gradient there, and
    that branch's own Eq. 12 step size grows without bound because there is
    nothing left to divide by. Under an output aggregation that unbounded step
    is added straight to the parameters. Here the other branch holds it: the
    block's step stays just under the *live* branch's own step size.

    "Just under" and not "at" is the arithmetic being honest. The starved
    branch's step size is not infinite -- the denominator floor caps it at
    ``eta / floor`` -- so it still claims its share of the ellipsoid, and the
    block gives up that share. At the published floor the share is a fraction
    of a percent, and it shrinks as the floor does.
    """

    starved = [
        {"delta": row["delta"], "actor": row["actor"], "critic": [0.0] * WIDTH}
        for row in SEQUENCE
    ]
    for step, (updates, _, block, branch) in enumerate(
        drive(optimizer(), starved), start=1
    ):
        alpha = float(block.step_size[0])
        live = float(branch["actor"].step_size[0])
        assert np.isfinite(alpha), step
        # The starved branch would have taken the whole of `eta / floor`.
        assert float(branch["critic"].step_size[0]) == pytest.approx(
            ETA_CRITIC / FLOOR, rel=1e-5
        )
        assert 0.99 * live <= alpha <= live * (1 + 1e-5), (step, alpha, live)
        assert np.all(np.isfinite(np.asarray(updates["w"])))


# ---------------------------------------------------- one metric, two sigmas
def test_the_metric_is_a_second_moment_of_the_summed_derivative():
    """``rho`` is the parameters' and not an objective's.

    Split per objective, an entry one head never touches has a second moment
    that decays at ``beta_rms`` with nothing to refresh it, and ``1/sqrt(nu)``
    reads "this objective does not use this entry" as "this entry is finely
    scaled". The summed derivative has no such entry. See issue 87.
    """

    first = SEQUENCE[0]
    _, state, _, _ = drive(optimizer(), SEQUENCE[:1])[0]
    summed = np.asarray(first["actor"], dtype=np.float32) + np.asarray(
        first["critic"], dtype=np.float32
    )
    # The first sample of a corrected average arrives whole.
    np.testing.assert_allclose(
        np.asarray(state.nu["w"][0]), summed**2, rtol=1e-6, atol=1e-8
    )


def test_sigma_is_each_branch_s_own_gradient_in_the_shared_metric():
    """No statistic is summed across branches before it is squared.

    Which is the property that closes the anti-alignment case: a ``sigma``
    taken from the summed derivative carries ``2 <p_actor, rho p_critic>``, and
    on a coordinate the two heads' credit opposes on that term is negative and
    shrinks the denominator the step size divides by. There is no such term
    here, and this pins it by recomputing each branch's ``sigma`` from that
    branch's derivative alone.
    """

    first = SEQUENCE[0]
    _, state, _, branch = drive(optimizer(), SEQUENCE[:1])[0]
    rho = 1.0 / (np.sqrt(np.asarray(state.nu["w"][0])) + EPS)
    for name in BRANCHES:
        own = np.asarray(first[name], dtype=np.float32)
        np.testing.assert_allclose(
            float(branch[name].sigma_bar[0]),
            float(np.sum(rho * own**2)),
            rtol=1e-5,
        )


def test_each_branch_averages_sigma_at_its_own_decay():
    """A branch's ``sigma_bar`` forgets at the rate its trace forgets at.

    Two decays is half of what makes the two branches two. If they shared one,
    the actor's statistic would be averaged at the critic's horizon and the
    step size would be sized against a history neither objective has.
    """

    fast = drive(optimizer(), SEQUENCE)
    slowed = drive(
        JointIntentionalOptimizer(
            SETTINGS, decays={"actor": 0.1, "critic": DECAY_CRITIC}, signals=SIGNALS
        ),
        SEQUENCE,
        decays={"actor": 0.1, "critic": DECAY_CRITIC},
    )
    # The critic's branch saw the same decay and the same inputs; the actor's
    # did not, so exactly one of the two statistics is allowed to have moved.
    last_fast, last_slow = fast[-1][3], slowed[-1][3]
    assert float(last_fast["actor"].sigma_bar[0]) != pytest.approx(
        float(last_slow["actor"].sigma_bar[0]), rel=1e-6
    )


# ------------------------------------------------------------ what is refused
def test_an_untraced_term_is_refused():
    """The entropy direction belongs inside the derivative the actor traces."""

    rule = optimizer()
    state = rule.init(PARAMS, streams=1)
    with pytest.raises(ValueError, match="no untraced term"):
        rule.update(
            delta=jnp.zeros((1,)),
            traces={name: PARAMS for name in BRANCHES},
            derivatives={name: PARAMS for name in BRANCHES},
            direct={"actor": PARAMS},
            step=1,
            params=PARAMS,
            state=state,
        )


def test_a_branch_without_an_allowance_is_refused():
    with pytest.raises(ValueError, match="no eta_torso"):
        JointIntentionalOptimizer(
            SETTINGS, decays={"torso": 0.5}, signals={"torso": TD}
        )


def test_a_signal_that_is_not_one_of_the_two_is_refused():
    with pytest.raises(ValueError, match="not one of the signals"):
        JointIntentionalOptimizer(
            SETTINGS, decays={"actor": 0.5}, signals={"actor": "entropy"}
        )


def test_a_decay_without_a_signal_is_refused():
    with pytest.raises(ValueError, match="one decay and one signal per branch"):
        JointIntentionalOptimizer(
            SETTINGS, decays={"actor": 0.5, "critic": 0.5}, signals={"actor": ADVANTAGE}
        )


# ------------------------------------------------------------------- streams
def test_every_stream_carries_its_own_step_size():
    """The statistics are per stream, and only the finished update is averaged."""

    taken = drive(optimizer(), SEQUENCE, streams=3)
    _, state, block, _ = taken[-1]
    assert block.step_size.shape == (3,)
    assert state.sigma_bar["actor"].shape == (3,)
    assert jax.tree.leaves(state.nu)[0].shape == (3, WIDTH)
    # One row per stream of the same numbers, so the three step sizes agree and
    # the averaged update is one stream's.
    assert float(block.step_size[0]) == pytest.approx(float(block.step_size[2]))
