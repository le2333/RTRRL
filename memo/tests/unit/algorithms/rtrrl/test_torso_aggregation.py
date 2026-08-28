"""Where the two heads' credit for the shared torso is combined.

Four configurations, from two independent choices -- the position the actor's
and the critic's contributions are added at, and the rule each is added under::

                       IU              ObGD
    before the rule    input_iu        input_obgd
    after the rule     output_iu       output_obgd

The claim that makes them four rather than two spellings of two is that neither
rule is linear in what it is given, so ``Rule(a + b)`` is not
``Rule(a) + Rule(b)``. ``test_the_two_positions_are_not_two_names_for_one_rule``
is that claim, driven with every other difference between the positions -- the
trace decays above all -- held equal, so the disagreement it finds can only be
the position.

Two levels of oracle:

the components, below
    :class:`Trace` and :class:`IntentionalOptimizer` or ``make_bounded_rule``,
    driven by hand over the same transitions. Each branch of an output
    aggregation is one single-path learner and has to be exactly that learner's
    numbers; the finished torso update has to be their elementwise sum and
    nothing else.
a real graph
    that a run document can reach all four, that the states stay apart under
    resets, several streams and a resumption, and that what a run files says
    which position produced it.

``tests/unit/algorithms/rtrrl/test_rtrrl_assembly.py`` holds the input position
against everything that was true of the torso before there was a second one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import expand
from memorax.rl.intentional import (
    ADVANTAGE,
    TD,
    IntentionalOptimizer,
    IntentionalUpdate,
)
from memorax.rl.traces import CARRIED, CURRENT, Trace
from memorax.rl.updates import (
    AdaptiveObBoundFixed,
    ObBound,
    ObGDStep,
    Sgd,
    make_bounded_rule,
)
from tests.support.builders import graph_of
from tests.support.environments import TinyContinuousEnv
from tests.support.numerics import flattened

STREAMS = 2
WIDTH = 3
PARAMS = {"w": jnp.zeros((WIDTH,), dtype=jnp.float32)}

GAMMA = 0.9
# Different on purpose: an output branch traces at its own head's rate, and a
# test that gave the three lambdas one value could not see that it does.
LAMBDA_PI = 0.8
LAMBDA_V = 0.5
LAMBDA_RNN = 0.6

ETA_ACTOR = 0.05
ETA_CRITIC = 0.5
LR = 1e-3
KAPPA = 2.0

IU_ACTOR = IntentionalUpdate(eta=ETA_ACTOR)
IU_CRITIC = IntentionalUpdate(eta=ETA_CRITIC)
OBGD_ACTOR = ObGDStep(bound=ObBound(kappa=KAPPA), lr=LR)
OBGD_CRITIC = ObGDStep(bound=ObBound(kappa=KAPPA / 2), lr=LR * 2)
# The plain ObGD bound is `lr / max(1, |delta| * ||z||_1 * lr * kappa)`, and
# while the maximum picks one the rule is `lr * delta * z` -- linear, and
# therefore the same at both positions. It is the bound that makes it not, so
# the case about the two positions being different algorithms runs at settings
# the bound actually reaches, and checks that it did.
OBGD_SATURATED = ObGDStep(bound=ObBound(kappa=50.0), lr=0.5)


def config(optimizer, **overrides):
    """An RTRRL configuration that differs only in what the torso steps under."""

    settings = {
        "num_envs": STREAMS,
        "gamma": GAMMA,
        "lambda_pi": LAMBDA_PI,
        "lambda_v": LAMBDA_V,
        "lambda_rnn": LAMBDA_RNN,
        "eta_f": 1.0,
        "torso_grad_clip": 0.0,
        "torso_optimizer": optimizer,
    }
    return rtrrl.RTRRLConfig(**{**settings, **overrides})


def streamed(rows):
    return {"w": jnp.asarray(rows, dtype=jnp.float32)}


def scalar(value):
    return jnp.asarray([value] * STREAMS, dtype=jnp.float32)


# One sequence, and a reset partway through it so the traces have a boundary to
# drop. The two heads' contributions differ in direction and in size, because
# an aggregation that summed them before either mattered would look right on
# any sequence where they were the same vector.
SEQUENCE = [
    {
        "actor": [[1.0, -2.0, 0.5], [0.5, 0.5, -1.0]],
        "critic": [[0.25, 1.0, -0.5], [-1.0, 0.25, 0.75]],
        "delta": 0.5,
        "reset": 0.0,
        "emphasis": 1.0,
    },
    {
        "actor": [[-0.5, 0.25, 1.5], [2.0, -0.5, 0.25]],
        "critic": [[1.0, 0.5, 0.5], [0.25, -1.5, 0.5]],
        "delta": -1.5,
        "reset": 0.0,
        "emphasis": 0.9,
    },
    {
        "actor": [[0.1, -0.1, 0.2], [-0.2, 0.3, 0.1]],
        "critic": [[-0.3, 0.2, 0.1], [0.4, 0.1, -0.2]],
        "delta": 2.0,
        "reset": 1.0,
        "emphasis": 1.0,
    },
    {
        "actor": [[1.5, 0.5, -1.5], [0.5, 1.0, 0.5]],
        "critic": [[-0.5, 1.5, 0.5], [1.0, -0.5, 1.5]],
        "delta": 0.75,
        "reset": 0.0,
        "emphasis": 0.81,
    },
]


def branch_derivatives(transition):
    return {
        "actor": streamed(transition["actor"]),
        "critic": streamed(transition["critic"]),
    }


def joint_derivative(transition):
    """What an input aggregation traces: the two contributions added.

    The pullback is linear, so summing the two cotangents and pulling back once
    -- which is what the input aggregation does -- reaches the same vector as
    summing the two pulled-back derivatives. That equality is what lets one
    sequence drive both positions; see
    ``test_the_two_positions_pull_back_the_same_credit``.
    """

    branches = branch_derivatives(transition)
    return jax.tree.map(lambda a, b: a + b, branches["actor"], branches["critic"])


def drive(aggregation, sequence, *, direct=None):
    """Every transition through one aggregation, applying each update as it comes."""

    split = isinstance(aggregation, rtrrl.OutputAggregation)
    params = PARAMS
    traces = aggregation.initial_traces(params)
    state = aggregation.init(params, traces)
    taken = []
    for step, transition in enumerate(sequence, start=1):
        delta = scalar(transition["delta"])
        aggregated = aggregation.step(
            params=params,
            carried=traces,
            traced=(
                branch_derivatives(transition)
                if split
                else joint_derivative(transition)
            ),
            direct=None if direct is None else direct(transition, split=split),
            delta=delta,
            sign=jnp.sign(delta),
            step=step,
            state=state,
            reset=scalar(transition["reset"]),
            emphasis=scalar(transition["emphasis"]),
        )
        params = jax.tree.map(lambda old, up: old + up, params, aggregated.update)
        traces, state = aggregated.traces, aggregated.state
        taken.append(aggregated)
    return params, taken


# ------------------------------------------------- one branch, on its own terms
def intentional_reference(settings, *, decay, signal, derivatives):
    """One single-path intentional learner, driven over the same transitions.

    The pair RTRRL builds for an intentional selection: the accumulating trace
    read after this step's derivative has joined it, and the optimizer that
    steps along what the trace hands back.
    """

    trace = Trace(decay=GAMMA * decay, reads=CURRENT, emphasized=False)
    optimizer = IntentionalOptimizer(settings, decay=GAMMA * decay, signal=signal)
    carried = trace.initial(PARAMS, STREAMS)
    state = optimizer.init(PARAMS, streams=STREAMS)
    updates = []
    for step, (transition, derivative) in enumerate(
        zip(SEQUENCE, derivatives), start=1
    ):
        used, carried = trace.stepped(
            carried,
            derivative,
            reset=scalar(transition["reset"]),
            emphasis=scalar(transition["emphasis"]),
        )
        update, state, _ = optimizer.update(
            delta=scalar(transition["delta"]),
            trace=used,
            derivative=derivative,
            direct=None,
            step=step,
            params=PARAMS,
            state=state,
        )
        updates.append(update)
    return updates


def obgd_reference(step_settings, *, decay, derivatives):
    """One single-path ObGD learner, over RTRRL's own trace.

    ``make_bounded_rule`` is the rule StreamAC answers to and the one the torso
    reaches; nothing about the bound is re-implemented here, which is the point
    of the reference.
    """

    trace = Trace(decay=GAMMA * decay, reads=CARRIED, emphasized=True)
    rule = make_bounded_rule(bound=step_settings.bound, base=Sgd(lr=step_settings.lr))
    carried = trace.initial(PARAMS, STREAMS)
    state = rule.init(params={"one": PARAMS}, traces={"one": carried})
    updates = []
    for step, (transition, derivative) in enumerate(
        zip(SEQUENCE, derivatives), start=1
    ):
        used, carried = trace.stepped(
            carried,
            derivative,
            reset=scalar(transition["reset"]),
            emphasis=scalar(transition["emphasis"]),
        )
        result = rule.apply(
            {"one": used},
            None,
            state,
            delta=scalar(transition["delta"]),
            derivative={"one": derivative},
            step=step,
            params={"one": PARAMS},
        )
        state = result.state
        updates.append(result.updates["one"])
    return updates


def assert_exactly(actual, expected, what):
    for step, (got, wanted) in enumerate(zip(actual, expected), start=1):
        np.testing.assert_allclose(
            np.asarray(got["w"]),
            np.asarray(wanted["w"]),
            rtol=0,
            atol=0,
            err_msg=f"{what}: step {step}",
        )


# ------------------------------------------------------------------ the pullback
def test_the_two_positions_pull_back_the_same_credit():
    """Both read one forward and one sensitivity; only the sum moves.

    The input position adds the two cotangents and pulls back once, which is
    the arithmetic every recorded run answers to and is kept. The output
    position pulls each back on its own. A linear pullback makes the two agree
    on the total, which is what lets one sequence drive both -- and is also why
    the difference between the positions has to come from the rules rather than
    from here.
    """

    matrix = jnp.asarray(np.arange(2 * WIDTH, dtype=np.float32).reshape(2, WIDTH) / 3.0)

    def upstream(cotangent):
        return ({"w": cotangent @ matrix},)

    actor = jnp.asarray([1.0, -2.0], dtype=jnp.float32)
    critic = jnp.asarray([0.5, 0.25], dtype=jnp.float32)

    joint = rtrrl.InputAggregation(config(IU_CRITIC), IU_CRITIC, clip=0.0).cotangents(
        upstream, actor=actor, critic=critic
    )
    split = rtrrl.OutputAggregation(
        config(rtrrl.OutputSteps(actor=IU_ACTOR, critic=IU_CRITIC)),
        rtrrl.OutputSteps(actor=IU_ACTOR, critic=IU_CRITIC),
        clip=0.0,
    ).cotangents(upstream, actor=actor, critic=critic)

    np.testing.assert_allclose(
        np.asarray(joint["w"]),
        np.asarray(split["actor"]["w"] + split["critic"]["w"]),
        rtol=1e-6,
        atol=1e-7,
    )


# --------------------------------------------------------- the branches, exactly
def test_each_intentional_branch_is_a_single_branch_learner_exactly():
    """Two learners, each the one it would be alone, and one elementwise sum.

    Each branch is measured against a whole single-path intentional learner --
    its own trace at its own head's decay, its own second moments, its own
    clipping statistic, its own dynamic step size -- driven over that head's
    contribution and nothing else. The finished torso update is then required
    to be the two added leaf by leaf, with no bound, no norm and no rescaling
    between them.
    """

    steps = rtrrl.OutputSteps(actor=IU_ACTOR, critic=IU_CRITIC)
    _, taken = drive(rtrrl.OutputAggregation(config(steps), steps, clip=0.0), SEQUENCE)

    branches = {
        "actor": intentional_reference(
            IU_ACTOR,
            decay=LAMBDA_PI,
            signal=ADVANTAGE,
            derivatives=[branch_derivatives(t)["actor"] for t in SEQUENCE],
        ),
        "critic": intentional_reference(
            IU_CRITIC,
            decay=LAMBDA_V,
            signal=TD,
            derivatives=[branch_derivatives(t)["critic"] for t in SEQUENCE],
        ),
    }
    for name, expected in branches.items():
        assert_exactly(
            [aggregated.taken[name].updates[name] for aggregated in taken],
            expected,
            f"the {name} branch",
        )
    assert_exactly(
        [aggregated.update for aggregated in taken],
        [
            {"w": actor["w"] + critic["w"]}
            for actor, critic in zip(branches["actor"], branches["critic"])
        ],
        "the finished torso update",
    )


def test_each_obgd_branch_is_a_single_branch_learner_exactly():
    """The same claim under the bound, with the two branches' bounds unequal.

    ``kappa`` and the base rate differ between the branches, so a run that had
    quietly shared one bound's statistics between them would not reproduce
    either reference. The sum is required to be an addition and only that:
    each bound constrains its own branch, and the total may exceed both.
    """

    steps = rtrrl.OutputSteps(actor=OBGD_ACTOR, critic=OBGD_CRITIC)
    _, taken = drive(rtrrl.OutputAggregation(config(steps), steps, clip=0.0), SEQUENCE)

    branches = {
        "actor": obgd_reference(
            OBGD_ACTOR,
            decay=LAMBDA_PI,
            derivatives=[branch_derivatives(t)["actor"] for t in SEQUENCE],
        ),
        "critic": obgd_reference(
            OBGD_CRITIC,
            decay=LAMBDA_V,
            derivatives=[branch_derivatives(t)["critic"] for t in SEQUENCE],
        ),
    }
    for name, expected in branches.items():
        assert_exactly(
            [aggregated.taken[name].updates[name] for aggregated in taken],
            expected,
            f"the {name} branch",
        )
    assert_exactly(
        [aggregated.update for aggregated in taken],
        [
            {"w": actor["w"] + critic["w"]}
            for actor, critic in zip(branches["actor"], branches["critic"])
        ],
        "the finished torso update",
    )


def test_the_input_position_is_the_joint_rule_over_the_joint_derivative():
    """One trace at ``lambda_rnn``, one rule state, one step, both rules.

    The input position is what the torso has always done and what every
    recorded run answers to, so it is held against the same single-path
    references the branches are -- driven over the summed derivative, at the
    torso's own decay, and with nothing else in the way.
    """

    joint = [joint_derivative(transition) for transition in SEQUENCE]

    _, intentional = drive(
        rtrrl.InputAggregation(config(IU_CRITIC), IU_CRITIC, clip=0.0), SEQUENCE
    )
    assert_exactly(
        [aggregated.update for aggregated in intentional],
        intentional_reference(
            IU_CRITIC, decay=LAMBDA_RNN, signal=TD, derivatives=joint
        ),
        "input_iu",
    )

    _, bounded = drive(
        rtrrl.InputAggregation(config(OBGD_CRITIC), OBGD_CRITIC, clip=0.0), SEQUENCE
    )
    assert_exactly(
        [aggregated.update for aggregated in bounded],
        obgd_reference(OBGD_CRITIC, decay=LAMBDA_RNN, derivatives=joint),
        "input_obgd",
    )


# ------------------------------------------- the claim that makes them four
@pytest.mark.parametrize(
    ("settings", "name"),
    [(IU_CRITIC, "iu"), (OBGD_SATURATED, "obgd")],
)
def test_the_two_positions_are_not_two_names_for_one_rule(settings, name):
    """``Rule(a + b) != Rule(a) + Rule(b)``, in situ and with nothing else moving.

    Both branches are given the same settings and the three lambdas are pinned
    to one value, so the two positions run identical traces at an identical
    decay under identical rules over identical contributions. Everything that
    could differ between them has been removed except where the sum happens.
    What is left is the non-linearity: the intentional update's step size is
    derived from statistics of what it is handed, and ObGD's bound reads the
    norm of what it is handed, so neither distributes over an addition.

    ObGD's is conditional and the case says so by running where the condition
    holds. Below its bound the rule is a rate times a trace, which does
    distribute, and the two positions then agree to the last bit -- a true
    statement about the rule and a useless one to have written this test on.
    """

    same = {"lambda_pi": 0.7, "lambda_v": 0.7, "lambda_rnn": 0.7}
    steps = rtrrl.OutputSteps(actor=settings, critic=settings)

    joint, _ = drive(
        rtrrl.InputAggregation(config(settings, **same), settings, clip=0.0),
        SEQUENCE,
    )
    split, _ = drive(
        rtrrl.OutputAggregation(config(steps, **same), steps, clip=0.0),
        SEQUENCE,
    )

    if name == "obgd":
        _, bounded = drive(
            rtrrl.InputAggregation(config(settings, **same), settings, clip=0.0),
            SEQUENCE,
        )
        sizes = [
            float(np.min(np.asarray(aggregated.taken.metrics["step_size"])))
            for aggregated in bounded
        ]
        assert min(sizes) < settings.lr, (
            "the bound never engaged, so this sequence could not tell the two "
            "positions apart even if they were different algorithms"
        )

    apart = float(jnp.max(jnp.abs(joint["w"] - split["w"])))
    scale = float(jnp.max(jnp.abs(joint["w"])))
    assert apart > 1e-6 * max(scale, 1.0), (
        f"{name} reached the same parameters at both positions, which would "
        "make the two aliases rather than two algorithms"
    )


def test_the_intentional_branches_carry_their_own_state_and_their_own_eta():
    """Nothing is shared: not the statistics, and not the intended reduction.

    The critic's branch steps along a TD error and carries no advantage scale;
    the actor's normalizes an advantage and does. Two second moments, two
    clipping statistics, two dynamic step sizes, and two ``eta``s that a run
    can set apart -- which is the configuration the published actor-critic has
    and the joint torso cannot express.
    """

    steps = rtrrl.OutputSteps(actor=IU_ACTOR, critic=IU_CRITIC)
    aggregation = rtrrl.OutputAggregation(config(steps), steps, clip=0.0)
    _, taken = drive(aggregation, SEQUENCE)
    last = taken[-1]

    actor = last.state["actor"]["actor"]
    critic = last.state["critic"]["critic"]
    assert actor.advantage_scale is not None
    assert critic.advantage_scale is None
    assert not np.allclose(np.asarray(actor.nu["w"]), np.asarray(critic.nu["w"]))
    assert not np.allclose(np.asarray(actor.sigma_bar), np.asarray(critic.sigma_bar))
    # The clipping statistic is the one thing the two agree on, and agreeing is
    # correct: it is a running average of the TD error, both branches are
    # credited by the same TD error, and both average it at the same rate. Two
    # copies of one number is what independent state looks like here.
    np.testing.assert_allclose(
        np.asarray(actor.delta_square), np.asarray(critic.delta_square)
    )

    sizes = {
        name: np.asarray(last.taken[name].metrics["step_size"][name])
        for name in ("actor", "critic")
    }
    assert not np.allclose(
        sizes["actor"], sizes["critic"]
    ), "the two branches derived one step size, so they are not two learners"


def test_changing_one_branch_leaves_the_other_untouched():
    """A branch's settings reach that branch and stop there.

    The actor's ``eta`` is moved and the critic's is not; the critic's state,
    its update and its step size have to come back bit for bit unchanged. This
    is what "two rule states over one parameter group" means, and it is the
    property a shared statistic scaled by two constants would fail.
    """

    def run(eta):
        steps = rtrrl.OutputSteps(actor=IntentionalUpdate(eta=eta), critic=IU_CRITIC)
        return drive(rtrrl.OutputAggregation(config(steps), steps, clip=0.0), SEQUENCE)

    _, reference = run(ETA_ACTOR)
    _, changed = run(ETA_ACTOR * 4)

    for step, (before, after) in enumerate(zip(reference, changed), start=1):
        np.testing.assert_array_equal(
            np.asarray(before.taken["critic"].updates["critic"]["w"]),
            np.asarray(after.taken["critic"].updates["critic"]["w"]),
            err_msg=f"the critic branch moved at step {step}",
        )
        np.testing.assert_array_equal(
            np.asarray(before.traces["critic"]["w"]),
            np.asarray(after.traces["critic"]["w"]),
            err_msg=f"the critic trace moved at step {step}",
        )
    assert not np.array_equal(
        np.asarray(reference[-1].taken["actor"].updates["actor"]["w"]),
        np.asarray(changed[-1].taken["actor"].updates["actor"]["w"]),
    ), "the actor branch did not move, so the run under test changed nothing"


def test_the_entropy_direction_reaches_the_actor_branch_and_no_other():
    """Whose objective the term belongs to decides which branch carries it.

    The critic's branch is handed no untraced direction at all, so its update
    has to be the same with the entropy present as without it; the actor's has
    to differ. Under the intentional rule the direction is folded into the
    traced derivative rather than added beside it, which is where the paper's
    policy gradient puts it.
    """

    steps = rtrrl.OutputSteps(actor=IU_ACTOR, critic=IU_CRITIC)
    aggregation = rtrrl.OutputAggregation(config(steps), steps, clip=0.0)

    def entropy(transition, *, split):
        term = streamed([[0.05, -0.05, 0.1]] * STREAMS)
        return {"actor": term} if split else term

    _, without = drive(aggregation, SEQUENCE)
    _, with_entropy = drive(aggregation, SEQUENCE, direct=entropy)

    for step, (bare, folded) in enumerate(zip(without, with_entropy), start=1):
        np.testing.assert_array_equal(
            np.asarray(bare.taken["critic"].updates["critic"]["w"]),
            np.asarray(folded.taken["critic"].updates["critic"]["w"]),
            err_msg=f"the entropy reached the critic branch at step {step}",
        )
    assert not np.array_equal(
        np.asarray(without[0].taken["actor"].updates["actor"]["w"]),
        np.asarray(with_entropy[0].taken["actor"].updates["actor"]["w"]),
    ), "the entropy direction never reached the actor branch"


# ------------------------------------------------------ the bound, made readable
def test_the_bound_statistics_are_the_terms_the_step_size_is_a_quotient_of():
    """A small rate is attributable rather than only observable.

    ``step_size`` alone says a rate was small and not why. It is ``lr`` over
    ``max(1, delta_bar * trace_sum * lr * kappa)`` and none of those terms can
    be recovered from the quotient, so each is reported: this holds the four
    against the arithmetic they came from, and against the step size they
    produce, on a sequence where the bound engages on some steps and not on
    others.
    """

    steps = rtrrl.OutputSteps(actor=OBGD_SATURATED, critic=OBGD_SATURATED)
    _, taken = drive(rtrrl.OutputAggregation(config(steps), steps, clip=0.0), SEQUENCE)

    engaged = 0
    for index, aggregated in enumerate(taken, start=1):
        for name in ("actor", "critic"):
            result = aggregated.taken[name]
            reading = result.metrics["obgd"]
            size = np.asarray(result.metrics["step_size"])
            pressure = np.asarray(reading.bound_denominator)

            np.testing.assert_allclose(
                pressure,
                np.asarray(reading.delta_bar)
                * np.asarray(reading.trace_sum)
                * OBGD_SATURATED.lr
                * OBGD_SATURATED.bound.kappa,
                rtol=1e-6,
                err_msg=f"{name} step {index}: the denominator is not its own terms",
            )
            np.testing.assert_allclose(
                size,
                OBGD_SATURATED.lr / np.maximum(1.0, pressure),
                rtol=1e-6,
                err_msg=f"{name} step {index}: the step size is not what it bounds to",
            )
            np.testing.assert_allclose(
                np.asarray(reading.bound_scale),
                size / OBGD_SATURATED.lr,
                rtol=1e-6,
                err_msg=f"{name} step {index}: the scale is not what survived",
            )
            # The bound is active exactly where the product exceeds one, which
            # is why the product is reported before the maximum rather than
            # after: afterwards a run approaching its bound and a run nowhere
            # near it are the same number.
            assert np.all((pressure > 1.0) == (size < OBGD_SATURATED.lr - 1e-12))
            engaged += int(np.any(pressure > 1.0))

    assert engaged, "the bound never engaged, so half of these assertions are vacuous"
    assert engaged < 2 * len(taken) * 2, (
        "the bound engaged on every step and stream, so the inactive branch of "
        "the maximum was never taken"
    )


def test_the_plain_bound_normalizes_by_nothing_and_reports_no_second_moment():
    """A reading exists where the quantity does, down to which bound was named.

    The adaptive bounds divide the trace by ``sqrt(v_hat)``, and the mean of
    that denominator is what separates a rate held down by the second moment
    from one held down by the trace itself. The plain bound has no such
    denominator, and reporting a one there would be a normalization a reader
    could not tell from an absent one.
    """

    plain = rtrrl.OutputSteps(actor=OBGD_ACTOR, critic=OBGD_CRITIC)
    adaptive_step = ObGDStep(
        bound=AdaptiveObBoundFixed(kappa=2.0, beta2=0.999, eps=1e-8), lr=LR
    )
    adaptive = rtrrl.OutputSteps(actor=adaptive_step, critic=adaptive_step)

    _, without = drive(
        rtrrl.OutputAggregation(config(plain), plain, clip=0.0), SEQUENCE
    )
    _, with_moment = drive(
        rtrrl.OutputAggregation(config(adaptive), adaptive, clip=0.0), SEQUENCE
    )

    assert without[-1].taken["actor"].metrics["obgd"].second_moment_rms is None
    moment = with_moment[-1].taken["actor"].metrics["obgd"].second_moment_rms
    assert moment is not None
    assert np.all(np.asarray(moment) > 0.0)

    # And the normalization reaches the L1 the bound reads, which is what makes
    # the two trace sums different numbers rather than one number twice.
    assert not np.allclose(
        np.asarray(without[-1].taken["actor"].metrics["obgd"].trace_sum),
        np.asarray(with_moment[-1].taken["actor"].metrics["obgd"].trace_sum),
    )


# ---------------------------------------------------------- a whole run document
def iu_branch(prefix, eta):
    return {f"{prefix}.eta": eta}


def obgd_branch(prefix, *, kappa, lr):
    return {
        f"{prefix}.bound.kind": "adaptive_ob_fixed",
        f"{prefix}.bound.adaptive_ob_fixed.kappa": kappa,
        f"{prefix}.bound.adaptive_ob_fixed.beta2": 0.999,
        f"{prefix}.bound.adaptive_ob_fixed.eps": 1e-8,
        f"{prefix}.lr": lr,
    }


# `expand` fills every leaf a selected branch declares from the low end of its
# search domain, so only what the case is about is spelled out here. The four
# modes' remaining settings are the published ones by construction.
MODES = {
    "input_iu": {**iu_branch("torso.optimizer.input_iu", ETA_CRITIC)},
    "input_obgd": obgd_branch("torso.optimizer.input_obgd", kappa=2.0, lr=1e-3),
    "output_iu": {
        **iu_branch("torso.optimizer.output_iu.actor", ETA_ACTOR),
        **iu_branch("torso.optimizer.output_iu.critic", ETA_CRITIC),
    },
    "output_obgd": {
        **obgd_branch("torso.optimizer.output_obgd.actor", kappa=2.0, lr=1e-3),
        **obgd_branch("torso.optimizer.output_obgd.critic", kappa=1.0, lr=2e-3),
    },
}

LIVE = {
    "eta_f": 1.0,
    "eta_pi": 1.0,
    "lambda_pi": 0.9,
    "lambda_v": 0.9,
    "lambda_rnn": 0.9,
    "entropy_rate": 1e-5,
}


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


def assembled(mode, *, num_envs=2, overrides=None):
    parameters = expand(
        rtrrl.PARAMETERS,
        {
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 4,
            "torso.backbone.lru.hidden_dim": 2,
            "torso.backbone.lru.differentiation.kind": "exact_rtrl",
            "torso.optimizer.kind": mode,
            # Every rule the torso may now step under sizes or bounds its own
            # step, so the outer clip is off for all four.
            "torso.grad_clip": 0.0,
            "torso.follow": 0.25,
            "actor.optimizer.kind": "adam",
            "actor.optimizer.adam.lr": 5e-4,
            "critic.optimizer.kind": "adam",
            "critic.optimizer.adam.lr": 5e-4,
            "actor.head.kind": "state_std",
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": False,
            **LIVE,
            **MODES[mode],
            **(overrides or {}),
        },
    )
    return assemble(
        rtrrl.RTRRL,
        BuildRequest(
            parameters=parameters,
            environment=EnvironmentSpec(
                id="tiny", backend=None, observed=None, episode_length=8
            ),
            num_envs=num_envs,
            record=rtrrl.OBSERVATIONS.trajectory_fields,
        ),
        environment_factory=tiny_environment,
    )


def test_an_input_run_files_the_bound_statistics_of_the_rule_it_selected():
    """The joint path reports whichever of the two self-sizing rules it took.

    Both derive their own step size and both have statistics behind it that the
    step size does not carry, so the position is not what decides whether they
    are filed -- the rule is.
    """

    bounded = set(assembled("input_obgd").observations.series)
    for quantity in (
        "trace_sum",
        "delta_bar",
        "bound_denominator",
        "bound_scale",
        "second_moment_rms",
    ):
        assert f"update.torso.obgd.{quantity}" in bounded, quantity
    assert not any(".intentional." in name for name in bounded if "torso" in name)

    intentional = set(assembled("input_iu").observations.series)
    assert "update.torso.intentional.sigma_bar" in intentional
    assert not any(".obgd." in name for name in intentional)


@pytest.mark.parametrize("mode", sorted(MODES))
def test_every_mode_is_reachable_from_a_run_configuration(mode):
    """All four survive the parameter surface and a scan of real transitions.

    The arithmetic is driven against references above. What is asserted here is
    everything between a run document and that arithmetic: that the branch
    names resolve, that the nested bound of an ObGD branch is built rather than
    left as the name it selected, that the whole graph -- torso traces, two
    readouts, emphasis, resets -- stays finite, and that the torso moved.
    """

    built = assembled(mode)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 32)

    for path, leaf in flattened(stepped.core).items():
        assert np.all(np.isfinite(leaf)), f"{path} went non-finite"
    assert np.all(np.isfinite(metrics.update.td_error))

    before = flattened(state.core.torso.params)
    after = flattened(stepped.core.torso.params)
    assert any(
        not np.array_equal(leaf, after[path]) for path, leaf in before.items()
    ), "the torso never moved"


@pytest.mark.parametrize("mode", sorted(MODES))
def test_every_series_a_mode_declares_arrives_with_a_value(mode):
    """Declared and produced are the same set, in both directions, per mode.

    A name in the schema and not in what a step files is a series the driver
    looks for and never finds, and it fails a run rather than a test -- the
    declaration travels to Runtime on the built graph, so nothing local
    compares the two unless something like this does.

    ``flattened`` drops a ``None`` leaf, which is what makes this the check
    rather than a restatement of the schema: a reading the kernel declares and
    then never fills is absent here and present in ``observations.series``, and
    the two sets stop being equal. Asserting the names exist would not have
    noticed, which is how a whole rule's telemetry was declared and dropped on
    the way into the metrics it is filed under.
    """

    built = assembled(mode)
    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, 4)

    produced = {
        name.lstrip(".").replace("/.", ".").replace("/", ".")
        for name in flattened({"forward": metrics.forward, "update": metrics.update})
    }
    assert produced == set(built.observations.series)


@pytest.mark.parametrize("mode", ["input_obgd", "output_obgd"])
def test_the_bound_statistics_reach_the_metrics_a_run_files(mode):
    """And the values are the bound's own, not zeros standing in for it.

    The case above compares two sets of names. This reads the numbers: a rate
    that survived is in ``(0, 1]``, a denominator is not negative, and the L1
    the bound reads is above zero once a trace exists at all -- so a reading
    wired to the wrong quantity, or to a tree of zeros, fails here rather than
    being filed for the length of a run.
    """

    built = assembled(mode)
    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, 8)

    torso = metrics.update.torso
    # The base rate each path was configured with, from the same place the run
    # document got it.
    branches = (
        {"": (torso, MODES[mode]["torso.optimizer.input_obgd.lr"])}
        if mode == "input_obgd"
        else {
            "actor": (
                torso.actor,
                MODES[mode]["torso.optimizer.output_obgd.actor.lr"],
            ),
            "critic": (
                torso.critic,
                MODES[mode]["torso.optimizer.output_obgd.critic.lr"],
            ),
        }
    )
    for name, (branch, lr) in branches.items():
        reading = branch.obgd
        assert reading.bound_scale is not None, f"{name}: no bound statistics arrived"
        scale = np.asarray(reading.bound_scale)
        assert np.all(scale > 0.0) and np.all(scale <= 1.0 + 1e-6), name
        assert np.all(np.asarray(reading.bound_denominator) >= 0.0), name
        assert np.all(np.asarray(reading.delta_bar) >= 1.0), name
        assert np.any(np.asarray(reading.trace_sum) > 0.0), f"{name}: the L1 is zero"
        assert np.all(np.asarray(reading.second_moment_rms) > 0.0), name
        # And it is this branch's own rate it is a fraction of.
        np.testing.assert_allclose(
            scale, np.asarray(branch.step_size) / lr, rtol=1e-5, err_msg=name
        )


@pytest.mark.parametrize("mode", ["output_iu", "output_obgd"])
def test_an_output_run_files_its_branches_and_no_joint_reading(mode):
    """A reading exists where the quantity does, and the position decides where.

    Under an output aggregation there is no joint derivative, no joint trace
    and no joint step size, so the four names that would report them are absent
    and each branch reports its own. A run that could only see the summed
    update could not tell a branch that stopped contributing from one whose
    contribution the other cancelled.
    """

    filed = set(assembled(mode).observations.series)

    for name in ("grad_norm.recurrence", "trace_norm.recurrence", "step_size"):
        assert f"update.torso.{name}" not in filed, name
    for branch in ("actor", "critic"):
        assert f"update.torso.{branch}.step_size" in filed
        assert f"update.torso.{branch}.grad_norm.recurrence" in filed
        assert f"update.torso.{branch}.trace_norm.recurrence" in filed

    def under(rule):
        return {
            name
            for name in filed
            if name.startswith("update.torso.") and f".{rule}." in name
        }

    if mode == "output_iu":
        intentional = under("intentional")
        assert "update.torso.actor.intentional.advantage_scale" in intentional
        # The critic's branch steps along a TD error, so it has no advantage to
        # normalize and files no scale for one.
        assert "update.torso.critic.intentional.advantage_scale" not in intentional
        assert "update.torso.critic.intentional.sigma_bar" in intentional
        assert not under("obgd"), "an intentional run filed bound statistics"
    else:
        assert not under("intentional"), "ObGD filed intentional readings"
        for branch in ("actor", "critic"):
            for quantity in (
                "trace_sum",
                "delta_bar",
                "bound_denominator",
                "bound_scale",
                "second_moment_rms",
            ):
                assert f"update.torso.{branch}.obgd.{quantity}" in filed, quantity
        # Each branch's, and no joint one: there is no joint bound to report.
        assert not any(
            name.startswith("update.torso.obgd.") for name in filed
        ), "an output aggregation filed a joint bound statistic"


def test_the_two_branches_stay_apart_across_streams_resets_and_a_resumption():
    """What a checkpoint is here: the state crossing an invocation boundary.

    Both branches' traces and both rule states have to be in what is handed
    back, have to survive a serialization round trip, and have to leave the
    transitions that follow exactly as they would have been. Run over several
    streams and an episode length short enough that resets land inside the
    window, because a reset that dropped one branch's trace and not the other's
    would show up nowhere else.
    """

    built = assembled("output_iu", num_envs=3)
    graph = graph_of(built)
    state = built.program.init(jax.random.key(0))
    keys = jax.random.split(jax.random.key(1), 12)

    uninterrupted, _ = jax.lax.scan(graph.train_step, state, keys)
    halfway, _ = jax.lax.scan(graph.train_step, state, keys[:6])

    saved = serialization.to_state_dict(halfway)
    assert set(saved["core"]["torso"]["traces"]) == {"actor", "critic"}
    held = saved["core"]["rule"]["torso"]
    assert set(held) == {"actor", "critic"}
    for branch in ("actor", "critic"):
        assert {"nu", "momentum", "sigma_bar", "delta_square"} <= set(
            held[branch][branch]
        ), f"the {branch} branch's intentional state is not all of it"

    restored = serialization.from_state_dict(halfway, saved)
    resumed, _ = jax.lax.scan(graph.train_step, restored, keys[6:])

    got = flattened(resumed.core)
    for path, leaf in flattened(uninterrupted.core).items():
        np.testing.assert_array_equal(got[path], leaf, err_msg=path)

    # And the two traces are genuinely two: the same tree twice would make
    # every assertion above pass while there was only one learner.
    traces = halfway.core.torso.traces
    assert not np.allclose(
        np.asarray(flattened(traces["actor"])[next(iter(flattened(traces["actor"])))]),
        np.asarray(
            flattened(traces["critic"])[next(iter(flattened(traces["critic"])))]
        ),
    )


def test_the_torso_is_written_once_and_the_sum_is_not_bounded_again():
    """One parameter write, one projection, one followed copy, and no third bound.

    The update the parameters received has to be the two branch updates added
    leaf by leaf -- so a clip, a norm or a rescale applied to the total would
    fail here even if it happened to be inert on most steps.
    """

    steps = rtrrl.OutputSteps(actor=OBGD_ACTOR, critic=OBGD_CRITIC)
    _, taken = drive(rtrrl.OutputAggregation(config(steps), steps, clip=0.0), SEQUENCE)

    for step, aggregated in enumerate(taken, start=1):
        summed = (
            aggregated.taken["actor"].updates["actor"]["w"]
            + aggregated.taken["critic"].updates["critic"]["w"]
        )
        np.testing.assert_array_equal(
            np.asarray(aggregated.update["w"]),
            np.asarray(summed),
            err_msg=f"something touched the sum at step {step}",
        )


# ------------------------------------------------------------------- refusals
@pytest.mark.parametrize("mode", sorted(MODES))
def test_a_clip_over_a_self_sizing_step_is_refused_with_the_reason(mode):
    """Every rule the torso may step under now bounds or sizes its own step.

    A configuration that also clipped the finished update would be applying a
    second bound it never declared, and one it could not account for, so the
    build refuses it rather than accepting a run that is not the algorithm it
    names.
    """

    with pytest.raises(ValueError, match="second, undeclared bound"):
        assembled(mode, overrides={"torso.grad_clip": 1.0})


def test_the_old_iu_branch_name_is_refused_rather_than_translated():
    """``iu`` named what is now ``input_iu`` and is not quietly read as it.

    The name had one meaning and now has a position in it. A run document that
    still carries the old one is refused with the branches it could have named,
    so the migration is a thing someone did rather than a thing that happened.
    """

    with pytest.raises(KeyError, match="input_iu"):
        assemble(
            rtrrl.RTRRL,
            BuildRequest(
                parameters={
                    **expand(rtrrl.PARAMETERS, {"torso.optimizer.kind": "input_iu"}),
                    "torso.optimizer.kind": "iu",
                },
                environment=EnvironmentSpec(
                    id="tiny", backend=None, observed=None, episode_length=8
                ),
                num_envs=1,
            ),
            environment_factory=tiny_environment,
        )


def test_the_declared_surface_names_the_position_rather_than_implying_it():
    """Four branches a run selects by name, and no way to end up at one by default.

    The position is in the branch name, so a configuration cannot arrive at an
    output aggregation by setting a parameter that happens to exist, and cannot
    arrive at an input one by leaving one out.
    """

    branches = set(rtrrl.RTRRL_TORSO_OPTIMIZERS.branches)
    assert branches == {
        "adam",
        "sgd",
        "d_rtrrl",
        "input_iu",
        "input_obgd",
        "output_iu",
        "output_obgd",
    }
    assert "iu" not in branches
    # Adam stays first, because a configuration that names no optimizer is
    # filled from the front of the search domain.
    assert next(iter(rtrrl.RTRRL_TORSO_OPTIMIZERS.branches)) == "adam"

    declared = set(expand(rtrrl.PARAMETERS, {"torso.optimizer.kind": "output_obgd"}))
    for branch in ("actor", "critic"):
        assert f"torso.optimizer.output_obgd.{branch}.lr" in declared
        assert f"torso.optimizer.output_obgd.{branch}.bound.kind" in declared
    assert not any(
        name.startswith("torso.optimizer.input_") for name in declared
    ), "an unselected position's parameters came back filled in"


def test_the_head_optimizers_cannot_name_a_torso_position():
    """A readout has one derivative and one trace; there is nothing to aggregate.

    The torso's two contribution optimizers are a different concept from the
    heads' own, and offering the four positions where a head selects would let
    a configuration say something that has no meaning there.
    """

    assert set(rtrrl.RTRRL_OPTIMIZERS.branches) == {"adam", "sgd", "d_rtrrl", "iu"}


def test_a_run_document_reaches_the_same_graph_the_config_builds():
    """The declared surface and the direct construction are one algorithm.

    ``config(...)`` builds the aggregation from Python objects and the cases
    above drive it; a run document builds it from names. This is the one place
    that says the two arrive at the same component.
    """

    graph = graph_of(assembled("output_obgd"))
    aggregation = graph.core.aggregation
    assert isinstance(aggregation, rtrrl.OutputAggregation)
    assert aggregation.position == rtrrl.OUTPUT
    assert set(aggregation.paths) == {"actor", "critic"}
    for name, path in aggregation.paths.items():
        assert isinstance(path.step, ObGDStep), name
        assert isinstance(path.step.bound, object) and not isinstance(
            path.step.bound, str
        ), f"the {name} branch's bound was left as the name it selected"

    assert isinstance(
        graph_of(assembled("input_iu")).core.aggregation, rtrrl.InputAggregation
    )
