"""Plain SGD as one of RTRRL's update rules.

SGD is the rule with nothing in it: no preconditioner, no carried statistic, no
bound of its own. That is exactly why it is worth a file. Every other rule here
can be told apart by the arithmetic it adds, and this one can only be told apart
by the arithmetic it does *not* add -- so the assertions are about what reaches
the parameters unchanged, and about the two things a bare rate can still get
wrong on the way there: the sign it steps in, and where the torso's outer clip
sits relative to it.

The rules are reached through ``make_rules`` rather than through the private
chain builder, because half of what is asserted is routing rather than
arithmetic: which block's rate is which, and that the outer clip is the torso's
alone.

Streams are the env axis, axis 0 of every trace leaf. The combination
``delta * trace + direct`` happens per stream and the mean over streams is taken
after it, which is the order every rule here shares and the one a single-stream
test could not see.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from memorax.algorithms.rtrrl_aaai import RTRRLConfig, make_rules
from memorax.rl.intentional import IntentionalUpdate
from memorax.rl.updates import Adam, ObBound, ObGDStep, Sgd

LR = 0.1
STREAMS = 3
DELTA = (0.5, -2.0, 0.25)


def block(key, *, streams=STREAMS, scale=1.0):
    """One block's trace, shaped as the rule is handed it: ``{name: tree}``."""

    kernel_key, bias_key = jax.random.split(key)
    return {
        "kernel": scale * jax.random.normal(kernel_key, (streams, 4, 3)),
        "bias": scale * jax.random.normal(bias_key, (streams, 3)),
    }


def params_of(traced):
    """The parameters a trace is a trace of: the same tree without the env axis."""

    return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape[1:]), traced)


def rules(*, torso, actor=None, critic=None, clip=0.0, streams=STREAMS):
    """The three block rules an RTRRL configuration builds."""

    return make_rules(
        RTRRLConfig(
            num_envs=streams,
            torso_optimizer=torso,
            actor_optimizer=torso if actor is None else actor,
            critic_optimizer=torso if critic is None else critic,
            torso_grad_clip=clip,
        )
    )


def stepped(rule, traced, delta, *, direct=None):
    """One application of a rule, with everything a rule is handed."""

    params = params_of(traced)
    state = rule.init(params=params, traces=traced)
    return rule.apply(
        traced,
        direct,
        state,
        delta=jnp.asarray(delta, dtype=jnp.float32),
        # The instantaneous derivative every rule is handed beside the trace. A
        # bare rate reads neither it nor the parameters.
        derivative=traced,
        step=1,
        params=params,
    )


def combined(traced, delta, direct=None):
    """``mean(delta * trace + direct, env)`` -- the direction a rate is given.

    Written out in numpy rather than reused from ``memorax.rl.updates`` so the
    expectation is not the implementation restated.
    """

    weights = np.asarray(delta, dtype=np.float32)

    def one(trace, immediate):
        shaped = weights.reshape((-1,) + (1,) * (np.ndim(trace) - 1))
        return np.mean(shaped * np.asarray(trace) + immediate, axis=0)

    if direct is None:
        return jax.tree.map(lambda trace: one(trace, 0.0), traced)
    return jax.tree.map(lambda trace, imm: one(trace, np.asarray(imm)), traced, direct)


def scaled(tree, rate):
    return jax.tree.map(lambda leaf: rate * leaf, tree)


def global_norm(tree):
    return float(
        np.sqrt(sum(np.sum(np.square(leaf)) for leaf in jax.tree.leaves(tree)))
    )


def assert_close(actual, expected, what):
    got = jax.tree.leaves(actual)
    wanted = jax.tree.leaves(expected)
    assert len(got) == len(wanted), f"{what}: the trees have different leaves"
    for index, (taken, meant) in enumerate(zip(got, wanted)):
        np.testing.assert_allclose(
            taken, meant, rtol=1e-6, atol=1e-7, err_msg=f"{what}: leaf {index}"
        )


# ------------------------------------------------------------------- the step
@pytest.mark.parametrize("with_direct", (False, True), ids=["traced", "and-direct"])
def test_the_step_is_the_rate_over_the_combined_direction(with_direct):
    """``lr * mean(delta * trace + direct, env)``, and nothing else.

    That is the whole of standard SGD under RTRRL's contract: the algorithm
    hands over a trace, a TD error and whatever ascends untraced, and the rule
    multiplies the finished direction by one number.
    """

    traced = {"torso": block(jax.random.key(0))}
    direct = {"torso": block(jax.random.key(1))} if with_direct else None

    taken = stepped(rules(torso=Sgd(lr=LR))["torso"], traced, DELTA, direct=direct)

    assert_close(taken.updates, scaled(combined(traced, DELTA, direct), LR), "the step")


def test_the_streams_are_averaged_after_each_one_s_direction_is_finished():
    """Per stream, then averaged -- which is not averaging the traces first.

    With one TD error per stream the two orders differ whenever the errors do,
    so a rule that averaged before weighting would fail here and pass every
    single-stream assertion in this file.
    """

    traced = {"torso": block(jax.random.key(2))}
    delta = (3.0, -1.0, 0.0)

    taken = stepped(rules(torso=Sgd(lr=LR))["torso"], traced, delta)

    assert_close(taken.updates, scaled(combined(traced, delta), LR), "per stream")
    averaged_first = jax.tree.map(
        lambda leaf: LR * float(np.mean(delta)) * np.mean(np.asarray(leaf), axis=0),
        traced,
    )
    assert any(
        not np.allclose(a, b)
        for a, b in zip(jax.tree.leaves(taken.updates), jax.tree.leaves(averaged_first))
    ), "the two orders agree on these numbers, so the case proves nothing"


def test_the_rate_ascends_where_the_packaged_optax_sgd_would_descend():
    """The sign convention, stated against the thing that would have broken it.

    RTRRL applies ``parameter + update``, so a rule hands back an ascent.
    ``optax.sgd`` folds a negation into its rate, because optax also adds its
    updates and is descending a gradient; reaching for it instead of a positive
    scale would have reversed every run that selected this rule, silently and
    without changing a single shape.
    """

    traced = {"torso": block(jax.random.key(3))}

    taken = stepped(rules(torso=Sgd(lr=LR))["torso"], traced, DELTA)

    ascent = combined(traced, DELTA)
    params = params_of(traced)
    packaged = optax.sgd(LR)
    descent, _ = packaged.update(ascent, packaged.init(params), params)
    assert_close(taken.updates, scaled(descent, -1.0), "the ascent direction")


def test_the_rate_is_reported_under_the_name_every_rule_reports_a_step_by():
    traced = {"torso": block(jax.random.key(4))}
    taken = stepped(rules(torso=Sgd(lr=LR))["torso"], traced, DELTA)
    np.testing.assert_allclose(taken.metrics["step_size"], np.full(STREAMS, LR))


def test_standard_sgd_carries_nothing_from_one_step_into_the_next():
    """No momentum and no second moment: the absence is the specification.

    A rule that began accumulating something would still satisfy every
    single-step assertion above, so the repeated step is the one that says it.
    """

    rule = rules(torso=Sgd(lr=LR))["torso"]
    traced = {"torso": block(jax.random.key(5))}
    params = params_of(traced)
    state = rule.init(params=params, traces=traced)

    carried = state
    for step in range(1, 4):
        taken = rule.apply(
            traced,
            None,
            carried,
            delta=jnp.asarray(DELTA, dtype=jnp.float32),
            derivative=traced,
            step=step,
            params=params,
        )
        carried = taken.state
        # The same inputs give the same step, however many have gone before it.
        assert_close(taken.updates, scaled(combined(traced, DELTA), LR), f"step {step}")
    assert not jax.tree.leaves(carried), "a bare rate carries no statistic"


# -------------------------------------------------------------- the outer clip
@pytest.mark.parametrize("clip", (0.0, 0.25, 100.0), ids=["off", "binding", "slack"])
def test_the_torso_clip_bounds_the_finished_direction_before_the_rate(clip):
    """Where ``grad_clip`` sits for SGD is where it sits for Adam.

    The bound is on the combined torso direction -- after the streams are
    averaged, before the rate is applied -- so a clip of ``c`` bounds the
    *direction*, and the step that follows it is at most ``lr * c`` long.
    """

    traced = {"torso": block(jax.random.key(6), scale=10.0)}
    direct = {"torso": block(jax.random.key(7))}

    taken = stepped(
        rules(torso=Sgd(lr=LR), clip=clip)["torso"], traced, DELTA, direct=direct
    )

    ascent = combined(traced, DELTA, direct)
    length = global_norm(ascent)
    shrink = min(1.0, clip / length) if clip else 1.0
    assert_close(taken.updates, scaled(ascent, LR * shrink), f"clip={clip}")
    if clip == 0.25:
        assert length > clip, "the binding case does not bind"
        np.testing.assert_allclose(global_norm(taken.updates), LR * clip, rtol=1e-5)

    # And the same claim said as the optax chain the two settings are declared
    # against, which is the reference the Adam path already answered to.
    chain = optax.chain(
        *((optax.clip_by_global_norm(clip),) if clip else ()), optax.scale(LR)
    )
    params = params_of(traced)
    reference, _ = chain.update(ascent, chain.init(params), params)
    assert_close(taken.updates, reference, f"the optax chain at clip={clip}")


def test_the_two_heads_take_no_outer_clip_of_their_own():
    """Only the torso declares one, and selecting SGD does not add one elsewhere.

    Asserted on a direction long enough that a clip would show: the heads take
    the full ``lr * direction`` while the torso, on the same numbers, does not.
    """

    trace = block(jax.random.key(8), scale=10.0)
    built = rules(torso=Sgd(lr=LR), clip=0.25)

    for name in ("actor", "critic"):
        taken = stepped(built[name], {name: trace}, DELTA)
        assert_close(
            taken.updates,
            scaled(combined({name: trace}, DELTA), LR),
            f"{name} was clipped",
        )

    torso = stepped(built["torso"], {"torso": trace}, DELTA)
    unclipped = scaled(combined({"torso": trace}, DELTA), LR)
    assert global_norm(torso.updates) < global_norm(
        unclipped
    ), "the torso clip did not bind, so the heads' freedom proves nothing"


# -------------------------------------------------------- three separate blocks
def test_each_block_steps_at_its_own_rate():
    """A rate per block, not one setting reaching three.

    The same trace and the same TD error into all three; three numbers out.
    """

    rates = {"torso": 1e-1, "actor": 1e-2, "critic": 1e-3}
    built = rules(
        torso=Sgd(lr=rates["torso"]),
        actor=Sgd(lr=rates["actor"]),
        critic=Sgd(lr=rates["critic"]),
    )
    shared = block(jax.random.key(9))

    for name, rate in rates.items():
        taken = stepped(built[name], {name: shared}, DELTA)
        assert_close(
            taken.updates,
            scaled(combined({name: shared}, DELTA), rate),
            f"{name} did not step at its own rate",
        )


def test_a_block_stepped_by_sgd_leaves_a_block_stepped_by_adam_alone():
    """The mixed configuration, at the level where mixing could cross wires.

    Each block builds its own rule over its own state, so an Adam head beside
    an SGD torso is the Adam head it would have been beside another Adam.
    """

    adam = Adam(lr=LR)
    mixed = rules(torso=Sgd(lr=LR), actor=adam, critic=adam)
    alone = rules(torso=adam, actor=adam, critic=adam)
    shared = block(jax.random.key(10))

    for name in ("actor", "critic"):
        assert_close(
            stepped(mixed[name], {name: shared}, DELTA).updates,
            stepped(alone[name], {name: shared}, DELTA).updates,
            f"{name} changed when the torso did",
        )


@pytest.mark.parametrize("clip", (0.0, 0.25), ids=["off", "binding"])
def test_adam_still_steps_the_chain_it_stepped_before_sgd_joined_the_family(clip):
    """Letting SGD in must not have moved Adam by a bit.

    The same reference the Adam path has always been the composition of: the
    outer clip first, then the preconditioner, then the positive rate.
    """

    base = Adam(lr=LR, b1=0.9, b2=0.999, eps=1e-8)
    traced = {"torso": block(jax.random.key(11), scale=10.0)}

    taken = stepped(rules(torso=base, clip=clip)["torso"], traced, DELTA)

    chain = optax.chain(
        *((optax.clip_by_global_norm(clip),) if clip else ()),
        optax.scale_by_adam(b1=base.b1, b2=base.b2, eps=base.eps),
        optax.scale(base.lr),
    )
    ascent = combined(traced, DELTA)
    params = params_of(traced)
    reference, _ = chain.update(ascent, chain.init(params), params)
    assert_close(taken.updates, reference, f"adam at clip={clip}")


# ------------------------------------------------------- the clip, swept
# `torso.grad_clip` is a number an ensemble may vary between its members, and a
# member's parameters arrive as tracers because the members share one graph.
# Everything above builds a rule from a value Python can read; everything below
# builds one from a value it cannot, which is the same rule and used to be a
# `TracerBoolConversionError` at construction.
#
# R2.2's stage-one SGD study is the shape that found it: one job, four vmapped
# configurations, `grad_clip` drawn from `[0.0, 0.3, 1.0, 3.0, 10.0]`. Zero is
# among them on purpose -- it is the value whose meaning is "no clip", and the
# one a rule assembled behind `if clip:` could not carry.
SWEPT = (0.0, 0.3, 1.0, 3.0)


def swept(clips, *, base: Sgd | Adam = Sgd(lr=LR), traced, delta=DELTA, direct=None):
    """One rule per member, built inside the map from that member's clip.

    ``EnsembleRuntime`` maps a build, not a built graph, because a value closed
    into a graph is not an argument vmap can map over. This is that arrangement
    in miniature: the clip is the mapped axis and the chain is assembled under
    the trace.
    """

    def member(clip):
        return stepped(
            rules(torso=base, clip=clip)["torso"], traced, delta, direct=direct
        ).updates

    return jax.vmap(member)(jnp.asarray(clips, dtype=jnp.float32))


def member_of(mapped, index):
    return jax.tree.map(lambda leaf: leaf[index], mapped)


def test_a_swept_clip_reaches_an_update_without_being_concretised():
    """The failure this replaces: four trials, none of which reached a step.

    Nothing about the values is asserted here -- the next test does that. What
    this one fixes is that the four are *reachable at all*, which is what an
    ensemble round of R2.2 lost when the chain's length depended on the number.
    """

    traced = {"torso": block(jax.random.key(12), scale=10.0)}

    taken = swept(SWEPT, traced=traced)

    for leaf in jax.tree.leaves(taken):
        assert leaf.shape[0] == len(SWEPT), "the member axis did not survive"
        assert np.all(np.isfinite(leaf)), "a member stepped to a non-finite update"


@pytest.mark.parametrize("with_direct", (False, True), ids=["traced", "and-direct"])
def test_every_swept_member_is_the_configuration_it_stands_for(with_direct):
    """A member of a sweep is the single run it was drawn to stand for.

    Asserted against the rule built from the same number as a Python float,
    which is the configuration a one-trial job would have run. If these came
    apart, an HPO round would be scoring four runs nobody could reproduce.
    """

    traced = {"torso": block(jax.random.key(13), scale=10.0)}
    direct = {"torso": block(jax.random.key(14))} if with_direct else None

    mapped = swept(SWEPT, traced=traced, direct=direct)

    for index, clip in enumerate(SWEPT):
        alone = stepped(
            rules(torso=Sgd(lr=LR), clip=clip)["torso"], traced, DELTA, direct=direct
        )
        assert_close(member_of(mapped, index), alone.updates, f"member at clip={clip}")


def test_a_swept_zero_is_still_the_absence_of_a_clip():
    """The disabled semantics, held against the member beside it that binds.

    Zero is no bound rather than a bound at zero, and under a sweep it is a
    number carried through the chain rather than a transform left out of it --
    so it is worth saying that the number still means what its absence meant.
    """

    traced = {"torso": block(jax.random.key(15), scale=10.0)}

    mapped = swept((0.0, 0.25), traced=traced)

    unbounded = scaled(combined(traced, DELTA), LR)
    assert_close(member_of(mapped, 0), unbounded, "the swept zero clipped something")
    assert global_norm(member_of(mapped, 1)) < global_norm(
        unbounded
    ), "the bound beside it did not bind, so the zero proves nothing"


def test_adam_is_the_same_chain_whether_its_clip_was_read_or_swept():
    """The other rate that carries this clip, at the same two settings.

    Adam's arithmetic is unchanged by a clip becoming mappable, and the chain
    it answers to is still the one composed by hand in the test above.
    """

    base = Adam(lr=LR, b1=0.9, b2=0.999, eps=1e-8)
    traced = {"torso": block(jax.random.key(16), scale=10.0)}

    mapped = swept(SWEPT, base=base, traced=traced)

    for index, clip in enumerate(SWEPT):
        chain = optax.chain(
            *((optax.clip_by_global_norm(clip),) if clip else ()),
            optax.scale_by_adam(b1=base.b1, b2=base.b2, eps=base.eps),
            optax.scale(base.lr),
        )
        ascent = combined(traced, DELTA)
        params = params_of(traced)
        reference, _ = chain.update(ascent, chain.init(params), params)
        assert_close(member_of(mapped, index), reference, f"adam at clip={clip}")


@pytest.mark.parametrize(
    "torso",
    (
        IntentionalUpdate(eta=1.0),
        ObGDStep(lr=1.0, bound=ObBound(kappa=2.0)),
    ),
    ids=["intentional", "obgd"],
)
def test_a_rule_that_bounds_its_own_step_refuses_a_swept_clip_as_a_sweep(torso):
    """Not one member at a time: the whole sweep, named as the mistake it is.

    These two rules require ``grad_clip: 0``, so a sweep over this leaf offers
    them a domain in which every value but one is already refused. Under a
    trace there is no value to read and no member to name, and the honest
    refusal is of the sweep -- said at construction, with the two rules that
    would have accepted it.
    """

    def member(clip):
        return rules(torso=torso, clip=clip)

    with pytest.raises(ValueError, match="group sweeping grad_clip"):
        jax.vmap(member)(jnp.asarray(SWEPT, dtype=jnp.float32))

    # And a clip it *can* read is still refused by its own message, which is
    # the one that names the number.
    with pytest.raises(ValueError, match="second, undeclared bound"):
        rules(torso=torso, clip=1.0)
    rules(torso=torso, clip=0.0)


@pytest.mark.parametrize("clip", (0.0, 0.25), ids=["off", "binding"])
def test_a_clip_that_can_be_read_still_decides_the_chain_at_build_time(clip):
    """A rate under no clip builds no clip, and the state layout says so.

    Carrying the bound is what a *swept* clip needs, and it costs a position in
    the chain -- which is a position in the rule's carried state, because that
    state is the chain's tuple. Carrying it unconditionally would therefore
    move Adam's moments one place along in every run that never asked for a
    clip, including both readouts, which are built with ``clip=0.0`` written
    out and can never be handed a tracer at all.

    So the two are pinned here rather than left to be noticed: the state a
    configuration builds is the state of the chain it declares, and the chain
    at zero is the one with nothing in front of the preconditioner.
    """

    base = Adam(lr=LR, b1=0.9, b2=0.999, eps=1e-8)
    traced = {"torso": block(jax.random.key(17))}
    params = params_of(traced)

    rule = rules(torso=base, clip=clip)["torso"]
    reference = optax.chain(
        *((optax.clip_by_global_norm(clip),) if clip else ()),
        optax.scale_by_adam(b1=base.b1, b2=base.b2, eps=base.eps),
        optax.scale(base.lr),
    )

    assert jax.tree.structure(
        rule.init(params=params, traces=traced)
    ) == jax.tree.structure(reference.init(params))
