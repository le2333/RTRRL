"""The D-RTRRL rule: the trace says where, the learning rate says how far.

There is no upstream implementation to drive this against, so nothing here
compares two copies of a guess. Every assertion is a property the rule is
*defined* by and that a rearrangement of its arithmetic would break: the length
of the step, the direction of the step, what a degenerate input does, and which
values provably cannot reach the step at all.

The last kind is the point of several of these. ``eta_f`` and a neighbouring
block's trace are both quantities that used to move a step and no longer can,
and a comment saying so is not checkable. Asserting that two configurations
give bit-identical updates pins the inertness where a reader of the diff would
have to re-derive it.

Streams are the env axis, axis 0 of every trace leaf. Most cases use one stream,
where ``||dtheta|| == eta`` holds exactly; the multi-stream case is separate
because averaging unit vectors is deliberately not a unit vector.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.rl.updates import DRTRRL, make_d_rtrrl_rule

ETA = 0.01
EPS = 1e-8


def blocks(key, *, streams=1, actor_scale=1.0):
    """Two blocks under one rule group, shaped the way RTRRL groups them."""

    actor_key, critic_key = jax.random.split(key)
    return {
        "actor": {
            "kernel": actor_scale * jax.random.normal(actor_key, (streams, 4, 3)),
            "bias": actor_scale * jax.random.normal(actor_key, (streams, 3)),
        },
        "critic": {"kernel": jax.random.normal(critic_key, (streams, 4, 1))},
    }


def stepped(traces, delta, *, direct=None, clip=0.0, **settings):
    """One application of the rule, with the defaults the surface declares."""

    rule = make_d_rtrrl_rule(DRTRRL(eta=ETA, **settings), clip=clip)
    return rule.apply(
        traces,
        direct,
        rule.init(params=None, traces=traces),
        delta=jnp.asarray(delta, dtype=jnp.float32),
        step=1,
        params=None,
    )


def norm(tree):
    return float(
        jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(tree)))
    )


def leaves(tree):
    """Every leaf under the name it is spelled with, so two trees can be paired."""

    return {
        "/".join(str(getattr(entry, "key", entry)) for entry in path): leaf
        for path, leaf in jax.tree.flatten_with_path(tree)[0]
    }


def scaled_to(tree, length):
    """One block rescaled so its norm is exactly ``length``."""

    return jax.tree.map(lambda leaf: leaf * (length / norm(tree)), tree)


@pytest.mark.parametrize("denominator", ["shifted", "clamped"])
@pytest.mark.parametrize("surprise", [0.5, -0.5, 7.0, -300.0], ids=str)
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3], ids=["tiny", "unit", "huge"])
def test_the_traced_step_is_eta_long_whatever_the_surprise_or_the_trace(
    denominator, surprise, scale
):
    """The defining property: length is the learning rate and nothing else.

    Neither the size of the TD error nor the size of the trace reaches the
    distance moved. This is the whole reason the rule exists -- it is what
    ``eta * delta * z_hat`` did not do, and what took that version non-finite.

    The claim is about the *traced* part, which is why ``direct`` is absent
    here: the entropy direction does not pass through the trace and is not
    normalized, so it is a length this property says nothing about. It is
    decomposed on its own further down rather than left to blur this one.

    Both denominators are checked because both are offered. Neither is exactly
    ``eta`` -- ``shifted`` is short by ``eps/(||z||+eps)`` and ``clamped`` is
    exact only once ``||z|| >= eps`` -- and at these trace scales the gap is
    far below the tolerance. Where it is not below the tolerance is the
    separate concern of the two tests that follow.
    """

    traces = jax.tree.map(
        lambda leaf: scale * leaf, blocks(jax.random.key(0))
    )
    output = stepped(traces, [surprise], denominator=denominator)

    for name, block in output.updates.items():
        np.testing.assert_allclose(norm(block), ETA, rtol=1e-5, err_msg=name)


def test_where_eps_sits_is_which_denominator_it_is():
    """The two denominators are two rules, and a run answers to one of them.

    Preregistration named ``||z|| + eps``; this issue's body named
    ``max(||z||, eps)``. They are not the same arithmetic near zero, so the
    choice is declared and this is what declaring it buys: at a trace whose
    norm is exactly ``eps`` the shifted form has already halved the step while
    the clamped form is still taking the whole of it. Substituting one for the
    other silently would move a recorded run by that factor.
    """

    traces = blocks(jax.random.key(9))
    at_eps = {name: scaled_to(block, EPS) for name, block in traces.items()}

    shifted = stepped(at_eps, [1.0], denominator="shifted")
    clamped = stepped(at_eps, [1.0], denominator="clamped")

    # One trace of norm eps, normalized two ways: eps/(eps+eps) against eps/eps.
    ratio = norm(shifted.updates) / norm(clamped.updates)
    np.testing.assert_allclose(ratio, 0.5, rtol=1e-3)

    # And they agree again once the trace clears eps by enough to drown it.
    ordinary = (
        stepped(traces, [1.0], denominator="shifted"),
        stepped(traces, [1.0], denominator="clamped"),
    )
    np.testing.assert_allclose(
        norm(ordinary[0].updates), norm(ordinary[1].updates), rtol=1e-6
    )


def test_the_direction_is_the_traces_own_turned_by_the_sign_of_the_td_error():
    """Trace times TD error is the direction, and it is the only direction."""

    traces = blocks(jax.random.key(1))
    for surprise, turn in ((2.5, 1.0), (-2.5, -1.0)):
        output = stepped(traces, [surprise])
        for name in traces:
            moved = leaves(output.updates[name])
            trace = {path: leaf[0] for path, leaf in leaves(traces[name]).items()}
            cosine = sum(
                float(jnp.sum(moved[path] * trace[path])) for path in trace
            ) / (norm(output.updates[name]) * norm(trace))
            np.testing.assert_allclose(cosine, turn, rtol=1e-5, err_msg=name)


@pytest.mark.parametrize("denominator", ["shifted", "clamped"])
@pytest.mark.parametrize(
    "traces, delta, why",
    [
        (None, [0.0], "a TD error of exactly zero"),
        ("zero", [1.5], "a trace of exactly zero"),
    ],
    ids=["no_td", "no_trace"],
)
def test_a_degenerate_input_gives_exactly_no_step(denominator, traces, delta, why):
    """Both degenerate cases give zero, under both denominators, finitely.

    A rule that moves a fixed distance every step has to be told when not to,
    and there are exactly two such moments: nothing was surprising, or nothing
    has been credited yet.

    Neither denominator has any advantage here, which is worth asserting
    because it is easy to assume otherwise: ``0 / (0 + eps)`` is exactly zero
    for the same reason ``0 / eps`` is. The two forms differ near zero, not at
    it, and that difference is the subject of its own test.
    """

    built = blocks(jax.random.key(2))
    if traces == "zero":
        built = jax.tree.map(jnp.zeros_like, built)
    output = stepped(built, delta, denominator=denominator)

    for name, block in output.updates.items():
        for path, leaf in leaves(block).items():
            assert jnp.all(jnp.isfinite(leaf)), f"{why}: {name}/{path} is not finite"
            assert jnp.all(leaf == 0.0), f"{why}: {name}/{path} moved"


def test_the_traced_and_direct_contributions_are_separable_and_only_one_is_fixed():
    """The fixed-step property belongs to the trace, and entropy is not in it.

    The direct directions do not pass through the trace and are not
    normalized, so a step taken with entropy on is not ``eta`` long. Reporting
    the two contributions apart is what keeps that from quietly falsifying the
    property the rule is chosen for: the traced part is ``eta``, the direct
    part is whatever the entropy gradient happens to be times ``eta``, and the
    step is their sum.
    """

    traces = blocks(jax.random.key(11))
    entropy = jax.tree.map(
        lambda leaf: 0.3 * jnp.ones_like(leaf), traces
    )

    traced_only = stepped(traces, [1.0])
    both = stepped(traces, [1.0], direct=entropy)

    for name in traces:
        # The traced contribution, alone, is the length the rule promises.
        np.testing.assert_allclose(norm(traced_only.updates[name]), ETA, rtol=1e-5)

        # The direct contribution is exactly `eta * d`, recoverable by
        # difference -- so it is additive and identifiable rather than folded
        # into the normalization.
        for path, leaf in leaves(both.updates[name]).items():
            recovered = leaf - leaves(traced_only.updates[name])[path]
            np.testing.assert_allclose(
                recovered,
                ETA * leaves(entropy[name])[path][0],
                rtol=1e-5,
                err_msg=f"{name}/{path}",
            )

        # And the total is therefore not eta, which is the thing that must not
        # be claimed by accident. It is not bounded above by eta either -- the
        # direct term can point with the trace or against it, so the sum can
        # land on either side. All that is asserted is that it stops being the
        # promised length.
        assert not np.isclose(norm(both.updates[name]), ETA, rtol=1e-3), (
            f"{name}: entropy left the step exactly eta long by coincidence; "
            "pick a direct term that actually moves it"
        )


def test_the_td_error_scale_cannot_change_a_signed_step():
    """``eta_f`` is inert under the default, and that is a contract.

    RTRRL hands the torso group ``delta * eta_f``. Under ``magnitude: sign``
    the rule reads ``sign(eta_f * delta)``, which for any positive ``eta_f`` is
    ``sign(delta)`` -- so the knob that used to set the torso-to-heads step
    ratio no longer sets anything, and that ratio has to be carried by the two
    ``eta`` values instead. Asserting bit-equality here is what stops the knob
    from being quietly tuned against a step it cannot move.
    """

    traces = blocks(jax.random.key(3), streams=3)
    delta = [0.7, -1.9, 4.0]

    plain = stepped(traces, delta)
    scaled = stepped(traces, [100.0 * value for value in delta])

    for name in plain.updates:
        for path, leaf in leaves(plain.updates[name]).items():
            assert np.array_equal(leaf, leaves(scaled.updates[name])[path]), (
                f"eta_f reached the step through {name}/{path}"
            )


def test_the_td_error_scale_does_reach_the_unsafe_ablation():
    """The same knob is live under ``td_error``, which is why it is a branch.

    If both settings ignored the scale there would be one rule with a spare
    name. This is the difference the ablation exists to measure.
    """

    traces = blocks(jax.random.key(3), streams=3)
    delta = [0.7, -1.9, 4.0]

    plain = stepped(traces, delta, magnitude="td_error")
    scaled = stepped(traces, [100.0 * value for value in delta], magnitude="td_error")

    assert not np.allclose(
        leaves(plain.updates["critic"])["kernel"],
        leaves(scaled.updates["critic"])["kernel"],
    )


def test_one_blocks_trace_cannot_spend_anothers_step():
    """``scope: block`` is what keeps two readouts from sharing a unit ball.

    The heads group holds the actor and the critic together. Normalized as one
    tree, an actor trace a thousand times the critic's takes nearly the whole
    step and the critic stops learning -- so the two scopes are two algorithms,
    and the default has to be the one where a block's own trace is all that
    sets its direction.
    """

    ordinary = blocks(jax.random.key(4))
    loud = blocks(jax.random.key(4), actor_scale=1000.0)

    by_block = (stepped(ordinary, [1.0]), stepped(loud, [1.0]))
    for path, leaf in leaves(by_block[0].updates["critic"]).items():
        assert np.array_equal(leaf, leaves(by_block[1].updates["critic"])[path]), (
            f"the actor's trace reached the critic through {path}"
        )

    together = (
        stepped(ordinary, [1.0], scope="group"),
        stepped(loud, [1.0], scope="group"),
    )
    assert not np.allclose(
        leaves(together[0].updates["critic"])["kernel"],
        leaves(together[1].updates["critic"])["kernel"],
    ), "under one norm the critic must feel the actor, or the scopes are one scope"


def test_streams_that_disagree_take_a_shorter_step_than_streams_that_agree():
    """Averaging unit vectors is not a unit vector, and that is the point.

    ``||dtheta|| == eta`` is a per-stream statement. Across streams the rule
    averages, so streams pointing different ways shorten the step on their own.
    """

    agreeing = blocks(jax.random.key(5), streams=1)
    agreeing = jax.tree.map(
        lambda leaf: jnp.repeat(leaf, 4, axis=0), agreeing
    )
    disagreeing = blocks(jax.random.key(6), streams=4)

    surprise = [1.0] * 4
    np.testing.assert_allclose(
        norm(stepped(agreeing, surprise).updates["critic"]), ETA, rtol=1e-5
    )
    assert norm(stepped(disagreeing, surprise).updates["critic"]) < ETA


def test_the_direct_directions_are_the_only_part_a_clip_can_reach():
    """Why ``grad_clip`` stays on: the traced part is bounded, this is not.

    Entropy does not pass through the trace and is not normalized, so it is the
    one term in this rule that can be any length at all.
    """

    traces = blocks(jax.random.key(7))
    huge = jax.tree.map(lambda leaf: 1e6 * jnp.ones_like(leaf), traces)

    unclipped = stepped(traces, [1.0], direct=huge)
    clipped = stepped(traces, [1.0], direct=huge, clip=1.0)

    assert norm(unclipped.updates) > 1.0
    np.testing.assert_allclose(norm(clipped.updates), 1.0, rtol=1e-5)


def test_the_rate_is_reported_under_the_name_every_rule_reports_it_under():
    """One reading means one thing whichever rule is in the way."""

    output = stepped(blocks(jax.random.key(8)), [1.0, -1.0, 0.0])
    np.testing.assert_allclose(output.metrics["step_size"], ETA)
