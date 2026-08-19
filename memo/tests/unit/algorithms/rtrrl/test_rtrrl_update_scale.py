"""The six per-group numbers R2 characterizes a collapse with.

A collapse is read off the fixed-evaluation curve, but what it is *explained*
by is what the updates were doing around it: how big a step the trace and the
TD error asked for, how much of it the rule's scale handling granted, and how
far the parameters actually moved. Those three are only an explanation if each
is measured rather than inferred from the others, so each is checked here
against a quantity computed outside the graph from the states either side of a
step.

Every case runs one transition at a time, because that is the only way to hold
the incoming trace and the outgoing parameters of the same step. One stream,
so that a per-stream reading and a whole-group one are the same number and a
disagreement is a disagreement about arithmetic rather than about averaging.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.networks.sequence import PLACES
from memorax.readings import taken
from memorax.runtime import EpisodeTracker
from tests.support.builders import D_RTRRL, C, assemble_rtrrl, graph_of
from tests.support.numerics import flattened

ETA_F = 1.0
STEPS = 6
BLOCKS = ("torso", "actor", "critic")


def norm(tree) -> float:
    return float(
        jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(tree)))
    )


def walked(optimizer=None, steps=STEPS):
    """One transition at a time, keeping what each step was handed and did.

    The first step is dropped throughout: it steps with the initial trace,
    which is zero, so every scale reading on it is a degenerate zero that says
    nothing about the arithmetic under test.
    """

    built = assemble_rtrrl(optimizer={"eta_f": ETA_F, **(optimizer or {})})
    state = built.program.init(jax.random.key(0))
    walk = []
    for step in range(steps):
        before = state
        state, metrics = built.program.train(jax.random.key(step + 1), state, 1)
        walk.append((before, state, metrics))
    return walk[1:]


def reading(metrics, block, name) -> float:
    """One block's scale reading at the single transition of a one-step call."""

    return float(np.asarray(getattr(getattr(metrics.update, block), name)).ravel()[0])


@pytest.mark.parametrize("block", BLOCKS)
def test_the_raw_update_is_the_td_error_times_the_trace_that_was_stepped_with(block):
    """``m_raw = ||delta * z||``, against the trace the step actually read.

    Two things are pinned. The product is the one the issue names -- a norm of
    the weighted trace, which for a scalar TD error is ``|delta| * ||z||`` --
    and the ``z`` in it is the *incoming* trace rather than the one this step
    accumulated into. The trace advances after the update uses it, so reading
    the advanced one would report a quantity no step ever took, and would do it
    plausibly enough that nothing downstream would notice.
    """

    for before, _, metrics in walked():
        delta = float(np.asarray(metrics.update.td_error).ravel()[0])
        scale = ETA_F if block == "torso" else 1.0
        trace = getattr(before.core, block).traces

        np.testing.assert_allclose(
            reading(metrics, block, "raw_update_norm"),
            abs(delta * scale) * norm(trace),
            rtol=1e-5,
            err_msg=block,
        )
        np.testing.assert_allclose(
            reading(metrics, block, "used_trace_norm"), norm(trace), rtol=1e-5
        )
        np.testing.assert_allclose(
            reading(metrics, block, "abs_td_error"), abs(delta * scale), rtol=1e-5
        )


def test_the_torsos_td_error_is_the_one_its_own_group_was_handed():
    """``abs_td_error`` is per group, and the two groups are handed different ones.

    RTRRL scales the torso's TD error by ``eta_f`` before its rule sees it. A
    single run-wide ``|delta|`` would be the same number under three names and
    would misreport the torso's by exactly that factor.
    """

    for _, _, metrics in walked({"eta_f": 4.0}):
        head = reading(metrics, "actor", "abs_td_error")
        np.testing.assert_allclose(
            reading(metrics, "torso", "abs_td_error"), 4.0 * head, rtol=1e-5
        )
        np.testing.assert_allclose(
            reading(metrics, "critic", "abs_td_error"), head, rtol=1e-5
        )


@pytest.mark.parametrize("optimizer", [None, D_RTRRL], ids=["original", "fixed_step"])
@pytest.mark.parametrize("block", BLOCKS)
def test_the_realized_update_is_the_distance_the_parameters_moved(optimizer, block):
    """``||dtheta||``, measured on the parameters rather than on the intent.

    This is the reading that makes the other five falsifiable. The raw update
    and the multiplier are what the rule meant to do; a rule that clipped
    somewhere the telemetry does not know about, or an outer bound applied
    after the reported factor, shows up here as a realized norm that is not the
    product -- and the fork arms are compared on exactly this number.
    """

    for before, after, metrics in walked(optimizer):
        moved = jax.tree.map(
            lambda old, new: new - old,
            getattr(before.core, block).params,
            getattr(after.core, block).params,
        )
        # The tolerance is float32 cancellation, not slack in the claim: an
        # update a ten-thousandth the size of the parameter it lands on is
        # recovered by subtraction with several of its digits gone, while the
        # reading measures the update itself. Anything wrong with the reading
        # is wrong by a factor, not by a last bit.
        np.testing.assert_allclose(
            reading(metrics, block, "realized_update_norm"),
            norm(moved),
            rtol=1e-4,
            atol=1e-9,
            err_msg=f"{block} did not move what it reported",
        )


def test_under_the_saturated_arm_the_multiplier_is_what_normalizes_the_trace():
    """``sign``'s factor is ``C / (||z|| + eps)``, and the reading says so.

    Asserting the product rather than the factor is deliberate: what the arm is
    defined by is that the step comes out ``C`` long whatever the trace was,
    and the multiplier is exactly the number that makes that true. The outer
    clip is off, so this factor is the only one in the way.
    """

    for _, _, metrics in walked({**D_RTRRL, "torso.grad_clip": 0.0}):
        for block in BLOCKS:
            product = reading(metrics, block, "clip_multiplier") * reading(
                metrics, block, "used_trace_norm"
            )
            np.testing.assert_allclose(product, C, rtol=1e-4, err_msg=block)


def test_the_clip_fraction_is_the_indicator_of_the_bound_having_bitten():
    """Zero and one at a transition; what it is a fraction of is the scope.

    Both directions are driven, because a reading that is always one is as
    useless as one that is always zero: a clip far below the raw update binds
    on every step, and one far above it binds on none. The threshold in between
    is what R2 reports, and it is only informative if the two ends work.
    """

    def fractions(clip):
        return [
            (
                reading(metrics, "torso", "clip_fraction"),
                reading(metrics, "torso", "clip_multiplier"),
            )
            for _, _, metrics in walked({"torso.grad_clip": clip})
        ]

    for fraction, multiplier in fractions(1e-6):
        assert fraction == 1.0
        assert multiplier < 1.0

    for fraction, multiplier in fractions(1e6):
        assert fraction == 0.0
        np.testing.assert_allclose(multiplier, 1.0)


def test_the_original_clip_binds_the_group_and_is_reported_against_its_blocks():
    """The original clip is a group-wide bound, and the reading does not pretend.

    ``clip_by_global_norm`` scales the whole group by one factor, so the actor
    and the critic -- which step as one group -- read the same multiplier. That
    is the truth about the bound rather than a limitation of the reading, and
    it is asserted so that a later per-block bound cannot be introduced without
    this saying that the meaning changed.
    """

    for _, _, metrics in walked({"torso.grad_clip": 1e-6}):
        np.testing.assert_allclose(
            reading(metrics, "actor", "clip_multiplier"),
            reading(metrics, "critic", "clip_multiplier"),
        )
        # The heads group is not clipped at all, and the torso's clip is not
        # the heads': one bound per group, and the torso's is the tight one.
        assert reading(metrics, "torso", "clip_multiplier") < reading(
            metrics, "actor", "clip_multiplier"
        )


def test_every_scale_reading_reaches_the_metric_names_the_catalog_declares():
    """A reading the kernel produces but nothing can name is not observability."""

    for block in BLOCKS:
        for name in (
            "abs_td_error",
            "used_trace_norm",
            "raw_update_norm",
            "clip_multiplier",
            "clip_fraction",
            "realized_update_norm",
        ):
            path = f"update.{block}.{name}"
            assert path in rtrrl.TRAINING_METRICS
            # The catalog declares the episode reduction of each, which is what
            # `metrics.jsonl` keeps and what a score may name; the step and
            # window scopes of the same path are the dashboard's.
            assert f"train/episode/{path}" in rtrrl.METRICS
            assert path in rtrrl.OBSERVATIONS.series


def test_a_run_that_does_not_ask_for_a_scale_reading_does_not_compute_one():
    """The declaration gates the arithmetic, as it does for every other reading.

    Six readings for each of three blocks is eighteen extra series on every
    transition of a million-step run. They are cheap, but "cheap" is not the
    contract: what a run takes is what its declaration says, and a reading that
    could not be turned off would be a cost no configuration could decline.
    """

    reports = replace(
        rtrrl.Reports(),
        torso=replace(rtrrl.BlockReports(), raw_update_norm=False),
    )
    built = assemble_rtrrl()
    graph_of(built).core.reports = reports

    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, 2)

    assert metrics.update.torso.raw_update_norm is None
    assert metrics.update.actor.raw_update_norm is not None
    assert "update.torso.raw_update_norm" not in taken(reports, parts=PLACES)
    assert "update.actor.raw_update_norm" in taken(reports, parts=PLACES)


def test_the_readings_are_finite_through_a_run_that_actually_learns():
    """Non-finite telemetry is a result, so it must not be produced by accident."""

    for _, _, metrics in walked(D_RTRRL):
        for block in BLOCKS:
            for name in (
                "abs_td_error",
                "used_trace_norm",
                "raw_update_norm",
                "clip_multiplier",
                "clip_fraction",
                "realized_update_norm",
            ):
                value = reading(metrics, block, name)
                assert np.isfinite(value), f"{block}.{name} went non-finite"


def test_the_scale_readings_travel_with_the_episode_they_belong_to():
    """They are per-transition series, so an episode carries one value of each."""

    built = assemble_rtrrl()
    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, 24)

    tracker = EpisodeTracker(
        observations=built.observations, num_envs=1, max_episode_steps=16
    )
    episodes = list(tracker.consume(metrics, start_env_steps=0).completed)
    assert episodes, "no episode completed, so nothing carried a series"
    for episode in episodes:
        for block in BLOCKS:
            name = f"update.{block}.realized_update_norm"
            assert name in episode.series
            assert len(episode.series[name]) == len(episode.rewards)
    assert set(rtrrl.TRAINING_METRICS) <= set(episodes[0].series)


def test_the_flattened_state_is_untouched_by_taking_the_readings():
    """Diagnostics that move a number are not diagnostics."""

    quiet = assemble_rtrrl()
    state = quiet.program.init(jax.random.key(0))
    stepped, _ = quiet.program.train(jax.random.key(1), state, 4)

    again = quiet.program.train(jax.random.key(1), state, 4)[0]
    left, right = flattened(stepped), flattened(again)
    assert all(np.array_equal(left[path], right[path]) for path in left)
