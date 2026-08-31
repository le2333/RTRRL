"""The third position: two credits that meet inside the rule.

``input_iu`` sums the two cotangents before anything reads them and steps one
rule over the sum. ``output_iu`` keeps them apart all the way to the parameters
and adds two finished updates, with nothing bounding the sum. ``joint_iu``
keeps everything apart that belongs to an objective -- the derivative, the
trace, the signal, ``sigma_bar``, ``eta`` -- and shares the two things that
belong to the parameters: one ``rho``, and one step size that honours both
branches' intended fractions.

``tests/unit/components/test_joint_intentional_update.py`` holds the step size
against the equation it is derived from. What is held here is the wiring: which
trees reach it, which decays they were accumulated at, where its readings are
filed, and that a whole run of it stays finite on the configuration issue 87 was
raised against.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import expand
from memorax.rl.traces import CARRIED
from tests.support.builders import graph_of
from tests.support.environments import TinyContinuousEnv

ETA_ACTOR = 0.05
ETA_CRITIC = 0.5

LAMBDA_PI = 0.9
LAMBDA_V = 0.7
LAMBDA_RNN = 0.5
GAMMA = 0.9

# `expand` fills anything unset from the low end of its search domain, which
# would leave every scale at zero and the torso never stepping at all.
LIVE = {
    "gamma": GAMMA,
    "eta_f": 1.0,
    "eta_pi": 1.0,
    "lambda_pi": LAMBDA_PI,
    "lambda_v": LAMBDA_V,
    "lambda_rnn": LAMBDA_RNN,
    "entropy_rate": 1e-5,
}

SHARED = {
    "clip": 20.0,
    "beta_rms": 0.999,
    "beta_clip": 0.9998,
    "beta_advantage": 0.9998,
    "beta_momentum": 0.0,
    "denominator_floor": 1e-8,
    "eps": 1e-8,
}


def iu(prefix, eta):
    """One published intentional selection, as a run document spells it."""

    return {f"{prefix}.eta": eta, **{f"{prefix}.{k}": v for k, v in SHARED.items()}}


def torso_branch(position):
    """The torso's selection at whichever position its credit is combined."""

    if position == "joint_iu":
        return {
            "torso.optimizer.kind": "joint_iu",
            "torso.optimizer.joint_iu.eta_actor": ETA_ACTOR,
            "torso.optimizer.joint_iu.eta_critic": ETA_CRITIC,
            **{f"torso.optimizer.joint_iu.{k}": v for k, v in SHARED.items()},
        }
    if position == "input_iu":
        return {
            "torso.optimizer.kind": "input_iu",
            **iu("torso.optimizer.input_iu", ETA_CRITIC),
        }
    return {
        "torso.optimizer.kind": "output_iu",
        **iu("torso.optimizer.output_iu.actor", ETA_ACTOR),
        **iu("torso.optimizer.output_iu.critic", ETA_CRITIC),
    }


def parameters(position, overrides=None):
    return expand(
        rtrrl.PARAMETERS,
        {
            **LIVE,
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 4,
            "torso.backbone.lru.hidden_dim": 2,
            "torso.backbone.lru.differentiation.kind": "exact_rtrl",
            # A rule that sizes its own step refuses a second, outer bound.
            "torso.grad_clip": 0.0,
            "torso.follow": 1.0,
            **torso_branch(position),
            "actor.optimizer.kind": "iu",
            **iu("actor.optimizer.iu", ETA_ACTOR),
            "critic.optimizer.kind": "iu",
            **iu("critic.optimizer.iu", ETA_CRITIC),
            "actor.head.kind": "bounded",
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": False,
            **(overrides or {}),
        },
    )


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


def assembled(position="joint_iu", *, num_envs=1, overrides=None):
    return assemble(
        rtrrl.RTRRL,
        BuildRequest(
            parameters=parameters(position, overrides),
            environment=EnvironmentSpec(
                id="tiny", backend=None, observed=None, episode_length=8
            ),
            num_envs=num_envs,
            record=rtrrl.OBSERVATIONS.trajectory_fields,
        ),
        environment_factory=tiny_environment,
    )


def finite(tree) -> bool:
    return all(
        bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in jax.tree.leaves(tree)
    )


def leaves(tree):
    return {
        jax.tree_util.keystr(path): np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


# ------------------------------------------------------------------- wiring
def test_the_position_is_two_recurrences_reaching_one_rule():
    """Which is what "combined inside the rule" means, spelled out.

    Every other aggregation keys its recurrences and its rules alike -- one
    each at the input, one each per branch at the output. This is the one place
    they do not pair, and the pairing failing is the position rather than an
    oversight: two objectives accumulate two traces, and one step size has to
    read both of them at once.
    """

    aggregation = graph_of(assembled()).core.aggregation

    assert aggregation.position == rtrrl.INSIDE
    assert sorted(aggregation.recurrences) == ["actor", "critic"]
    assert sorted(aggregation.rules) == [rtrrl.JOINT]
    assert sorted(rtrrl.make_rules(graph_of(assembled()).core.cfg)) == [
        "actor",
        "critic",
        "torso",
    ]


def test_each_trace_runs_at_its_own_head_s_decay_and_reads_the_carried_one():
    """A trace belongs to the objective, not to the block it credits.

    ``gamma * lambda_pi`` for the actor's and ``gamma * lambda_v`` for the
    critic's, and neither at the joint path's ``lambda_rnn``, which is a
    setting this position has no path for. Both read the carried trace, for the
    reason every rule in this repository does: RTRRL runs one forward per
    transition, so the error's own derivative joined the trace last step.
    """

    recurrences = graph_of(assembled()).core.aggregation.recurrences

    assert recurrences["actor"].decay == pytest.approx(GAMMA * LAMBDA_PI)
    assert recurrences["critic"].decay == pytest.approx(GAMMA * LAMBDA_V)
    for name, recurrence in recurrences.items():
        assert recurrence.reads == CARRIED, name
        # The intentional update is derived against the unemphasized
        # recurrence, and the joint step is derived against Eq. 12.
        assert not recurrence.emphasized, name


def test_a_second_bound_over_the_finished_step_is_refused():
    """The rule sizes its own step, so an outer clip is an undeclared bound."""

    with pytest.raises(ValueError, match="second, undeclared bound"):
        assembled(overrides={"torso.grad_clip": 1.0})


def test_it_is_not_another_spelling_of_the_output_position():
    """Same two etas, same two traces, same two signals -- different step.

    The only thing that differs is how the two branches' step sizes reach the
    parameters: two whole updates added there, or one step size that had to
    satisfy both. If those agreed there would be nothing here to choose.
    """

    torsos = {}
    for position in ("joint_iu", "output_iu"):
        built = assembled(position, num_envs=2)
        state = built.program.init(jax.random.key(0))
        stepped, _ = built.program.train(jax.random.key(1), state, 32)
        torsos[position] = leaves(stepped.core.torso.params)

    moved = [
        path
        for path, leaf in torsos["joint_iu"].items()
        if not np.array_equal(leaf, torsos["output_iu"][path])
    ]
    assert moved, "the two positions left the torso in the same place"


# ------------------------------------------------------------------ readings
def test_every_reading_is_filed_at_the_level_that_produced_it():
    """Both levels of ``TorsoUpdate`` are on, and each carries what it has.

    One ``rho``, one clipped error and one step at the block; two derivatives,
    two traces, two ``sigma_bar`` and two Eq. 12 step sizes at the branches. A
    name at the wrong level would be either a number nothing produced or two
    numbers filed as one.
    """

    built = assembled(num_envs=2)
    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, 16)
    torso = metrics.update.torso

    # The block's.
    assert torso.step_size is not None
    assert torso.intentional.denominator is not None
    assert torso.intentional.rms_scale is not None
    assert torso.intentional.clipped_delta is not None
    assert torso.intentional.update_norm is not None
    # Two of each, so nothing at the block.
    assert torso.grad_norm is None
    assert torso.trace_norm is None
    assert torso.intentional.sigma_bar is None
    assert torso.intentional.trace_quadratic is None

    for name in ("actor", "critic"):
        branch = getattr(torso, name)
        assert branch.grad_norm is not None, name
        assert branch.trace_norm is not None, name
        assert branch.step_size is not None, name
        assert branch.intentional.sigma_bar is not None, name
        assert branch.intentional.trace_quadratic is not None, name
        # One each, so nothing at a branch.
        assert branch.intentional.clipped_delta is None, name
        assert branch.intentional.rms_scale is None, name
        assert branch.intentional.update_norm is None, name

    # The running scale a normalized advantage is divided by exists for the
    # branch whose signal is an advantage and nowhere else.
    assert torso.actor.intentional.advantage_scale is not None
    assert torso.critic.intentional.advantage_scale is None


def test_the_block_never_outsteps_either_branch_over_a_whole_run():
    """``alpha <= min_b alpha_b``, read off a run rather than off the optimizer.

    Which is the property that closes issue 87's output-position failure: a
    branch whose credit for an entry collapses drives its own Eq. 12 step size
    up without bound, and here the other branch is still holding the block.
    """

    built = assembled(num_envs=2)
    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, 128)
    torso = metrics.update.torso

    alpha = np.asarray(torso.step_size)
    each = np.minimum(
        np.asarray(torso.actor.step_size), np.asarray(torso.critic.step_size)
    )
    assert np.all(alpha <= each * (1 + 1e-4)), (
        float(np.max(alpha - each)),
        "the block stepped further than a branch alone would have",
    )


# -------------------------------------------------- a bounded continuous action
def test_the_first_transition_of_a_joint_run_moves_nothing():
    """Both traces are empty, so both branches have nothing to spend along."""

    built = assembled()
    graph = graph_of(built)
    state = built.program.init(jax.random.key(0))
    stepped, _ = graph.train_step(state, jax.random.key(1))

    for block in ("torso", "actor", "critic"):
        before = leaves(getattr(state.core, block).params)
        after = leaves(getattr(stepped.core, block).params)
        moved = [
            path
            for path, leaf in after.items()
            if not np.array_equal(leaf, before[path])
        ]
        assert not moved, f"{block} moved on its first transition: {moved}"


def test_a_bounded_gaussian_run_keeps_every_joint_quantity_finite():
    """The configuration issue 87 was raised against, at a unit test's size.

    The bounded policy head, an intentional rule on all three blocks, and
    enough transitions for every running statistic to have moved off the zero
    it was allocated at. What has to stay finite is not only the parameters --
    optimizer state, the actions the environment was handed, and every reading
    a run would be scored on.
    """

    built = assembled(num_envs=2)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 256)

    assert finite(stepped.core.torso.params), "the torso's parameters"
    assert finite(stepped.core.actor.params), "the actor's parameters"
    assert finite(stepped.core.critic.params), "the critic's parameters"
    assert finite(stepped.core.rule), "the rules' state"
    assert finite(stepped.core.torso.traces), "the torso's traces"
    assert finite(metrics.interaction.action), "the actions"
    assert finite(metrics.update), "the update readings"
    assert finite(metrics.forward), "the forward readings"

    # And the rule really engaged, so the run was not 256 quiet steps.
    assert float(jnp.max(jnp.abs(metrics.update.torso.step_size))) > 0.0

    # The watch agrees, which is the reading a run reporting `-1e30` is read by.
    for name, first in (metrics.update.finiteness or {}).items():
        assert int(jnp.max(first)) == 0, f"{name} went non-finite"


def test_a_joint_run_does_not_amplify_the_error_it_sized_itself_against():
    """A whole graph, over enough updates for a wrong sign to be visible.

    The bound is loose on purpose. It is not a claim about how well this tiny
    environment is learned; it is the difference between a run that is learning
    and one whose step is pointed at its own error, and nothing lands between.
    """

    built = assembled(num_envs=2)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 256)

    assert finite(stepped.core), "the run left the finite numbers"
    error = float(jnp.max(jnp.abs(metrics.update.td_error)))
    assert error < 1e3, f"the TD error reached {error:.3g}"
    value = float(jnp.max(jnp.abs(metrics.forward.critic.value)))
    assert value < 1e3, f"the value reached {value:.3g}"
