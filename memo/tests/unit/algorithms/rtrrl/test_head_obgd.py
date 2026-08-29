"""ObGD as the rule a whole agent steps under, readouts included.

``tests/unit/algorithms/rtrrl/test_torso_aggregation.py`` holds ObGD on the
shared torso, where the question is *where* two contributions meet. A readout
has one contribution and no such question, so what is left to establish is
narrower and is here: that the rule reaches the two heads at all, that what it
does there is the published bounded step and not an approximation of it, and
that a head's bound is as readable as the torso's.

The arm this exists for is the update-rule comparison. An ObGD arm that ran
Adam on both readouts would be measuring "Adam heads plus an ObGD torso", which
is not on the same axis as the Adam, SGD and intentional arms it is compared
with -- every one of those steps all three blocks under one rule. See #77.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import expand, flatten
from memorax.rl.traces import CARRIED, Trace
from memorax.rl.updates import (
    AdaptiveObBoundFixed,
    ObGDStep,
    Sgd,
    make_bounded_rule,
)
from tests.support.builders import graph_of
from tests.support.environments import TinyContinuousEnv
from tests.support.numerics import flattened
from tests.support.parameters import kinds

STREAMS = 2
GAMMA = 0.99
LAMBDA_PI = 0.9
LAMBDA_V = 0.8
KAPPA = 2.0
LR = 1e-3


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


def bounded(prefix, *, kappa=KAPPA, lr=LR, bound="adaptive_ob_fixed"):
    """One block's ObGD selection, spelled the way a run document spells it."""

    settings = {
        f"{prefix}.kind": "obgd",
        f"{prefix}.obgd.lr": lr,
        f"{prefix}.obgd.bound.kind": bound,
        f"{prefix}.obgd.bound.{bound}.kappa": kappa,
    }
    if bound != "ob":
        settings[f"{prefix}.obgd.bound.{bound}.beta2"] = 0.999
        settings[f"{prefix}.obgd.bound.{bound}.eps"] = 1e-8
    return settings


#: The arm #77 freezes: one rule on all three blocks, the published bound, and
#: the torso credited at the position the reference RTRRL credits it at.
EXPERIMENT_TWO = {
    **bounded("actor.optimizer"),
    **bounded("critic.optimizer"),
    "torso.optimizer.kind": "input_obgd",
    "torso.optimizer.input_obgd.lr": LR,
    "torso.optimizer.input_obgd.bound.kind": "adaptive_ob_fixed",
    "torso.optimizer.input_obgd.bound.adaptive_ob_fixed.kappa": KAPPA,
    "torso.optimizer.input_obgd.bound.adaptive_ob_fixed.beta2": 0.999,
    "torso.optimizer.input_obgd.bound.adaptive_ob_fixed.eps": 1e-8,
}


def assembled(overrides=None, *, num_envs=STREAMS):
    parameters = expand(
        rtrrl.PARAMETERS,
        {
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 4,
            "torso.backbone.lru.hidden_dim": 2,
            "torso.backbone.lru.differentiation.kind": "exact_rtrl",
            # Every rule that bounds or sizes its own step refuses an outer one.
            "torso.grad_clip": 0.0,
            "torso.follow": 0.25,
            "actor.head.kind": "state_std",
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": False,
            "gamma": GAMMA,
            "eta_f": 1.0,
            "eta_pi": 1.0,
            "lambda_pi": LAMBDA_PI,
            "lambda_v": LAMBDA_V,
            "lambda_rnn": 0.9,
            "entropy_rate": 1e-5,
            **EXPERIMENT_TWO,
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


# ------------------------------------------------------------- the declaration
@pytest.mark.parametrize(
    "entry",
    ("rtrrl", "rtrrl_ctrnn_rflo", "rtrrl_lstm_rflo", "rtrrl_ssm_rflo"),
)
@pytest.mark.parametrize("block", ("actor", "critic"))
def test_every_entry_offers_obgd_on_both_readouts(entry, block):
    """Per entry, because the failure worth catching is one entry offering less.

    An entry that declared a narrower family than the one it names would let a
    run document select ObGD on the torso and not on a head, which is the arm
    this is here to make expressible.
    """

    module = __import__(f"entries.{entry}", fromlist=["PARAMETERS"])
    declared = module.PARAMETERS

    assert "obgd" in set(kinds(declared, f"{block}.optimizer"))
    flat = flatten(declared)
    assert f"{block}.optimizer.obgd.lr" in flat
    assert f"{block}.optimizer.obgd.bound.kind" in flat
    for name in ("ob", "adaptive_ob", "adaptive_ob_fixed"):
        assert f"{block}.optimizer.obgd.bound.{name}.kappa" in flat, name


def test_a_readout_and_the_torso_share_one_bound_family():
    """One family, so a bound means the same thing wherever it is selected.

    Two families would be two `adaptive_ob_fixed`s that could drift apart, and
    a run comparing a head against a torso would be comparing two rules with
    one name.
    """

    flat = flatten(rtrrl.PARAMETERS)
    for name in ("ob", "adaptive_ob", "adaptive_ob_fixed"):
        head = {
            key.split(f"obgd.bound.{name}.")[1]
            for key in flat
            if key.startswith(f"actor.optimizer.obgd.bound.{name}.")
        }
        torso = {
            key.split(f"input_obgd.bound.{name}.")[1]
            for key in flat
            if key.startswith(f"torso.optimizer.input_obgd.bound.{name}.")
        }
        assert head and head == torso, name


# ------------------------------------------------------------- the arithmetic
def test_a_readout_steps_the_published_bounded_rule_exactly():
    """A head's ObGD is one single-path bounded learner and nothing else.

    Driven against ``make_bounded_rule`` over the same trace, at the head's own
    decay, with the entropy direction left untraced the way the published
    implementation applies it. The rule under test is reached through
    ``make_rules``, so what is checked is the routing as well as the arithmetic.
    """

    step = ObGDStep(
        bound=AdaptiveObBoundFixed(kappa=KAPPA, beta2=0.999, eps=1e-8), lr=LR
    )
    cfg = rtrrl.RTRRLConfig(
        num_envs=STREAMS,
        gamma=GAMMA,
        lambda_pi=LAMBDA_PI,
        lambda_v=LAMBDA_V,
        actor_optimizer=step,
        critic_optimizer=step,
        torso_optimizer=step,
        torso_grad_clip=0.0,
    )
    rules = rtrrl.make_rules(cfg)
    traces = rtrrl.make_head_traces(cfg)

    # The pairing first: ObGD takes RTRRL's own trace, at that head's decay.
    for name, decay in (("actor", LAMBDA_PI), ("critic", LAMBDA_V)):
        assert traces[name] == Trace(
            decay=GAMMA * decay, reads=CARRIED, emphasized=True
        ), name

    params = {"w": jnp.zeros((3,), dtype=jnp.float32)}
    derivative = {"w": jnp.asarray([[1.0, -2.0, 0.5], [0.5, 0.5, -1.0]], jnp.float32)}
    direct = {"w": jnp.asarray([[0.05, -0.05, 0.1], [0.1, 0.0, -0.1]], jnp.float32)}
    delta = jnp.asarray([0.5, -1.5], jnp.float32)
    reset = jnp.zeros((STREAMS,), jnp.float32)
    emphasis = jnp.ones((STREAMS,), jnp.float32)

    for name, decay in (("actor", LAMBDA_PI), ("critic", LAMBDA_V)):
        trace = Trace(decay=GAMMA * decay, reads=CARRIED, emphasized=True)
        carried = trace.initial(params, STREAMS)
        used, _ = trace.stepped(carried, derivative, reset=reset, emphasis=emphasis)

        rule = rules[name]
        state = rule.init(params={name: params}, traces={name: carried})
        taken = rule.apply(
            {name: used},
            {name: direct},
            state,
            delta=delta,
            derivative={name: derivative},
            step=1,
            params={name: params},
        )

        reference = make_bounded_rule(bound=step.bound, base=Sgd(lr=step.lr))
        expected = reference.apply(
            {"one": used},
            {"one": direct},
            reference.init(params={"one": params}, traces={"one": carried}),
            delta=delta,
            derivative={"one": derivative},
            step=1,
            params={"one": params},
        )
        np.testing.assert_array_equal(
            np.asarray(taken.updates[name]["w"]),
            np.asarray(expected.updates["one"]["w"]),
            err_msg=f"{name} is not the bounded rule it selected",
        )


def test_the_whole_agent_steps_one_rule_and_moves():
    """The arm itself: three blocks, one rule, a real scan.

    This is what an ObGD arm of the update-rule comparison runs. Every block
    moves, nothing goes non-finite, and each block's rule is the bounded one --
    a run where a head had quietly stayed on Adam would still move and still be
    finite, so the rules are checked as well as the run.
    """

    built = assembled()
    graph = graph_of(built)
    assert isinstance(graph.core.aggregation, rtrrl.InputAggregation)
    for name in ("actor", "critic"):
        assert isinstance(
            getattr(graph.core.cfg, f"{name}_optimizer"), ObGDStep
        ), f"{name} did not select ObGD"

    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 32)

    for path, leaf in flattened(stepped.core).items():
        assert np.all(np.isfinite(leaf)), f"{path} went non-finite"
    for block in ("torso", "actor", "critic"):
        before = flattened(getattr(state.core, block).params)
        after = flattened(getattr(stepped.core, block).params)
        assert any(
            not np.array_equal(leaf, after[path]) for path, leaf in before.items()
        ), f"{block} never moved"
        assert np.all(np.isfinite(getattr(metrics.update, block).step_size))


# ------------------------------------------------------------------ telemetry
@pytest.mark.parametrize("block", ("actor", "critic"))
def test_a_readout_reports_the_terms_its_step_size_is_a_quotient_of(block):
    """A head's bound is as readable as the torso's, and for the same reason.

    Without these a head's low effective rate cannot be attributed either --
    and in an update-rule comparison the readouts are half of what is being
    compared.
    """

    built = assembled()
    filed = set(built.observations.series)
    for quantity in (
        "trace_sum",
        "delta_bar",
        "bound_denominator",
        "bound_scale",
        "second_moment_rms",
    ):
        assert f"update.{block}.obgd.{quantity}" in filed, quantity

    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, 8)

    reading = getattr(metrics.update, block).obgd
    assert reading.bound_scale is not None, "the bound statistics never arrived"
    scale = np.asarray(reading.bound_scale)
    assert np.all(scale > 0.0) and np.all(scale <= 1.0 + 1e-6)
    assert np.all(np.asarray(reading.delta_bar) >= 1.0)
    assert np.any(np.asarray(reading.trace_sum) > 0.0)
    assert np.all(np.asarray(reading.second_moment_rms) > 0.0)
    np.testing.assert_allclose(
        scale,
        np.asarray(getattr(metrics.update, block).step_size) / LR,
        rtol=1e-5,
    )


def test_a_run_files_exactly_the_series_its_schema_names():
    """Declared and produced are one set, for the arm as a whole.

    The same check the aggregation suite makes per mode, over the configuration
    this file is about: a name in the schema and not in what a step files is a
    series the driver looks for and never finds.
    """

    built = assembled()
    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, 4)

    produced = {
        name.lstrip(".").replace("/.", ".").replace("/", ".")
        for name in flattened({"forward": metrics.forward, "update": metrics.update})
    }
    assert produced == set(built.observations.series)
    assert not any(".intentional." in name for name in produced)


# -------------------------------------------------------------- what it is not
def test_a_readout_selecting_a_rate_is_unchanged_by_any_of_this():
    """Adam on the heads still reaches the parameters it always did.

    The point of adding a branch is that it is a branch: a configuration that
    did not select it must be the run it was before, so this drives the default
    rate arm and asserts it files no bound statistics at all.
    """

    built = assembled(
        {
            "actor.optimizer.kind": "adam",
            "actor.optimizer.adam.lr": 5e-4,
            "critic.optimizer.kind": "adam",
            "critic.optimizer.adam.lr": 5e-4,
        }
    )
    filed = set(built.observations.series)
    assert not any(
        name.startswith(("update.actor.obgd.", "update.critic.obgd.")) for name in filed
    )
    assert "update.torso.obgd.bound_scale" in filed, "the torso's are still filed"

    state = built.program.init(jax.random.key(0))
    stepped, _ = built.program.train(jax.random.key(1), state, 8)
    for path, leaf in flattened(stepped.core).items():
        assert np.all(np.isfinite(leaf)), path
