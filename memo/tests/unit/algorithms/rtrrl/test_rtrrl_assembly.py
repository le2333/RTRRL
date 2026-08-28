from importlib import import_module
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from flax import serialization

import memorax
from entries import rtrrl as entry
from memorax import algorithms
from memorax.algorithms import RTRRL
from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.networks.sequence import PLACES
from memorax.observability.metrics import metric_names
from memorax.parameters import expand, flatten
from memorax.readings import taken
from tests.support.builders import graph_of
from tests.support.environments import TinyContinuousEnv
from tests.support.numerics import flattened
from tests.support.parameters import kinds


def assert_tree_equal(actual, expected, what):
    """Assert two trees hold the same leaves bit for bit."""

    got, wanted = flattened(actual), flattened(expected)
    assert set(got) == set(wanted), f"{what}: the trees have different leaves"
    moved = [
        path for path, leaf in wanted.items() if not np.array_equal(got[path], leaf)
    ]
    assert not moved, f"{what}: {moved} moved"


C = 1.0

# `expand` fills anything unset from the low end of its search domain, which
# leaves `eta_f`, `eta_pi` and every `lambda` at zero. That is harmless for the
# structural assertions the shared fixture was written for, and fatal for these:
# `eta_f == 0` makes the torso's TD error zero, so `sign` of it is zero and the
# torso never takes a traced step at all. Anything asserting that a step was
# taken has to say what these are.
LIVE = {
    "eta_f": 1.0,
    "eta_pi": 1.0,
    "lambda_pi": 0.9,
    "lambda_v": 0.9,
    "lambda_rnn": 0.9,
    "entropy_rate": 1e-5,
}

TORSO_LR = 1e-2
HEAD_LR = 5e-3

SGD = {
    **LIVE,
    "torso.optimizer.kind": "sgd",
    "torso.optimizer.sgd.lr": TORSO_LR,
    "actor.optimizer.kind": "sgd",
    "actor.optimizer.sgd.lr": HEAD_LR,
    "critic.optimizer.kind": "sgd",
    "critic.optimizer.sgd.lr": HEAD_LR,
}

D_RTRRL = {
    **LIVE,
    "torso.optimizer.kind": "d_rtrrl",
    "torso.optimizer.d_rtrrl.c": C,
    "torso.optimizer.d_rtrrl.magnitude": "sign",
    "torso.optimizer.d_rtrrl.scope": "block",
    "torso.optimizer.d_rtrrl.eps": 1e-8,
    "actor.optimizer.kind": "d_rtrrl",
    "actor.optimizer.d_rtrrl.c": C,
    "actor.optimizer.d_rtrrl.magnitude": "sign",
    "actor.optimizer.d_rtrrl.scope": "block",
    "actor.optimizer.d_rtrrl.eps": 1e-8,
    "critic.optimizer.kind": "d_rtrrl",
    "critic.optimizer.d_rtrrl.c": C,
    "critic.optimizer.d_rtrrl.magnitude": "sign",
    "critic.optimizer.d_rtrrl.scope": "block",
    "critic.optimizer.d_rtrrl.eps": 1e-8,
}


# The published actor-critic's two intended reductions: a policy step aiming
# at roughly five percent of the log-probability, a value step at half the TD
# error. They are different numbers, and a surface that could not say so could
# not express the configuration every reported result came from.
ETA_ACTOR = 0.05
ETA_CRITIC = 0.5

INTENTIONAL = {
    **LIVE,
    # The intentional update sets its own step size, so the outer bound on the
    # finished torso step has to be off before one can be selected at all.
    "torso.grad_clip": 0.0,
    "torso.optimizer.kind": "iu",
    "torso.optimizer.iu.eta": ETA_CRITIC,
    "torso.optimizer.iu.clip": 20.0,
    "torso.optimizer.iu.beta_rms": 0.999,
    "torso.optimizer.iu.beta_clip": 0.9998,
    "torso.optimizer.iu.beta_advantage": 0.9998,
    "torso.optimizer.iu.eps": 1e-8,
    "actor.optimizer.kind": "iu",
    "actor.optimizer.iu.eta": ETA_ACTOR,
    "actor.optimizer.iu.clip": 20.0,
    "actor.optimizer.iu.beta_rms": 0.999,
    "actor.optimizer.iu.beta_clip": 0.9998,
    "actor.optimizer.iu.beta_advantage": 0.9998,
    "actor.optimizer.iu.eps": 1e-8,
    "critic.optimizer.kind": "iu",
    "critic.optimizer.iu.eta": ETA_CRITIC,
    "critic.optimizer.iu.clip": 20.0,
    "critic.optimizer.iu.beta_rms": 0.999,
    "critic.optimizer.iu.beta_clip": 0.9998,
    "critic.optimizer.iu.beta_advantage": 0.9998,
    "critic.optimizer.iu.eps": 1e-8,
}

# One block per rule, which is the configuration that would find a rule reading
# a neighbour's state. The torso keeps its outer clip: only the intentional
# update refuses one, and here it is the critic's, which declares none.
MIXED = {
    **LIVE,
    "torso.optimizer.kind": "sgd",
    "torso.optimizer.sgd.lr": TORSO_LR,
    "actor.optimizer.kind": "adam",
    "actor.optimizer.adam.lr": HEAD_LR,
    "critic.optimizer.kind": "iu",
    "critic.optimizer.iu.eta": ETA_CRITIC,
    "critic.optimizer.iu.clip": 20.0,
    "critic.optimizer.iu.beta_rms": 0.999,
    "critic.optimizer.iu.beta_clip": 0.9998,
    "critic.optimizer.iu.beta_advantage": 0.9998,
    "critic.optimizer.iu.eps": 1e-8,
}


def parameters(backbone="lru", differentiation="exact_rtrl", optimizer=None):
    branch = f"torso.backbone.{backbone}"
    return expand(
        rtrrl.PARAMETERS,
        {
            "torso.backbone.kind": backbone,
            **({f"{branch}.feature_dim": 4} if backbone == "lru" else {}),
            f"{branch}.hidden_dim": 2,
            f"{branch}.differentiation.kind": differentiation,
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 1e-3,
            "torso.grad_clip": 1.0,
            "torso.follow": 0.25,
            "actor.optimizer.kind": "adam",
            "actor.optimizer.adam.lr": 5e-4,
            "critic.optimizer.kind": "adam",
            "critic.optimizer.adam.lr": 5e-4,
            **(optimizer or {}),
            "actor.head.kind": "state_std",
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": False,
        },
    )


def assembled(
    backbone="lru", differentiation="exact_rtrl", record=None, optimizer=None
):
    return assemble(
        rtrrl.RTRRL,
        BuildRequest(
            parameters=parameters(backbone, differentiation, optimizer),
            environment=EnvironmentSpec(
                id="tiny",
                backend=None,
                observed=None,
                episode_length=8,
            ),
            num_envs=1,
            record=(rtrrl.OBSERVATIONS.trajectory_fields if record is None else record),
        ),
        environment_factory=tiny_environment,
    )


def run_document(*, every_steps=None, total_steps=50, episode_length=7):
    rerun = (
        None if every_steps is None else SimpleNamespace(log_every_steps=every_steps)
    )
    return SimpleNamespace(
        algorithm=SimpleNamespace(
            parameters={"gamma": 0.9},
            environment=SimpleNamespace(
                id="tiny",
                backend="test",
                observed=[0, 1],
                episode_length=episode_length,
                kwargs={},
            ),
            num_envs=2,
        ),
        training=SimpleNamespace(
            seed=7,
            total_steps=total_steps,
            chunk_steps=10,
        ),
        evaluation=SimpleNamespace(every_steps=10, episodes=3, chunk_steps=4, seed=11),
        logging=SimpleNamespace(aim=SimpleNamespace(training=None), rerun=rerun),
    )


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


@pytest.mark.parametrize("backbone", ("lru", "rtu"))
def test_each_recurrent_backbone_registers_its_differentiation_scope(backbone):
    assert set(kinds(rtrrl.PARAMETERS, "torso.backbone")) == {"lru", "rtu"}
    assert set(
        kinds(rtrrl.PARAMETERS, f"torso.backbone.{backbone}.differentiation")
    ) == {"exact_rtrl", "tbptt"}


@pytest.mark.parametrize("backbone", ("lru", "rtu"))
@pytest.mark.parametrize("kind", ("exact_rtrl", "tbptt"))
def test_rtrrl_builds_the_selected_kernel_scoped_differentiation(backbone, kind):
    differentiation = import_module("memorax.networks.differentiation")
    built = assembled(backbone, kind)
    torso = graph_of(built).core.torso
    selected = torso._differentiation

    assert isinstance(selected, differentiation.RecurrentDifferentiation)
    if kind == "tbptt":
        assert type(selected) is differentiation.TruncatedBPTT
    else:
        assert type(selected).__module__ == (
            f"memorax.networks.sequence_models.{backbone}"
        )

    state = built.program.init(jax.random.key(0))
    differentiation_state = state.core.torso.recurrence.differentiation_state
    assert (differentiation_state is None) is (kind == "tbptt")


def test_every_block_chooses_from_one_set_of_rules():
    """Three blocks, one family, and standard SGD among what it offers.

    The blocks declare the same restriction of the shared step family, so a
    rule reaching one of them reaches all three. Asserting the three sets are
    equal is what keeps a rule from arriving in the torso alone, which is the
    shape the surface took when only the torso's optimizer was being worked on.
    """

    offered = {"sgd", "adam", "d_rtrrl", "iu"}
    declared = flatten(rtrrl.PARAMETERS)
    for block in ("torso", "actor", "critic"):
        assert set(kinds(rtrrl.PARAMETERS, f"{block}.optimizer")) == offered
        assert f"{block}.optimizer.sgd.lr" in declared

    # And the order the rules are offered in is load-bearing: a configuration
    # naming no optimizer is filled from the front of the domain, so a rule
    # arriving ahead of Adam would change every unpinned run without changing a
    # line of it.
    unpinned = expand(rtrrl.PARAMETERS, {})
    for block in ("torso", "actor", "critic"):
        assert unpinned[f"{block}.optimizer.kind"] == "adam"


@pytest.mark.parametrize("backbone", ("lru", "rtu"))
def test_the_sgd_optimizer_is_reachable_from_a_run_configuration(backbone):
    """The plain rate, from a run document through to a step that moves.

    Its arithmetic is driven in ``test_rtrrl_sgd_rule``. What is asserted here
    is everything between a configuration and that arithmetic: that ``sgd``
    survives the parameter surface on either backbone, that each block steps at
    the rate it declared rather than at a sibling's, and that a scan of real
    transitions -- torso trace, two readouts, emphasis, and four episode
    endings in thirty-two steps -- stays finite and moves every block.
    """

    built = assembled(backbone, optimizer=SGD)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 32)

    for path, leaf in flattened(stepped.core).items():
        assert np.all(np.isfinite(leaf)), f"{path} went non-finite"
    assert np.all(np.isfinite(metrics.update.td_error))

    for block, rate in (("torso", TORSO_LR), ("actor", HEAD_LR), ("critic", HEAD_LR)):
        np.testing.assert_allclose(getattr(metrics.update, block).step_size, rate)

    for block in ("torso", "actor", "critic"):
        before = flattened(getattr(state.core, block).params)
        after = flattened(getattr(stepped.core, block).params)
        assert any(
            not np.array_equal(leaf, after[path]) for path, leaf in before.items()
        ), f"{block} never moved"


def test_a_rate_does_not_reach_anything_the_algorithm_carries():
    """Selecting SGD changes the parameters and nothing around them.

    The eligibility, the emphasis and the recurrent sensitivity are the
    algorithm's, accumulated from the parameters as they stood when the
    transition was taken -- so after one step from one init on one key they
    cannot depend on which rule spent the direction. Asserting it against Adam
    pins the boundary the rule contract draws: a rule is handed a direction and
    hands back an update, and reads none of the state that produced it.
    """

    def one_step(optimizer):
        built = assembled(optimizer=optimizer)
        state = built.program.init(jax.random.key(0))
        return graph_of(built).train_step(state, jax.random.key(1))[0]

    under_sgd, under_adam = one_step(SGD), one_step(dict(LIVE))

    for block in ("torso", "actor", "critic"):
        assert_tree_equal(
            getattr(under_sgd.core, block).traces,
            getattr(under_adam.core, block).traces,
            f"{block} eligibility",
        )
    assert_tree_equal(
        under_sgd.core.torso.recurrence,
        under_adam.core.torso.recurrence,
        "the torso recurrence and its sensitivity",
    )
    assert_tree_equal(under_sgd.core.emphasis, under_adam.core.emphasis, "emphasis")

    # And the parameters did move apart, so the agreement above is about the
    # state rather than about two rules that took the same step.
    stepped = flattened(under_sgd.core.torso.params)
    other = flattened(under_adam.core.torso.params)
    assert any(
        not np.array_equal(leaf, other[path]) for path, leaf in stepped.items()
    ), "the two rules moved the torso identically, so nothing was told apart"


def test_a_resumed_sgd_run_is_the_run_it_would_have_been():
    """A rate carries no statistic, and a checkpoint has to survive that too.

    Runtime trains in chunks, so every run is already a sequence of
    resumptions. A rule state holding nothing is the case a serialization round
    trip is most likely to get quietly wrong: an empty subtree is easy to drop
    and impossible to notice from the numbers afterwards.
    """

    built = assembled(optimizer=SGD)
    graph = graph_of(built)
    state = built.program.init(jax.random.key(0))
    keys = jax.random.split(jax.random.key(1), 8)

    uninterrupted, _ = jax.lax.scan(graph.train_step, state, keys)
    halfway, _ = jax.lax.scan(graph.train_step, state, keys[:4])

    assert not flattened(halfway.core.rule), "a bare rate carries no statistic"
    saved = serialization.to_state_dict(halfway)
    # The eligibility is still the block's rather than the rule's, which is
    # what an empty rule state must not be read as meaning.
    assert set(saved["core"]["torso"]) >= {"params", "traces"}

    restored = serialization.from_state_dict(halfway, saved)
    assert_tree_equal(restored, halfway, "restored state")
    resumed, _ = jax.lax.scan(graph.train_step, restored, keys[4:])
    assert_tree_equal(resumed.core, uninterrupted.core, "core")


def test_the_three_blocks_may_each_be_stepped_by_a_different_rule():
    """One run, three rules: an SGD torso, an Adam actor, an intentional critic.

    Nothing in the configuration ties the blocks together, and this is where
    that stops being a claim about the declaration and becomes one about the
    graph: three rules, three states, three step sizes, and every block moving.
    """

    built = assembled(optimizer=MIXED)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 32)

    for path, leaf in flattened(stepped.core).items():
        assert np.all(np.isfinite(leaf)), f"{path} went non-finite"

    # Each rule carries what its own kind carries, over its own block and no
    # other: nothing for the rate, Adam's two moments for the actor, the
    # intentional statistics for the critic.
    assert not flattened(stepped.core.rule["torso"]), "the sgd torso carried state"
    assert set(stepped.core.rule["actor"][0].mu) == {"actor"}
    assert set(stepped.core.rule["critic"]) == {"critic"}
    assert {"nu", "sigma_bar"} <= set(
        serialization.to_state_dict(stepped)["core"]["rule"]["critic"]["critic"]
    )

    # The torso reports the rate it declared; the critic derives its own, and
    # a derived one that never moved would mean the two were not told apart.
    np.testing.assert_allclose(metrics.update.torso.step_size, TORSO_LR)
    derived = np.asarray(metrics.update.critic.step_size).reshape(-1)
    assert not np.allclose(derived, derived[0]), "the critic's step never moved"

    for block in ("torso", "actor", "critic"):
        before = flattened(getattr(state.core, block).params)
        after = flattened(getattr(stepped.core, block).params)
        assert any(
            not np.array_equal(leaf, after[path]) for path, leaf in before.items()
        ), f"{block} never moved"


def test_the_d_rtrrl_optimizer_is_reachable_from_a_run_configuration():
    """The rule is only real if a run document can name it and a step can take it.

    The rule's own arithmetic is driven in ``tests/unit/components``. What is
    asserted here is everything between a configuration and that arithmetic:
    that ``d_rtrrl`` survives the parameter surface, that the two groups
    build a rule each, and that a scan of real transitions through the whole
    graph -- torso trace, two readouts, emphasis, resets -- stays finite.
    Finiteness is the claim the version this replaces could not make.
    """

    built = assembled(optimizer=D_RTRRL)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 32)

    for path, leaf in flattened(stepped.core).items():
        assert np.all(np.isfinite(leaf)), f"{path} went non-finite"
    assert np.all(np.isfinite(metrics.update.td_error))
    # The rate every rule reports under one name, and here a constant one.
    for block in ("torso", "actor", "critic"):
        np.testing.assert_allclose(getattr(metrics.update, block).step_size, C)

    # Every block took a step. Without this the assertions above are satisfied
    # by a run that never moved: a TD error of zero reaching `sign` leaves the
    # traced update at exactly zero, which is finite and constant and means
    # nothing was learned.
    for block in ("torso", "actor", "critic"):
        before = flattened(getattr(state.core, block).params)
        after = flattened(getattr(stepped.core, block).params)
        assert any(
            not np.array_equal(leaf, after[path]) for path, leaf in before.items()
        ), f"{block} never moved"


def test_the_intentional_optimizer_is_reachable_from_a_run_configuration():
    """The same claim as above for the intentional update, which needs more.

    Its arithmetic is driven in ``tests/unit/components``. Everything between a
    run document and that arithmetic is here: that ``iu`` survives the
    parameter surface, that each of the three blocks gets an optimizer of its
    own, that a scan of real transitions stays finite, and that the algorithm
    still holds every eligibility trace -- the optimizer reads one, and reading
    is not owning.
    """

    built = assembled(optimizer=INTENTIONAL)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 32)

    for path, leaf in flattened(stepped.core).items():
        assert np.all(np.isfinite(leaf)), f"{path} went non-finite"
    for block in ("torso", "actor", "critic"):
        reading = getattr(metrics.update, block).intentional
        assert np.all(np.asarray(reading.non_finite) == 0.0), block
        # The trace is still the algorithm's, and it is not empty: an
        # optimizer that had quietly started keeping its own would leave this
        # one at the zeros it was allocated with.
        trace = getattr(stepped.core, block).traces
        assert trace is not None
        assert any(np.any(leaf != 0) for leaf in flattened(trace).values()), block

        before = flattened(getattr(state.core, block).params)
        after = flattened(getattr(stepped.core, block).params)
        assert any(
            not np.array_equal(leaf, after[path]) for path, leaf in before.items()
        ), f"{block} never moved"


def test_each_block_gets_an_intentional_state_and_an_eta_of_its_own():
    """Three learners, three step sizes, and one advantage scale between them.

    Nothing is shared: not the state, and not the intended reduction. The
    published actor-critic sets ``eta`` to 0.05 for the policy and 0.5 for the
    value function and sweeps them separately, so a surface that gave the two
    heads one number could not express it -- which is what this asserts is no
    longer the case, by reading the two step sizes back out of one run.

    The critic, whose signal is a TD error rather than an advantage, carries no
    advantage scale at all.
    """

    built = assembled(optimizer=INTENTIONAL)
    state = built.program.init(jax.random.key(0))
    stepped, metrics = built.program.train(jax.random.key(1), state, 8)

    graph = graph_of(built)
    assert graph.cfg.actor_optimizer.eta == ETA_ACTOR
    assert graph.cfg.critic_optimizer.eta == ETA_CRITIC

    sizes = {
        block: np.asarray(getattr(metrics.update, block).step_size).reshape(-1)
        for block in ("torso", "actor", "critic")
    }
    assert not np.allclose(sizes["actor"], sizes["critic"])
    assert not np.allclose(sizes["torso"], sizes["critic"])

    assert np.all(np.asarray(metrics.update.actor.intentional.advantage_scale) >= 0)
    assert metrics.update.critic.intentional.advantage_scale is None
    assert metrics.update.torso.intentional.advantage_scale is None
    assert stepped.core.rule["critic"]["critic"].advantage_scale is None


def test_the_entropy_direction_joins_the_trace_only_under_an_intentional_step():
    """Where the entropy term goes is the algorithm's, and it is not one place.

    RTRRL's own rules take it untraced, on the step it arises, which is what
    the published RTRRL does: the trace after a transition is the same whether
    the entropy coefficient is zero or not. The paper's intentional policy
    gradient is one derivative -- the log-probability and the entropy together,
    signed by the TD error -- and the trace has to accumulate that sum, so the
    same comparison has to come out different.

    Read on the trace rather than on the parameters, because the parameters
    move under both and only the trace says which route the term took.
    """

    def actor_trace(optimizer):
        loud = assembled(optimizer={**optimizer, "entropy_rate": 1e-2})
        quiet = assembled(optimizer={**optimizer, "entropy_rate": 0.0})
        traces = []
        for built in (loud, quiet):
            state = built.program.init(jax.random.key(0))
            stepped, _ = built.program.train(jax.random.key(1), state, 1)
            traces.append(flattened(stepped.core.actor.traces))
        return traces

    with_entropy, without = actor_trace(D_RTRRL)
    for path, leaf in with_entropy.items():
        np.testing.assert_array_equal(
            leaf, without[path], err_msg=f"{path} traced the entropy under d_rtrrl"
        )

    with_entropy, without = actor_trace(INTENTIONAL)
    assert any(
        not np.array_equal(leaf, without[path]) for path, leaf in with_entropy.items()
    ), "the entropy direction never reached the intentional trace"


def test_an_intentional_run_files_exactly_the_series_its_schema_names():
    """The declaration follows the selection, in both directions.

    ``tests/test_readings.py`` holds this for the default configuration. The
    intentional one is where it could break: it is the first selection that
    changes which readings exist, so the schema handed to Runtime is the built
    graph's rather than the class's, and a name in one and not the other fails
    a run on a series that never arrives.
    """

    built = assembled(optimizer=INTENTIONAL)
    state = built.program.init(jax.random.key(0))
    _, metrics = built.program.train(jax.random.key(1), state, 4)

    produced = {
        name.lstrip(".").replace("/.", ".").replace("/", ".")
        for name in flattened({"forward": metrics.forward, "update": metrics.update})
    }
    assert produced == set(built.observations.series)
    assert "update.actor.intentional.sigma_bar" in produced
    assert "update.actor.step_size" in produced
    # A catalog advertises every reading some configuration files. This is the
    # configuration that files all of them; the default one, which produces no
    # intentional state at all, files strictly fewer.
    available = set(taken(rtrrl.AVAILABLE_REPORTS, parts=PLACES))
    assert set(built.observations.series) == available
    assert set(assembled().observations.series) < available


def test_a_resumed_run_carries_the_intentional_state_unchanged():
    """What a checkpoint is here: the state crossing an invocation boundary.

    Runtime trains in chunks, so every run is already a sequence of resumptions
    from a state that was handed back. The state serialized and read back has
    to be the same state -- the trace, the second moment, the clipping
    statistic and the advantage scale included -- and the transitions that
    follow it have to be the transitions that would have followed anyway.

    Driven on one key sequence from both sides, so what is compared is the
    resumption and not two different streams of random numbers.
    """

    built = assembled(optimizer=INTENTIONAL)
    graph = graph_of(built)
    state = built.program.init(jax.random.key(0))
    keys = jax.random.split(jax.random.key(1), 8)

    uninterrupted, _ = jax.lax.scan(graph.train_step, state, keys)
    halfway, _ = jax.lax.scan(graph.train_step, state, keys[:4])

    saved = serialization.to_state_dict(halfway)
    held = set(saved["core"]["rule"]["actor"]["actor"])
    assert {
        "nu",
        "sigma_bar",
        "delta_square",
        "advantage_scale",
    } <= held, "the intentional state is not all of it in what a checkpoint would hold"
    # And the eligibility is where it always was, under the block rather than
    # under the rule.
    assert set(saved["core"]["actor"]) >= {"params", "traces"}
    restored = serialization.from_state_dict(halfway, saved)
    assert_tree_equal(restored, halfway, "restored state")

    resumed, _ = jax.lax.scan(graph.train_step, restored, keys[4:])
    assert_tree_equal(resumed.core.rule, uninterrupted.core.rule, "intentional state")
    assert_tree_equal(resumed.core, uninterrupted.core, "core")


def test_the_intentional_update_refuses_a_second_bound_on_its_step():
    """A clip over an intentional step is a different, undeclared algorithm.

    The step size is derived from what the step is supposed to spend. Cutting
    the finished step back to a fixed length spends something else, and the run
    would still be recorded as a reproduction of the published rule -- so it is
    refused where it is asked for rather than applied quietly.
    """

    with pytest.raises(ValueError, match="second, undeclared bound"):
        assembled(optimizer={**INTENTIONAL, "torso.grad_clip": 1.0})


def test_the_two_arms_differ_in_exactly_the_magnitude_they_keep():
    """The R3 comparison in miniature: same C, same everything, one difference.

    This is the property the formal arms are read for, asserted here on a tiny
    environment so it is checkable without a cluster. Over a run of real
    transitions:

    ``sign``
        every step moves the torso the same distance, whatever that step's TD
        error was. The realized update norm is flat.
    ``td_out``
        every step moves it a distance proportional to ``|delta|``. The
        realized update norm divided by ``|delta|`` is what is flat instead.

    The outer clip is off here. At ``grad_clip == C`` it binds on nearly every
    step and would flatten ``td_out`` too, which is the interaction worth
    knowing about and not the thing being measured.
    """

    def realized(magnitude):
        built = assembled(
            optimizer={
                **D_RTRRL,
                "torso.optimizer.d_rtrrl.magnitude": magnitude,
                "actor.optimizer.d_rtrrl.magnitude": magnitude,
                "critic.optimizer.d_rtrrl.magnitude": magnitude,
                "torso.grad_clip": 0.0,
            }
        )
        state = built.program.init(jax.random.key(0))
        moves, surprises = [], []
        for step in range(8):
            before = flattened(state.core.critic.params)
            state, metrics = built.program.train(jax.random.key(step + 1), state, 1)
            after = flattened(state.core.critic.params)
            moves.append(
                float(
                    np.sqrt(
                        sum(
                            np.sum(np.square(np.asarray(after[k]) - np.asarray(leaf)))
                            for k, leaf in before.items()
                        )
                    )
                )
            )
            surprises.append(abs(float(np.asarray(metrics.update.td_error).ravel()[0])))
        # The first step steps with the initial trace, which is zero.
        return np.array(moves[1:]), np.array(surprises[1:])

    flat, flat_delta = realized("sign")
    scaled, scaled_delta = realized("td_out")

    # The TD errors are not all equal, or neither claim below means anything.
    assert flat_delta.std() > 0.05 * flat_delta.mean(), "no spread in |delta| to see"

    # `sign`: the distance is C and says nothing about the surprise.
    np.testing.assert_allclose(flat, C, rtol=1e-4)

    # `td_out`: the distance is proportional to the surprise instead. The trace
    # is long enough to be clipped throughout, which is what makes the ratio
    # exactly C rather than merely increasing.
    np.testing.assert_allclose(scaled / scaled_delta, C, rtol=1e-4)
    assert scaled.std() > 0.05 * scaled.mean(), "td_out did not vary with |delta|"


def test_the_shared_recurrent_direction_is_formed_before_it_is_normalized():
    """The torso's two sources are summed first and the resultant normalized.

    RTRRL's torso is fed by both readouts: ``upstream(actor_upward +
    critic_upward)``. The sum happens before the pullback and there is one
    torso trace, so the norm that is removed is the resultant's.

    The failure this is written against is normalizing the two sources
    separately and adding them afterwards. That version is *invariant* to the
    actor's source magnitude -- scaling the actor's sensitivity would rescale
    a vector that is about to be divided by its own norm, and the torso would
    not feel it. So asymmetric sensitivities are exactly what tells the two
    orderings apart, and ``eta_pi`` is the knob that makes them asymmetric:
    the actor's traced objective is ``eta_pi * log_prob``, so its upward
    gradient is proportional to it while the critic's is not.

    Two update steps, because the first one steps with the initial trace,
    which is zero.
    """

    def torso_after(sensitivity):
        built = assembled(optimizer={**D_RTRRL, "eta_pi": sensitivity})
        state = built.program.init(jax.random.key(0))
        return built.program.train(jax.random.key(1), state, 2)[0].core.torso.params

    balanced, actor_loud = torso_after(1.0), torso_after(50.0)

    moved = [
        path
        for path, leaf in flattened(balanced).items()
        if not np.array_equal(leaf, flattened(actor_loud)[path])
    ]
    assert moved, (
        "the torso step did not feel the actor's source magnitude, which is "
        "what separately normalizing the two sources would look like"
    )


def test_a_signed_step_does_not_read_the_torso_td_scaling():
    """``eta_f`` is inert end to end, not only where the sign is taken.

    The rule-level assertion pins the arithmetic; this one pins the wiring, so
    that a future change routing ``eta_f`` somewhere other than the TD error
    handed to the torso rule is caught here rather than in a sweep over a knob
    that turns out to move nothing.
    """

    def trained(magnitude, scaling):
        built = assembled(
            optimizer={
                **D_RTRRL,
                "torso.optimizer.d_rtrrl.magnitude": magnitude,
                "eta_f": scaling,
            }
        )
        state = built.program.init(jax.random.key(0))
        return built.program.train(jax.random.key(1), state, 16)[0]

    signed = (trained("sign", 1.0), trained("sign", 100.0))
    assert_tree_equal(signed[0].core.torso.params, signed[1].core.torso.params, "eta_f")

    # The control, without which the assertion above would also pass if `eta_f`
    # never reached the torso rule at all. Under the ablation the same knob has
    # to move the same parameters, or this test is measuring the wiring being
    # absent rather than the sign discarding it. `td_out` keeps |delta|, so it
    # is the arm where the same knob must still be live.
    ablated = (trained("td_out", 1.0), trained("td_out", 100.0))
    moved = [
        path
        for path, leaf in flattened(ablated[0].core.torso.params).items()
        if not np.array_equal(leaf, flattened(ablated[1].core.torso.params)[path])
    ]
    assert moved, "eta_f reaches the torso rule under neither setting: it is unwired"


def test_rtrrl_declares_parameters_and_observations_beside_its_graph():
    assert RTRRL is rtrrl.RTRRL
    assert entry.PARAMETERS is rtrrl.PARAMETERS
    assert entry.METRICS is rtrrl.METRICS
    assert rtrrl.PARAMETERS
    assert rtrrl.TRAINING_METRICS == taken(rtrrl.REPORTS, parts=PLACES)
    # The catalog advertises every reading some configuration files, which is
    # more than the default configuration files: the intentional optimizer
    # carries state that has no counterpart under Adam, and an Adam run's
    # schema must not name a series it is never going to produce. So these two
    # are no longer one list, and the wider one is what is published.
    available = taken(rtrrl.AVAILABLE_REPORTS, parts=PLACES)
    assert rtrrl.METRICS == metric_names("train", available) + metric_names("eval")
    assert set(rtrrl.TRAINING_METRICS) < set(available)


def test_observation_schema_separates_episode_and_trajectory_fields():
    schema = rtrrl.OBSERVATIONS

    assert schema.episode_fields == frozenset(
        (schema.reward, schema.done, schema.terminal, *schema.series)
    )
    assert schema.trajectory_fields == frozenset(
        (schema.observation, schema.next_observation, schema.action)
    )
    assert set(rtrrl.TRAINING_METRICS) <= schema.episode_fields


def test_only_the_current_rtrrl_contract_is_public():
    assert memorax.RTRRL is algorithms.RTRRL is rtrrl.RTRRL
    for obsolete in ("EvalSummary", "IndependentRTRRL"):
        assert obsolete not in algorithms.__all__
        assert obsolete not in memorax.__all__


def test_entry_only_projects_the_run_config_for_assembly_and_runtime():
    config = run_document(every_steps=10)

    request = entry.build_request(config)
    schedule = entry.runtime_config(config)

    assert request.parameters is config.algorithm.parameters
    assert request.environment.id == "tiny"
    assert request.num_envs == 2
    assert schedule.total_steps == 50
    assert schedule.chunk_steps == 10
    assert schedule.evaluation_episodes == 3
    assert schedule.evaluation_chunk_steps == 4
    assert schedule.evaluation_seed == 11
    assert schedule.num_envs == 2
    assert schedule.seed == 7


def test_entry_expands_the_rerun_interval_into_the_steps_it_names():
    config = run_document(every_steps=10)

    request = entry.build_request(config)
    schedule = entry.runtime_config(config)

    # The schedule begins at the first interval, and the environment's own
    # episode limit is what bounds a train call.
    assert schedule.trajectory_at_steps == (10, 20, 30, 40, 50)
    assert schedule.max_episode_steps == 7
    assert request.record == rtrrl.OBSERVATIONS.trajectory_fields


def test_a_run_without_rerun_asks_for_no_sample_and_keeps_no_walk():
    config = run_document(every_steps=None)

    assert entry.build_request(config).record == frozenset()
    assert entry.runtime_config(config).trajectory_at_steps == ()


def test_a_graph_keeps_the_walk_only_when_the_build_asked_for_it():
    kept = assembled()
    dropped = assembled(record=frozenset())

    state = kept.program.init(jax.random.key(0))
    _, walked = kept.program.train(jax.random.key(1), state, 2)
    state = dropped.program.init(jax.random.key(0))
    _, plain = dropped.program.train(jax.random.key(1), state, 2)

    for reading in (walked, plain):
        assert reading.interaction.reward.shape == (2, 1)
        assert reading.interaction.done.shape == (2, 1)
    assert walked.interaction.observation is not None
    assert walked.interaction.next_observation is not None
    assert walked.interaction.action is not None
    assert plain.interaction.observation is None
    assert plain.interaction.next_observation is None
    assert plain.interaction.action is None

    # Runtime is handed the schema the graph will actually answer.
    assert kept.observations is rtrrl.OBSERVATIONS
    assert dropped.observations.trajectory_fields == frozenset()
    assert dropped.observations.episode_fields == rtrrl.OBSERVATIONS.episode_fields


def test_generic_assembly_closes_one_shared_torso_rtrrl_graph():
    built = assembled()

    graph = graph_of(built)
    assert isinstance(graph.core.torso, rtrrl.Torso)
    assert graph.core.actor is not graph.core.critic
    assert graph.cfg.torso_optimizer.lr == 1e-3
    assert graph.cfg.actor_optimizer.lr == 5e-4
    assert graph.cfg.critic_optimizer.lr == 5e-4
    assert built.observations is rtrrl.OBSERVATIONS

    state = built.program.init(jax.random.key(0))
    trained, metrics = built.program.train(jax.random.key(1), state, 2)
    opened = built.program.open_evaluation(jax.random.key(2), trained)
    _, evaluated = built.program.evaluate(jax.random.key(3), opened, 2)

    assert int(trained.step) == 2
    assert metrics.interaction.reward.shape == (2, 1)
    assert evaluated.interaction.reward.shape == (2, 1)


def test_program_exposes_no_learning_interaction():
    built = assembled()
    state = built.program.init(jax.random.key(0))

    advanced, metrics = built.program.interact(jax.random.key(1), state)

    assert metrics.interaction.reward.shape == (1,)
    assert int(advanced.update_step) == int(state.update_step)
    assert int(advanced.step) == int(state.step)
    assert not np.array_equal(
        np.asarray(advanced.timestep.obs), np.asarray(state.timestep.obs)
    )
    assert (
        int(advanced.env_state.step_count[0]) == int(state.env_state.step_count[0]) + 1
    )


def test_interaction_moves_no_learned_quantity():
    built = assembled()
    state = built.program.init(jax.random.key(0))
    before = state.core

    advanced, _ = built.program.interact(jax.random.key(2), state)

    assert_tree_equal(advanced.core.torso.params, before.torso.params, "torso params")
    assert_tree_equal(advanced.core.torso.traces, before.torso.traces, "torso traces")
    assert_tree_equal(
        advanced.core.torso.slow_params, before.torso.slow_params, "torso slow params"
    )
    assert_tree_equal(advanced.core.actor.params, before.actor.params, "actor params")
    assert_tree_equal(advanced.core.actor.traces, before.actor.traces, "actor traces")
    assert_tree_equal(
        advanced.core.critic.params, before.critic.params, "critic params"
    )
    assert_tree_equal(
        advanced.core.critic.traces, before.critic.traces, "critic traces"
    )
    assert_tree_equal(advanced.core.rule, before.rule, "rule state")
    assert_tree_equal(advanced.scales, state.scales, "normalization scales")


def test_semantic_subgraphs_do_not_expose_the_old_generic_network_layer():
    assert not hasattr(rtrrl, "Network")

    graph = graph_of(assembled())

    for subgraph in (graph.core.torso, graph.core.actor, graph.core.critic):
        assert not hasattr(subgraph, "block")
        assert not hasattr(subgraph, "module")
        assert not hasattr(subgraph, "credit")
