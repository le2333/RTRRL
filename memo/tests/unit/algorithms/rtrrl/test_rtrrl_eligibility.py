"""What each condition keeps of the trace, stated as an identity per condition.

The four conditions are one factorial: the update's direction comes from the
accumulated trace or from this transition's gradient, and so, independently,
does its size. So each one has an identity that says which -- a norm that
survives, a direction that survives -- and none of them needs the kernel to run
to be checked. What the kernel does with them is next door in ``test_rtrrl``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.algorithms.rtrrl_aaai import BLOCKS
from memorax.parameters import KIND, Choice, Parameter, expand
from memorax.rl.intentional import IntentionalUpdate
from memorax.rl.updates import DRTRRL, Adam
from tests.support.parameters import branch

STREAMS = 2
EPS = 1e-8


def block(seed: int = 0):
    """One block's trace and gradient: two leaves, two streams, nothing equal.

    Two leaves of different shape, because a rescaling that reached them
    separately would still satisfy every claim about one of them. Two streams
    scaled apart, because one norm over both would satisfy every claim about
    their sum.
    """

    keys = jax.random.split(jax.random.key(seed), 4)
    stream = jnp.array([1.0, 7.0])[:, None, None]
    trace = {
        "cell": {"kernel": stream * jax.random.normal(keys[0], (STREAMS, 3, 2))},
        "readout": stream[..., 0] * jax.random.normal(keys[1], (STREAMS, 4)),
    }
    gradient = {
        "cell": {"kernel": 5.0 * jax.random.normal(keys[2], (STREAMS, 3, 2))},
        "readout": 0.2 * jax.random.normal(keys[3], (STREAMS, 4)),
    }
    return trace, gradient


def readout(kind: str, **overrides) -> rtrrl.EligibilityReadout:
    """The component an experiment naming ``kind`` would be built with."""

    direction, magnitude = rtrrl.ELIGIBILITY_SOURCES[kind]
    return rtrrl.EligibilityReadout(
        direction=direction, magnitude=magnitude, **overrides
    )


def norms(tree):
    """One L2 norm per stream, written out rather than taken from the kernel."""

    return jnp.sqrt(
        sum(
            jnp.sum(jnp.square(leaf.reshape(STREAMS, -1)), axis=1)
            for leaf in jax.tree.leaves(tree)
        )
    )


def factors(effective, chosen):
    """What each of a block's numbers was multiplied by, per stream."""

    return jnp.concatenate(
        [
            (after / before).reshape(STREAMS, -1)
            for after, before in zip(
                jax.tree.leaves(effective), jax.tree.leaves(chosen)
            )
        ],
        axis=1,
    )


# --------------------------------------------- the two conditions that rescale


@pytest.mark.parametrize(
    "kind,keeps,replaces",
    [("direction", "trace", "gradient"), ("gain", "gradient", "trace")],
)
def test_a_mixed_condition_keeps_one_candidate_s_direction_at_the_other_s_size(
    kind, keeps, replaces
):
    """Which is what "direction only" and "gain only" each mean, numerically."""

    trace, gradient = block()
    candidates = {"trace": trace, "gradient": gradient}

    effective = readout(kind, eps=EPS).read(trace, gradient)

    scaled = factors(effective, candidates[keeps])
    for stream in range(STREAMS):
        # One direction, because one positive number multiplies every leaf.
        assert float(jnp.ptp(scaled[stream])) == pytest.approx(0.0, abs=1e-5)
        assert float(scaled[stream][0]) > 0
    assert norms(effective) == pytest.approx(norms(candidates[replaces]), rel=1e-5)


def test_the_rescaling_is_one_number_per_block_and_not_one_per_stream_axis():
    """A block is rescaled as a whole, and each stream on its own.

    The shared torso arrives with both readouts' cotangents already summed
    into it. A factor computed per leaf would weigh parts of one block against
    each other, and a factor computed over the streams together would weigh
    two independent copies of the algorithm against each other.
    """

    trace, gradient = block()

    effective = readout("direction", eps=EPS).read(trace, gradient)

    scaled = factors(effective, trace)[:, 0]
    expected = norms(gradient) / (norms(trace) + EPS)
    assert scaled == pytest.approx(expected, rel=1e-5)
    assert float(scaled[0]) != pytest.approx(float(scaled[1]))


def test_an_empty_trace_rescales_to_nothing_rather_than_to_a_division_by_zero():
    """The first transition of every run, and every one after an ending."""

    trace, gradient = block()
    empty = jax.tree.map(jnp.zeros_like, trace)

    for kind in ("direction", "gain"):
        effective = readout(kind, eps=EPS).read(empty, gradient)
        leaves = jax.tree.leaves(effective)
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
        assert float(jnp.max(norms(effective))) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------- the two conditions that rescale not


@pytest.mark.parametrize("kind,candidate", [("trace", 0), ("gradient", 1)])
def test_a_pure_condition_hands_back_the_candidate_it_names(kind, candidate):
    """Not that candidate rescaled to its own norm: no guard enters the baseline.

    ``||z|| / (||z|| + eps)`` is a hair below one, and a baseline standing a
    hair away from the published algorithm would put that hair into every
    comparison the ablation is for.
    """

    candidates = block()

    effective = readout(kind, eps=EPS).read(*candidates)

    assert effective is candidates[candidate]


def test_a_source_that_is_neither_candidate_is_refused():
    with pytest.raises(ValueError, match="from the trace or from the gradient"):
        rtrrl.EligibilityReadout(direction="momentum")


# ------------------------------------------- what the rule underneath can read


UNIT_TRACE = DRTRRL(c=1.0, magnitude="sign", scope="block")
INTENTIONAL = IntentionalUpdate(eta=0.5)


def config(**optimizers) -> rtrrl.RTRRLConfig:
    """One config with Adam everywhere except where a test says otherwise."""

    return rtrrl.RTRRLConfig(
        num_envs=1,
        **{f"{name}_optimizer": optimizers.get(name, Adam(lr=1e-3)) for name in BLOCKS},
    )


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("rule", (UNIT_TRACE, INTENTIONAL), ids=("d_rtrrl", "iu"))
@pytest.mark.parametrize("kind", ("direction", "gain"))
def test_a_mixture_under_a_rule_that_cannot_read_a_length_is_refused(kind, rule, block):
    """The size half is only measurable through a rule that reads a length.

    ``sign`` exports whatever it is handed as a unit vector. The intentional
    update divides by a statistic carrying the same factor its step carries, so
    the two cancel. Under either, a mixture steps exactly as the pure condition
    it rescales does -- two of the four cells would be duplicates of the other
    two, at a million steps and five seeds each.
    """

    with pytest.raises(ValueError, match="cannot read the length"):
        rtrrl.refuse_unreadable_conditions(config(**{block: rule}), readout(kind))


@pytest.mark.parametrize("kind", tuple(rtrrl.ELIGIBILITY_SOURCES))
def test_the_arm_that_clips_a_length_rather_than_replacing_it_takes_every_condition(
    kind,
):
    """``td_out`` passes a trace shorter than ``c`` through at its own length."""

    clipped = DRTRRL(c=1.0, magnitude="td_out", scope="block")

    rtrrl.refuse_unreadable_conditions(config(torso=clipped), readout(kind))


@pytest.mark.parametrize("rule", (UNIT_TRACE, INTENTIONAL), ids=("d_rtrrl", "iu"))
def test_a_pure_condition_is_taken_by_every_rule(rule):
    """Nothing is refused for its own sake: the two pure cells are readable
    under every rule, because they differ in direction and not in length."""

    for kind in ("trace", "gradient"):
        for block in BLOCKS:
            rtrrl.refuse_unreadable_conditions(config(**{block: rule}), readout(kind))


# ------------------------------------------------------------ what is declared


def test_the_four_conditions_are_declared_as_one_choice():
    assert set(rtrrl.ELIGIBILITY_SOURCES) == {
        "trace",
        "gradient",
        "direction",
        "gain",
    }
    assert set(rtrrl.ELIGIBILITY_BRANCHES) == set(rtrrl.ELIGIBILITY_SOURCES)
    assert set(branch(rtrrl.PARAMETERS, "eligibility")) == {
        KIND,
        "direction",
        "gain",
    }, "only the conditions that divide by a norm declare anything under them"


def test_only_the_published_condition_is_searched_and_it_is_what_a_run_defaults_to():
    """A condition is an experiment's statement, not a dimension of a study."""

    declared = branch(rtrrl.PARAMETERS, "eligibility")[KIND]

    assert isinstance(declared, Parameter)
    assert isinstance(declared.search, Choice)
    assert declared.search.values == ("trace",)
    assert expand(rtrrl.PARAMETERS)[f"eligibility.{KIND}"] == "trace"
