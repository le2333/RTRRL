"""Which of the two LRUs float32 hurts more, measured against float64.

``test_lru_parity.py`` establishes that ours and the published LRU compute the
same quantities, and then allows each leaf a few last bits because they compute
them in different orders. That file can say the gaps are reassociation, on the
argument that a constant distributed over a sum is exact in arithmetic and not in
float32. It cannot say so from measurement: at one precision there are two
numbers and no third to place them against, so the argument stands on reasoning
about the code, and the code is what is in question.

This is the third number. Both implementations are run again on the same draw in
double precision, which turns one comparison into three that answer separate
questions.

The first is whether the two formulas are the same at all. If they are, doubling
the precision takes the gap between them down with it -- a reassociation error is
bounded by the machine epsilon of the format it happens in -- and what was a last
bit or two at float32 lands near 1e-16 at float64. If instead the two computed
different quantities, the gap would be whatever the difference is and would
barely move. It lands at one or two epsilons, so the parity file's allowances are
reassociation and are now measured to be, rather than argued to be.

The second is which single-precision run is closer to the answer. That question
was asked and it needs a reference to be answerable at all, so the float64 run
becomes one: it is the same arithmetic on the same inputs with the rounding
pushed nine decimal digits down, which makes the float32 errors measurable
against it rather than against each other. The answer is neither. Both sit at
around one float32 epsilon after forty steps, and which of them is closer changes
from leaf to leaf and seed to seed, with several exact ties -- so the ordering of
operations costs each of them the same accuracy, and there is no fidelity to
trade against the cost of either order.

The third is the honesty of the second. Using our float64 run as the reference
would favour us if the choice mattered, so the first test's bound is what licenses
it: the two float64 runs agree to around 1e-16 while the errors being compared are
around 1e-7, so either could serve and the answer would not change. That is the
order the three sit in, and the reason the reference test comes first.

This module needs ``jax_enable_x64``, which is a process-wide flag JAX will not
scope to a block, so it is skipped unless the process was started with it. CI
runs it as its own step; ``memo-ci.yml`` has the invocation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from conftest import flattened
from test_lru_parity import (
    FEATURES,
    HIDDEN,
    drawn,
    expected_sensitivity,
    input_gain,
    inputs,
    our_side,
    our_step,
    paper_side,
    paper_step,
    widened,
)

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="needs JAX_ENABLE_X64=1, which is process-wide and so is its own CI step",
)

# Longer than the parity file's five. Reassociation error accumulates along the
# stream -- every allowance there lands at the third step or later -- and this
# measures the accumulation rather than checking it stays inside a bound, so it
# is given room to grow into.
STEPS = 40

# What a reassociation of float32 can reach once it has run for STEPS steps,
# relative to the size of the leaf. Around 1e-7 is one float32 epsilon; the bound
# is loose because this test does not exist to pin the error down, only to
# establish that both errors are of that order and that neither implementation is
# in a different regime from the other.
SINGLE = 1e-4

# What two float64 runs of the same formula in different orders may differ by,
# relative. The four seeds reach 2.0e-16, 1.9e-16, 1.9e-16 and 4.5e-16 -- one or
# two epsilons of the format, after forty steps -- so this is roughly nine times
# the worst of them. Loose enough not to depend on which order XLA picks on a
# given host, and eleven orders of magnitude tighter than SINGLE, which is the
# margin the whole comparison rests on.
DOUBLE = 4e-15


def relative(got, wanted) -> float:
    """How far apart two leaves are, scaled by the size of what they hold.

    Relative and not in last bits: the two sides of every comparison here are in
    different formats, and a last-bit count is a statement about one format.
    """

    got, wanted = jnp.asarray(got), jnp.asarray(wanted)
    scale = float(jnp.max(jnp.abs(wanted.astype(jnp.complex128))))
    gap = float(
        jnp.max(jnp.abs(got.astype(jnp.complex128) - wanted.astype(jnp.complex128)))
    )
    return gap / max(scale, 1e-30)


def run(seed: int, dtype):
    """Both implementations over one stream, and everything worth comparing.

    Returns ours and theirs separately, each flattened to named leaves, with
    theirs already mapped through ``expected_sensitivity`` so that the two are
    quantities of the same kind. The mapping is the parity file's, imported rather
    than restated, so this cannot be measuring a second reading of their code.
    """

    layer, paper_params, paper_carry = paper_side(seed, dtype=dtype)
    core, our_params, our_carry, sensitivity = our_side(seed, dtype=dtype)
    # In the precision the run is in, not the precision it was drawn in. The
    # mapping between their influence matrices and our sensitivities is arithmetic
    # on these, so a float32 draw here would hold the comparison at float32
    # however the two implementations were run -- which is what it did at first.
    values = widened(drawn(seed), dtype)
    xs, _ = inputs(seed, steps=STEPS)
    xs = xs.astype(jnp.finfo(dtype).dtype)

    ours, theirs = {}, {}
    for step, x in enumerate(xs):
        paper_carry, paper_y = paper_step(layer, paper_params, paper_carry, x)
        our_carry, our_y, sensitivity = our_step(
            core, our_params, our_carry, sensitivity, x
        )
        if step == len(xs) - 1:
            ours = flattened(
                {
                    "h": our_carry.state[0, 0],
                    "y": our_y[0, 0],
                    **jax.tree.map(lambda leaf: leaf[0, 0], sensitivity),
                }
            )
            theirs = flattened(
                {
                    "h": paper_carry[0],
                    "y": paper_y,
                    **expected_sensitivity(values, paper_carry[1]),
                }
            )
    return ours, theirs


@pytest.mark.parametrize("seed", range(4))
def test_the_two_formulas_agree_when_the_format_stops_rounding(seed):
    """The parity file's premise, measured.

    Every allowance over there is justified as reassociation. If that is what it
    is, it is bounded by the epsilon of the format, and moving to float64 has to
    take it with it. So the same two implementations on the same draw, in double
    precision, are required to agree eight orders of magnitude tighter than the
    single-precision comparison needs -- which no pair of different formulas would
    do, and which nothing but the format changed to achieve.
    """

    ours, theirs = run(seed, jnp.complex128)
    apart = sorted(
        ((relative(ours[name], theirs[name]), name) for name in theirs), reverse=True
    )
    worst, where = apart[0]
    print(f"\nseed={seed}, {STEPS} steps, float64: worst {worst:.3g} at {where}")
    assert worst < DOUBLE, (
        f"seed={seed}: at float64 ours and the published LRU are still {worst:.3g} "
        f"apart relatively at {where}, which is too far for the same formula in a "
        "different order and means the parity file's allowances are covering a "
        "difference in what is computed:\n"
        + "\n".join(f"  {gap:.3g}  {name}" for gap, name in apart)
    )


@pytest.mark.parametrize("seed", range(4))
def test_neither_order_is_meaningfully_more_faithful(seed):
    """Both float32 runs against the float64 answer, per leaf.

    Ours forms the whole input projection and unrolls it by an associative scan,
    chaining each parameter's derivative into the Jacobian before accumulating.
    Theirs steps the recurrence and chains at the gradient. Neither order is
    obviously the more accurate one and the difference between them is visible in
    the parity file only as a gap with no direction, so this gives it one.

    Reported rather than ranked. The assertion is that both are within the
    accumulated float32 error the format allows, which is the claim that matters:
    the reassociation is not costing either implementation accuracy beyond what
    single precision costs anyway. A winner at 1e-7 would be a property of this
    draw, not of the arithmetic.
    """

    single_ours, single_theirs = run(seed, jnp.complex64)
    truth, _ = run(seed, jnp.complex128)

    lines, worst = [], 0.0
    for name in sorted(truth):
        ours = relative(single_ours[name], truth[name])
        theirs = relative(single_theirs[name], truth[name])
        worst = max(worst, ours, theirs)
        closer = "ours" if ours < theirs else "theirs"
        lines.append(f"  {name:12s} ours {ours:9.3g}   theirs {theirs:9.3g}   {closer}")
    report = f"\nseed={seed}, {STEPS} steps, against float64:\n" + "\n".join(lines)

    print(report)
    assert worst < SINGLE, (
        f"a float32 run is {worst:.3g} away from the float64 answer relatively, "
        f"which is past what accumulated single-precision rounding explains:{report}"
    )


def test_the_reference_is_not_the_thing_it_measures():
    """And float64 is far enough down to be called the answer.

    The test above measures two float32 runs against our float64 run. That is
    only a measurement of them, rather than partly of the reference, if the
    reference's own uncertainty is negligible at the scale being reported. So the
    two float64 runs are required to sit far below the float32 errors, and the
    margin is asserted rather than assumed.
    """

    ours64, theirs64 = run(0, jnp.complex128)
    ours32, _ = run(0, jnp.complex64)

    between = max(relative(ours64[name], theirs64[name]) for name in theirs64)
    measured = max(relative(ours32[name], ours64[name]) for name in ours64)
    assert between * 1e3 < measured, (
        f"the two float64 runs are {between:.3g} apart and the float32 error being "
        f"measured against them is {measured:.3g}, which is not the three orders of "
        "magnitude that would make either of them serviceable as the reference"
    )


def test_the_comparison_would_notice_a_changed_formula():
    """And the float64 bound is a bound, not a number nothing could exceed.

    ``test_the_two_formulas_agree_when_the_format_stops_rounding`` passes at
    1e-13. If it would also pass on two implementations that genuinely computed
    different things, it would be asserting nothing. So the correction that makes
    the two comparable is removed -- leaving the published revision's missing
    exponential in place, which is a real difference and the one this repository
    has already had to account for -- and the bound is required to catch it.
    """

    layer, paper_params, paper_carry = paper_side(0, dtype=jnp.complex128)
    core, our_params, our_carry, sensitivity = our_side(0, dtype=jnp.complex128)
    values = widened(drawn(0), jnp.complex128)
    xs, _ = inputs(0, steps=4)
    xs = xs.astype(jnp.float64)

    for x in xs:
        paper_carry, _ = paper_step(layer, paper_params, paper_carry, x)
        our_carry, _, sensitivity = our_step(
            core, our_params, our_carry, sensitivity, x
        )
    uncorrected = expected_sensitivity(values, paper_carry[1], correcting=False)
    gap = relative(sensitivity["B_real"][0, 0], uncorrected["B_real"])
    assert gap > DOUBLE, (
        f"dropping the missing exponential moved the influence matrix for B by "
        f"{gap:.3g} relatively, which is inside the float64 bound, so that bound "
        "would not have noticed the two implementations computing different things"
    )
    # And the factor that was dropped is the one named, at the width the leaves
    # here have, so the difference above is that defect and not a shape accident.
    assert input_gain(drawn(0)).shape == (HIDDEN,)
    assert sensitivity["B_real"][0, 0].shape == (HIDDEN, FEATURES)
