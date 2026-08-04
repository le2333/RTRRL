# Numerical Testbench Design

## Goal

A package for asserting that two theoretically equivalent computations agree,
structured as a hardware testbench rather than as a test suite: one stimulus,
two implementations, probes at corresponding points, and a verdict at each
probe.

The package is a sibling of `training-sdk`, installed by path rather than
published, so it is a real package boundary without a release process. Working
name `testbench`.

## What is wrong with the current arrangement

`memo/tests/conftest.py` already carries the comparison arithmetic in forty
lines, and it serves all twelve test files. The arithmetic is not the problem.
The problem is that a comparison cannot be read at the place it is written.

Taking `test_lru_parity.py:387-422` as the example, six things are invisible
there:

The direction is positional. `watch(worst, A, B, ...)` gives no indication of
which side is ours and which is the reference; reversing them raises nothing and
only inverts the wording of every report.

The compared quantity is buried in shape plumbing. In
`flattened(jax.tree.map(lambda leaf: leaf[0, 0], sensitivity))` only
`sensitivity` carries meaning; the rest strips a batch axis and a time axis.

The tolerance is a distant name. `INFLUENCE` is defined three hundred lines
away, and the eighteen lines explaining why its entries are 2.0, 4.0 and 8.0 are
visible only there.

The unit is not in the code. `4.0` is last bits, `1e-13` is a relative gap and
`1e6` is a ratio between two of those; all three read as bare floats.

The accumulator is a mutable dictionary split across three statements —
declared, mutated in the loop, asserted after it. Omitting the last statement
leaves the test permanently green.

The names do not describe the judgement. `watch` says nothing about what it
does, and the `within` of `assert_within` is last bits, which cannot be read off
the name.

Across the suite the distant-constant problem accounts for about a dozen
uppercase tables: `READOUT`, `INFLUENCE`, `CREDITED`, `UNROLLED`, `ALLOWED`,
`FRAMEWORKS`, `SINGLE`, `DOUBLE`, `TRAVELS`, `BITS`, `REASSOCIATED`.

The seventh thing is not legibility. `conftest.last_bits:33` subtracts after
casting both sides with `.astype(np.float64)`, and on complex input numpy
discards the imaginary part, warns, and continues. A disagreement of exactly
`1j` is therefore measured as a gap of zero. The quantities `test_lru_parity.py`
exists to compare are complex — the hidden state and all five sensitivities are
`complex64` — so its central comparison has been reading real parts only, and
the run's warning count is where the evidence has been sitting.

That bug is worth more than an argument, because it is exactly the failure this
design is organised around. It is not a threshold set too loosely. It is a
comparison that ran, reported nothing, and established nothing, and no
adjustment to any of the eleven tables above would have revealed it. The library
answers it in two places: one widening rule, tested on the input that used to
return zero, in the one function that measures a gap; and probe pairing that
refuses a comparison rather than quietly passing it.

## The verdict: three classes

Per-leaf tolerance budgets are replaced by three classes.

**Unbiased.** The two leaves are bit-identical. This is the default and it is
what most of `test_blocks.py` and the untoleranced sections of the golden
comparison already assert.

The name is not a statistical claim, and the statistical reading was considered
and rejected. Rounding error is zero-mean for reassociated sums and products of
a sign-symmetric draw, because IEEE arithmetic is exact under negation, but that
symmetry does not survive a non-odd function, and this algorithm applies `exp`
to three parameters and divides the bounded step by `sqrt(v_hat) + 1e-6`.
Differing reduction trees break it too: a tree reduction has systematically
smaller error than a sequential one, so the difference carries a component tied
to the data rather than a symmetric one. Separately, testing a mean needs far
more samples than testing a maximum; four seeds can exclude only a gross bias.
So bias is a signal worth reporting and not a verdict worth issuing, and a
genuinely rounding-level gap that happens to be biased must not fail.

**Rounding.** The gap is of the kind that rounding produces. Two conditions are
required. One is about the size of the gap and is always the same; the other is
about the nature of the gap and comes in a cheap form and a decisive form, of
which only the cheap one is built now.

The first is magnitude: the gap is within a few last bits, with one global
threshold of 8. This is one number where the suite currently has about a dozen
per-leaf entries, and it is nominally looser than several of them — `INFLUENCE`
allows `nu_log` 2.0 and `READOUT` allows `h` 2.0.

Very little detection power is actually lost, because the per-leaf spread is not
calibration. Three things say so.

`test_lru_parity.py:104` states how the entries were obtained: each is twice the
worst of four seeds, then rounded to a power of two. The maximum of four samples
has enormous variance, so the distance between 2.0 and 4.0 is within what a
different four draws would produce. To use 2.0 as an instrument one would have
to distinguish a fourfold growth in `nu_log`'s rounding from a change of seed,
and four samples cannot do that.

The prose accompanying those entries contradicts them. Line 99 says `theta_log`
is the widest of the group because its chain factor is a rotation; the table
three lines below gives `theta_log` 4.0 and `gamma_log` 8.0.

The mechanism that prose offers predicts the wrong order. The chain factors are
`-exp(nu_log) * Lambda` and `1j * exp(theta_log) * Lambda`, both complex, against
`exp(gamma_log)` and `exp(gamma_log) / gamma_log`, both real. A real scaling
reassociates more benignly than a complex one and a rotation worst of all, which
predicts `theta_log` above `nu_log` above `gamma_log`. Measured, `gamma_log` is
the widest of the three. A mechanism that does fit is visible in the reference's
recursion at `memo/memorax/networks/sequence_models/upstream_lru.py:127-135`:
`new_grad_lambda` adds `h_tminus1` with no inner reduction, `new_grad_B` adds an
outer product with none either, and `new_grad_gamma` adds `inputs @ B.T`, a
length-`FEATURES` contraction over signed values that carries a reassociation and
a cancellation of its own. That is a hypothesis from reading the recursion; the
file establishes neither it nor the one it states.

There is also a precedent inside the same file. `CREDITED` already gives all five
recurrent parameters 8.0 and both readout matrices 4.0, so at the gradient the
per-leaf discrimination was already given up in favour of one number per group.
A global threshold brings `INFLUENCE` to where `CREDITED` already is.

The second is growth: the relative gap must grow as the accumulation lengthens.
Rounding accumulates, so the relative gap grows roughly as the square root of
the number of accumulated terms. A multiplicative difference in what is computed
gives a constant relative gap instead — the missing exponential in the published
influence matrix for `B` scales it by a fixed `gamma_log / exp(gamma_log)`
whether the stream is five steps or forty. Two independent axes lengthen the
accumulation and both are used: the stream length, and the width over which each
contraction reduces.

**Anomalous.** Everything else, and it fails.

### What the growth condition is and is not

It is a weak discriminator and must be described as one. Between five and forty
steps the square-root model predicts a factor of about three, so the two
hypotheses are separated by roughly one order of magnitude once both axes are
used — against the eight orders the precision test below achieves. A threshold
placed in that window will admit false positives and false negatives.

It is kept because it costs nothing and because it is blind in a different place
than the magnitude condition is. The magnitude condition cannot see a small
constant relative difference; the growth condition can see exactly that, which
is the class the missing exponential belongs to. Neither is decisive; the pair
covers more than either.

### The precision test, deferred

There is a third condition of the same shape, and where it is available it
replaces the growth condition rather than joining it. It asks the same question
— is this gap the kind rounding produces — and answers it decisively, so a
comparison that can afford it has no use for the weaker screen.

It recomputes the comparison at double precision. Float32's epsilon is 1.19e-7
and float64's is 2.22e-16, a factor of 5.4e8, so a rounding gap must shrink by
that order while a difference in what is computed does not move at all.
`test_lru_precision.py:246-290` implements this with a threshold of 1e6 against
a worst measured leaf of 3.4e7, which is more than two orders of margin on
either side.

The magnitude condition is required in every case regardless, because neither
the growth check nor the precision check can tell an amplified rounding error
from an ordinary one, and an error that scales correctly while being enormous is
a statement about the conditioning of the computation rather than about two
implementations agreeing.

It is deferred because it requires the computation to be precision-parametric
and `memo/memorax/` names `float32` or `complex64` in 137 places. The fixture
must accept a dtype from the first day anyway, so resuming it later costs
nothing beyond threading the dtype through whichever computation is being
compared.

What is given up meanwhile is recorded rather than forgotten. Same-precision
multi-case judgement can only see a difference where the draws land in the
region it manifests. This repository has already been bitten by that class:
`test_paper_parity.py:13-15` records that the fork moved `eps` out of a square
root, which "while the second moment is near eps changes the step by a factor,
not by a rounding" — invisible on any draw that stays away from `eps`. The
guard `test_the_missing_exponential_is_a_real_difference` exists for the same
reason. Multi-case lowers the chance that every draw sits at a degenerate point;
it does not remove it, and sampling coverage becomes a standing assumption of
the whole scheme.

## Fixture and cases, fixture first

The fixture is what mounts two implementations so they present the same
interface. The cases are the points it is driven at. They are separate modules,
and the fixture is designed first.

They are currently one file. `test_lru_parity.py` is 682 lines: the fixture runs
from line 73 to line 355 and the cases begin at the first parametrize on line
357, so the seam is already clean and nearly halfway. That the fixture wants to
be a module is not a hypothesis — `test_lru_precision.py:48-60` imports eleven
names from `test_lru_parity` to reuse it, which is what reuse looks like when
there is nowhere to reuse from.

### Why the fixture comes first

Every verdict is a statement about behaviour along an axis, so an axis the
fixture does not expose is a verdict that cannot be issued. The growth condition
is impossible without a stream-length axis and a width axis; the precision test
is impossible without a dtype axis; multi-case is impossible without a seed
axis. Deciding the axes is therefore the commitment, and it has to be made
before the cases exist rather than discovered underneath them.

### The axes

`seed`, for multi-case. `steps`, for growth along the stream. `width`, for
growth along each contraction. `dtype`, for the deferred precision test. `arm`,
for selecting which implementation is mounted.

Four of these already exist in `test_lru_parity.py`: `inputs` takes `steps`,
`paper_side` and `our_side` take `dtype`, `our_side` takes `cell`, and every
entry point takes `seed`. Width does not — `HIDDEN` and `FEATURES` are module
constants at lines 73 and 74. Adding it is the one axis this design requires
that is not already there, and it is required because it is the second
independent way to lengthen an accumulation, which is what makes the growth
condition worth having at all.

`our_side` also carries `skip`, which is not an axis but a knob one vacuity
guard needs. Knobs like that belong to the fixture too, but they are named
separately so that the case modules can tell an axis from a lever.

### What belongs to each

The fixture owns mounting, injection, stepping, probe extraction, and the
correspondence model. The correspondence model belongs here rather than with the
cases because making two differently shaped things present the same probe names
is exactly what a fixture does.

It also owns the shape helpers, because stripping a leading batch axis and a
leading time axis is this repository's convention rather than a general one.

The cases own which points on the axes to visit, which probes to compare at,
which verdict each comparison demands, and the vacuity guards.

The package owns the three verdicts, driving a comparison along an axis and
judging how the relative gap moves, the injection totality check, probe pairing,
the scoreboard, and the flattening of a pytree to named leaves. None of these
depend on what this repository's algorithms compute, which is what makes the
package boundary defensible on a single sample: the correspondence models and
the shape conventions, the two things that would have been overfitted to the
LRU, are exactly the two the fixture keeps.

### The risk

Designing a fixture before all its cases exist can overfit it to the cases that
happen to be at hand. The mitigation is that the axes are the only real
commitment and they are derived from the verdicts rather than from the current
tests; everything else in the fixture can move without disturbing a case.

## Architecture

### Stimulus

One draw covering every parameter, injected into both implementations, plus the
input sequence and the initial carry.

For a neural network graph those three exhaust the degrees of freedom, so
injecting at the module boundary is complete injection and no graph surgery is
required. This is why the difficulty the testbench analogy suggests — reaching
into a computation graph to drive an internal node — does not arise on the
stimulus side.

The injection must be total in both directions: every leaf of each parameter
tree is covered by the draw, and every drawn value is used by at least one side.
`_inject` at `test_lru_parity.py:157-182` already implements exactly this, keyed
on the last element of each path so that neither side's module nesting is
written into the test. It is the equivalent of checking that no port is left
floating, and it is what prevents a comparison that passes because it compared
nothing.

### Probes

Two mechanisms, in order of preference.

Function and module boundaries, which every current test uses and which cost
nothing.

Flax's `sow` into the `intermediates` collection, for taps inside a module. The
plumbing for this already exists in the repository: seven algorithm modules
apply with `mutable=["intermediates"]` and log the result. There is exactly one
`.sow()` call in the whole repository, at
`memo/memorax/networks/blocks/router.py:33`, so that collection is effectively
empty today. Adding taps is therefore cheaper than it looks — the wiring is laid
and unused.

A reference kept verbatim cannot be instrumented at all. `upstream_lru.py` may
not be edited, on the grounds recorded in its own docstring, so its probes are
whatever it already returns. Here that is enough by luck: the influence matrices
ride in the carry it hands back at `:199`. A future reference may keep its
intermediates private, and then its observation points are fixed by its author
rather than chosen by us. The package must treat the available probe set as an
input, not as something it can widen.

### Correspondence

The part no tool can generate, and the main work of any parity effort.

Two equivalent implementations usually decompose differently, so their internal
nodes do not correspond even when their endpoints do. In the LRU case ours
chains each parameter's derivative into the Jacobian before accumulating and the
reference chains afterwards at the gradient, so the reference's three influence
matrices have no counterpart on our side at all until they are translated.
`expected_sensitivity` at `test_lru_parity.py:294-320` is that translation:
twenty-seven lines of arithmetic on their matrices, producing our five
sensitivities.

The package cannot write these, and it should not pretend to. What it enforces
is that the correspondence is complete: every probe on one side is paired with
something on the other, and an unpaired probe is an error rather than a silently
skipped leaf.

### Scoreboard

Accumulates across the stream, keeping the worst gap each leaf reached and the
step it reached it at, then issues one verdict per leaf at the end.
`watch` and `assert_explained` at `test_lru_parity.py:323-355` are the
prototype.

Watching the whole stream rather than stopping at the first disagreement is
required rather than stylistic. The influence matrices start at zero and the
initial hidden state is zero with them, so at the first step the quantities
built from the previous carry are zero on both sides and agree for a reason that
proves nothing.

The scoreboard is closed by an explicit call, not by a context manager. A
context manager would make the closing impossible to forget, but it moves the
failure's line number from the step that diverged to the end of the block, and
the diagnostic value of the line number is worth more than the safety.

## Boundaries

**Golden snapshots cannot use the rounding verdict at all.** Both of its
conditions need the comparison rerun somewhere the recording cannot follow. The
growth condition needs the accumulation lengthened, and the recorded side of
`test_stream_ac_golden.py` is a fixed artifact of three steps at one width from
2026-07-17. The precision condition needs a double-precision counterpart, and
recomputing the past at higher precision is not something a recording supports.
That comparison stays on the unbiased verdict plus the five hand-set allowances
it already carries, and those five stay hand-set.

This is an argument for what a rebuilt snapshot generator should record. A
recording taken at two stream lengths, two widths and two precisions makes the
whole verdict available to the golden comparison; a recording of one point does
not, whatever tooling is later put around it.

**Bit-identity across environments is not claimed.** It is not achievable in
float32 under XLA. The golden test only asserts that the JAX version matches a
fixed recording, so the snapshot is a regression test rather than proof of
cross-environment bit identity.

**The deferred precision test needs precision to be injectable.**
`memo/memorax/` names `float32` or `complex64` in 137 places, so precision is
not a free parameter there today. The surface that actually needs threading is
much smaller than that number suggests: the LRU already takes a dtype through
`our_side` and `paper_side`, `test_blocks.py` needs it for the single leaf in
`REASSOCIATED`, and the torch comparison is per-tensor and therefore easy.

Whenever it is resumed, a comparison whose double-precision arm silently stayed
in float32 must be reported as vacuous rather than as passing, which means
verifying the dtype of every leaf on the high-precision side. This is the
failure that looks exactly like success, and the repository has already met it:
`test_lru_precision.py:151-154` records that a float32 draw held the whole
comparison at float32 however the two implementations were run, "which is what
it did at first".

**Cross-process and cross-environment value capture is out of scope.** Both arms
run in one process. Capturing a reference's values in its own environment and
comparing them later is a separate design, and it is the one that would be
needed for a reference whose dependencies cannot coexist with ours.

## Migration

The fixture is extracted first, because the axes it exposes decide which
verdicts the cases can ask for. That extraction is mostly a move:
`test_lru_parity.py:73-355` becomes a fixture module, the eleven names
`test_lru_precision.py` imports from a test module start coming from it
instead, and the one addition is the width axis, which means turning `HIDDEN`
and `FEATURES` from module constants into parameters.

Then one comparison at a time, with the old and new spellings coexisting.

The first is `INFLUENCE`, at `test_lru_parity.py:387-422`. It is the right one
to start with because its five hand-set budgets and eighteen lines of supporting
prose are the clearest example of what the three verdicts replace, and because
the accumulation it measures is the one the growth condition was chosen for: the
influence matrices start at zero and build along the stream, so lengthening the
stream is guaranteed to move a rounding gap.

Converting it also settles the contradiction those eighteen lines contain, since
the claim that `theta_log` is the widest and the table that makes `gamma_log`
widest cannot both survive into a scheme with one threshold. The contradiction is
itself the argument for the scheme: numbers set by measurement and justified by
prose written beside them will drift apart, and nothing in the file can notice.

Success for that first conversion is judged by reading the converted file. The
quantity, the direction and the kind of agreement being asserted should all be
visible at the comparison itself; the eighteen lines should be gone rather than
relocated; and the case module should contain no mounting, no injection and no
translation between the two implementations.
