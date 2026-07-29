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

Taking `test_lru_parity.py:398-415` as the example, six things are invisible
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

## The verdict: three classes, measured rather than declared

Per-leaf tolerance budgets are replaced by three classes.

**Unbiased.** The two leaves are bit-identical. This is the default and it is
what most of `test_blocks.py` and the untoleranced sections of the golden
comparison already assert.

**Rounding.** The gap behaves the way floating-point rounding behaves. Two
conditions are required and both must hold.

The first is that the gap travels with the format's epsilon. Float32's epsilon
is 1.19e-7 and float64's is 2.22e-16, a factor of 5.4e8, so recomputing at
double precision must shrink the gap by that order. A difference in what is
computed does not move at all. The threshold is 1e6, which separates the two
hypotheses with more than two orders of margin on either side; the worst leaf
measured in `test_lru_precision.py` reaches 3.4e7.

The second is that the gap is only a few last bits within each format, with the
threshold at 8. This condition is not redundant. An amplified rounding error
scales with the format and is still enormous, and that is a statement about the
conditioning of the computation rather than about the two implementations
agreeing. Without this condition the first one would pass it.

**Anomalous.** Everything else, and it fails.

Both thresholds are adopted from `test_lru_precision.py:105-122`, where they
were set from measurement rather than chosen. They are the only two numbers the
package carries, and their exact placement barely matters because the two
hypotheses they separate are eight orders of magnitude apart.

This is what removes the dozen tolerance tables and their justifying prose. A
comparison stops saying "four last bits are allowed here, and here is a
paragraph arguing that reassociation is why" and starts saying "the difference
here must be of the kind that rounding produces", which is then measured.

## The structural consequence

The rounding verdict cannot be applied to two arrays. It requires the same
computation at two precisions, so what it takes is a pair of
*precision-parametric computations* — callables that accept a dtype and return
named leaves — rather than a pair of values.

This makes the package's central object a pair of computations, not a pair of
arrays. `run(seed, dtype)` in `test_lru_precision.py:140-180` is the working
prototype of that shape: it takes a dtype, drives both implementations over one
stream, and returns each side flattened to named leaves.

The unbiased and anomalous verdicts still work on plain values, so comparisons
that never needed a tolerance are unaffected.

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
`_inject` at `test_lru_parity.py:153-178` already implements exactly this, keyed
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
`expected_sensitivity` at `test_lru_parity.py:290-316` is that translation:
twenty-seven lines of arithmetic on their matrices, producing our five
sensitivities.

The package cannot write these, and it should not pretend to. What it enforces
is that the correspondence is complete: every probe on one side is paired with
something on the other, and an unpaired probe is an error rather than a silently
skipped leaf.

### Scoreboard

Accumulates across the stream, keeping the worst gap each leaf reached and the
step it reached it at, then issues one verdict per leaf at the end.
`watch` and `assert_explained` at `test_lru_parity.py:319-350` are the
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

## What lives where

The package carries the vocabulary and the machinery: the three verdicts, the
precision-parametric comparison, the totality check on stimulus injection, probe
pairing, and the scoreboard. None of these depend on what this repository's
algorithms compute.

The repository keeps two things. Its shape helpers, because stripping a leading
batch axis and a leading time axis is a local convention rather than a general
one. And every correspondence model, because each is a statement about one
specific pair of implementations.

## Boundaries

**Golden snapshots cannot use the rounding verdict.** The recorded side of
`test_stream_ac_golden.py` is a fixed float32 artifact from 2026-07-17. There is
no double-precision counterpart and none can be made after the fact, because
recomputing the past at higher precision is not something a recording supports.
That comparison stays on the unbiased verdict plus the five hand-set allowances
it already carries. A future recording that stores both precisions becomes
eligible, and recording both should be the default when the snapshot generator
is rebuilt.

**Bit-identity across environments is not claimed.** It is not achievable in
float32 under XLA, and this is already the repository's position: the golden
test asserts the jax version matches the recording, and
`memo/docs/superpowers/specs/2026-07-20-rtrrl-numerical-modularization-design.md`
states that snapshots are regression tests rather than proof.

**Precision must be injectable before the rounding verdict applies.**
`memo/memorax/` names `float32` or `complex64` in 137 places, so precision is
not currently a free parameter there. The surface that actually needs threading
is much smaller than that number suggests: the LRU already takes a dtype through
`our_side` and `paper_side`, `test_blocks.py` needs it for the single leaf in
`REASSOCIATED`, and the torch comparison is per-tensor and therefore easy. Any
comparison whose double-precision arm silently stays in float32 must be reported
as vacuous rather than as passing, which means verifying the dtype of every leaf
on the high-precision side.

**Cross-process and cross-environment value capture is out of scope.** Both arms
run in one process. Capturing a reference's values in its own environment and
comparing them later is a separate design, and it is the one that would be
needed for a reference whose dependencies cannot coexist with ours.

## Migration

One file at a time, with the old and new spellings coexisting.

The first target is the `INFLUENCE` comparison in
`test_lru_parity.py:383-415`. It is the right one to start with because the LRU
is already precision-parametric, so the rounding verdict can be applied to it
without changing anything under `memo/memorax/` first, and because its five
hand-set budgets with eighteen lines of supporting prose are the clearest
example of what the three verdicts are meant to replace.

Success for that first conversion is judged by reading the converted file: the
quantity, the direction, and the kind of agreement being asserted should all be
visible at the comparison itself, and the eighteen lines should be gone rather
than relocated.
