# Running a round as one graph

A sweep over seeds and non-structural hyperparameters is today one Batch job per
trial. Each job holds one GPU, and each of those GPUs runs a graph sized for one
environment. On `g6.xlarge` that is an L4 kept busy by a CartPole, which is the
shape a GPU is worst at: the device is paid for by arithmetic wide enough to
fill it, and one member is not wide.

The members of a round differ only in leaf values. Their graph is the same
graph. So the round can be one compiled program with a member axis, and the
device can be filled by the sweep itself rather than by making any single run
bigger.

This is the design for that, with the existing path left standing.

## What was measured first

The design rests on one claim: that `assemble` can be called inside `jax.vmap`,
with tracers standing in for the leaves being swept, and produce a working
graph. That is not obvious -- the parameters reach the graph at build time, and
a build that reads a value is a build that can demand a concrete one.

It holds. Four DRQN members, one learning rate each:

```
float leaves in the trained state: 50
leaves that differ across members: 48 of 50
largest leaf, per-member L2 against the lr=0 member:
    lr=1.0     ||delta|| = 15.765826
    lr=0.1     ||delta|| =  5.659760
    lr=0.0     ||delta|| =  0.000000
VERDICT: sweep reaches each member
```

Every member ran from the same PRNG key, so the learning rate is the only thing
that varied, and the trained parameters diverge with it monotonically. A sweep
that had been silently broadcast would have produced four bit-identical members;
48 of 50 leaves differ.

The control matters because the obvious check does not work here. CartPole pays
1.0 every step, so a fixed-length chunk's return sums to its own length for any
policy at all -- the first version of this experiment "showed" four identical
members and was measuring nothing.

## Why the parameter layer needs no change

`read_parameters` assigns leaves straight through:

```python
values[item.name] = params[key]
```

No coercion, no validation. A `jnp` array in the manifest mapping arrives in the
component's dataclass as an array, and `optax.adam(base.lr, ...)` accepts it.

`read_branch` reads `str(params[key])` for the `kind` of a `structure`, which
keeps every branch selection concrete by construction. So the boundary the
design needs -- *values may vary across members, choices may not* -- is already
where it has to be, and it is enforced by the existing code rather than by a new
rule someone has to remember.

That boundary is also already declared. `param(valid=, search=)` is a value;
`structure(branches=)` is a choice; `group(of=)` is a scope. "Non-structural
hyperparameter" is not a new concept to introduce, it is `param`.

## Where it actually breaks

The algorithms coerce. `drqn.graph` does

```python
gamma=float(parameters["gamma"]),
```

and a tracer through `float()` is a `ConcretizationTypeError`. There are 36 such
call sites:

| algorithm | sites |
| --- | --- |
| `drqn.py` | 10 |
| `rtrrl_aaai.py` | 10 |
| `r2d2.py` | 12 |
| `stream_ac.py` | 4 |

They divide cleanly, and the division is the real content of this design:

**Shape-determining — must stay concrete.** `replay.capacity`,
`replay.minimum_size`, `replay.batch_size`, `learning.truncated.length`,
`core.*.hidden_dim`, `core.*.feature_dim`. These size arrays. A member axis
cannot vary them, because members of one vmap share their shapes.

**Value-only — may be traced.** `gamma`, `grad_clip`, `epsilon_start`,
`epsilon_end`, `evaluation_epsilon`, `target.update_period`,
`exploration.epsilon_decay_steps`, every `lr`, and RTRRL's `lambda_*`, `eta_*`,
`entropy_rate`. These are read arithmetically. Dropping the `float()` and
letting the value through is the whole change at each one.

So "non-structural" is necessary but not sufficient: a swept leaf must be a
`param` **and** not shape-determining. `hidden_dim` is a `param` and cannot be
swept.

### Saying which is which

Two ways to keep a shape-determining leaf out of a sweep.

*Declare it.* Add `param(..., shapes=True)` and have the launcher refuse a sweep
that names one. Backward compatible -- the flag defaults false -- and it costs
marking roughly eight leaves across four algorithms. The error arrives before a
job is submitted and can say which leaf and why.

*Discover it.* Trace once at launch with tracers in every swept position and let
JAX raise. The error names the leaf too, and nothing has to be declared or kept
in sync.

Declaring is better here, because the second is only as good as the reachability
of the code path that reads the leaf: a leaf read inside a branch this
configuration does not select would trace clean and fail later, on a
configuration that does select it. Do both if the marking proves tedious --
declaration for the message, the trace as the backstop.

## What equivalence to ask for

The obvious acceptance criterion -- a member reproduces the same seed under the
single-member driver, digit for digit -- is unreachable, and building to it
would have meant chasing a bug that is not there.

`jax.vmap` rewrites a computation into batched operations; XLA compiles those to
different kernels; different kernels reduce in a different order. On DRQN the
divergence shows up as an episode one step longer than the driver's, about a
hundred and fifty steps in, once the accumulated last-place difference flips an
action. A **one-member** ensemble diverges too, at the same episode and in the
same direction as a three-member one, which is what rules out the batching of
several members as the cause.

The property to hold instead is that a member is a function of its seed and of
nothing else about the round -- not its size, not the member's index in it, not
which other seeds it travelled with. That one is testable, and it holds: seed 3
reports identical episodes alone, in a pair, in a round of three, and at index 2
of a round of five.

It is also the property that matters. Bit-identity with the driver buys a
comparison nobody needs to make; independence from the round is what keeps a
reported number from depending on how its sweep happened to be packed.

The consequence to write down: **an ensemble run is not comparable digit-for-digit
with an acceptance number taken on the driver.** Comparisons across the two
paths are statistical, and a re-run of an acceptance file under the ensemble
will not reproduce its recorded score exactly.

## The shape of the change

### Untouched

`assemble`, `BuildRequest`, `BuiltAlgorithm`, `Program`, `Driver`, `Runtime`,
`read_parameters`, `read_branch`, every existing entry, the existing worker path,
and every experiment file. A run that names `drqn` today gets exactly the driver
it gets today.

### New: an ensemble driver beside the existing one

`Driver` jits the five arrows and threads one key. The ensemble driver vmaps a
function that *builds and then runs*, because vmap maps over arguments and a
value closed into a graph is not an argument:

```python
def member(swept, key, steps):
    parameters = {**fixed, **dict(zip(paths, swept))}
    program = assemble(definition, request(parameters)).program
    state = program.init(key)
    return program.train(key, state, steps)
```

Everything below the member axis is unchanged code. `EpisodeTracker` stays as it
is and is applied per member on the host, over arrays already computed --
members are not streams and must not be folded into `num_envs`, because streams
within a member share a policy and members do not.

### New: an entry that takes a group

The worker already walks a manifest of runs:

```python
for config_uri in manifest["runs"]:
```

and runs each in its own subprocess. The ensemble path adds a manifest that
names a *group* the entry receives whole, rather than a list the worker
iterates. The control plane's notion of a trial does not change: N members still
produce N results, N metric streams and N scores. What changes is that one job
computes them together.

That keeps `scoring.py` and `settle` untouched, which is worth more than it
looks: the scorer is where a subtle change would be least visible.

### Contract

Bump 11 to 12 with one optional field naming the group and the swept paths. An
image built before the bump ignores it; an experiment written before the bump
does not set it.

## What the round already gives us

An earlier reading in this conversation was wrong and is worth correcting here,
because it changes what is possible: TPE was described as sampling trial by
trial, which would have left only the seed axis vmappable.

`HPO.ask()` returns `tuple[SampledTrial, ...]` -- a whole round. TPE here is
round-sequential, not trial-sequential: it reads the previous round's scores and
proposes the next round as a batch. So **the vmappable unit is a round**, under
TPE as much as under grid or random, and the seed axis nests inside it.

One qualification: a round may contain trials that differ *structurally*, if the
space includes a `kind`. Those cannot share a graph. Partition a round by its
structural signature and vmap each partition; a space with no structural choice
-- which is what a seed and non-structural sweep is -- is one partition.

## Staging

Each stage is useful alone and leaves the tree working.

1. **Seeds only, no coercion changes.** Seeds need nothing from the parameter
   layer: the graph is bit-identical and only the key varies. This is the whole
   mechanism -- vmapped build, per-member tracking, N results from one job --
   with none of the 36 call sites touched. It is also most of the win, since a
   sweep is usually seeds times a handful of settings.
2. **Mark the shape-determining leaves.** Mechanical, and it makes stage 3 safe
   to do incrementally.
3. **Uncoerce the value-only leaves, one algorithm at a time.** DRQN first: it
   is the entry a GPU pays for, and it has a working acceptance file to compare
   against.
4. **Partition a round by structural signature.** Only needed when a space that
   picks a `kind` is swept.

Stop after 1 and the infrastructure is strictly better than today. Stop after 3
and the original goal is met.

## What this does not settle

**Whether it is faster.** Filling an L4 with members is the reason to do this,
but nothing here measures it. `experiments/drqn gpu smoke.yaml` prices a single
member against CPU and is the baseline the ensemble has to beat; the comparison
is per-member wall clock at a given member count, not the job's own.

**Memory** is settled, and it is not the constraint. One member's initial state,
measured on the configurations the acceptance files describe:

| entry | one member | largest leaf | members per 24 GiB |
| --- | --- | --- | --- |
| DRQN | 0.582 MiB | replay `observation` `(1, 8224, 4)` | ~42,000 |
| RTRRL | 0.090 MiB | LRU `C_real` / `C_imag` `(32, 32)` | ~273,000 |

RTRRL is the smaller of the two, against the expectation that carrying a
jacobian would make it the larger. Its torso is a single-layer LRU at hidden 32,
so the influence it carries is `(1, 32, 32)` -- four kilobytes -- and the
recurrent kernel it multiplies is the same size. DRQN is bigger only because
replay holds 8224 transitions of a four-dimensional observation.

At these sizes the member count is set by when the device saturates, not by when
it runs out of room, and choosing it is a throughput measurement rather than a
memory budget. What remains unmeasured is the per-step chunk: `train` stacks its
observations over `chunk_steps`, so a 5000-step chunk is the term that actually
scales with the member axis. It is still small -- tens of scalars per step -- but
it is the one to watch, not replay.

**Failure granularity.** One member diverging to NaN does not stop the others,
which is good, but they also cannot be retried separately -- the job is the
round. `non_finite: worst` already gives such a member a score rather than an
error, so this is a cost to accept rather than a bug to fix.

**RTRRL's compile time**, rather than its memory. On CPU it takes 20 seconds to
assemble and 32 to reach the end of an evaluation, against DRQN's 14 -- the
jacobian-in-scan is expensive to compile even where it is cheap to hold. vmap
compiles once for any member count, so this is a fixed cost rather than a
growing one, but it is paid on every job and it is the reason a round should be
large rather than a sweep be many rounds.
