# Episode sampling throughput

DRQN performs one learner update per environment transition, so replay's
per-update work is multiplied by the length of the run and by nothing else.
The R1.1 5M throughput run on `StatelessCartPoleEasy` asks for
`replay.capacity: 400000`, which is fifty times what the acceptance files
use, and it did not reach its first 100k evaluation promptly.

This document reports what in that path grew with the capacity, what it cost,
and what is left after issue 62. Numbers come from
`docs/episode-sampling-throughput.py`, which measures the replay path and
nothing else — no network, no environment, no optimiser.

## What was being paid

`EpisodeWindowBuffer.sample` chose its minibatch by Gumbel top-k: an
independent Gumbel score for every slot in the episode index, a `top_k` over
the scores, and — to decide which slots were eligible to be scored — a mask
built by comparing every record's `start` against the eviction threshold. The
index is sized so it can never wrap before the transition ring does, which
makes it `capacity + max_episode_length` records long. Every one of those was
touched once per learner update, which is once per environment step.

`can_sample` was on the same footing. It reads `retained`, which summed the
lengths of every stored record, and it is read on *every* step through the
`lax.cond` that decides whether to update — including the warmup steps, where
there is nothing to draw at all.

Both are also in the first trace, so the size showed up in compilation before
it showed up in any step.

## What replaces it

Storage did not change. The index did:

- it is `[streams, capacity]` with a commit counter per stream, so a stream's
  records are sorted by `start` and the still-stored ones are a contiguous run
  found by binary search rather than a mask built by a scan;
- episodes long enough to draw a window from are recorded a second time in an
  index of their own, decided once at the ending, so eligibility is a range and
  a count rather than a predicate over records;
- the minibatch is drawn from `[0, N)` by Floyd's algorithm, which is uniform
  over subsets of size `B` without replacement in `B` draws;
- `retained` is read off the boundary — logical time is contiguous, so a
  stream's stored episodes tile the interval from the oldest one's start to
  where its open episode began — rather than summed over records.

Nothing about the draw's distribution changes; that is the subject of
`tests/unit/buffers/test_episode_window_sampling.py`, which compares the new
draws against the Gumbel top-k sampler they replace and against the uniform
distribution over subsets that both are supposed to be.

## Measurements

AMD Ryzen AI 9 365, JAX 0.10.0 on CPU, Python 3.12.12, one stream, episodes of
100 transitions, TBPTT window 10, minibatch 8. 2000 transitions per pass,
every pass inside a `lax.scan` because that is where the learner runs it —
timing a jitted `add` one call at a time measures XLA copying the ring out to a
fresh buffer, an O(capacity) cost the training loop does not pay.

`warmup` is `add` per transition. `steady state` is `add`, `can_sample`, and a
draw under the same `lax.cond` the learner reads `can_sample` through, on a
buffer past its own wrap. `scored` and `ranked` are the two selections timed
on their own: `scored` is the Gumbel-and-`top_k` this replaces, `ranked` is
Floyd's algorithm.

| capacity | warmup us/step | steady state us/step | scored us/step | ranked us/step |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 0.04 | 29.52 | 140.73 | 11.96 |
| 65536 | 0.07 | 33.17 | 618.94 | 8.31 |
| 400000 | 0.04 | 42.49 | 3005.10 | 8.56 |

| capacity | warmup compile s | steady state compile s | scored compile s | ranked compile s |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 0.06 | 0.85 | 0.50 | 0.35 |
| 65536 | 0.06 | 0.82 | 1.46 | 0.37 |
| 400000 | 0.07 | 0.88 | 6.26 | 0.44 |

Reading it:

- **The selection was the whole of the growth, and it was linear.** `scored`
  rises 141 → 619 → 3005 µs as the capacity rises 8192 → 65536 → 400000; the
  ratios are 4.4x and 4.9x against capacity ratios of 8x and 6.1x, which is
  linear growth in the index length over a fixed floor. `ranked` sits at
  8–12 µs at every capacity, which is the cost of a scan of eight steps and
  not a reading of the buffer at all.
- **At the sweep's capacity, selection alone was ~3 ms per transition.** A
  100k-transition segment is about 300 seconds of Gumbel and `top_k` — before
  the network, the environment, or the optimiser — and the 5M run is fifty of
  those. That is the observation the issue opens with.
- **The replay path no longer has a term in the capacity.** 29.5 → 33.2 →
  42.5 µs per transition across a 49x range: 1.4x, against the 21x the
  selection alone rose over the same range. What is left of the slope is not
  index work — every index read is `log2` of the records or smaller — but the
  gather of `B x window` transitions out of a ring that is 49x larger and
  therefore 49x colder in cache. It is bounded by memory locality rather than
  by a loop over the buffer, and the jaxpr assertion in
  `tests/unit/buffers/test_episode_window_sampling.py` says the same thing
  exactly: no value the draw produces grows with the capacity at all.
- **Warmup was never the problem and still is not.** Storing a transition is
  an in-place write to the ring and an occasional scatter into the index:
  0.04–0.07 µs per transition at every capacity.
- **Compilation stops growing too.** `scored` compiles in 0.50 s at 8192 and
  6.26 s at 400000; the buffer's whole pass stays at 0.82–0.88 s. This is the
  part that lands before the first step rather than on it, and it is what a
  run that appears to hang before its first evaluation is partly waiting on.

The `scored` column is if anything generous to what it measures: it is the
selection only, and the sampler it stands for also built its eligibility mask
over the same index, as did every `can_sample` on every warmup step.

## What this does not say

- It is a CPU reading on one machine. The shape of the claim — one term linear
  in capacity, removed — is a property of the graph and is also asserted
  against the jaxpr in
  `tests/unit/buffers/test_episode_window_sampling.py::test_a_draw_costs_the_minibatch_and_not_the_buffer`,
  which needs no clock. The absolute microseconds are this machine's.
- The tables measure replay, not DRQN. What the learner's step costs with them
  in it is the last section, which is also where the second term linear in the
  capacity -- outside the sampler, and found only by measuring the whole step
  -- is reported and dealt with.
- **The R1.1 benchmark has not been rerun.** The issue asks for the 100k
  five-seed `StatelessCartPoleEasy` DRQN run *after this lands*; it is an AWS
  Batch job and is not reachable from a development checkout. Until it has
  run, the claim here is about the sampler's cost and not about the
  benchmark's throughput.

## End to end, and the second term this found

The tables above are replay on its own. `docs/drqn-throughput.py` assembles the
`drqn` entry and times its `train`, so what it reports is the whole step —
network, environment and solver included — and a difference between two
checkouts is a difference in the run. LSTM-32, TBPTT-10, minibatch 8, one
environment, `gymnax::CartPole-v1` at `episode_length: 200`; the sweep's
`StatelessCartPoleEasy` needs the `popjym` extra, which is not installed here,
and differs in the observation the network reads rather than in what a step
costs. 3000 steps a pass, the second pass timed, two readings of each.

Microseconds per environment transition:

| capacity | `origin/main` | sampler only | sampler and order |
| ---: | ---: | ---: | ---: |
| 8192 | 788, 745 | 557, 698 | 529, 563 |
| 400000 | 6833, 6522 | 3395, 3258 | 618, 622 |

At the sweep's capacity that is **6.7 ms a step down to 0.62 ms**, about 150
steps a second up to about 1600, and — the part that matters more than the
factor — a step that costs what it costs at 8192: 1.1x for 49x the replay.

The middle column is why there are two commits here rather than one. Replacing
the sampler halved the step and left it seven times more expensive at 400k than
at 8192, on a sampler these tables show is flat. The rest was not in the
buffer's functions but in how the learner's step read them:

```python
core_state, update = self._maybe_update(update_key, ..., state.buffer_state)
buffer_state = self.buffer.add(state.buffer_state, transition)
```

The update must read replay as it stood *before* this transition — a learner
that stores whole episodes may not draw the episode it has just finished — and
reading it off the pre-add state keeps that state live across the add, so the
new transition cannot be written into the ring in place. XLA copies the whole
ring instead, once per step, which is 16 MB a step at 400k. Order alone, same
buffer, same transitions (`SAMPLING_ORDER=1`):

| capacity | add, then read us/step | read, then add us/step |
| ---: | ---: | ---: |
| 8192 | 14.1 | 25.7 |
| 400000 | 39.0 | 315.4 |

What makes this cheap to fix is the index the first commit introduced. "Replay
as of before this transition" used to be a property of the arrays and is now a
property of four counters, so the step adds first and reads `as_before`: the
rings as they are now, under `committed`, `drawable`, `open_start` and
`written` as they were. The view is exact rather than close —

- the records an add writes sit one position past the ones a pre-add search
  covers, so nothing reads them;
- the record it overwrites is the one slot per stream the index holds back for
  exactly this (`first_stored_fn`), which is affordable because replay stores
  at most `committed_capacity` episodes and the ring is `max_episode_length`
  longer than that;
- the transition it writes lands past anything a pre-add window can reach,
  which is what the transition ring's own slack was already for.

`tests/unit/buffers/test_episode_window_sampling.py` holds it on all three
counts, comparing the view against the state it stands for — same key, same
windows, same warmup, same answer to whether a draw is possible — including on
a buffer whose index ring has wrapped so that the add really does overwrite the
oldest record a search could reach. The identity of the arrays is asserted too,
because a copy would compare equal to what it copies and a copy per step is the
entire cost this exists to avoid.
