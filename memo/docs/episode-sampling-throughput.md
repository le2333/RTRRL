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
- It measures replay, not DRQN. How much of the 5M run's wall clock this
  returns depends on what the rest of an update costs, which at
  `StatelessCartPoleEasy`'s network sizes is not large enough to hide 2.9 ms.
- **The R1.1 benchmark has not been rerun.** The issue asks for the 100k
  five-seed `StatelessCartPoleEasy` DRQN run *after this lands*; it is an AWS
  Batch job and is not reachable from a development checkout. Until it has
  run, the claim here is about the sampler's cost and not about the
  benchmark's throughput.
