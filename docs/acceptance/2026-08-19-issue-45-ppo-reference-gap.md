# Issue 45: the Brax PPO reference, and the protocol gap that stops it

Issue 45 asks for Brax PPO beside Original RTRRL as a performance-positive
reference, and says how: reuse an already published, validated PPO
container/config **if its evaluation contract can satisfy the global formal
protocol**, do not reimplement PPO in the memo image for uniform packaging, and
if the pinned image's catalog lacks exact fixed-eval support, *track the gap
rather than silently changing evaluation*.

This is that record. Nothing in this branch adds a PPO arm, and the reason is
the second half of that instruction.

## What exists

- `rtrrl/ppo_baseline.py` runs Brax's own PPO, with `rtrrl/config/ppo_*.yml`
  covering masked HalfCheetah across a batch/learning-rate/entropy grid. It is
  the validated reference the issue means, and it is not reimplemented here.
- `rtrrl/catalog.json` is the AAAI image's catalog. It declares **one** entry,
  `rtrrl_aaai`, at deployment contract 5. There is no `ppo` entry in any
  image's catalog, so the current facility has no pinned PPO image to reuse:
  the baseline runs as a script, from the era before the entry/catalog
  boundary existed.
- Its evaluation is Brax's `num_evals`, chosen in `_default_num_evals` as "a
  recording probe, not a training hyperparameter", and reported as
  `eval/episode_reward`.

## The gap

The formal protocol wants a fixed number of *completed episodes* per
checkpoint --- five for Brax --- at boundaries every 10k environment steps,
with the same evaluation seeds archived and paired across methods. Two things
stand between the PPO baseline and that:

1. **Episode counts are not exact.** Brax PPO evaluates for a number of
   *steps*, as does this repository's own runtime (`evaluation.rollout_steps`).
   On a variable-length task that is not a fixed number of episodes, and the
   difference is not noise: an early policy that falls over ends many short
   episodes inside one rollout and a late one may not finish a single episode.
   This is the gap issue #46 names, and #46 owns closing it in the shared
   runtime component. Closing it here, for one arm, would produce a PPO curve
   measured under a protocol nothing else in the study is measured under.

2. **There is no catalog entry to pin.** Reaching the PPO baseline through the
   current control plane means an entry, a catalog and an image digest. That
   work is packaging, and the issue explicitly rules out doing it for
   packaging's sake --- so it is worth doing only once the evaluation contract
   is the formal one, at which point the entry is built against that contract
   rather than adapted to it afterwards.

## What was done instead, and what it is safe to claim

The R2 side of issue 45 --- the collapse characterization of Original RTRRL,
its per-group update telemetry, the collapse detector and the R3.4 checkpoint
forks --- does not depend on the PPO arm. It is delivered in full and reads
only Original RTRRL's own fixed-evaluation curve.

What cannot be claimed without the PPO arm is the *performance-positive*
half: that the collapse is a property of the update rule rather than of the
masked task being unlearnable. That claim stays open, and it is unblocked by
#46 rather than by anything in this branch.

## When this is picked up again

After #46 lands exact episode-count evaluation:

1. Check the pinned PPO image's catalog again before building anything --- the
   issue's order is reuse first.
2. If an entry is still needed, it wraps `rtrrl/ppo_baseline.py`'s call into
   Brax PPO. It does not re-derive PPO.
3. The PPO arm's fixed-evaluation seeds are paired with the RTRRL arms', which
   is what makes the two curves comparable at a checkpoint rather than only in
   aggregate.
