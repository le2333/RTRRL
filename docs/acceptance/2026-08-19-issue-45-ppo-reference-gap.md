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
with the evaluation seed declared so that two methods are measured on paired
episodes. Issue #46 has landed exactly that, and it landed it **in memo's
runtime**: `evaluation.episodes`, a rollout advanced until every named slot
holds a complete episode, and an evaluation key stream of its own.

The PPO baseline does not run through that runtime. It is
`rtrrl/ppo_baseline.py` calling Brax's own PPO, and it evaluates on Brax's
`num_evals` --- a number of *evaluations*, each a step-bounded rollout. On a
variable-length task that is not a fixed number of episodes, and the
difference is not noise: an early policy that falls over ends many short
episodes inside one rollout, and a late one may not finish a single episode.

So the remaining distance is not the protocol; it is that PPO is on the other
side of the entry boundary from where the protocol now lives:

1. **There is no catalog entry to pin.** `rtrrl/catalog.json` declares one
   entry, `rtrrl_aaai`, at deployment contract 5. Reaching the PPO baseline
   through the current control plane means an entry, a catalog and an image
   digest.

2. **Building one is packaging, which the issue rules out doing for its own
   sake.** It is worth doing now that the evaluation contract is the formal
   one --- an entry built against it rather than adapted to it afterwards ---
   but it is a piece of work with its own shape, and doing it inside this
   issue would have meant a PPO arm measured under whichever protocol was
   convenient at the time.

## What was done instead, and what it is safe to claim

The R2 side of issue 45 --- the collapse characterization of Original RTRRL,
its per-group update telemetry, the collapse detector and the R3.4 checkpoint
forks --- does not depend on the PPO arm. It is delivered in full and reads
only Original RTRRL's own fixed-evaluation curve.

What cannot be claimed without the PPO arm is the *performance-positive*
half: that the collapse is a property of the update rule rather than of the
masked task being unlearnable. That claim stays open. Nothing blocks it any
more --- #46's evaluation is in --- so what it needs is the entry, and the
paragraph below is the order to do it in.

## When this is picked up again

1. Check the pinned PPO image's catalog again before building anything --- the
   issue's order is reuse first, and the catalog is what says whether there is
   anything to reuse.
2. If an entry is still needed, it wraps `rtrrl/ppo_baseline.py`'s call into
   Brax PPO. It does not re-derive PPO.
3. The PPO arm's fixed-evaluation seeds are paired with the RTRRL arms', which
   is what makes the two curves comparable at a checkpoint rather than only in
   aggregate.
