# Algorithm and Component Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the algorithm side into components that satisfy the declaration contract, and make `stream_ac` run from the target experiment file. This is spec phase 3, which merges what were phases 3 and 5.

**Architecture:** A component is a frozen dataclass of declarations plus the code it configures, and it knows neither what else it sits beside nor what its caller calls it. A network is a sequence of them rather than three named slots. The optimiser is two axes, a bound and a base, each an instance per role. The normaliser is a running estimator and a discounted trace, each an instance per stream. The kernel holds those instances instead of flat fields, so nothing has to be reconciled on the way in.

**Tech Stack:** Python 3.12, JAX, Flax, optax, dataclasses, uv.

## Global Constraints

- Spec `2026-07-30-configuration-surface-design.md` is the authority. Read §4 and §5 before writing code.
- **TDD.** Write the failing test, run it, see it red, then implement and see it green. Two runs, never one.
- **Tests take their parameters through the phase 2 loader.** Declare a tree, `expand` it, hand the flat manifest to the entry. A hand-built parameter dictionary bypasses the loader and so never finds out whether a declaration and its reading agree.
- **`stream_ac` only.** `upstream_stream_ac`, `rtrrl` and `rtrrl_aaai` keep their `SPACE` and stay red. Do not migrate them, and do not weaken a shared module to keep them importable.
- Every component must pass `memo/tests/test_component_contract.py` as it stands. That file asserts contract properties and names no field, so a new component joins the tuple at the top and nothing else changes.
- Do not write explanations into code or configuration files.
- Tests run locally in WSL with virtualenvs and caches outside the repository. After changing `training-sdk`, reinstall it into memo's environment or memo keeps importing the old copy.
- Recorded scores stop being comparable: parameter counts and the placement of nonlinearities both change. `test_hopper_reproduction.py` asserts against a recorded run and has to be rethought, not repaired.
- Stage explicit paths; never `git add -A`.

---

## File Structure

- `memo/memorax/environments/brax.py`: the wrapper carries out brax's `truncation` and the observation from before the auto-reset, and takes `episode_length` from the environment section instead of the literal 1000.
- `memo/memorax/rl/td.py`: `td0` takes `terminal` and `gamma` rather than a discount its caller already multiplied.
- `memo/memorax/rl/updates.py`: `make_bounded_rule(*, bound, base)` over component instances. `make_obgd_rule`, its `rule` string and `BOUNDED_RULES` are deleted.
- `memo/memorax/rl/normalization.py`: a running estimator that centres or does not, and a discounted trace. Neither names an observation or a reward.
- `memo/memorax/networks/`: components compose into a sequence. `sequence.py` holds the composition, `components.py` the leaves, `backbones.py` what each branch contributes. `Network` and `FeatureExtractor` move to `memorax/algorithms/slots.py`; `torso.py` and `blocks/stack.py` are deleted.
- `memo/memorax/algorithms/stream_ac.py`: `StreamACConfig` holds component instances.
- `memo/entries/stream_ac.py`: declares structures, reads components with `read_branch`, composes. No translation table, no reconciliation.

---

### Task 1: The Kernel Takes Optimiser Components

**Interfaces:**
- Consumes: `ObBound`, `AdaptiveObBound`, `Sgd`, `Adam` from `memorax/rl/updates.py`.
- Produces: `make_bounded_rule(*, bound, base)`; `StreamACConfig` with `actor_bound`, `actor_base`, `critic_bound`, `critic_base`.

- [x] **Step 1: Write the failing tests**

The rule takes the components and the names the configuration surface uses. `bound=None`
is the unbounded path. Asserting on `BOUNDED_RULES` or on a string named `obgd` means the
old surface survived and the test is wrong.

Two roles asking for different bounds is ordinary, not an error: build a config with an
`ObBound` on one role and an `AdaptiveObBound` on the other and assert it constructs.

- [x] **Step 2: Implement and delete the old path**

`make_bounded_rule` replaces `make_obgd_rule`. `BOUNDED_RULES` goes, and with it the
entry's `_RULES` table, `_optimizer` and `_rate`.

- [x] **Step 3: Green, and the golden failures are still exactly five**

---

### Task 2: Terminal and Done Are Two Signals

**Interfaces:**
- Consumes: brax's `truncation`, and the state the auto-reset wrapper replaces.
- Produces: a wrapper that emits both endings and the true next observation; `td0(*, reward, value, next_value, terminal, gamma)`.

- [x] **Step 1: Write the failing tests**

Step an environment to its step limit and assert the wrapper reports the ending as a
truncation, not a termination, and hands back the observation from before the reset.
Step it into a fall and assert the opposite. Then assert the TD error bootstraps at a
truncation and does not at a termination -- with the same reward on both, since the
environment returns one either way.

Asserting on a single `done` means the two are still conflated and the test is wrong.

- [x] **Step 2: Implement**

`BraxGymnaxWrapper.step` returns an empty `info` today and throws both away.
`bootstrap_discount = gamma * (1 - next_done)` in the kernel becomes the TD component's
own `gamma * (1 - terminal)`; the carry and the traces keep resetting on `done`, since
both endings end an episode.

Gamma moves to the algorithm's declarations rather than a component's: the bootstrap and
the discounted reward trace both read it and have to agree, and this design has no way to
say that between two components.

`episode_length` comes from the environment section. It defines the task -- the same
policy's return under a limit of 500 and of 1000 is not the same number -- so a literal
in a wrapper is the wrong place for it.

- [x] **Step 3: Green**

**What landed, beyond what was planned.** Two things followed from the split and
are recorded here because they are not in the steps above.

Restarting moved out of the environment entirely. The wrapper first grew brax's
auto-reset so the observation the episode ended in could survive it; then the
reset moved to the top of the acting step, where the carry and the traces already
restart on the same flag, and the wrapper shrank back below where it started. An
episode now begins from a freshly drawn initial state rather than the one stored
at the first reset, which had collapsed hopper's initial distribution to one point
per stream for a whole run.

The metrics containers were regrouped by what produced them -- interaction,
forward, update -- because terminal was the second field the training and
evaluation containers had to carry twice, and absence in the old shape could not
distinguish "no update ran" from "this container lacks the field".

---

### Task 3: The Normaliser Stops Naming Its Streams

**Interfaces:**
- Produces: a running estimator parameterised by `center`, and a discounted trace carrying `gamma` and `reset_on_done`.

- [x] **Step 1: Write the failing tests**

One estimator, instantiated twice. Centred, it subtracts the mean; uncentred it only
scales. Fed a discounted trace it is the reward path; fed values directly it is the
observation path. No method and no field may contain the word observation or reward.

- [x] **Step 2: Implement**

`_update_observation`/`_normalize_observation` and `_update_reward`/`_scale_reward`
collapse to one pair. `step`'s two near-identical blocks go with them.

- [x] **Step 3: Green**

---

### Task 4: A Network Is a Sequence

**Interfaces:**
- Produces: a sequence whose step is `(carries, x) -> (carries, y)`; `FFN`, `LayerNorm`, and the activations as components.

**The shape, worked out against the code it replaces.** Written down here because
the analysis is most of the work and the wiring is the rest.

`Network.__call__` takes `observation, done, action, reward, initial_carry` and hands
every one of them to all three slots, so a slot that wants none of them still has to
accept them, and a slot that wants `done` gets it whether or not it is recurrent. The
sequence's step is `(carries, x) -> (carries, y)` and nothing else crosses it. What a
component needs beyond `x` it declares, and the sequence supplies only what it declared.

**The carry is a list, one entry per component.** A stateless component returns the entry
it was given; a recurrent one returns a new one. That is what makes "a stateless
component ignores the carry" checkable rather than a convention.

**The credit wraps the recurrent component, not the sequence.** `make_credit(kind, core)`
takes the thing whose Jacobian the sensitivity is carried through, and today the kernel
hands it `network.torso`. Under a sequence it is handed the one recurrent entry, which
the sequence has to be able to name. A sequence with two recurrent components is refused
here rather than silently credited wrongly -- exact RTRL across two of them needs a dense
cross-layer sensitivity, which the spec defers.

**`meta_rl` moves out.** Concatenating the previous action and reward with the
observation is input composition; it happens in the kernel, where those values already
are, and the sequence sees one vector.

**Blast radius.** `memorax/networks/{network,feature_extractor,torso}.py`; `_forward` in
both kernels; the credit call in both; the `network()` builders in three entries;
`test_blocks`, `test_paper_parity`, `test_lru_parity`, `test_stream_ac_golden` and
`test_algorithms`, all of which build a network by naming the three slots.

- [x] **Step 1: Write the failing tests**

Composition is by list, not by three slots. A stateless component ignores the carry; a
recurrent one is the only thing that contributes to it. `done` reaches the recurrent
component because it declares that input, not because the network pushes it at every
stage. Nothing carries `action`, `reward` or an embedding dictionary through. A sequence
with two recurrent components is refused.

`memo/tests/test_sequence.py`, fourteen tests, red at collection before the modules
existed.

- [x] **Step 2: Implement, and delete what named its call sites**

`Network`, `FeatureExtractor`, `TORSOS` and `make_torso` are deleted. `meta_rl`'s
concatenation of the previous action and reward is input composition and moves outside
the network.

- [x] **Step 3: Green**

**What landed, beyond what was planned.** Four things, recorded here because they
are decisions rather than steps.

*A component declares recurrence and what it reads.* `recurrent` and `reads` are
class attributes, not fields, so a component says whether it carries anything and
what it needs beyond `x`; the sequence supplies only what was declared, which is
how `done` reaches the recurrence and reaches nothing else. Detection had to be a
declaration -- what the tests check is the behaviour it licenses, that every
entry but one comes back the way it went in.

*The three slots are gone, and what only they could reach went with them.*
They were briefly kept beside the kernels that still speak them; that was
overruled, since those kernels are getting rewritten anyway. `Network`,
`FeatureExtractor`, `torso.py` and `blocks/stack.py` are all deleted. What that
costs, written down so it is not rediscovered:

- `entries/upstream_stream_ac.py` and `entries/rtrrl.py` no longer import. They
  were already refused by `discover()` for declaring `SPACE`, so the catalog
  test's one red is unchanged; the failure is now at import instead.
- `test_upstream_stream_ac.py` joins the excluded set: it drives upstream's
  kernel end to end and there is no network for it to drive.
- `test_blocks.py` keeps every block comparison -- upstream is built with
  `None` networks, because those blocks are arithmetic on quantities already
  computed and none of them reaches one. Two comparisons are gone: the
  truncated gradient against upstream's, which was the seam the rest of the
  file leaves open, and the same-seed same-start claim. Both need upstream's
  forward. `test_exact_credit_is_not_the_truncated_one` survives on a state
  built by our own `init`.
- `test_algorithms.py` loses RTRRL's two programs and its two gate ablations.
  RTRRL routes its three-domain gradient by slot name -- `RECURRENT_DOMAINS` is
  `("feature_extractor", "torso")` -- so it cannot take a sequence without
  being rewritten. Restore them with that rewrite.

*One seed no longer buys both kernels the same start.* Flax draws a parameter
from the path of the module holding it, and a position in a sequence is not
spelled the way a named slot is. The composition is identical -- the truncated
gradient is still upstream's leaf for leaf, through a rename in the test -- but
the draw is not, so `stream_ac` and `upstream_stream_ac` can no longer be
compared at a single seed. Task 5 ends that comparison anyway by putting each
backbone back the way its source has it. The test now asserts what survived and
says in its name that the rest did not.

*Per-part gradient norms are per place, not per component.* `PARTS` was three
slot names; a sequence has as many parts as it has components and the count
changes with the backbone, while `METRICS` is a module constant the catalog
reads. So `Sequence.split` groups a tree into `before`, `recurrence` and
`after` -- which is the distinction `subtree_norms` was split for in the first
place -- and a declared metric name stays a name whichever backbone runs.

---

### Task 5: The Three Backbones Match Their Sources

**Interfaces:**
- Produces: `rtu`, `lru` and `mlp` branches whose sequences are their published ones.

- [ ] **Step 1: Write the failing parity tests**

Each branch is driven against its source rather than asserted to look like it:
`rtu` against `arXiv 2605.24709`'s Masked MuJoCo settings, `lru` against RTRRL AAAI's
`OnlineLRULayer` path including the SiLU after the readout, `mlp` against
streaming-drl's two `Linear → LayerNorm → LeakyReLU` blocks. `test_paper_parity.py`
covers the optimiser and the normalisation today and nothing about architecture.

- [ ] **Step 2: Implement**

`stream_ac`'s backbone offers `rtu` and `mlp`; `lru` belongs to the rtrrl line.
`RTUCell` and `LRUCell` lose `activation_fn`: RTU's nonlinearity is part of its
recurrence and LRU has none.

- [ ] **Step 3: Green**

---

### Task 6: `stream_ac` Runs From the Target Experiment File

- [ ] **Step 1: Write the failing test**

Load `experiments/streamac template.yaml` through the control plane, resolve it against
`stream_ac.PARAMETERS`, sample a trial, and run the entry on that manifest for a few
steps. That is the whole loader and the whole entry in one assertion.

- [ ] **Step 2: Reconcile the template with what the entry declares**

- [ ] **Step 3: Green**

---

## Open, decide before Task 6

The template scores on `eval/episode/return_per_step`, which phase 4 introduces. Either
phase 4 lands first, or Task 5 scores on a metric this entry already reports and the
template is adjusted when phase 4 arrives. Nothing else in the template depends on
phase 4.

`evaluation.num_envs` still has no consumer: `drive()` evaluates on the training stream
count. Spec §8 has it open, and running from the target file is the first thing that
makes it visible.

## Self-Review

**Spec coverage.** Covers §4's component rules, the three backbone sources, and §5's
bound and base decomposition. Does not cover the metric surface, which is phase 4, nor
multiple recurrent layers, which the spec defers because exact RTRL across two of them
needs a dense cross-layer sensitivity.

**Scope.** One entry. Every task names what it deletes, so the old path cannot survive
beside the new one.
