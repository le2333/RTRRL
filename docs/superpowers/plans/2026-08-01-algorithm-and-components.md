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

- `memo/memorax/rl/updates.py`: `make_bounded_rule(*, bound, base)` over component instances. `make_obgd_rule`, its `rule` string and `BOUNDED_RULES` are deleted.
- `memo/memorax/rl/normalization.py`: a running estimator that centres or does not, and a discounted trace. Neither names an observation or a reward.
- `memo/memorax/networks/`: components compose into a sequence. `Network`, `torso.py`'s `TORSOS`/`make_torso`, and `FeatureExtractor` are deleted.
- `memo/memorax/algorithms/stream_ac.py`: `StreamACConfig` holds component instances.
- `memo/entries/stream_ac.py`: declares structures, reads components with `read_branch`, composes. No translation table, no reconciliation.

---

### Task 1: The Kernel Takes Optimiser Components

**Interfaces:**
- Consumes: `ObBound`, `AdaptiveObBound`, `Sgd`, `Adam` from `memorax/rl/updates.py`.
- Produces: `make_bounded_rule(*, bound, base)`; `StreamACConfig` with `actor_bound`, `actor_base`, `critic_bound`, `critic_base`.

- [ ] **Step 1: Write the failing tests**

The rule takes the components and the names the configuration surface uses. `bound=None`
is the unbounded path. Asserting on `BOUNDED_RULES` or on a string named `obgd` means the
old surface survived and the test is wrong.

Two roles asking for different bounds is ordinary, not an error: build a config with an
`ObBound` on one role and an `AdaptiveObBound` on the other and assert it constructs.

- [ ] **Step 2: Implement and delete the old path**

`make_bounded_rule` replaces `make_obgd_rule`. `BOUNDED_RULES` goes, and with it the
entry's `_RULES` table, `_optimizer` and `_rate`.

- [ ] **Step 3: Green, and the golden failures are still exactly five**

---

### Task 2: The Normaliser Stops Naming Its Streams

**Interfaces:**
- Produces: a running estimator parameterised by `center`, and a discounted trace carrying `gamma` and `reset_on_done`.

- [ ] **Step 1: Write the failing tests**

One estimator, instantiated twice. Centred, it subtracts the mean; uncentred it only
scales. Fed a discounted trace it is the reward path; fed values directly it is the
observation path. No method and no field may contain the word observation or reward.

- [ ] **Step 2: Implement**

`_update_observation`/`_normalize_observation` and `_update_reward`/`_scale_reward`
collapse to one pair. `step`'s two near-identical blocks go with them.

- [ ] **Step 3: Green**

---

### Task 3: A Network Is a Sequence

**Interfaces:**
- Produces: a sequence whose step is `(carries, x) -> (carries, y)`; `FFN`, `LayerNorm`, and the activations as components.

- [ ] **Step 1: Write the failing tests**

Composition is by list, not by three slots. A stateless component ignores the carry; a
recurrent one is the only thing that contributes to it. `done` reaches the recurrent
component because it declares that input, not because the network pushes it at every
stage. Nothing carries `action`, `reward` or an embedding dictionary through.

- [ ] **Step 2: Implement, and delete what named its call sites**

`Network`, `FeatureExtractor`, `TORSOS` and `make_torso` are deleted. `meta_rl`'s
concatenation of the previous action and reward is input composition and moves outside
the network.

- [ ] **Step 3: Green**

---

### Task 4: The Three Backbones Match Their Sources

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

### Task 5: `stream_ac` Runs From the Target Experiment File

- [ ] **Step 1: Write the failing test**

Load `experiments/streamac template.yaml` through the control plane, resolve it against
`stream_ac.PARAMETERS`, sample a trial, and run the entry on that manifest for a few
steps. That is the whole loader and the whole entry in one assertion.

- [ ] **Step 2: Reconcile the template with what the entry declares**

- [ ] **Step 3: Green**

---

## Open, decide before Task 5

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
