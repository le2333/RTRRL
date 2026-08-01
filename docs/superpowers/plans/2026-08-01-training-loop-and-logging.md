# Training Loop and Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report on episode boundaries. This is spec phase 4, and it supersedes `2026-08-01-metric-surface.md`, which planned a rename on top of epoch aggregation — an arrangement the spec no longer has.

**Architecture:** A completed episode emits one event carrying that episode's statistics. There is no other reporting occasion. The episode is the only window with training meaning: the environment gives its boundary, where a chunk is a scheduling unit and an epoch is an evaluation interval. Aim takes the statistics; rerun takes the series they were reduced from. Both are keyed by the episode and neither sees a chunk.

**Tech Stack:** Python 3.12, JAX, Aim, rerun, uv.

## Global Constraints

- Spec `2026-07-30-configuration-surface-design.md` §6 is the authority.
- **TDD.** Write the failing test, run it, see it red, then implement and see it green.
- Independent of phases 2 and 3: it touches the loop and the sinks, not the parameter surface. It may run before or after phase 3.
- Do not write explanations into code or configuration files.
- Tests run locally in WSL with virtualenvs and caches outside the repository.
- Stage explicit paths; never `git add -A`.

---

## File Structure

- `memo/runner/loop.py`: `drive` reports when an episode ends rather than when an epoch does. `named_scalars` goes; it flattens the env axis along with time, which is what loses the per-stream distinction the episode gives for free.
- `memo/runner/episodes.py`: `complete_episodes` already cuts a summary into episodes per stream. It becomes the thing that drives reporting rather than a step inside evaluation.
- `training-sdk/src/training_sdk/sinks/aim.py`: `log_episode` stops being a no-op and takes the statistics; `report` and `every_steps` go.
- `training-sdk/src/training_sdk/sinks/rerun.py`: takes the episode's series.
- `training-sdk/src/training_sdk/contract.py`: `LoggingConfig` loses `every_steps`.
- `memo/entries/*.py`: `TRAINING_METRICS` names the fields to reduce over an episode; `METRICS` is derived rather than hand-written.

---

### Task 1: An Episode Emits Its Statistics

**Interfaces:**
- Produces: for each completed episode, `length`, `return`, `return_per_step`, and the variance of its per-step reward.

- [ ] **Step 1: Write the failing tests**

Drive a run whose episodes have known lengths and rewards, and assert one event per
completed episode with those four values. An episode cut off by the step budget emits
nothing.

Assert the events are per stream: two streams whose episodes end at different steps give
two independent series, not one averaged number. A test that asserts a single scalar per
epoch is testing the arrangement being removed.

- [ ] **Step 2: Implement**

`drive` reports on episode completion. `named_scalars` is deleted along with the epoch
report; `jnp.nanmean` over the whole array is what flattened time and stream together.

- [ ] **Step 3: Green**

---

### Task 2: The Algorithm's Diagnostics Reduce Over the Same Window

**Interfaces:**
- Produces: each declared step-level diagnostic reduced over the episode, as a mean and a variance.

- [ ] **Step 1: Write the failing tests**

A diagnostic recorded per step arrives as two numbers per episode. A mean alone cannot
say whether a value sat steady or swung, which is the whole reason for the second.

- [ ] **Step 2: Implement**

The fields a declared metric needs must be recorded, so `record_trajectory` follows from
the declarations rather than being a separate flag that defaults to off and that no entry
sets. Left as it is, `reward` and `done` are never recorded and every new metric reports
nothing at all, silently.

- [ ] **Step 3: Green**

---

### Task 3: Names Say Their Window

**Interfaces:**
- Produces: `<phase>/<window>/<quantity>`, with `episode` the only window.

- [ ] **Step 1: Write the failing tests**

Every reported name has three parts and its middle one is `episode`. A name with `step`
in the middle means something is still claiming a granularity that no longer exists.

`METRICS` is derived from what the entry declares plus what the loop contributes, not
hand-written beside it. Assert they agree by construction: an entry cannot report a name
it has not declared, nor declare one it does not report.

- [ ] **Step 2: Implement, and migrate what names a metric**

Every `score.metric` in `experiments/*.yaml` and in both templates.

- [ ] **Step 3: Green**

---

### Task 4: Aim Takes Statistics, Rerun Takes the Series

**Interfaces:**
- Produces: `log_episode` carrying the statistics to Aim and the series to rerun; `LoggingConfig` without `every_steps`.

- [ ] **Step 1: Write the failing tests**

An episode's statistics reach Aim at the cumulative step where it ended. Its per-step
series reaches rerun. `every_steps` is gone from the contract, and an experiment naming
it is refused.

- [ ] **Step 2: Implement**

`AimSink.log_episode` is a no-op today, which is why episode detail has never reached
Aim. `every_steps` discarded writes of an already-computed mean while its name claimed a
stride it never took.

- [ ] **Step 3: Green**

---

## Open

The AAAI arm's `eval_model` discards episode length, so it cannot report
`return_per_step`. It does not run this loop, so it declares what it can; the five-arm
comparison reads `eval/episode/return`. Recorded rather than resolved.

`evaluation.num_envs` still has no consumer: `drive()` evaluates on the training stream
count. Spec §8 has it open and this plan does not close it.

## Self-Review

**Spec coverage.** Covers §6 entirely: the reporting occasion, the four quantities, the
variance, the three-part names, the split between the two sinks, and the removal of
`every_steps`. Does not cover per-environment reward decomposition, which §6 leaves to
later because it needs metric declarations to vary with the environment.

**Independence.** Touches the loop, the sinks and the contract's logging section. It does
not read the parameter tree, so it neither depends on nor blocks phase 3.
