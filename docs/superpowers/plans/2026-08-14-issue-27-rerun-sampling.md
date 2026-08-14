# Issue 27 Bounded Sampled-Episode Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RTRRL sampled-episode reporting exact across chunk and budget boundaries while keeping scan outputs and random-key arrays independent of reporting-epoch length.

**Architecture:** Algorithms return bounded transition chunks and expose one no-learning behavior-policy interaction. Runtime owns a per-stream episode tracker, routes all complete episodes to scalar reporting, routes only requested complete trajectories to Rerun, and uses the no-learning operation solely to finish a final sampled episode after budget. Entry expands the periodic Rerun configuration into explicit sample points and requests heavy trajectory fields only when Rerun is enabled.

**Tech Stack:** Python 3.11, JAX, Flax, NumPy, pytest, Aim, Rerun SDK, uv.

## Global Constraints

- Do not alter RTRRL objectives, eligibility traces, optimizer groups, update order, defaults, or AAAI parity behavior.
- A sample at a `done` boundary selects the next episode after reset.
- The periodic schedule excludes step zero; 50M with interval 10M means exactly 10M, 20M, 30M, 40M, and 50M.
- Peak reporting memory is bounded by one bounded chunk, stream count, maximum episode length, and active current episodes; it does not grow with `epoch_steps` or total training steps.
- Do not materialize a reporting-epoch-length `StepMetrics` tree or random-key array.
- Do not execute Python callbacks, logger calls, or Rerun I/O from a JAX scan.
- Budget-end completion uses the behavior policy and current environment/recurrent state, but does not update parameters, optimizer state, eligibility traces, normalization statistics, update counters, or the training budget.
- Aim continues to receive evaluation metrics at configured epoch boundaries, including `eval/episode/return`.
- Runs without `logging.rerun` emit no sampled trajectories and no RRD artifacts.
- Add no fallback, compatibility alias, broad exception handler, or silent recovery path.
- Do not run AWS, Docker, paid remote acceptance, or workflow dispatch without a new explicit authorization.
- Keep virtual environments and caches outside the repository. In WSL use `UV_PROJECT_ENVIRONMENT=/tmp/memorax-issue27-venv` and `UV_CACHE_DIR=/tmp/uv-cache-memorax-issue27`.

---

## File structure

- `memo/memorax/runtime/tracker.py`: stateful, algorithm-neutral reconstruction and sampling of per-stream episodes.
- `memo/memorax/runtime/episode.py`: immutable completed-episode and sampled-trajectory values.
- `memo/memorax/runtime/program.py`: closed algorithm operations Runtime may schedule.
- `memo/memorax/runtime/driver.py`: bounded train/eval scheduling and final no-learning continuation.
- `memo/memorax/runtime/rollout.py`: retained stateless field-path helpers and evaluation cutting; no sampling policy.
- `memo/memorax/observability/reporting.py`: scalar/completed-episode versus sampled-trajectory fan-out.
- `memo/memorax/observability/sinks/rerun.py`: RRD serialization only.
- `memo/memorax/assembly.py`: closes each algorithm graph over the complete Program and requested trajectory paths.
- `memo/entries/rtrrl.py`, `memo/entries/stream_ac.py`: deployment projection into build and runtime contracts.
- Runtime unit tests live under `memo/tests/unit/runtime/`; algorithm interaction tests remain under their algorithm unit directories; backend serialization remains under `memo/tests/integration/observability/`.

---

### Task 1: Stateful multi-stream episode tracker

**Files:**
- Create: `memo/memorax/runtime/tracker.py`
- Modify: `memo/memorax/runtime/episode.py`
- Modify: `memo/memorax/runtime/__init__.py`
- Test: `memo/tests/unit/runtime/test_episode_tracker.py`
- Test: `memo/tests/unit/runtime/test_episode.py`

**Interfaces:**
- Consumes: `ObservationSchema` field paths and one bounded summary tree with leading `[time, environment]` axes.
- Produces:

```python
@dataclass(frozen=True)
class SampledTrajectory:
    episode: Episode
    sample_step: int
    post_budget: tuple[bool, ...]

@dataclass(frozen=True)
class TrackingResult:
    completed: tuple[Episode, ...]
    sampled: tuple[SampledTrajectory, ...]

class EpisodeTracker:
    def __init__(
        self,
        *,
        observations: ObservationSchema,
        num_envs: int,
        max_episode_steps: int,
        sample_steps: tuple[int, ...] = (),
        first_number: int = 1,
    ) -> None: ...

    def consume(
        self,
        summary: object,
        *,
        start_env_steps: int,
        post_budget: bool = False,
        report_completed: bool = True,
    ) -> TrackingResult: ...

    @property
    def pending_sample_steps(self) -> tuple[int, ...]: ...

    @property
    def next_number(self) -> int: ...
```

- Invariant: transition at scan row `r`, stream `e` has global position `start_env_steps + r * num_envs + e`; a sample is attached immediately before the transition at the same position.
- Invariant: `consume()` also applies a sample exactly equal to the chunk's ending boundary, so a final-budget sample becomes pending before continuation.

- [x] **Step 1: Write failing value-contract tests**

Add to `test_episode.py`:

```python
def test_sampled_trajectory_marks_every_transition_budget_side():
    episode = make_episode()
    sampled = SampledTrajectory(
        episode=episode,
        sample_step=1,
        post_budget=(False, True),
    )

    assert sampled.sample_step == 1
    assert sampled.post_budget == (False, True)


def test_sampled_trajectory_requires_one_budget_mark_per_transition():
    with pytest.raises(ValueError, match="post_budget"):
        SampledTrajectory(
            episode=make_episode(),
            sample_step=1,
            post_budget=(False,),
        )
```

- [x] **Step 2: Write the failing deterministic tracker test**

Create a `summary()` helper that returns NumPy arrays for observations, actions,
rewards, `done`, terminal, and a `td_error` series. Feed three chunks whose
first stream has two completed episodes in one chunk and whose selected episode
crosses the next boundary:

```python
def test_sampled_episode_survives_chunks_and_done_boundary_selects_next():
    tracker = EpisodeTracker(
        observations=OBSERVATIONS,
        num_envs=2,
        max_episode_steps=6,
        sample_steps=(4, 10),
    )

    first = tracker.consume(first_chunk(), start_env_steps=0)
    second = tracker.consume(second_chunk(), start_env_steps=6)
    third = tracker.consume(third_chunk(), start_env_steps=12)

    sampled = first.sampled + second.sampled + third.sampled
    assert [one.sample_step for one in sampled] == [4, 10]
    assert [one.episode.stream for one in sampled] == [0, 0]
    assert sampled[0].episode.observations == EXPECTED_EPISODE_AT_FOUR
    assert sampled[1].episode.observations == EXPECTED_EPISODE_AFTER_DONE_AT_TEN
    assert all(one.episode.terminals[-1] for one in sampled)
```

The fixture must make step 10 the boundary after a `done` for stream zero. Its
expected value begins with the next reset observation, not the preceding
episode.

- [x] **Step 3: Write failing bounded-slot and multi-episode tests**

```python
def test_one_stream_reuses_its_slot_for_multiple_episodes_in_one_chunk():
    tracker = make_tracker(sample_steps=())
    result = tracker.consume(chunk_with_two_env_zero_endings(), start_env_steps=0)

    assert [episode.stream for episode in result.completed] == [0, 0, 1]
    assert [len(episode.rewards) for episode in result.completed] == [2, 2, 4]


def test_unfinished_episode_is_retained_without_being_reported():
    tracker = make_tracker(sample_steps=(3,))
    first = tracker.consume(prefix_without_done(), start_env_steps=0)
    second = tracker.consume(suffix_with_done(), start_env_steps=4)

    assert first.completed == first.sampled == ()
    assert second.sampled[0].episode.observations == FULL_PREFIX_AND_SUFFIX


def test_episode_longer_than_declared_limit_fails_instead_of_truncating():
    tracker = make_tracker(max_episode_steps=2)
    with pytest.raises(ValueError, match="maximum episode length"):
        tracker.consume(three_steps_without_done(), start_env_steps=0)
```

- [x] **Step 4: Run the focused tests and verify RED**

Run:

```bash
uv run --project memo pytest \
  memo/tests/unit/runtime/test_episode.py \
  memo/tests/unit/runtime/test_episode_tracker.py -q
```

Expected: collection fails because `SampledTrajectory` and `EpisodeTracker` do
not exist.

- [x] **Step 5: Implement the immutable sampled value and tracker**

Implement `SampledTrajectory.__post_init__()` so it freezes the mask and checks
that its length equals `len(episode.rewards)`. In `tracker.py`, extract the
configured paths once per chunk with `runtime.rollout.read`, normalize scalar
series to `[time, environment]`, and process rows chronologically. Each stream
slot owns:

```python
@dataclass
class _OpenEpisode:
    start_env_steps: int
    observations: list[object]
    actions: list[object]
    rewards: list[float]
    terminals: list[bool]
    truncations: list[bool]
    series: dict[str, list[float]]
    post_budget: list[bool]
    samples: list[int]
```

On `done`, construct one `Episode`, construct one `SampledTrajectory` per
attached sample, increment the global episode number, and replace only that
stream's slot. Never retain a finalized slot.

- [x] **Step 6: Run focused and existing rollout tests**

Run:

```bash
uv run --project memo pytest \
  memo/tests/unit/runtime/test_episode.py \
  memo/tests/unit/runtime/test_episode_tracker.py \
  memo/tests/test_rollout.py -q
```

Expected: PASS.

- [x] **Step 7: Commit Task 1**

```bash
git add memo/memorax/runtime memo/tests/unit/runtime
git commit -m "feat: track sampled episodes across chunks"
```

---

### Task 2: No-learning behavior-policy interaction

**Files:**
- Modify: `memo/memorax/runtime/program.py`
- Modify: `memo/memorax/assembly.py`
- Modify: `memo/memorax/algorithms/rtrrl_aaai.py`
- Modify: `memo/memorax/algorithms/stream_ac.py`
- Modify: `memo/tests/support/programs.py`
- Modify: `memo/tests/test_loop.py`
- Test: `memo/tests/unit/algorithms/rtrrl/test_rtrrl_assembly.py`
- Test: `memo/tests/unit/algorithms/stream_ac/test_stream_ac_assembly.py`

**Interfaces:**
- Consumes: a final online algorithm state and one JAX PRNG key.
- Produces: `Program.interact(key, state) -> tuple[state, StepMetrics]` for one vectorized environment interaction with stochastic behavior-policy action and no learning mutation.
- The returned state may change only environment state, timestep/terminal state, normalization output values without changing normalization statistics, and actor/torso recurrent carry.

- [x] **Step 1: Write failing Program closure assertions**

Extend both assembly tests:

```python
def test_program_exposes_no_learning_interaction():
    built = assembled()
    state = built.program.init(jax.random.key(0))

    advanced, metrics = built.program.interact(jax.random.key(1), state)

    assert metrics.interaction.reward.shape == (1,)
    assert int(advanced.update_step) == int(state.update_step)
    assert int(advanced.step) == int(state.step)
```

For StreamAC use its existing update counter field. Assert environment state or
timestep observation changes so the test cannot pass through an identity stub.

- [x] **Step 2: Write failing learning-state preservation assertions**

Add a tree comparison helper in each algorithm test and assert exact equality
for parameters, traces, optimizer/rule state, and normalization statistics:

```python
before = state.core
advanced, _ = built.program.interact(jax.random.key(2), state)

assert_tree_equal(advanced.core.torso.params, before.torso.params)
assert_tree_equal(advanced.core.torso.traces, before.torso.traces)
assert_tree_equal(advanced.core.actor.params, before.actor.params)
assert_tree_equal(advanced.core.critic.params, before.critic.params)
assert_tree_equal(advanced.core.rule, before.rule)
assert_tree_equal(advanced.scales, state.scales)
```

Use the corresponding StreamAC actor/critic/rule field names rather than an
adapter layer.

- [x] **Step 3: Run the focused tests and verify RED**

Run:

```bash
uv run --project memo pytest \
  memo/tests/unit/algorithms/rtrrl/test_rtrrl_assembly.py \
  memo/tests/unit/algorithms/stream_ac/test_stream_ac_assembly.py -q
```

Expected: FAIL because `Program` has no `interact` member.

- [x] **Step 4: Implement each algorithm's semantic interaction**

Add `interact()` to RTRRL and StreamAC. The RTRRL shape is:

```python
def interact(self, key: Any, state: RTRRLState) -> tuple[RTRRLState, StepMetrics]:
    reset_key, action_key, env_key = jax.random.split(key, 3)
    state = self._reset(reset_key, state, update=False)
    observation = state.timestep.obs
    recurrence, action, _ = self.core.act(
        action_key, state.core, state.timestep, deterministic=False
    )
    obs, env_state, environment_reward, done, terminal, info = (
        self.environment.step(env_key, state.env_state, action)
    )
    obs, reward, _ = self.normalization.apply(
        state.scales, obs, environment_reward, done, update=False
    )
    next_timestep = Timestep(obs=obs, action=action, reward=reward, done=done)
    advanced = state.replace(
        timestep=next_timestep,
        terminal=terminal,
        env_state=env_state,
        core=state.core.replace(
            torso=state.core.torso.replace(recurrence=recurrence)
        ),
    )
    return advanced, StepMetrics(
        interaction=self._interaction(
            observation=observation,
            next_observation=next_timestep.obs,
            action=action,
            reward=environment_reward,
            done=done,
            terminal=terminal,
            info=info,
        )
    )
```

Implement the same semantics through StreamAC's own actor recurrence and
`core.sample_action`; do not introduce a cross-algorithm helper that knows their
state layouts.

- [x] **Step 5: Close assembly and test programs over the fourth operation**

Add the required field to `Program`:

```python
@dataclass(frozen=True)
class Program:
    init: Callable[..., Any]
    train: Callable[..., Any]
    evaluate: Callable[..., Any]
    interact: Callable[..., Any]
```

Set `interact=graph.interact` in `assemble()`. Extend `arithmetic_program()` to
return an interaction function and update its sole `Program(...)` test
construction. Do not provide a default or compatibility shim.

- [x] **Step 6: Run focused algorithm and loop tests**

Run:

```bash
uv run --project memo pytest \
  memo/tests/unit/algorithms/rtrrl/test_rtrrl_assembly.py \
  memo/tests/unit/algorithms/stream_ac/test_stream_ac_assembly.py \
  memo/tests/test_loop.py -q
```

Expected: PASS.

- [x] **Step 7: Run RTRRL numerical guard tests**

Run:

```bash
uv run --project memo pytest \
  memo/tests/test_rtrrl.py \
  memo/tests/test_rtrrl_parity.py \
  memo/tests/test_paper_parity.py -q
```

Expected: PASS with the pre-existing third-party warnings only.

- [x] **Step 8: Commit Task 2**

```bash
git add memo/memorax/assembly.py memo/memorax/runtime/program.py \
  memo/memorax/algorithms/rtrrl_aaai.py memo/memorax/algorithms/stream_ac.py \
  memo/tests/support/programs.py memo/tests/test_loop.py \
  memo/tests/unit/algorithms
git commit -m "feat: expose no-learning policy interaction"
```

---

### Task 3: Separate scalar episode and sampled trajectory reporting

**Files:**
- Modify: `memo/memorax/observability/protocols.py`
- Modify: `memo/memorax/observability/reporting.py`
- Modify: `memo/memorax/observability/sinks/rerun.py`
- Modify: `memo/entries/_observability.py`
- Modify: `memo/tests/support/fakes.py`
- Modify: `memo/tests/support/observability.py`
- Test: `memo/tests/unit/observability/test_reporter.py`
- Test: `memo/tests/integration/observability/test_rerun.py`
- Test: `memo/tests/integration/observability/test_entry_composition.py`

**Interfaces:**
- Consumes: `Episode` for scalar reduction and `SampledTrajectory` for trajectory backends.
- Produces:

```python
class TrajectorySink(Protocol):
    def log_trajectory(self, trajectory: SampledTrajectory) -> None: ...
    def close(self) -> None: ...

class Reporter:
    def log_episode(self, episode: Episode) -> None: ...
    def log_trajectory(self, trajectory: SampledTrajectory) -> None: ...
```

- `RerunSink(directory, *, metadata)` has no interval or stream arguments.
- RRD filenames are keyed by requested sample step:
  `train-sample-000010000000.rrd` for sample 10M.

- [x] **Step 1: Write failing Reporter routing tests**

Add a trajectory recorder to the support fakes and verify strict separation:

```python
def test_reporter_reduces_every_episode_but_routes_only_sampled_trajectories():
    scalars = ScalarRecorder()
    trajectories = TrajectoryRecorder()
    reporter = Reporter(scalar_sinks=(scalars,), trajectory_sinks=(trajectories,))
    episode = completed_episode()
    sampled = completed_trajectory(sample_step=8)

    reporter.log_episode(episode)
    reporter.log_trajectory(sampled)

    assert scalars.reports[0][1]["train/episode/return"] == 4.0
    assert trajectories.trajectories == [sampled]
    assert len(scalars.reports) == 1
```

- [x] **Step 2: Replace Rerun sampling tests with serialization tests**

Delete tests of `RerunSink._sampled()` behavior. Add:

```python
def test_rerun_serializes_the_runtime_selected_sample_and_budget_mask(tmp_path):
    sink = RerunSink(tmp_path, metadata=metadata())
    trajectory = completed_trajectory(
        sample_step=10_000_000,
        post_budget=(False, True),
    )

    sink.log_trajectory(trajectory)
    sink.close()

    path = tmp_path / "train-sample-000010000000.rrd"
    summary = RrdReader(path).store().summary()
    assert "/episode/post_budget" in summary
    assert "/episode/rewards" in summary
```

The metadata test must read the RRD and verify `sample_step`, stream, start, and
end values rather than asserting only that a file exists.

- [x] **Step 3: Run focused tests and verify RED**

Run:

```bash
uv run --project memo pytest \
  memo/tests/unit/observability/test_reporter.py \
  memo/tests/integration/observability/test_rerun.py -q
```

Expected: FAIL because the sampled-trajectory protocol and sink method do not
exist.

- [x] **Step 4: Implement the two reporting paths**

Add `TrajectorySink`, store `trajectory_sinks` in `Reporter`, and make
`log_trajectory()` fan out without computing scalar statistics. Refactor
`RerunSink` so `log_trajectory()` unwraps `trajectory.episode`, writes the
sample metadata, writes the boolean post-budget sequence, and performs no
sampling decision.

Update `build_reporter()` to pass Rerun sinks through `trajectory_sinks` and
remove `every_steps` and `num_envs` from the sink constructor. The interval
remains in the run config for Runtime projection in Task 5.

- [x] **Step 5: Update composition fixtures to use a sampled trajectory**

Add to `tests/support/observability.py`:

```python
def completed_trajectory(
    *,
    sample_step: int = 1,
    post_budget: tuple[bool, ...] = (False, False),
) -> SampledTrajectory:
    return SampledTrajectory(
        episode=completed_episode(),
        sample_step=sample_step,
        post_budget=post_budget,
    )
```

Change the entry composition test to call both `reporter.log_episode()` and
`reporter.log_trajectory()`. Assert one scalar record and one RRD.

- [x] **Step 6: Run observability tests**

Run:

```bash
uv run --project memo pytest \
  memo/tests/unit/observability \
  memo/tests/integration/observability/test_entry_composition.py \
  memo/tests/integration/observability/test_rerun.py -q
```

Expected: PASS.

- [x] **Step 7: Commit Task 3**

```bash
git add memo/memorax/observability memo/entries/_observability.py \
  memo/tests/support memo/tests/unit/observability \
  memo/tests/integration/observability
git commit -m "refactor: route sampled trajectories explicitly"
```

---

### Task 4: Bounded Runtime scheduling and final continuation

**Files:**
- Modify: `memo/memorax/runtime/driver.py`
- Modify: `memo/memorax/runtime/rollout.py`
- Modify: `memo/tests/support/fakes.py`
- Modify: `memo/tests/support/programs.py`
- Modify: `memo/tests/test_loop.py`
- Test: `memo/tests/unit/runtime/test_driver_sampling.py`

**Interfaces:**
- Consumes: the Task 1 `EpisodeTracker`, Task 2 `Program.interact`, and Task 3 reporter methods.
- Produces:

```python
@dataclass(frozen=True)
class RuntimeConfig:
    total_steps: int
    epoch_steps: int
    eval_steps: int
    num_envs: int
    seed: int
    max_episode_steps: int
    sample_steps: tuple[int, ...] = ()
```

- Runtime train call size is at most
  `max_episode_steps * num_envs`, additionally clipped to the current epoch.
- Runtime publishes `TrackingResult.completed` through `log_episode()` and
  `TrackingResult.sampled` through `log_trajectory()`.

- [ ] **Step 1: Write the failing memory-shape scheduling test**

Use a fake program whose train function records and rejects large requests:

```python
def test_long_epoch_is_executed_as_bounded_train_calls():
    requested: list[int] = []

    def train(key, state, num_steps):
        del key
        requested.append(num_steps)
        assert num_steps <= 8
        return state + num_steps, chunk_for(num_steps, num_envs=2)

    run_runtime(
        train=train,
        total_steps=80,
        epoch_steps=40,
        num_envs=2,
        max_episode_steps=4,
    )

    assert requested == [8] * 10
```

This is a behavior test. Do not inspect source text or monkeypatch
`jax.random.split`.

- [ ] **Step 2: Write the failing Runtime sampling test**

Build a deterministic two-stream program whose episode crosses two train
calls and whose sample at a done boundary must select the following episode:

```python
def test_runtime_reports_exact_sampled_episode_across_train_calls():
    recorder = EpisodeRecorder()
    runtime = deterministic_runtime(
        total_steps=24,
        epoch_steps=12,
        num_envs=2,
        max_episode_steps=3,
        sample_steps=(6, 12),
    )

    runtime.run(recorder)

    assert [one.sample_step for one in recorder.trajectories] == [6, 12]
    assert recorder.trajectories[1].episode.observations == NEXT_EPISODE_AFTER_DONE
```

- [ ] **Step 3: Write the failing final-budget continuation test**

The fake state's separate `updates` and `interactions` counters make accidental
training visible:

```python
def test_final_sample_is_finished_without_an_update():
    recorder = EpisodeRecorder()
    runtime, trained_states, interaction_states = final_sample_runtime(sample_step=8)

    runtime.run(recorder)

    sampled = recorder.trajectories[-1]
    assert trained_states[-1].updates == 4
    assert [state.updates for state in interaction_states] == [4, 4]
    assert sampled.post_budget == (False, False, True, True)
    assert sampled.episode.terminals[-1]
```

Also assert `interact` was called exactly twice and `train` was never called
after total step 8.

- [ ] **Step 4: Run focused tests and verify RED**

Run:

```bash
uv run --project memo pytest \
  memo/tests/unit/runtime/test_driver_sampling.py \
  memo/tests/test_loop.py -q
```

Expected: FAIL because `RuntimeConfig` has no bound/sample schedule and Runtime
still submits an epoch-sized train call.

- [ ] **Step 5: Implement bounded epoch scheduling**

Replace one train call per epoch with a nested loop:

```python
chunk_limit = config.max_episode_steps * config.num_envs
trained_steps = 0
for epoch_end in epochs:
    while trained_steps < epoch_end:
        chunk_steps = min(chunk_limit, epoch_end - trained_steps)
        key, chunk_key = jax.random.split(key)
        state, chunk = train(chunk_key, state, chunk_steps)
        result = tracker.consume(chunk, start_env_steps=trained_steps)
        self._publish(reporter, result)
        trained_steps += chunk_steps
    self._evaluate_at_boundary(..., done=epoch_end)
```

Preserve the existing evaluation state/date semantics. Create a fresh tracker
for each fresh evaluation rollout, seed evaluation numbering from its previous
value, and publish only its completed episodes.

- [ ] **Step 6: Implement final sampled-episode continuation**

After the last train chunk has applied the exact final boundary sample, copy the
final state into a local continuation variable. While
`tracker.pending_sample_steps` is non-empty, call the jitted `program.interact`
once, add a leading time axis to the returned metrics with `jax.tree.map`, and
consume it with `post_budget=True` and `report_completed=False`.

Limit this loop to `max_episode_steps` vectorized interactions. If a pending
sample remains, raise `ValueError("sampled episode exceeded maximum episode length")`.
Do not assign the continuation state back to the final training state.

- [ ] **Step 7: Remove train sampling from stateless rollout cutting**

Keep `read()` and any evaluation-only stateless helpers in `rollout.py`. Runtime
training must use `EpisodeTracker`; no call to `complete_episodes()` may treat a
train chunk boundary as reset. Update `test_loop.py` expectations so a training
episode spanning chunks is reported once and whole.

- [ ] **Step 8: Run Runtime and rollout tests**

Run:

```bash
uv run --project memo pytest \
  memo/tests/unit/runtime \
  memo/tests/test_loop.py \
  memo/tests/test_rollout.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 4**

```bash
git add memo/memorax/runtime memo/tests/support \
  memo/tests/unit/runtime memo/tests/test_loop.py memo/tests/test_rollout.py
git commit -m "feat: bound runtime scans and complete final samples"
```

---

### Task 5: Entry schedule projection and conditional trajectory fields

**Files:**
- Modify: `memo/memorax/runtime/program.py`
- Modify: `memo/memorax/assembly.py`
- Modify: `memo/memorax/algorithms/rtrrl_aaai.py`
- Modify: `memo/memorax/algorithms/stream_ac.py`
- Modify: `memo/entries/rtrrl.py`
- Modify: `memo/entries/stream_ac.py`
- Test: `memo/tests/unit/algorithms/rtrrl/test_rtrrl_assembly.py`
- Test: `memo/tests/unit/algorithms/stream_ac/test_stream_ac_assembly.py`
- Test: `memo/tests/integration/observability/test_entry_composition.py`
- Test: `infra/tests/test_experiment_hpo.py`

**Interfaces:**
- `ObservationSchema.episode_fields` contains reward, `done`, optional terminal, and configured scalar series paths.
- `ObservationSchema.trajectory_fields` contains non-`None` observation, next-observation, and action paths.
- `BuildRequest.record: frozenset[str]` carries requested heavy trajectory paths into the algorithm graph.
- Entry runtime projection expands `every_steps` into explicit sample points and sets `max_episode_steps` from the environment specification.

- [ ] **Step 1: Write failing observation-field partition tests**

```python
def test_observation_schema_separates_episode_and_trajectory_fields():
    schema = rtrrl.OBSERVATIONS

    assert schema.episode_fields == frozenset(
        (schema.reward, schema.done, schema.terminal, *schema.series)
    )
    assert schema.trajectory_fields == frozenset(
        (schema.observation, schema.next_observation, schema.action)
    )
```

Remove assertions against the old module-level `RECORD`; requested heavy paths
are now a build input, not an algorithm constant.

- [ ] **Step 2: Write failing Entry projection tests**

For each entry, construct one config with
`logging.rerun.every_steps=10`, `total_steps=50`, `num_envs=2`, and environment
episode length 7:

```python
request = entry.build_request(config)
schedule = entry.runtime_config(config)

assert request.record == entry_algorithm.OBSERVATIONS.trajectory_fields
assert schedule.sample_steps == (10, 20, 30, 40, 50)
assert schedule.max_episode_steps == 7
```

Construct a second config with `logging.rerun=None` and assert:

```python
assert entry.build_request(config).record == frozenset()
assert entry.runtime_config(config).sample_steps == ()
```

- [ ] **Step 3: Write the failing graph-recording test**

Assemble RTRRL and StreamAC twice, once with empty `record` and once with the
schema's trajectory fields. Invoke a short train chunk and assert observation,
next observation, and action are `None` in the first output and arrays in the
second. Reward and `done` must be arrays in both.

- [ ] **Step 4: Run focused tests and verify RED**

Run:

```bash
uv run --project memo pytest \
  memo/tests/unit/algorithms/rtrrl/test_rtrrl_assembly.py \
  memo/tests/unit/algorithms/stream_ac/test_stream_ac_assembly.py \
  memo/tests/integration/observability/test_entry_composition.py -q
```

Expected: FAIL because schema partition properties, `BuildRequest.record`, and
runtime sample projection do not exist.

- [ ] **Step 5: Implement schema partition and graph record injection**

Replace `ObservationSchema.required_fields` with explicit properties:

```python
@property
def episode_fields(self) -> frozenset[str]: ...

@property
def trajectory_fields(self) -> frozenset[str]: ...
```

Add `record: frozenset[str] = frozenset()` to `BuildRequest`. Pass it as the
keyword-only `record` argument to each algorithm's `graph()` method, then to the
algorithm constructor. Delete the static `RECORD` constant and do not add an
alias. Algorithm topology, reports, and scalar readings remain unchanged.

- [ ] **Step 6: Implement Entry projection**

In each entry use one local helper with this exact schedule rule:

```python
rerun = config.logging.rerun
sample_steps = (
    ()
    if rerun is None
    else tuple(range(rerun.every_steps, runtime.total_steps + 1, rerun.every_steps))
)
```

Set `BuildRequest.record` to `algorithm.OBSERVATIONS.trajectory_fields` only
when `rerun is not None`. Set `RuntimeConfig.max_episode_steps` from
`config.algorithm.environment.episode_length` and pass `sample_steps`.

- [ ] **Step 7: Preserve Infra's explicit Rerun-off behavior**

Extend `infra/tests/test_experiment_hpo.py` so an experiment with
`enable_rerun: false` produces run configs with no `logging.rerun` member for
every trial. Do not make Runtime infer HPO mode and do not change HPO sampling.

- [ ] **Step 8: Run entry, assembly, and Infra tests**

Run:

```bash
uv run --project memo pytest \
  memo/tests/unit/algorithms/rtrrl/test_rtrrl_assembly.py \
  memo/tests/unit/algorithms/stream_ac/test_stream_ac_assembly.py \
  memo/tests/integration/observability/test_entry_composition.py -q
uv run --project infra pytest infra/tests/test_experiment_hpo.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 5**

```bash
git add memo/memorax/runtime/program.py memo/memorax/assembly.py \
  memo/memorax/algorithms/rtrrl_aaai.py memo/memorax/algorithms/stream_ac.py \
  memo/entries/rtrrl.py memo/entries/stream_ac.py \
  memo/tests/unit/algorithms memo/tests/integration/observability \
  infra/tests/test_experiment_hpo.py
git commit -m "feat: project sampled trajectory requirements"
```

---

### Task 6: End-to-end sampled RTRRL acceptance and regression gate

**Files:**
- Create: `memo/tests/integration/observability/test_rtrrl_sampled_runtime.py`
- Modify: `memo/tests/test_loop.py`
- Modify: `memo/tests/integration/observability/test_rerun.py`
- Modify: `docs/superpowers/specs/2026-08-14-issue-27-rerun-sampling-design.md` only if implementation discovered a contract correction; otherwise leave it unchanged.

**Interfaces:**
- Consumes: the complete Entry → assembly → Runtime → Reporter path.
- Produces: local evidence that a fixed run yields one RRD per requested sample,
  evaluation remains at epoch boundaries, post-budget steps do not add updates,
  and an Rerun-disabled run emits no RRD.

- [ ] **Step 1: Write the failing fixed-run integration test**

Assemble the real RTRRL graph against `TinyContinuousEnv`, then use a real
`Runtime`, `Reporter`, `MetricsSink`, and `RerunSink`. Configure the tiny
environment with horizon 3, one stream, three evaluation steps, and a schedule
scaled from 50M/10M to 50/10:

```python
def test_fixed_run_writes_five_samples_and_five_evaluations(tmp_path):
    runtime = assembled_tiny_rtrrl_runtime(
        total_steps=50,
        epoch_steps=10,
        sample_steps=(10, 20, 30, 40, 50),
    )

    with sampled_reporter(tmp_path) as reporter:
        runtime.run(reporter)

    assert len(list((tmp_path / "rerun").glob("*.rrd"))) == 5
    records = read_metrics(tmp_path / "metrics.jsonl")
    evaluations = [
        record for record in records
        if "eval/episode/return" in record["metrics"]
    ]
    assert [record["step"] for record in evaluations] == [10, 20, 30, 40, 50]
```

The horizon-3 fixture makes the sample at 50 require one post-budget
transition. Read that RRD and assert its post-budget sequence is
`(False, False, True)`.

- [ ] **Step 2: Write the failing Rerun-disabled integration test**

```python
def test_run_without_sample_points_emits_scalars_but_no_rerun(tmp_path):
    runtime = sampled_runtime(sample_steps=())

    with scalar_only_reporter(tmp_path) as reporter:
        runtime.run(reporter)

    assert (tmp_path / "metrics.jsonl").is_file()
    assert not (tmp_path / "rerun").exists()
```

- [ ] **Step 3: Run integration tests and verify their initial result**

Run:

```bash
uv run --project memo pytest \
  memo/tests/integration/observability/test_rtrrl_sampled_runtime.py -q
```

Expected before adding the fixture/composition: FAIL because the acceptance
helpers and complete path do not yet exist. After implementing only the minimum
fixture and composition code, rerun and expect PASS.

- [ ] **Step 4: Run focused Runtime, observability, and entry suites**

Run:

```bash
uv run --project memo pytest \
  memo/tests/unit/runtime \
  memo/tests/unit/observability \
  memo/tests/unit/algorithms/rtrrl \
  memo/tests/integration/observability/test_entry_composition.py \
  memo/tests/integration/observability/test_rerun.py \
  memo/tests/integration/observability/test_rtrrl_sampled_runtime.py \
  memo/tests/test_loop.py memo/tests/test_rollout.py -q
```

Expected: PASS.

- [ ] **Step 5: Run RTRRL numerical and parity suites**

Run:

```bash
uv run --project memo pytest \
  memo/tests/test_rtrrl.py \
  memo/tests/test_rtrrl_parity.py \
  memo/tests/test_paper_parity.py \
  memo/tests/test_recurrent_differentiation.py -q
```

Expected: PASS with no numerical tolerance changes.

- [ ] **Step 6: Run static checks**

Run:

```bash
uv run --project memo ruff check memo
uv run --project infra ruff check infra
```

Expected: PASS.

- [ ] **Step 7: Run the complete local non-external suite**

Run:

```bash
JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --project memo pytest memo/tests \
  -m "not external and not service and not container" -q
uv run --project infra pytest infra/tests -q
```

Expected: PASS. Record the exact test counts and pre-existing warnings in the
task report; do not describe a timed-out or partial run as a pass.

- [ ] **Step 8: Commit Task 6**

```bash
git add memo/tests/integration/observability/test_rtrrl_sampled_runtime.py \
  memo/tests/test_loop.py memo/tests/integration/observability/test_rerun.py \
  docs/superpowers/specs/2026-08-14-issue-27-rerun-sampling-design.md
git commit -m "test: accept bounded sampled RTRRL runs"
```

If the design document did not change, omit it from `git add`.

- [ ] **Step 9: Prepare but do not execute remote acceptance**

Report the local commit SHA and the exact future remote acceptance commands and
configuration values needed for:

- masked Hopper CPU with a long reporting interval;
- a fixed five-sample/five-evaluation run;
- an HPO run with no Rerun artifacts.

Stop before any push, image build, workflow dispatch, AWS Batch submission, or
paid resource mutation and request explicit authorization.
