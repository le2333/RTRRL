# R2D2 Full BPTT and TBPTT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the retired R2D2 implementation with a platform-native recurrent discrete Q-learning algorithm that reuses the pure LRU/RTU forward components and supports canonical R2D2 TBPTT plus a full-episode BPTT branch.

**Architecture:** `R2D2` owns environment interaction, prioritized replay, update cadence, and train/eval scans; `Core` owns online/target parameters and one sequence update; `QFunction` owns recurrent input encoding, the selected pure LRU/RTU backbone, and linear/dueling Q readout. The two learning branches share sequence alignment, Double-Q n-step targets, masked loss, target updates, and priority calculation, but construct recurrent gradients separately.

**Tech Stack:** Python 3.12, JAX, Flax, Optax, Flashbax, Gymnax spaces, Pytest, PyYAML, the existing Memorax assembly/runtime/catalog contracts.

**Design:** `docs/superpowers/specs/2026-08-14-r2d2-full-tbptt-design.md`

## Global Constraints

- Work on the development checkout. Set `UV_PROJECT_ENVIRONMENT` and `UV_CACHE_DIR` to directories outside the repository before running tests.
- Do not modify `../memorax-upstream`, StreamAC, RTRRL, recurrent differentiation implementations, LRU/RTU cell mathematics, or their registered online-differentiation choices.
- Do not import Acme, Reverb, TensorFlow, RLax, or another training framework. Reference behavior is translated into small local pure functions and tested numerically.
- Do not use `BACKBONE_FAMILY`, `RecurrentDifferentiation`, or `TruncatedBPTT`. R2D2 uses `backbone()` and ordinary Flax reverse-mode differentiation over replay sequences.
- Do not add broad exception handling, fallback sampling, silent numerical recovery, or compatibility aliases. Invalid states remain visible.
- Do not call `lox` or any logger from R2D2. The algorithm returns readings; Runtime and the entry-owned reporter perform logging.
- The public action space is discrete. Assembly obtains `action_dim` from `context.action_space.n`; a non-discrete space therefore fails at the graph boundary.
- One vectorized environment interaction adds `num_envs` records. Once replay is sampleable, that interaction performs one learner update. `target.update_period` counts learner updates, not environment transitions.
- `done` resets the environment and recurrence. `terminal` alone removes bootstrap value, so time-limit truncation remains bootstrap-eligible.
- A raw replay window contains `T` executed transitions. Before it reaches `Core`, R2D2 converts it to a learner sequence containing `T + 1` chronological recurrent inputs plus one recorded bootstrap input per transition. The extra bootstrap inputs equal the shifted chronological inputs during ordinary continuation and preserve the pre-reset terminal observation at a time-limit truncation. This keeps the design's `T + 1` alignment without bootstrapping from a reset observation.
- Every mathematical function that is intended to reproduce Acme/SEED behavior gets a hand-calculated numerical test before it is used by `Core`.
- Each task ends with its focused tests and a commit. Do not combine tasks merely because they edit the same algorithm file.

---

### Task 1: Freeze transformed Double-Q sequence mathematics

**Files:**
- Create: `memo/tests/unit/algorithms/r2d2/test_r2d2_math.py`
- Modify: `memo/memorax/algorithms/r2d2.py`

**Interfaces:**

```python
signed_hyperbolic(x, epsilon=1e-3) -> Array
signed_parabolic(x, epsilon=1e-3) -> Array
double_q_n_step_targets(
    rewards,
    terminals,
    online_q,
    target_q,
    valid,
    *,
    gamma,
    n_step,
    transform,
    inverse_transform,
) -> Array
masked_sequence_loss(td_error, valid, importance_weights) -> Array
sequence_priorities(td_error, valid, *, max_weight) -> Array
```

- [ ] **Step 1: Write the transform round-trip and monotonicity tests**

Create tests over `[-100.0, -2.0, 0.0, 3.0, 100.0]` asserting:

```python
transformed = signed_hyperbolic(values)
np.testing.assert_allclose(signed_parabolic(transformed), values, rtol=2e-5)
assert np.all(np.diff(np.asarray(transformed)) > 0)
```

- [ ] **Step 2: Write a hand-calculated Double-Q n-step test**

Use one batch, three valid transitions, `gamma=0.5`, and `n_step=2`. Choose online next-Q rows whose argmax actions are `[1, 0, 1]` and target rows where those selected values differ from `max(target_q)`. Assert the targets are:

```python
expected_t0 = reward[0] + 0.5 * reward[1] + 0.25 * target_next[1]
expected_t1 = reward[1] + 0.5 * reward[2] + 0.25 * target_next[2]
expected_t2 = reward[2] + 0.5 * target_next[2]
```

The final target is deliberately shorter than `n_step`; it proves the last `n_step - 1` valid starts are retained.

- [ ] **Step 3: Write terminal-versus-truncation tests**

For identical rewards and Q arrays, compare `terminals=[False, True, False]` with `terminals=[False, False, False]`. Assert a true terminal removes every bootstrap and later reward reached through it, while a separate `done` value is not accepted by this function and cannot affect the result.

- [ ] **Step 4: Write masked loss and max/mean priority tests**

For `td_error=[[1.0, -3.0, 100.0], [2.0, 4.0, 6.0]]`, `valid=[[1, 1, 0], [1, 1, 1]]`, `importance=[0.5, 1.0]`, and `max_weight=0.75`, assert:

```python
per_sequence_half_mse = [0.5 * (1.0**2 + 3.0**2) / 2, 0.5 * (2.0**2 + 4.0**2 + 6.0**2) / 3]
expected_loss = np.mean(np.array(per_sequence_half_mse) * [0.5, 1.0])
expected_priority = [0.75 * 3.0 + 0.25 * 2.0, 0.75 * 6.0 + 0.25 * 4.0]
```

The masked `100.0` must affect neither value.

- [ ] **Step 5: Run the tests and observe the legacy implementation fail**

Run from the repository root:

```powershell
uv run --project memo pytest memo/tests/unit/algorithms/r2d2/test_r2d2_math.py -v
```

Expected: import failures for the new pure-function names and failure of the old fixed-horizon target at the tail.

- [ ] **Step 6: Replace only the legacy math section**

Implement signed-hyperbolic transform and its analytic inverse. Implement `double_q_n_step_targets` with a static `jax.lax.fori_loop` over `n_step`: accumulate rewards while both the start and traversed step are valid and no prior terminal has occurred; select the bootstrap action from `online_q[:, 1:]`; read its value from `target_q[:, 1:]`; inverse-transform bootstrap values before return accumulation; transform the completed target. Stop gradients on the returned target.

Implement masked reduction using `sum(masked) / maximum(sum(mask), 1)` per sequence. Do not add a fallback for an all-invalid learner sample; replay validity prevents one from being produced.

When R2D2 writes the returned sequence priorities back to replay, add the fixed constant `1e-6`. Do not add that epsilon inside `sequence_priorities`, so the reported mathematical priority remains the exact max/mean value tested above.

- [ ] **Step 7: Run focused tests**

```powershell
uv run --project memo pytest memo/tests/unit/algorithms/r2d2/test_r2d2_math.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit the reference math**

```powershell
git add memo/memorax/algorithms/r2d2.py memo/tests/unit/algorithms/r2d2/test_r2d2_math.py
git commit -m "test: fix r2d2 sequence mathematics"
```

---

### Task 2: Make prioritized replay express valid recurrent windows

**Files:**
- Create: `memo/tests/unit/buffers/test_prioritised_episode_buffer.py`
- Create: `memo/tests/unit/algorithms/r2d2/test_r2d2_replay.py`
- Modify: `memo/memorax/buffers/prioritised_episode_buffer.py`
- Modify: `memo/memorax/algorithms/r2d2.py`

**Interfaces:**

```python
@struct.dataclass(frozen=True)
class ReplayTransition:
    observation: Any
    previous_action: Any
    previous_reward: Any
    episode_start: Any
    action: Any
    reward: Any
    next_observation: Any
    done: Any
    terminal: Any
    actor_recurrence: Any

@struct.dataclass(frozen=True)
class RecurrentInputs:
    observation: Any
    previous_action: Any
    previous_reward: Any
    episode_start: Any

@struct.dataclass(frozen=True)
class LearnerSequence:
    inputs: RecurrentInputs       # every leaf begins [batch, T + 1]
    bootstrap_inputs: RecurrentInputs  # every leaf begins [batch, T]
    actions: Any                  # [batch, T]
    rewards: Any                  # [batch, T]
    dones: Any                    # [batch, T]
    terminals: Any                # [batch, T]
    valid: Any                    # [batch, T]
    initial_recurrence: Any
    probabilities: Any
    indices: Any
    buffer_size: Any

learner_sequence(sample, *, transition_count, full_episode) -> LearnerSequence
completed_episode_starts(experience, *, transition_count) -> Array
tbptt_starts(experience, *, burn_in_length) -> Array
```

- [ ] **Step 1: Test that buffer readiness means a real eligible start exists**

Construct a prioritized buffer with `sample_sequence_length=4`, minimum length four, and a `get_start_flags` returning all false. Add four records and assert `buffer.can_sample(state)` is false. Repeat with one true eligible start carrying positive priority and assert it is true.

- [ ] **Step 2: Test that sampling never silently returns position zero**

With eligible starts at two nonzero positions and distinct stored observations, draw 32 samples and assert every sampled first observation belongs to those two starts. Remove the current `_fallback_uniform` expectation entirely.

- [ ] **Step 3: Test TBPTT learner-sequence construction**

Create a raw window of four transitions. Call `learner_sequence(sample, transition_count=4, full_episode=False)` and assert it has five chronological recurrent inputs; the first four come from each transition's current fields; the fifth uses the fourth transition's `next_observation`, executed action, and received reward. Because that observation is still the pre-reset successor, its episode-start flag is false even when the fourth transition ended. Assert `bootstrap_inputs[t]` is built from transition `t`'s recorded next observation/action/reward with a false episode-start flag. Assert actions/rewards/dones/terminals retain only the four executed transitions, `buffer_size` is preserved, and `initial_recurrence` is the first stored actor recurrence.

- [ ] **Step 4: Test time-limit alignment explicitly**

Put a `done=True, terminal=False` transition in the middle of the window with `next_observation=[9, 9]`; make the following record's current reset observation `[-9, -9]` and `episode_start=True`. Assert the chronological next input is the reset record while that transition's `bootstrap_inputs` contains `[9, 9]` and `episode_start=False`. The Core test in Task 4 must prove the truncated transition bootstraps from this recorded input, not from the shifted reset input.

- [ ] **Step 5: Test full-episode start and padding masks**

For an episode of three transitions padded to `transition_count=5`, assert only the position with `episode_start=True` and a `done` within the next five records is eligible. Assert `learner_sequence(sample, transition_count=5, full_episode=True)` returns `valid=[1, 1, 1, 0, 0]` and uses no stored actor recurrence. A partial episode with no `done` in the window must produce no eligible start even after the buffer reaches `minimum_size`.

For TBPTT, assert `tbptt_starts` accepts arbitrary positions with at least `burn_in_length + 1` transitions before the next `done`, and rejects positions whose episode ends during burn-in. This guarantees every sampled TBPTT window contains at least one valid learner step after the burn-in slice.

- [ ] **Step 6: Run the tests and observe failures**

```powershell
uv run --project memo pytest memo/tests/unit/buffers/test_prioritised_episode_buffer.py memo/tests/unit/algorithms/r2d2/test_r2d2_replay.py -v
```

Expected: current `can_sample` reports only length readiness, current sampling has a position-zero fallback, and the R2D2 replay dataclasses/conversion do not exist.

- [ ] **Step 7: Implement eligibility-aware prioritized replay**

In `make_prioritised_episode_buffer`, replace Flashbax's length-only `can_sample` partial with a local function that combines:

```python
length_ready = can_sample(state, min_length_time_axis=min_length_time_axis)
start_flags = get_start_flags(state.experience)
positions = _valid_start_mask(state, sample_sequence_length)[None, :]
priorities = _get_priorities_for_positions(
    state.sum_tree_state,
    add_batch_size,
    max_length_time_axis,
)
eligible = start_flags & positions & (priorities > 0)
return length_ready & jnp.any(eligible)
```

In `prioritised_episode_sample`, sample directly from normalized masked priorities. Delete `_fallback_uniform` and its `jax.lax.cond`; the caller's readiness branch is the sole sampling guard.

- [ ] **Step 8: Implement R2D2 replay records and conversion**

Add the three dataclasses beside R2D2 state declarations. `completed_episode_starts` must combine `episode_start` with a forward rolling `done` check of exactly `transition_count` records. `tbptt_starts` must reject windows whose first ending occurs at or before the last burn-in record. `learner_sequence` must build the final chronological input from the last included transition's next fields, build every transition's recorded bootstrap input, compute a prefix-valid mask through the first `done`, preserve sampled probability/index/current buffer size, and select stored recurrence only for TBPTT.

- [ ] **Step 9: Run focused tests**

```powershell
uv run --project memo pytest memo/tests/unit/buffers/test_prioritised_episode_buffer.py memo/tests/unit/algorithms/r2d2/test_r2d2_replay.py -v
```

Expected: all tests pass.

- [ ] **Step 10: Commit replay semantics**

```powershell
git add memo/memorax/buffers/prioritised_episode_buffer.py memo/memorax/algorithms/r2d2.py memo/tests/unit/buffers/test_prioritised_episode_buffer.py memo/tests/unit/algorithms/r2d2/test_r2d2_replay.py
git commit -m "fix: make recurrent replay sample real windows"
```

---

### Task 3: Build the semantic recurrent Q-function

**Files:**
- Create: `memo/tests/unit/algorithms/r2d2/test_r2d2_q_function.py`
- Modify: `memo/memorax/algorithms/r2d2.py`

**Interfaces:**

```text
DuelingQHead(action_dim).__call__(hidden) -> q_values
QFunction.init(keys, timestep) -> (params, recurrence)
QFunction.reset(key, recurrence) -> fresh_recurrence
QFunction.apply(params, length_one_timestep, recurrence)
    -> (advanced_recurrence, q_values)
QFunction.unroll(params, timesteps, recurrence)
    -> (final_recurrence, q_values)
QFunction._unroll_with_recurrences(params, timesteps, recurrence)
    -> (final_recurrence, q_values, post_input_recurrences)
```

- [ ] **Step 1: Write input-encoding tests**

For two actions, assert the encoder concatenates observation, one-hot previous action, previous reward, and episode-start flag in that order. Assert the one-hot width is exactly `action_dim` and output dtype is floating point.

- [ ] **Step 2: Write linear and dueling readout tests**

Initialize each head on a `[batch, time, hidden]` tensor and assert output shape `[batch, time, action_dim]`. For the dueling head, inspect the returned values numerically by applying the module and assert its output equals:

```python
value + advantage - advantage.mean(axis=-1, keepdims=True)
```

- [ ] **Step 3: Write the public QFunction contract tests**

Parameterize over `lru` and `rtu`. Build the sequence as:

```text
encoded input -> FFN(feature_dim) -> LayerNorm -> Tanh
              -> backbone(kind, features=feature_dim, hidden_dim=hidden_dim)
              -> selected Q head
```

Assert `init` returns parameters and recurrence, `apply` returns one new recurrence and `[batch, 1, action_dim]`, and `apply` equals `unroll` on the same length-one input and recurrence. Assert `_unroll_with_recurrences` returns the same final recurrence/Q values as `unroll` and one post-input recurrence per time position.

- [ ] **Step 4: Write episode-boundary recurrence tests**

Run two observations sequentially, then rerun the second with `episode_start=True`. Assert the resulting Q values equal those obtained from a fresh recurrence on the second input. This establishes reset-before-consume semantics for both LRU and RTU.

- [ ] **Step 5: Run the tests and observe missing interfaces**

```powershell
uv run --project memo pytest memo/tests/unit/algorithms/r2d2/test_r2d2_q_function.py -v
```

- [ ] **Step 6: Implement the private Q graph**

Keep the input encoder, `DuelingQHead`, and QFunction in `r2d2.py`. Build recurrent components with `backbone()` only. Wrap linear `DiscreteQNetwork` or the dueling module after the recurrence. Implement `_unroll_with_recurrences` as a `jax.lax.scan` of the same length-one Flax application used by `apply`; `unroll` drops the collected carry trajectory and returns only final recurrence/Q values. `reset` returns `network.initialize_carry(key, input_shape)` with the same batch width and does not modify parameters.

- [ ] **Step 7: Run focused tests**

```powershell
uv run --project memo pytest memo/tests/unit/algorithms/r2d2/test_r2d2_q_function.py -v
```

- [ ] **Step 8: Commit the Q-function**

```powershell
git add memo/memorax/algorithms/r2d2.py memo/tests/unit/algorithms/r2d2/test_r2d2_q_function.py
git commit -m "feat: add the r2d2 recurrent q graph"
```

---

### Task 4: Implement canonical R2D2 TBPTT in Core

**Files:**
- Create: `memo/tests/unit/algorithms/r2d2/test_r2d2_tbptt.py`
- Create: `memo/tests/unit/algorithms/r2d2/test_r2d2_core_contract.py`
- Modify: `memo/memorax/algorithms/r2d2.py`

**Interfaces:**

```python
@struct.dataclass(frozen=True)
class CoreState:
    update_step: Any
    recurrence: Any
    params: Any
    target_params: Any
    optimizer_state: Any

@struct.dataclass(frozen=True)
class ForwardMetrics:
    selected_q: Any
    epsilon: Any

@struct.dataclass(frozen=True)
class UpdateMetrics:
    applied: Any
    loss: Any
    td_error: Any
    q_value: Any
    gradient_norm: Any
    importance_weight: Any
    priority: Any
```

```text
Core.init(keys, timestep) -> CoreState
Core.reset(key, state) -> CoreState
Core.act(key, state, timestep, epsilon=epsilon)
    -> (recurrence, action, ForwardMetrics)
Core.update_parameters(key, state, sample, step=step)
    -> (CoreState, UpdateMetrics, priorities)
```

- [ ] **Step 1: Test burn-in forward equality**

Using a small deterministic recurrent function, compare a complete forward unroll with a burn-in prefix followed by a learning suffix. Assert warmed recurrence and suffix Q values are numerically equal to the corresponding complete-unroll values.

- [ ] **Step 2: Test burn-in gradient truncation**

Differentiate the suffix loss with respect to inputs and parameters. Assert burn-in input gradients are zero after `stop_gradient(warmed_recurrence)`, suffix input gradients are nonzero, and recurrent parameter gradients remain nonzero because the same parameters are reused in the suffix.

- [ ] **Step 3: Test Acme/SEED sequence alignment through Core**

Supply a fake QFunction whose online and target unrolls return identifiable `[T + 1, action]` values. Assert Core uses online `q[:, :-1]` at executed replay actions, online `q[:, 1:]` for selector actions, and target `q[:, 1:]` for bootstrap evaluation during ordinary continuation. Add one `done=True, terminal=False` transition: assert Core applies its recorded `bootstrap_inputs` once from that transition's post-input online/target recurrence and substitutes those Q values for the shifted reset-observation Q values. A true terminal needs no substitution because its bootstrap is masked. Assert Core never initializes an independent next-observation unroll from zero.

- [ ] **Step 4: Test importance weighting and target update period**

Use two sequences with different probabilities and losses. Assert normalized importance weights alter their contribution exactly once at the sequence level. With `target.update_period=2`, assert target parameters remain unchanged after update one and copy online parameters after update two.

- [ ] **Step 5: Test Core boundaries**

Assert `Core.act` changes recurrence but leaves params, target params, optimizer state, and update step byte-for-byte tree-equal. Assert `Core.update_parameters` changes online params and optimizer state, increments `update_step` by one, returns one priority per sampled sequence, and has no environment or replay-state argument to mutate.

- [ ] **Step 6: Run the tests and observe failures**

```powershell
uv run --project memo pytest memo/tests/unit/algorithms/r2d2/test_r2d2_tbptt.py memo/tests/unit/algorithms/r2d2/test_r2d2_core_contract.py -v
```

- [ ] **Step 7: Implement focused TBPTT operations**

Add private functions or focused Core methods with these roles and no generic strategy classes:

```python
_burn_in(params, target_params, inputs, initial_recurrence)
_aligned_unroll(params, target_params, sample, recurrence, target_recurrence)
_loss_and_readings(params, target_params, sample, importance_weights)
_apply_optimizer(state, grads)
_update_target(params, target_params, next_update_step)
```

Burn in online and target networks over `burn_in_length`, stop both warmed carries, then unroll exactly `unroll_length + 1` recurrent inputs while retaining the post-input recurrence trajectory. Ordinary bootstrap Q values are the shifted unroll. At `done & ~terminal`, apply the recorded bootstrap input from that transition's retained recurrence and substitute its Q values before Double-Q selection/evaluation. Use the shared Task 1 functions for targets, loss, and priorities. Compute beta from the configured constant `replay.importance_sampling_exponent`, using the sampled probabilities and current replay size supplied in `LearnerSequence`.

- [ ] **Step 8: Implement Core state and public methods**

`Core.init` initializes online params, copies them to target params, initializes Optax state, and retains the acting recurrence. `Core.act` uses epsilon-greedy selection with independent random-action and epsilon keys. `Core.update_parameters` calls `jax.value_and_grad`, applies Optax updates, updates the target on learner-update count, and returns scalar summary readings plus per-sequence priorities.

- [ ] **Step 9: Run focused tests**

```powershell
uv run --project memo pytest memo/tests/unit/algorithms/r2d2/test_r2d2_math.py memo/tests/unit/algorithms/r2d2/test_r2d2_q_function.py memo/tests/unit/algorithms/r2d2/test_r2d2_tbptt.py memo/tests/unit/algorithms/r2d2/test_r2d2_core_contract.py -v
```

- [ ] **Step 10: Commit TBPTT Core**

```powershell
git add memo/memorax/algorithms/r2d2.py memo/tests/unit/algorithms/r2d2/test_r2d2_tbptt.py memo/tests/unit/algorithms/r2d2/test_r2d2_core_contract.py
git commit -m "feat: implement r2d2 tbptt updates"
```

---

### Task 5: Add the full-episode BPTT branch

**Files:**
- Create: `memo/tests/unit/algorithms/r2d2/test_r2d2_full_bptt.py`
- Modify: `memo/memorax/algorithms/r2d2.py`

**Interfaces:**

```python
Core._full_bptt_loss(params, target_params, sample, importance_weights)
    -> tuple[Array, UpdateReadings]
Core._tbptt_loss(params, target_params, sample, importance_weights)
    -> tuple[Array, UpdateReadings]
```

- [ ] **Step 1: Write an explicit full-unroll gradient oracle**

Define a tiny scalar recurrence in the test:

```python
def recurrence(weight, inputs):
    def step(hidden, value):
        hidden = jnp.tanh(weight * hidden + value)
        return hidden, hidden
    return jax.lax.scan(step, 0.0, inputs)[1]
```

Build the direct masked TD loss over a completed episode and compare `jax.grad` of that expression with `Core._full_bptt_loss` under an injected test QFunction. Assert numerical equality, not merely finiteness.

- [ ] **Step 2: Prove padding contributes nothing**

Change padded observations, actions, rewards, and target Q values by large constants while keeping `valid` fixed. Assert loss, priority, and parameter gradient do not change.

- [ ] **Step 3: Prove Full BPTT starts from fresh recurrence**

Put a nonzero stored actor recurrence in the sample and assert Full BPTT results are unchanged. Change the QFunction's initialized recurrence and assert results do change. This prevents accidental reuse of the TBPTT actor state.

- [ ] **Step 4: Prove the two branches meet at the limiting case**

For a complete episode, set TBPTT burn-in to zero and unroll length to the complete valid episode. Assert Full BPTT and TBPTT produce equal Q outputs, loss, priorities, and parameter gradients.

- [ ] **Step 5: Test all backbone/mode combinations complete an update**

Parameterize `backbone in {lru, rtu}` and `learning in {tbptt, full_bptt}`. Initialize a small real QFunction and one valid sample, call `Core.update_parameters`, and assert finite loss/gradient norm and a finite nonzero gradient in recurrent parameters.

- [ ] **Step 6: Run the tests and observe the missing branch**

```powershell
uv run --project memo pytest memo/tests/unit/algorithms/r2d2/test_r2d2_full_bptt.py -v
```

- [ ] **Step 7: Implement Full BPTT as a distinct gradient construction**

Initialize online and target recurrence through `QFunction.reset`; unroll the entire `episode_length + 1` input sequence; do not stop the online recurrence within the valid episode; use the shared alignment, target, masked loss, and priority functions. Select `_full_bptt_loss` or `_tbptt_loss` with the graph's static `learning.kind`, not with a runtime array branch and not through a generic learner class.

- [ ] **Step 8: Run focused gradient tests**

```powershell
uv run --project memo pytest memo/tests/unit/algorithms/r2d2/test_r2d2_full_bptt.py memo/tests/unit/algorithms/r2d2/test_r2d2_tbptt.py -v
```

- [ ] **Step 9: Commit Full BPTT**

```powershell
git add memo/memorax/algorithms/r2d2.py memo/tests/unit/algorithms/r2d2/test_r2d2_full_bptt.py
git commit -m "feat: add full episode bptt to r2d2"
```

---

### Task 6: Connect R2D2 interaction, replay, metrics, and Program

**Files:**
- Create: `memo/tests/unit/algorithms/r2d2/test_r2d2_algorithm.py`
- Create: `memo/tests/integration/algorithms/test_r2d2_smoke.py`
- Modify: `memo/memorax/algorithms/r2d2.py`

**State:**

```python
@struct.dataclass(frozen=True)
class R2D2State:
    step: Any
    timestep: Timestep
    episode_start: Any
    env_state: Any
    buffer_state: Any
    core: CoreState
```

- [ ] **Step 1: Test one interaction record**

Use `TinyDiscreteEnv` with one stream. After one `train_step`, assert replay stores the pre-action observation, previous action/reward, episode-start flag, executed action, environment reward, next observation, done, terminal, and the pre-action actor recurrence. Assert `state.step == 1`.

- [ ] **Step 2: Test update cadence and no-update metrics**

Before replay is sampleable, assert every training step returns `UpdateMetrics` with `applied=False` and zero scalar values in its remaining fields, and leaves params/optimizer/update count unchanged. On the first sampleable step, assert exactly one update runs, `applied=True`, update count increments once, and returned priorities are written back to the sampled replay indices.

- [ ] **Step 3: Test episode reset and truncation behavior**

Run through `TinyDiscreteEnv`'s horizon. Assert the next acting input has `episode_start=True`, previous action and reward are cleared by `EnvironmentStreams.persisted`, and recurrence resets before consuming the reset observation. Add a test adapter returning `done=True` and `terminal=False`; assert replay preserves that distinction.

- [ ] **Step 4: Test evaluation isolation**

Initialize, collect enough replay for one update, then evaluate. Assert evaluation uses a fresh environment and recurrence plus `evaluation_epsilon`, returns `StepMetrics.update is None`, and leaves the original training state's params, target params, optimizer state, replay state, step, and update step unchanged.

- [ ] **Step 5: Test Program-shaped scans**

Assert:

```python
state = algorithm.init(key)
trained, train_metrics = algorithm.train(key, state, num_steps=4)
eval_metrics = algorithm.evaluate(key, trained, num_steps=3)
assert train_metrics.interaction.reward.shape == (4, 1)
assert eval_metrics.interaction.reward.shape == (3, 1)
```

Use one stream so scan length equals environment steps. Add a two-stream assertion that scan time is `num_steps // num_envs`.

- [ ] **Step 6: Run the tests and observe the legacy contract fail**

```powershell
uv run --project memo pytest memo/tests/unit/algorithms/r2d2/test_r2d2_algorithm.py memo/tests/integration/algorithms/test_r2d2_smoke.py -v
```

- [ ] **Step 7: Implement top-level R2D2 flow**

Use `EnvironmentStreams` for reset/step/persisted transitions. `init` opens streams, creates the blank timestep with `episode_start=True` for every stream, initializes Core and replay. `train_step` resets streams selected by the previous timestep's `done`, consumes the retained `episode_start` in the recurrent input/replay record, calls `Core.act`, steps the environment, then sets the next state's episode-start flags to that transition's `done`. It adds one replay record per stream and uses `jax.lax.cond(buffer.can_sample(buffer_state), update_branch, no_update_branch, operand)` to perform zero or one update. Replay sampling and `set_priorities` remain here.

The epsilon schedule is a clipped linear interpolation from `epsilon_start` to `epsilon_end` over environment-step count. `evaluate_step` calls Core with `evaluation_epsilon`, steps no replay and performs no update. `train` and `evaluate` use `jax.lax.scan` exactly as current online algorithms do.

Delete the legacy `lox` calls, flat state, policy helper methods, warmup method, independent-next-sequence target unroll, old fixed-horizon return function, and state-returning evaluation path as this closed flow replaces them. Do not keep two R2D2 execution paths in the file.

- [ ] **Step 8: Declare observations and readings**

Add `Reports`, `REPORTS`, `TRAINING_METRICS`, `METRICS`, `RECORD`, and `OBSERVATIONS` beside the algorithm, following the current readings contract. Map interaction reward/done/terminal for episode cutting. Gate optional series for selected Q, epsilon, loss, TD error, Q value, gradient norm, importance weight, and priority. Training steps with no update emit `applied=False`; evaluation carries no update object.

- [ ] **Step 9: Run algorithm tests**

```powershell
uv run --project memo pytest memo/tests/unit/algorithms/r2d2 memo/tests/integration/algorithms/test_r2d2_smoke.py -v
```

- [ ] **Step 10: Commit the closed algorithm**

```powershell
git add memo/memorax/algorithms/r2d2.py memo/tests/unit/algorithms/r2d2/test_r2d2_algorithm.py memo/tests/integration/algorithms/test_r2d2_smoke.py
git commit -m "feat: close r2d2 over the memorax runtime"
```

---

### Task 7: Register the graph, entry, catalog, and experiment contract

**Files:**
- Create: `memo/entries/r2d2.py`
- Create: `experiments/r2d2 template.yaml`
- Create: `memo/tests/unit/algorithms/r2d2/test_r2d2_assembly.py`
- Create: `memo/tests/unit/environments/test_gymnax.py`
- Create: `memo/tests/integration/test_r2d2_round_trip.py`
- Modify: `memo/tests/support/builders.py`
- Modify: `memo/memorax/algorithms/r2d2.py`
- Modify: `memo/memorax/environments/gymnax.py`
- Modify: `memo/memorax/algorithms/__init__.py`
- Modify: `memo/memorax/__init__.py`
- Modify: `.github/workflows/memo-ci.yml`

**Parameter tree:**

```yaml
backbone:
  kind: [lru]
  lru:
    feature_dim: [128]
    hidden_dim: [128]
head:
  kind: [dueling]
learning:
  kind: [tbptt]
  tbptt:
    burn_in_length: [40]
    unroll_length: [80]
optimizer:
  kind: [adam]
  adam:
    lr: [0.001]
    b1: [0.9]
    b2: [0.999]
    eps: [1.0e-8]
replay:
  capacity: [100000]
  minimum_size: [50000]
  batch_size: [64]
  priority_exponent: [0.9]
  importance_sampling_exponent: [0.6]
  max_priority_weight: [0.9]
target:
  update_period: [2500]
returns:
  n_step: [5]
  value_transform:
    kind: [signed_hyperbolic]
gamma: [0.997]
exploration:
  epsilon_start: [0.4]
  epsilon_end: [0.01]
  epsilon_decay_steps: [1000000]
  evaluation_epsilon: [0.001]
```

The `rtu` branch declares the same two dimensions. `learning.full_bptt` has no child parameters. `returns.value_transform.identity` has no child parameters. Structure choices are pinned to one value in the experiment template.

- [ ] **Step 1: Write parameter and assembly contract tests**

Assert the flattened tree contains no `differentiation` path. Assert it contains both backbone branches, both head choices, both learning choices, and both value transforms. Expand one configuration for each learning mode, assemble against `TinyDiscreteEnv`, and assert the selected QFunction/backbone/head and Core learning kind match the manifest.

- [ ] **Step 2: Test the entry is composition-only**

Copy the narrow shape of `entries/rtrrl.py`: re-export `PARAMETERS` and `METRICS`, implement only `build_request`, `runtime_config`, `run`, and `main`, and assert it contains no Q, replay, target, loss, or gradient symbol.

- [ ] **Step 3: Test and align the Gymnax deployment adapter**

Assembly always supplies `observed`, `backend`, and `episode_length`, but the current Gymnax adapter forwards all three to `gymnax.make`. Write a test that calls the repository's environment factory for `gymnax::CartPole-v1` with those three deployment fields, asserts construction succeeds, asserts the action space is `Discrete`, and asserts the returned parameters carry the requested `max_steps_in_episode`.

Update `memorax/environments/gymnax.py` so its signature consumes `observed`, `backend`, and `episode_length`; discard only `backend`; apply `SelectObservationWrapper` when `observed` is not `None`; and replace `env_params.max_steps_in_episode` with the declared episode length. Forward only actual Gymnax constructor options to `gymnax.make`. This is deployment adaptation, not R2D2 logic.

- [ ] **Step 4: Write the R2D2 experiment template**

Use `entry: r2d2`, `environment.id: gymnax::CartPole-v1`, one stream, `backend: null`, `observed: null`, `episode_length: 32`, `total_steps: 128`, `epoch_steps: 64`, and `evaluation.steps: 32`. Pin `lru` with feature/hidden widths 32, `dueling`, `tbptt` with burn-in 4 and unroll 8, Adam at `1e-3`, replay capacity 1024/minimum 32/batch 4, target period 4, three-step identity-transform returns, gamma 0.99, and epsilon start/end/decay/evaluation values `0.2/0.05/128/0.0`. Configure one HPO round with one trial and score `eval/episode/return_per_step`. Include the existing logging, storage, compute, and contract-version-8 fields; the test replaces external endpoints with local fixtures rather than weakening the file's shape.

- [ ] **Step 5: Write the catalog/control-plane/assembly round trip**

Following the existing StreamAC round-trip stages, assert:

1. `build_catalog()` discovers `r2d2` automatically;
2. every experiment pin is declared by `entries.r2d2.PARAMETERS`;
3. every generated manifest contains exactly the selected branches;
4. `RunSpec` accepts it;
5. assembly builds and trains the tiny discrete graph; and
6. the configured score is in `entries.r2d2.METRICS`.

- [ ] **Step 6: Run the new platform tests and observe failures**

```powershell
uv run --project memo pytest memo/tests/unit/environments/test_gymnax.py memo/tests/unit/algorithms/r2d2/test_r2d2_assembly.py memo/tests/integration/test_r2d2_round_trip.py -v
```

- [ ] **Step 7: Implement parameter declarations and graph assembly**

Define R2D2-private dataclasses and `ComponentFamily` values in `r2d2.py` for backbone, Q head, learning mode, and value transform. Reuse `BASE_FAMILY.restricted("adam")` and `base_transform` for Adam. `R2D2.graph(parameters, components, context)` reads only scalar relations it owns, builds private selections through `ComponentBuilder`, creates the prioritized replay buffer with the selected mode's window/start predicate, and returns the closed R2D2 graph.

Set `R2D2.observations = OBSERVATIONS` so generic assembly can close the `BuiltAlgorithm`.

- [ ] **Step 8: Add entry, exports, builder, and static-check coverage**

Add `memo/entries/r2d2.py` using the exact composition shape tested in Step 2. Add `assemble_r2d2` to `memo/tests/support/builders.py`. Retain public `R2D2`, `R2D2Config`, and `R2D2State` exports but point them only to the new implementation. Add `memorax/algorithms/r2d2.py` to the `CHECKED` list in `memo-ci.yml`.

- [ ] **Step 9: Run platform tests**

```powershell
uv run --project memo pytest memo/tests/unit/environments/test_gymnax.py memo/tests/unit/algorithms/r2d2/test_r2d2_assembly.py memo/tests/integration/test_r2d2_round_trip.py memo/tests/test_template.py memo/tests/test_round_trip.py -v
```

- [ ] **Step 10: Build and inspect the catalog**

```powershell
uv run --project memo python -m deployment.catalog --print-label
```

Assert the printed JSON contains `r2d2`, `rtrrl`, and `stream_ac`, and the R2D2 parameter paths exactly equal `describe(PARAMETERS)` from the entry.

- [ ] **Step 11: Commit platform registration**

```powershell
git add memo/entries/r2d2.py 'experiments/r2d2 template.yaml' memo/tests/unit/environments/test_gymnax.py memo/tests/unit/algorithms/r2d2/test_r2d2_assembly.py memo/tests/integration/test_r2d2_round_trip.py memo/tests/support/builders.py memo/memorax/environments/gymnax.py memo/memorax/algorithms/r2d2.py memo/memorax/algorithms/__init__.py memo/memorax/__init__.py .github/workflows/memo-ci.yml
git commit -m "feat: register r2d2 across the platform"
```

---

### Task 8: Verify numerical, algorithm, and worker-facing behavior

**Files:**
- Modify only if a failing check identifies a concrete defect in a file introduced by Tasks 1-7.

- [ ] **Step 1: Run the complete R2D2 and replay suites**

```powershell
uv run --project memo pytest memo/tests/unit/algorithms/r2d2 memo/tests/unit/buffers/test_prioritised_episode_buffer.py memo/tests/integration/algorithms/test_r2d2_smoke.py memo/tests/integration/test_r2d2_round_trip.py -v
```

- [ ] **Step 2: Run the complete local Memorax suite**

```powershell
uv run --project memo pytest memo/tests -m "not external and not service and not container" -v
```

Do not replace this with only the focused R2D2 tests. The buffer and public exports are shared surfaces.

- [ ] **Step 3: Run service integration tests**

```powershell
uv run --project memo pytest memo/tests -m service -v
```

- [ ] **Step 4: Run all static checks used by Memo CI**

From `memo` with the same `CHECKED` paths as `.github/workflows/memo-ci.yml`:

```powershell
uv run pyright memorax/rl memorax/assembly.py memorax/building.py memorax/algorithms/contract.py memorax/algorithms/r2d2.py memorax/algorithms/rtrrl.py memorax/algorithms/stream_ac.py memorax/algorithms/upstream_stream_ac.py memorax/algorithms/independent_rtrrl.py memorax/utils/trees.py memorax/networks/backbones.py memorax/networks/components.py memorax/networks/initialization.py memorax/networks/initializers/sparse.py memorax/networks/readouts.py memorax/networks/sequence.py memorax/parameters.py memorax/observability memorax/runtime entries runner worker tests
uv run black --check memorax/algorithms/r2d2.py entries/r2d2.py tests/unit/algorithms/r2d2 tests/unit/buffers/test_prioritised_episode_buffer.py tests/integration/algorithms/test_r2d2_smoke.py tests/integration/test_r2d2_round_trip.py
uv run isort --check-only --profile=black memorax/algorithms/r2d2.py entries/r2d2.py tests/unit/algorithms/r2d2 tests/unit/buffers/test_prioritised_episode_buffer.py tests/integration/algorithms/test_r2d2_smoke.py tests/integration/test_r2d2_round_trip.py
uv run ruff check memorax/algorithms/r2d2.py entries/r2d2.py tests/unit/algorithms/r2d2 tests/unit/buffers/test_prioritised_episode_buffer.py tests/integration/algorithms/test_r2d2_smoke.py tests/integration/test_r2d2_round_trip.py
```

- [ ] **Step 5: Audit the implementation against the design**

Run:

```powershell
rg -n "BACKBONE_FAMILY|RecurrentDifferentiation|TruncatedBPTT|import acme|import rlax|except Exception|fallback|TODO|TBD" memo/memorax/algorithms/r2d2.py memo/memorax/buffers/prioritised_episode_buffer.py memo/tests/unit/algorithms/r2d2
```

Expected: no matches in the R2D2 implementation for forbidden dependencies, online differentiation, broad exceptions, fallbacks, or placeholders. A test description may mention a removed fallback; inspect that match rather than deleting the assertion.

Inspect `git diff --check` and `git status --short`. The only untracked paths allowed to remain are the user's pre-existing `.tmp/` and `memo/memorax/loggers/`; do not add, remove, or rewrite them.

- [ ] **Step 6: Commit only evidence-driven fixes**

If Steps 1-5 required code changes, stage exactly those changed files and commit:

```powershell
git commit -m "fix: complete r2d2 platform verification"
```

If no files changed, do not create an empty commit.

- [ ] **Step 7: Push and read CI before claiming completion**

Push the feature branch, then confirm the remote Memo CI ran the pushed commit and all static, algorithm, service, catalog, and existing external-comparison gates are green. A green run for an earlier SHA is not evidence for this implementation.

- [ ] **Step 8: Leave remote container/AWS acceptance for explicit authorization**

Do not dispatch a paid AWS Batch job in this plan. Report the locally and remotely verified commit and request the owner's go-ahead for image build and `dev-*` acceptance. The eventual acceptance must run both `learning.kind=tbptt` and `learning.kind=full_bptt` through `python -m worker` on the built image and report the image digest, catalog hash, run IDs, exit codes, and finite train/eval episode metrics.
