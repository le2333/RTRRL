# Composable Online Recurrent AC Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 JAX-first 通用入口，并先证明它在固定 seed 下逐步数值复现当前
RTRRL+Adam 与 StreamAC-RTU+OBGD；在 parity gate 通过前不实现 RTRRL-OBGD 或迁移
实验基础设施。

**Architecture:** 新代码旁路现有算法落地，保留旧 `RTRRL` 和 `StreamACRtrl` 作为
独立 oracle。共享层只提供不会改变数据流的 pure kernels；`MetaProgram` 与
`StandardProgram` 分别实现具体 step。完成逐叶 golden、短轨迹、evaluation 和 JIT
等价后，旧 builder 才翻译到新 program。

**Tech Stack:** Python 3.11、JAX、Flax、Optax、Distrax、lox、pytest、simple_parsing。

## Global Constraints

- 设计基准：
  `docs/superpowers/specs/2026-07-17-composable-online-recurrent-ac-design.md`。
- 同一 JAX/device/seed 下验证逐叶严格容差等价，不以 tree norm 代替。
- 旧算法在 parity 完成前不得改写；golden 必须从旧实现直接生成。
- `MetaProgram` 与 `StandardProgram` 是两个 concrete kernel，不建立万能 step。
- acting、bootstrap、differentiation forward 必须保持为三个不同数据流。
- optimizer、path label、closure 在 JIT 前构建；不得在 jitted `init()` 修改 Python 对象。
- legacy RTRRL 保留三 trace、emphasis、slow torso、Grouped Adam 和 action 语义。
- legacy StreamAC 保留 exact RTU local Jacobian、fresh trace 和 actor/critic whole-tree
  OBGD。
- legacy evaluation 默认 `fixed_vector_steps`、
  `reset_on_start=true`、`update_during_eval=true`。
- recorder、trajectory、Rerun、S3、plugin registry 和 RTRRL-OBGD 不属于本计划。
- 本计划不授权 Git commit；只有用户显式要求时才提交。

---

## File Map

创建：

```text
memorax/online_ac/
├── __init__.py
├── types.py
├── credit.py
├── td.py
├── traces.py
├── objectives.py
├── targets.py
├── updates.py
├── normalization.py
├── meta.py
├── standard.py
└── build.py

tests/online_ac/
├── conftest.py
├── golden.py
├── test_legacy_characterization.py
├── test_credit.py
├── test_components.py
├── test_meta_parity.py
├── test_standard_parity.py
├── test_evaluation_parity.py
├── test_legacy_builders.py
└── test_jit_contract.py

tests/online_ac/golden/
├── rtrrl_lru.json
├── rtrrl_lru.npz
├── stream_ac_rtu.json
└── stream_ac_rtu.npz
```

parity 通过后修改：

```text
memorax/algorithms/rtrrl.py
memorax/algorithms/stream_ac_rtrl.py
memorax/algorithms/__init__.py
experiments/base/experiment.py
```

不修改：

```text
memorax/networks/sequence_models/lru.py
memorax/networks/sequence_models/rtu.py
memorax/networks/sequence_models/rnn.py
memorax/networks/sequence_models/memoroid.py
memorax/environments/wrappers/normalize_observation.py
memorax/environments/wrappers/normalize_reward.py
```

这些文件是 parity oracle。

---

### Task 1: Freeze Legacy Oracles

**Files:**
- Create: `tests/online_ac/conftest.py`
- Create: `tests/online_ac/golden.py`
- Create: `tests/online_ac/test_legacy_characterization.py`
- Create: `tests/online_ac/golden/rtrrl_lru.{json,npz}`
- Create: `tests/online_ac/golden/stream_ac_rtu.{json,npz}`

**Interfaces:**
- Produces: `TinyContinuousEnv`, `TinyDiscreteEnv`
- Produces: `assert_tree_allclose(actual, expected, *, rtol, atol)`
- Produces: `save_golden(name, tree, metadata)` and `load_golden(name)`
- Produces: immutable legacy one-step and short-trajectory oracle data

- [ ] **Step 1: Add deterministic environments and stable tree paths**

```python
def flatten_with_paths(tree):
    pairs, treedef = jax.tree_util.tree_flatten_with_path(tree)
    return {
        "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path): leaf
        for path, leaf in pairs
    }, treedef

def assert_tree_allclose(actual, expected, *, rtol=1e-6, atol=1e-7):
    actual_leaves = flatten_with_paths(actual)[0]
    expected_leaves = flatten_with_paths(expected)[0]
    assert actual_leaves.keys() == expected_leaves.keys()
    for path in actual_leaves:
        np.testing.assert_allclose(
            np.asarray(actual_leaves[path]),
            np.asarray(expected_leaves[path]),
            rtol=rtol,
            atol=atol,
            err_msg=path,
        )
```

Tiny environments must have observation `(2,)`, continuous action `(2,)` or two discrete
actions, deterministic transitions, and a terminal third transition.

- [ ] **Step 2: Characterize current RTRRL and StreamACRtrl**

Tests must directly call existing `init`, private `_update_step`, `train`, and `evaluate`.
Capture:

```text
params, slow_torso, carry, sensitivity, traces, emphasis,
optimizer/OBGD state, timestep, env_state, counters,
sampled/logprob/env/feedback action, value, next_value, TD,
metric keys and dtypes
```

Cover RTRRL incoming/fresh timing and StreamAC adaptive/non-adaptive OBGD.

- [ ] **Step 3: Generate versioned golden files once**

Manifest fields:

```json
{
  "jax": "...",
  "jaxlib": "...",
  "backend": "cpu",
  "seed": 7,
  "algorithm": "rtrrl_lru",
  "steps": 3,
  "leaf_paths": []
}
```

Generation is an explicit script path in `golden.py`; tests must never rewrite files.

- [ ] **Step 4: Verify the oracle**

Run:

```bash
cd /home/ubuntu/trainer/streaming-rtrrl/memo
JAX_PLATFORM_NAME=cpu uv run pytest \
  tests/online_ac/test_legacy_characterization.py -v
```

Expected: all characterization tests pass and both manifests match the active versions.

---

### Task 2: Define Fixed Program Types and Exact Credit

**Files:**
- Create: `memorax/online_ac/__init__.py`
- Create: `memorax/online_ac/types.py`
- Create: `memorax/online_ac/credit.py`
- Create: `memorax/online_ac/td.py`
- Create: `tests/online_ac/test_credit.py`

**Interfaces:**
- Produces: `AgentProgram(init_fn, train_epoch_fn, evaluate_fn, state_schema, metric_schema)`
- Produces: `ActionDecision`
- Produces: `Transition`
- Produces: `JAXEnvAdapter(reset_fn, step_fn, env_params, build_context)`
- Produces: frozen `MetaProgramConfig` and `StandardProgramConfig`
- Produces: `make_exact_rtrl_credit(core)`
- Produces: `make_td0()`

- [ ] **Step 1: Write failing type and TD tests**

```python
def test_td0_uses_explicit_bootstrap_discount():
    td = make_td0()
    delta = td(
        reward=jnp.array([2.0]),
        value=jnp.array([1.0]),
        next_value=jnp.array([4.0]),
        bootstrap_discount=jnp.array([0.5]),
    )
    np.testing.assert_allclose(delta, [3.0])
```

Also assert that `AgentProgram` is host-only, `JAXEnvAdapter` contains only JAX-callable
reset/step closures plus explicit array parameters, and `ActionDecision` has sampled,
logprob, env, bootstrap-feedback and persisted-feedback actions.

- [ ] **Step 2: Implement frozen host types and pure TD**

`types.py` must not import Aim, experiment config, or algorithm implementations.

- [ ] **Step 3: Write failing exact-credit tests**

Required tests:

```text
test_lru_credit_delegates_to_memoroid_local_jacobian
test_rtu_credit_delegates_to_rnn_local_jacobian
test_phantom_changes_gradient_but_not_forward_value
test_rtu_reset_order_matches_legacy
test_bootstrap_state_can_be_discarded
test_differentiation_forward_replays_pre_acting_state
```

- [ ] **Step 4: Implement exact credit as delegation**

The factory must call:

```python
core.apply(
    {"params": params},
    inputs,
    done,
    carry,
    sensitivity=credit,
    method="local_jacobian",
)
```

Do not reproduce LRU/RTU Jacobian mathematics in `credit.py`.

- [ ] **Step 5: Verify**

```bash
JAX_PLATFORM_NAME=cpu uv run pytest tests/online_ac/test_credit.py -v
```

Expected: exact value and per-leaf gradient comparisons pass.

---

### Task 3: Implement Shared Pure Kernels

**Files:**
- Create: `memorax/online_ac/traces.py`
- Create: `memorax/online_ac/objectives.py`
- Create: `memorax/online_ac/targets.py`
- Create: `memorax/online_ac/updates.py`
- Create: `tests/online_ac/test_components.py`

**Interfaces:**
- Produces: `make_rtrrl_trace(config)`
- Produces: `make_stream_ac_trace(config)`
- Produces: `make_rtrrl_objective(config)`
- Produces: `make_stream_ac_objective(config)`
- Produces: `make_slow_subtree_target(config)`
- Produces: `make_grouped_adam(config, abstract_params)`
- Produces: `make_whole_tree_obgd(config)`

- [ ] **Step 1: Test the two trace equations by hand**

Use two envs, nonzero incoming traces, distinct lambdas, terminal/nonterminal masks.

```python
rtrrl_expected = decay * (1 - terminated_after) * old + emphasis * grad
stream_expected = gamma_lambda * (1 - reset_before) * old + grad
```

Assert RTRRL carried trace is always new while update trace selects incoming/fresh.
Assert StreamAC always uses fresh trace.

- [ ] **Step 2: Implement trace closures**

Closures capture lambdas/timing statically. They receive explicit boundary names, never a
generic `done`.

- [ ] **Step 3: Test objective domain routing**

Cover:

```text
RTRRL:
actor traced = eta_pi * logprob_scale * log_prob
critic traced = value
recurrent traced multiplied by eta_f only during update
entropy and prediction direct, not TD-scaled, not eta_f-scaled

StreamAC:
actor traced = log_prob + entropy_coefficient * sign(stop_gradient(delta)) * entropy
critic traced = value
```

- [ ] **Step 4: Implement objective closures**

Return fixed-domain trees:

```python
ObjectiveDirections(
    traced_by_domain=...,
    direct_by_domain=...,
    metrics=...,
)
```

- [ ] **Step 5: Test target and updates**

Required tests:

```text
test_slow_torso_used_by_all_three_forward_views
test_update_destination_is_fast_params
test_polyak_runs_after_fast_update
test_sensitivity_not_recomputed_after_polyak
test_grouped_adam_is_ascent
test_freeze_gamma_only_freezes_gamma_log
test_obgd_actor_and_critic_are_whole_tree_domains
test_obgd_adaptive_false_still_updates_v
test_obgd_computes_per_env_step_before_env_mean
```

- [ ] **Step 6: Implement target, Grouped Adam, and OBGD**

Use Optax for legacy Adam. Copy current OBGD expression and operation order exactly; do
not “simplify” reductions.

- [ ] **Step 7: Verify**

```bash
JAX_PLATFORM_NAME=cpu uv run pytest tests/online_ac/test_components.py -v
```

Expected: all hand-computed and legacy-comparison tests pass.

---

### Task 4: Implement MetaProgram and Prove RTRRL Parity

**Files:**
- Create: `memorax/online_ac/meta.py`
- Create: `tests/online_ac/test_meta_parity.py`

**Interfaces:**
- Produces: `MetaState`
- Produces: `MetaStepMetrics`
- Produces: `make_meta_program(parts, static_config) -> AgentProgram`

- [ ] **Step 1: Write init parity test**

Compare all leaves against `rtrrl_lru` golden:

```text
params, slow torso, carry, sensitivity, traces, Adam state,
I, timestep, env state, step/update_step, shape and dtype
```

- [ ] **Step 2: Implement MetaState and pure init**

Optimizer already exists in the closure. `init_fn` may create array state but must not
assign to `self` or mutate captured Python objects.

- [ ] **Step 3: Write one-step parity test**

Instrument the new step result with fixed-shape debug metrics and compare:

```text
acting output/state
sampled/logprob/clipped env action
bootstrap value and discarded state
differentiation Jacobian per leaf
carried/update traces and emphasis
direct gradients
Adam increments/state
fast params and slow torso
persisted feedback and counters
```

- [ ] **Step 4: Implement the concrete Meta step**

Required order:

```text
normalize/input → target views → acting forward → heads/sample →
env step → bootstrap forward → TD → differentiation forward inside jacobian →
three traces → Adam → slow target → persisted state/metrics
```

- [ ] **Step 5: Add terminal, incoming/fresh, clipping, prediction tests**

Use parameterized tests; every branch must compare to the old algorithm rather than only
to a local formula.

- [ ] **Step 6: Add three-step parity**

Compare each intermediate state, not only the final state.

- [ ] **Step 7: Verify**

```bash
JAX_PLATFORM_NAME=cpu uv run pytest tests/online_ac/test_meta_parity.py -v
```

Expected: init, one-step, terminal and three-step parity pass within declared tolerance.

---

### Task 5: Implement StandardProgram and Prove StreamAC-RTU Parity

**Files:**
- Create: `memorax/online_ac/standard.py`
- Create: `tests/online_ac/test_standard_parity.py`

**Interfaces:**
- Produces: `NetworkState`
- Produces: `StandardState`
- Produces: `StandardStepMetrics`
- Produces: `make_standard_program(parts, static_config) -> AgentProgram`

- [ ] **Step 1: Write init parity test**

Actor and critic must have independent params, carry, sensitivity, traces and OBGD `v`.

- [ ] **Step 2: Implement StandardState and init**

Invoke the same network recipe twice with different legacy RNG keys. Do not share Module
parameters or update state.

- [ ] **Step 3: Write one-step parity test**

Compare acting actor/critic states, sampled action, bootstrap value, TD, both Jacobians,
fresh traces, whole-tree step sizes, `v`, params and feedback.

- [ ] **Step 4: Implement concrete Standard step**

Required order:

```text
normalize/input → actor acting → critic acting → sample/env step →
critic bootstrap → TD → actor/critic differentiation forward inside jacobian →
fresh traces → critic whole-tree OBGD → actor whole-tree OBGD →
persisted state/metrics
```

- [ ] **Step 5: Cover adaptive on/off and discrete/continuous**

Also assert that wrapper-clipped action does not silently replace StreamAC feedback.

- [ ] **Step 6: Add three-step and terminal parity**

The first step starts with `reset_before_t=True`; test the trace one-step boundary.

- [ ] **Step 7: Verify**

```bash
JAX_PLATFORM_NAME=cpu uv run pytest tests/online_ac/test_standard_parity.py -v
```

Expected: all StandardProgram leaves match the StreamACRtrl oracle.

---

### Task 6: Parameterize Normalization and Evaluation

**Files:**
- Create: `memorax/online_ac/normalization.py`
- Create: `tests/online_ac/test_evaluation_parity.py`
- Modify: `memorax/online_ac/meta.py`
- Modify: `memorax/online_ac/standard.py`

**Interfaces:**
- Produces: `make_normalizer(config)`
- Produces: evaluation closures for both programs

- [ ] **Step 1: Test explicit normalizer against existing wrappers**

Compare step-by-step:

```text
observation mean/M2/count and normalized value
reward G/mean/M2/count and scaled reward
episode boundary behavior
raw episode return
```

Reward normalizer uses its own legacy gamma `0.99`, not algorithm gamma.

- [ ] **Step 2: Implement the explicit normalizer**

Only one owner may run. Builder validation must reject simultaneous wrapper and program
normalization.

- [ ] **Step 3: Test the two evaluation booleans**

Required cases:

```text
reset_on_start=True,  update_during_eval=True   # legacy
reset_on_start=False, update_during_eval=False  # copied/frozen
reset_on_start=False, update_during_eval=True   # copied/adaptive
reset_on_start=True,  update_during_eval=False  # build error
```

- [ ] **Step 4: Implement legacy fixed-vector-step evaluation**

Preserve:

```text
runtime key split → evaluation reset/eval split →
per-env reset keys → per-step keys
```

Use deterministic actions, evaluation-local carry/sensitivity, and advance sensitivity
through `local_jacobian`. Never update params, targets, traces or optimizer.

- [ ] **Step 5: Verify both algorithms against legacy evaluate**

```bash
JAX_PLATFORM_NAME=cpu uv run pytest tests/online_ac/test_evaluation_parity.py -v
```

Expected: legacy state/metrics match; all three legal normalization modes satisfy their
explicit contracts.

---

### Task 7: Build-Time Composition and Legacy Builders

**Files:**
- Create: `memorax/online_ac/build.py`
- Modify: `memorax/online_ac/__init__.py`
- Modify: `memorax/algorithms/__init__.py`
- Modify: `experiments/base/experiment.py`
- Create: `tests/online_ac/test_legacy_builders.py`

**Interfaces:**
- Produces: `build_meta_program(config: MetaProgramConfig, env: JAXEnvAdapter)`
- Produces: `build_standard_program(config: StandardProgramConfig, env: JAXEnvAdapter)`
- Preserves: `build_rtrrl_agent(cfg, env, env_params)`
- Preserves: `build_stream_ac_agent(cfg, env, env_params)`

- [ ] **Step 1: Test strict build validation**

Reject unsupported core/credit pairs, missing target subtree, mismatched domains, double
normalization, invalid evaluation booleans and conflicting legacy/new fields.

- [ ] **Step 2: Implement explicit factories**

Use exhaustive `isinstance`/tag matching before JIT. No registry or string dispatch enters
the program closure.

- [ ] **Step 3: Translate every RTRRL legacy field**

Include encoder/raw recipe, LRU/RTU dimensions, activation, gamma, eta, trace timing,
prediction, actor bounds/clipping, logprob reduction, freeze gamma, Adam and normalization.

- [ ] **Step 4: Translate every StreamACRtrl legacy field**

Include separate actor/critic network recipes, exact RTRL, entropy objective, trace lambda,
whole-tree OBGD/adaptive state and evaluation semantics.

- [ ] **Step 5: Keep old algorithms as oracle façades**

At this gate builders may return adapters around `AgentProgram`, but public lifecycle
signatures remain:

```python
init(key)
warmup(key, state, num_steps)
train(key, state, num_steps)
evaluate(key, state, num_steps)
```

- [ ] **Step 6: Verify old YAML/build paths**

```bash
JAX_PLATFORM_NAME=cpu uv run pytest tests/online_ac/test_legacy_builders.py -v
```

Expected: current RTRRL and StreamAC configs build without edits and one-step parity still
passes.

---

### Task 8: JIT, Compile-Reuse, and Final Parity Gate

**Files:**
- Create: `tests/online_ac/test_jit_contract.py`
- Modify only if failures prove necessary: files created in Tasks 2–7

**Interfaces:**
- Verifies the complete first-stage acceptance gate

- [ ] **Step 1: Test JIT lifecycle**

```python
init = jax.jit(program.init_fn)
train_epoch = jax.jit(program.train_epoch_fn)
evaluate = jax.jit(program.evaluate_fn)
```

Assert fixed state treedef/shape/dtype across scan.

- [ ] **Step 2: Test one trace per fixed shape**

Use a Python trace counter captured only for the test. Two epoch calls with identical
shapes must not increment it twice.

- [ ] **Step 3: Inspect JAXPR contract**

Assert no recorder, config parser, host callback or Python runtime state appears in JAXPR.
`AgentProgram`, Module and optimizer transform must not be State leaves.

- [ ] **Step 4: Run all parity tests**

```bash
cd /home/ubuntu/trainer/streaming-rtrrl/memo
JAX_PLATFORM_NAME=cpu uv run pytest tests/online_ac -v
```

Expected: zero failures.

- [ ] **Step 5: Run static and repository checks**

```bash
uv run pyright memorax/online_ac tests/online_ac
uv run pre-commit run --all-files
JAX_PLATFORM_NAME=cpu uv run pytest -v
```

Expected: zero errors/failures. If the repository still has no tests outside
`tests/online_ac`, pytest must at least collect and pass this new suite.

- [ ] **Step 6: Record parity evidence**

Create a test-output artifact containing:

```text
JAX/JAXLIB/backend
golden manifest hashes
RTRRL init/one-step/three-step/eval result
StreamAC init/one-step/three-step/eval result
JIT trace-count result
```

Do not start recorder/runtime/CLI migration or RTRRL-OBGD until this gate passes.

---

## Deferred Follow-Up Plans

Parity gate 通过后分别编写，不合并进本计划：

1. `ExperimentRuntime + tagged config + environment adapter + Aim recorder`；
2. legacy YAML/CLI/Aim/HPO 全链路迁移；
3. RTRRL-OBGD direct-gradient/domain 语义与实验；
4. episode trajectory/Rerun。

这保证“模块化入口存在”与“两个原算法已经复现”先成为可独立验收的软件增量。
