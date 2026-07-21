# RTRRL Numerical-Parity Modularization Implementation Plan

> **SUPERSEDED IN PART — 2026-07-20:** The later user decision replaces every
> instruction to modify or delegate `rtrrl/rtrrl.py`. That file must equal
> functional base `5f7ff4e` byte-for-byte and remains only a backup/reference.
> Memo's compatibility helpers and strict runtime remain under `memo/`. Task 11
> is replaced by the preservation contract and independent numerical audit
> below; Task 12 invokes Memo's runner directly.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Memorax the single modular implementation of AAAI25 LRU-RTRRL,
preserve legacy configuration and logging contracts, and retain Memorax
ablation branches without contaminating the strict numerical baseline.

**Architecture:** Convert `memorax.algorithms.rtrrl` from one module into a
package whose public exports remain unchanged. Pure functional components are
selected at construction time and closed over by lifecycle functions. The
AAAI25 process emits versioned fixtures; the Memorax process consumes them, so
different JAX/JAXLIB versions are never imported together.

**Tech Stack:** Python 3.12, JAX/JAXLIB, Flax, Optax, Distrax, Pytest, NumPy,
simple_parsing, AWS EC2 and AWS Batch.

## Global Constraints

- Work only in
  `/home/ubuntu/trainer/streaming-rtrrl-worktrees/rtrrl-numerical-modularization`.
- Do not edit the main checkout or any pre-existing worktree.
- Create one local commit per reviewed task. Pushing and final branch integration
  are allowed, but occur only after the corresponding verification gates pass.
- Do not run a complete reinforcement-learning environment on the local host.
- Do not expand the local host.
- Run JAX parity and finite-difference tests on an EC2 instance with at least
  4 vCPUs and 8 GiB RAM.
- Use `/home/ubuntu/trainer/RTRRL-AAAI25/rtrrl.py` as the oracle.
- Strict parity covers LRU only. Explicit CTRNN and no-RNN legacy selections
  fail during construction.
- Existing YAML files load without edits.
- Existing metric names, aggregation, and logger step semantics remain intact.
- Every production change follows red, green, refactor.

## Target File Structure

Create:

```text
memo/memorax/algorithms/rtrrl/
├── __init__.py
├── types.py
├── components.py
├── heads.py
├── lru.py
├── rules.py
├── state_machine.py
├── program.py
├── compatibility.py
└── legacy.py

memo/tests/rtrrl_parity/
├── __init__.py
├── assertions.py
├── oracle_capture.py
├── test_public_api.py
├── test_config_compat.py
├── test_heads_parity.py
├── test_lru_forward_parity.py
├── test_lru_credit_parity.py
├── test_rules_parity.py
├── test_init_parity.py
├── test_step_parity.py
├── test_program_contract.py
├── test_logging_compat.py
└── golden/
    ├── manifest.json
    └── aaai25_lru.npz
```

Modify:

```text
memo/memorax/algorithms/__init__.py
memo/memorax/algorithms/independent_rtrrl.py
memo/experiments/base/experiment.py
memo/experiments/rtrrl_hopper/run.py
memo/tests/online_ac/conftest.py
memo/tests/online_ac/test_legacy_builders.py
memo/tests/online_ac/test_meta_parity.py
memo/tests/test_independent_rtrrl.py
```

Delete only after public API parity is green:

```text
memo/memorax/algorithms/rtrrl.py
```

---

### Task 1: Establish the Oracle Fixture Contract

**Files:**
- Create: `memo/tests/rtrrl_parity/__init__.py`
- Create: `memo/tests/rtrrl_parity/assertions.py`
- Create: `memo/tests/rtrrl_parity/oracle_capture.py`
- Create: `memo/tests/rtrrl_parity/test_public_api.py`
- Create: `memo/tests/rtrrl_parity/golden/manifest.json`
- Create: `memo/tests/rtrrl_parity/golden/aaai25_lru.npz`

**Interfaces:**
- Produces: `load_oracle() -> tuple[dict[str, np.ndarray], dict[str, Any]]`
- Produces: `assert_tree_close(actual, expected, policy) -> None`
- Consumes: AAAI25 modules through an explicit `--oracle-root` argument.

- [ ] **Step 1: Write the fixture-loader tests**

Create `test_public_api.py` with tests that require:

```python
def test_oracle_manifest_pins_source_and_runtime():
    arrays, manifest = load_oracle()
    assert manifest["source"] == "RTRRL-AAAI25"
    assert manifest["commit"] == "4301943c349171d828d0fcf3e40944c286451415"
    assert manifest["algorithm"] == "lru"
    assert manifest["seed"] == 7
    assert manifest["dtype_policy"] == "float32-complex64"
    assert sorted(arrays) == manifest["leaf_paths"]


def test_oracle_fixture_has_required_sections():
    arrays, _ = load_oracle()
    required = {
        "heads/input",
        "heads/actor_loc",
        "heads/actor_scale",
        "heads/value",
        "lru/input",
        "lru/carry_before",
        "lru/carry_after",
        "lru/output",
        "credit/after_step_1",
        "credit/after_step_2",
        "init/action",
        "init/value",
        "step/td_error",
    }
    assert required <= arrays.keys()
```

- [ ] **Step 2: Run the tests and verify the missing-loader failure**

Run on the local host:

```bash
python3 -m pytest memo/tests/rtrrl_parity/test_public_api.py -q
```

Expected: import failure for `load_oracle`.

- [ ] **Step 3: Implement stable flattening and comparison helpers**

`assertions.py` must:

- flatten paths with JAX key names and tuple/list indices
- compare shape and dtype before values
- compare bool/int leaves exactly
- reject non-finite floats
- support exact, `(rtol, atol)`, or per-leaf ULP policies
- normalize signed zero before ULP comparison

Use the existing helpers in
`memo/tests/online_ac/test_standard_parity.py:51-152` as the starting
implementation, moving reusable logic rather than copying divergent versions.

- [ ] **Step 4: Implement `oracle_capture.py` as a standalone CLI**

The CLI signature is:

```python
def main(
    oracle_root: Path,
    output_dir: Path,
    seed: int = 7,
) -> None:
    ...
```

It must:

- prepend `oracle_root` to `sys.path`
- import AAAI25 `RNNActorCritic`, `OnlineLRULayer`, trace helpers, and optimizer
- use hidden size 2, input size 4, action size 2, batch size 1
- use deterministic explicit mock transitions rather than Brax
- save arrays with slash-delimited stable names
- write source commit, Python/JAX/JAXLIB/backend versions and leaf paths
- refuse to overwrite existing fixtures without `--overwrite`

- [ ] **Step 5: Generate the fixture on EC2**

Create or reuse an 8 GiB EC2 development instance. Install the AAAI25
environment separately from Memorax, then run:

```bash
python memo/tests/rtrrl_parity/oracle_capture.py \
  --oracle-root /workspace/RTRRL-AAAI25 \
  --output-dir memo/tests/rtrrl_parity/golden
```

Expected: `manifest.json` and `aaai25_lru.npz` with finite leaves.

- [ ] **Step 6: Run loader tests in the Memorax environment**

Expected: both tests pass without importing AAAI25.

---

### Task 2: Freeze the Existing Public API Before Splitting the Module

**Files:**
- Modify: `memo/tests/rtrrl_parity/test_public_api.py`
- Create: `memo/memorax/algorithms/rtrrl/__init__.py`
- Create: `memo/memorax/algorithms/rtrrl/legacy.py`
- Modify: `memo/memorax/algorithms/__init__.py`
- Modify: `memo/memorax/algorithms/independent_rtrrl.py`
- Delete after green: `memo/memorax/algorithms/rtrrl.py`

**Interfaces:**
- Preserves: `RTRRL`, `RTRRLConfig`, `RTRRLState`
- Preserves for internal consumers: `_find_leaf`, `_tree_norm`

- [ ] **Step 1: Add public API characterization**

Assert:

```python
def test_rtrrl_public_exports_remain_stable():
    from memorax.algorithms import RTRRL, RTRRLConfig, RTRRLState
    from memorax.algorithms.rtrrl import _find_leaf, _tree_norm

    assert RTRRL.__name__ == "RTRRL"
    assert RTRRLConfig.__name__ == "RTRRLConfig"
    assert RTRRLState.__name__ == "RTRRLState"
    assert callable(_find_leaf)
    assert callable(_tree_norm)
```

Also instantiate the existing tiny agent from
`memo/tests/online_ac/conftest.py:118-150`.

- [ ] **Step 2: Run the characterization test**

Expected: pass against the existing single-file module.

- [ ] **Step 3: Move the implementation mechanically**

Move the complete current `rtrrl.py` body to `rtrrl/legacy.py`. In
`rtrrl/__init__.py`, re-export:

```python
from .legacy import (
    RTRRL,
    RTRRLConfig,
    RTRRLState,
    _find_leaf,
    _tree_norm,
)

__all__ = [
    "RTRRL",
    "RTRRLConfig",
    "RTRRLState",
]
```

Update `independent_rtrrl.py` to import from
`.rtrrl` without changing names.

- [ ] **Step 4: Remove the conflicting `rtrrl.py` module**

Delete the old file only after the package contains the complete moved
implementation.

- [ ] **Step 5: Run all existing RTRRL characterization tests on EC2**

Run:

```bash
pytest \
  memo/tests/online_ac/test_legacy_characterization.py \
  memo/tests/online_ac/test_meta_parity.py \
  memo/tests/online_ac/test_legacy_builders.py \
  memo/tests/test_independent_rtrrl.py -q
```

Expected: no behavior change.

---

### Task 3: Normalize Legacy Configuration Without Changing Semantics

**Files:**
- Create: `memo/memorax/algorithms/rtrrl/compatibility.py`
- Create: `memo/tests/rtrrl_parity/test_config_compat.py`
- Modify: `memo/experiments/rtrrl_hopper/run.py`
- Modify: `memo/experiments/base/experiment.py`

**Interfaces:**
- Produces: `LegacyRTRRLConfig`
- Produces: `normalize_legacy_config(raw) -> LegacyRTRRLConfig`
- Produces: `to_component_config(legacy) -> RTRRLComponentConfig`

- [ ] **Step 1: Write failing tests for real YAML files**

Parameterize over:

- `rtrrl/config/rtrrl_hop_533.yml`
- `rtrrl/config/rtrrl_hop_534.yml`
- `memo/config/rtrrl_hopper_533.yml`
- `memo/config/rtrrl_hopper_newlru_base.yml`

Assert that:

- scientific-notation strings become floats
- nested optimizer and environment fields survive
- absent `rnn_model` becomes `"lru"`
- explicit `"lru"` is accepted
- explicit `"ctrnn"` and `None` raise `UnsupportedRTRRLBranch`
- legacy no-op fields remain present with a warning record
- unknown fields include their dotted path in the exception

- [ ] **Step 2: Run tests and verify missing normalizer failures**

- [ ] **Step 3: Implement frozen compatibility dataclasses**

Do not use free-form dictionaries after parsing. Include every field audited in
the design, including nested optimizer/environment fields and streaming additions
`run_name`, `align_action_logprob`, and `update_trace_before_td`.

- [ ] **Step 4: Add component-profile resolution**

Resolution rules:

```python
if profile == "aaai25_strict_lru":
    backbone = "aaai25_lru"
    trace_timing = "incoming"
    logprob_reduction = "mean"
elif profile == "memo_experimental":
    backbone = legacy.backbone
else:
    raise ValueError(...)
```

Explicit experimental overrides must be recorded in the effective config.

- [ ] **Step 5: Route the experiment builder through normalization**

Keep old caller signatures. Do not change training behavior yet.

- [ ] **Step 6: Run configuration and existing builder tests**

Expected: real YAMLs load unchanged and builders still construct.

---

### Task 4: Implement Parity-Verified Heads

**Files:**
- Create: `memo/memorax/algorithms/rtrrl/heads.py`
- Create: `memo/tests/rtrrl_parity/test_heads_parity.py`

**Interfaces:**
- Produces: `FADense`
- Produces: `RTRRLTDHead`
- Produces: `make_action_distribution(...)`

- [ ] **Step 1: Write fixture-based forward tests**

Use explicit fixture parameters rather than relying only on matching random
initializers. Assert actor loc, scale, value, log-probability, entropy, and
sampled action against the oracle.

- [ ] **Step 2: Add initializer and parameter-path tests**

Assert:

- actor uses no bias in the strict linear path
- critic bias behavior matches AAAI25
- kernels use the AAAI25 initializer
- continuous output has `2 * action_dim`
- strict distribution remains per-dimension `distrax.Normal`
- objective uses mean over action dimensions

- [ ] **Step 3: Run and observe failures against generic Memorax heads**

The test must demonstrate at least the actor-bias, initializer, and distribution
reduction differences.

- [ ] **Step 4: Implement the strict heads**

Port the mathematical behavior from:

- `RTRRL-AAAI25/models/neural_networks.py:14-76`
- `RTRRL-AAAI25/rtrrl.py:126-285`

Use module names and parameter paths recorded by fixtures.

- [ ] **Step 5: Run forward and VJP parity tests**

Expected: exact tree structure; float values within the initial narrow policy.

---

### Task 5: Implement the Strict LRU Forward Component

**Files:**
- Create: `memo/memorax/algorithms/rtrrl/lru.py`
- Create: `memo/tests/rtrrl_parity/test_lru_forward_parity.py`
- Create: `memo/memorax/algorithms/rtrrl/components.py`

**Interfaces:**
- Produces protocol: `RecurrentComponent`
- Produces: `AAAI25LRU`
- Produces: `LRUCarry`

- [ ] **Step 1: Define the component protocol**

The protocol exposes:

```python
class RecurrentComponent(Protocol):
    def initialize(self, key, input_shape): ...
    def forward(self, params, carry, inputs, reset): ...
    def credit(self, params, credit_state, carry, inputs, cotangent): ...
```

All dynamic values are arguments; component construction captures only static
dimensions and mode.

- [ ] **Step 2: Write strict forward tests**

Assert fixture parity for:

- `nu_log`, `theta_log`, `gamma_log`
- complex lambda
- normalized complex input matrix
- next hidden state
- `C_real/C_img` projection
- `D` skip
- SiLU output
- reset behavior

- [ ] **Step 3: Verify generic Memorax LRU fails the parity test**

Record failures caused by parameter names/layout, initializer order, carry
layout, or output path.

- [ ] **Step 4: Implement AAAI25 recurrence and readout**

Translate:

- `RTRRL-AAAI25/models/online_lru.py:17-121`
- `RTRRL-AAAI25/models/online_lru.py:240-300`

Do not implement credit/custom VJP in this task.

- [ ] **Step 5: Run eager and JIT forward parity**

Expected: eager within narrow tolerance; JIT differences, if any, measured but
not yet whitelisted.

---

### Task 6: Implement Strict LRU Online Credit

**Files:**
- Modify: `memo/memorax/algorithms/rtrrl/lru.py`
- Create: `memo/tests/rtrrl_parity/test_lru_credit_parity.py`

**Interfaces:**
- Produces: `LRUCreditState`
- Produces: custom VJP compatible with `AAAI25LRU.forward`

- [ ] **Step 1: Write two-step credit tests**

Compare after each step:

- lambda sensitivity
- gamma sensitivity
- B sensitivity
- `nu_log`, `theta_log`, `gamma_log`, `B_real`, and `B_img` gradients
- the negative imaginary sign for `B_img`

- [ ] **Step 2: Add finite-difference directional tests**

For each recurrent parameter group, use central differences with float32
`epsilon=1e-3`. Require:

- finite analytical and numerical directions
- cosine similarity at least `0.999`
- relative error at most `1e-2`

- [ ] **Step 3: Run tests and verify the forward-only component fails**

- [ ] **Step 4: Implement credit recurrence and custom VJP**

Translate:

- `RTRRL-AAAI25/models/online_lru.py:158-237`

Keep the compact AAAI25 credit state rather than Memorax's full Jacobian state.

- [ ] **Step 5: Run eager, JIT, and finite-difference tests on EC2**

Any systematic discrepancy requires formula investigation; do not widen
tolerance.

---

### Task 7: Extract Pure RTRRL Update Rules

**Files:**
- Create: `memo/memorax/algorithms/rtrrl/rules.py`
- Create: `memo/tests/rtrrl_parity/test_rules_parity.py`
- Modify when parity proves reuse: `memo/memorax/online_ac/td.py`
- Modify when parity proves reuse: `memo/memorax/online_ac/traces.py`
- Modify when parity proves reuse: `memo/memorax/online_ac/updates.py`
- Modify when parity proves reuse: `memo/memorax/online_ac/targets.py`

**Interfaces:**
- Produces: `td_error`
- Produces: `update_traces`
- Produces: `combine_update_directions`
- Produces: `update_emphasis_or_average_reward`
- Produces: `update_slow_target`

- [ ] **Step 1: Write table-driven scalar and tree tests**

Cover:

- terminal/non-terminal TD
- episodic emphasis
- average reward
- accumulated actor/RNN traces
- accumulated and Dutch critic traces
- `lambda_rnn == 0`
- incoming and fresh timing
- per-environment delta-before-mean reduction
- entropy and direct-gradient routing
- Polyak period 1 and 0.1

- [ ] **Step 2: Compare existing online_ac helpers**

Run each helper against the independent equations. Reuse a helper only when its
tests pass without adapters that reorder operations.

- [ ] **Step 3: Implement missing strict rules**

Use:

- `RTRRL-AAAI25/traces.py:11-53`
- `RTRRL-AAAI25/rtrrl.py:663-769`

- [ ] **Step 4: Add negative tests**

Tests must reject mean-before-delta, wrong trace timing, wrong entropy scaling,
and wrong parameter-domain scaling.

- [ ] **Step 5: Run the rule suite locally if measured memory permits**

These tests should avoid full network compilation.

---

### Task 8: Implement Historical Initialization and the Online State Machine

**Files:**
- Create: `memo/memorax/algorithms/rtrrl/types.py`
- Create: `memo/memorax/algorithms/rtrrl/state_machine.py`
- Create: `memo/tests/rtrrl_parity/test_init_parity.py`
- Create: `memo/tests/rtrrl_parity/test_step_parity.py`

**Interfaces:**
- Produces: `RTRRLComponentConfig`
- Produces: `RTRRLState`
- Produces: `make_init_fn(components, config, env)`
- Produces: `make_step_fn(components, config, env, debug)`

- [ ] **Step 1: Write initialization parity tests**

Compare key split order, params, optimizer state, initial action, initial value,
advanced recurrent state, zero traces, emphasis, and running statistics.

- [ ] **Step 2: Implement `make_init_fn`**

Match AAAI25 pre-sampling exactly. Do not use the current zero-action Memorax
initialization in strict mode.

- [ ] **Step 3: Write complete eager-step parity tests**

Use a deterministic mock environment adapter. Compare every persisted state
leaf and these observables:

- environment action
- model input
- sampled next action
- value target
- TD error
- gradients
- incoming and carried traces
- optimizer updates
- fast and slow parameters
- emphasis/average reward

- [ ] **Step 4: Implement `make_step_fn` as closure composition**

The factory captures static components and config. The returned function accepts
only `(state, key)` and returns explicit state and metrics.

- [ ] **Step 5: Add terminal, two-environment, and three-step tests**

Verify reset phase, reduction order, persisted feedback, and no one-step phase
drift.

- [ ] **Step 6: Add separate train and debug metric schemas**

`debug=True` may return full trees for at most three steps. Production mode
returns scalar/event data only.

- [ ] **Step 7: Run eager parity on EC2**

Expected: all fixture sections within reviewed tolerances.

---

### Task 9: Build the Closed Program and Preserve Logging

**Files:**
- Create: `memo/memorax/algorithms/rtrrl/program.py`
- Create: `memo/tests/rtrrl_parity/test_program_contract.py`
- Create: `memo/tests/rtrrl_parity/test_logging_compat.py`
- Modify: `memo/memorax/algorithms/rtrrl/__init__.py`
- Modify: `memo/experiments/base/experiment.py`

**Interfaces:**
- Produces: `build_rtrrl_program(config, components, env) -> AgentProgram`
- Preserves: legacy `RTRRL.init/train/evaluate` façade

- [ ] **Step 1: Write closure and JIT contract tests**

Assert:

- component selection occurs once at construction
- state is a fixed JAX pytree
- `step_fn` has no host callback
- production epoch output excludes full state history
- repeated fixed-shape calls do not retrace

- [ ] **Step 2: Write logging-contract tests**

Feed fixed synthetic step summaries and assert all historical metric keys,
aggregation, dtype, and logger step values.

- [ ] **Step 3: Implement the program builder**

Use `memorax.online_ac.types.AgentProgram` and `ActionDecision`. Keep
environment translation and logging outside mathematical components.

- [ ] **Step 4: Implement the legacy class façade**

`RTRRL`, `RTRRLConfig`, and `RTRRLState` remain importable. Methods delegate to
the constructed program rather than keeping a second update implementation.

- [ ] **Step 5: Run JIT, logging, and existing builder tests on EC2**

Expected: fixed schema, no large stacked debug trees, historical metrics intact.

---

### Task 10: Reintroduce Memorax Extension Branches

**Files:**
- Modify: `memo/experiments/base/experiment.py`
- Modify: `memo/memorax/algorithms/independent_rtrrl.py`
- Modify: `memo/tests/test_independent_rtrrl.py`
- Modify: `memo/tests/online_ac/test_meta_parity.py`
- Create or modify component adapters under:
  `memo/memorax/algorithms/rtrrl/components.py`

**Interfaces:**
- Produces components for RTU, encoder, bounded actor, clipping, frozen gamma,
  prediction, trace timing/reduction, and independent pathways.

- [ ] **Step 1: Add component-selection tests**

For every retained branch, assert effective config records the selected
component and that selection occurs before JIT.

- [ ] **Step 2: Adapt existing Memorax modules to component protocols**

Do not alter `aaai25_strict_lru`. Add adapters for:

- existing `Memoroid(LRUCell)` native LRU
- `RNN(RTUCell)`
- encoder/no-encoder feature extraction
- bounded/unbounded Gaussian heads
- prediction head
- shared/independent topology

- [ ] **Step 3: Preserve branch-specific invariants**

Run existing tests for independent ownership, zero cross-gradients, entropy
routing, bootstrap carry, terminal reset, and slow targets.

- [ ] **Step 4: Add strict-profile immutability tests**

Experimental flags must either produce a different profile name or be rejected
when combined with `aaai25_strict_lru`.

- [ ] **Step 5: Run the complete Memorax RTRRL test set on EC2**

Expected: strict parity remains green while all retained extensions construct
and satisfy their contracts.

---

### Task 11: Preserved External Copy and Memo Configuration Migration

**Files:**
- Preserve: `rtrrl/rtrrl.py` exactly as at `5f7ff4e`
- Modify: `memo/experiments/rtrrl_hopper/run.py`
- Modify: legacy builder tests and configuration tests

**Interfaces:**
- Memo's compatibility/configuration helpers accept old YAML mappings.
- The external script remains unchanged and retains its original mathematics.

- [ ] **Step 1: Add Memo-helper compatibility and preservation tests**

For representative YAML files, parse and build without starting a full
environment. Assert effective budgets, profile, logger settings, and optimizer
fields.

- [ ] **Step 2: Restore and freeze the external script**

Restore `rtrrl/rtrrl.py` byte-for-byte from `5f7ff4e`; enforce byte hash and
canonical AST hash. Keep Memo normalization, program construction, training,
evaluation, and logging tests independent of the external script.

- [ ] **Step 3: Verify Memo historical metric translation**

Run one mock epoch and compare the emitted metric dictionary against the
pre-refactor fixture.

- [ ] **Step 4: Verify all 686 RTRRL YAML files parse**

Parsing does not create environments or compile JAX. Produce a report of:

- accepted files
- unsupported explicit branches
- unknown fields
- deprecated no-op fields

Acceptance requires no edits for configurations in the supported LRU scope.

- [ ] **Step 5: Audit the preserved LRU path against AAAI25**

Use separate processes/environments and deterministic inputs. Distinguish
matching forward/trace behavior, differing actor-gradient semantics, matching
optional trace order, intentionally different configuration fields, and
unverified branches. Do not infer complete-step parity from source similarity.

---

### Task 12: Final Verification and Resource Report

**Files:**
- Create: `memo/docs/rtrrl-numerical-parity-report.md`
- Update: fixture manifest only if reviewed runtime calibration requires it

**Interfaces:**
- Produces evidence for every acceptance criterion in the design.

- [ ] **Step 1: Run the complete targeted EC2 suite**

Run:

```bash
pytest memo/tests/rtrrl_parity -q
pytest memo/tests/online_ac -q
pytest memo/tests/test_independent_rtrrl.py -q
```

Record test counts, failures, duration, peak RSS, JAX/JAXLIB, and backend.

- [ ] **Step 2: Run a short Batch or EC2 Memo Brax integration smoke**

Use one environment, strict LRU, and the shortest budget that compiles and
emits training/evaluation metrics. Invoke Memo's strict runner directly, never
the preserved external script. This is integration validation, not the oracle.

- [ ] **Step 3: Compare CPU eager, CPU JIT, and selected Batch backend**

Document exact leaves and any calibrated ULP bounds. Do not generalize bounds
from one leaf to an entire tree.

- [ ] **Step 4: Verify repository state**

Confirm:

- only the new worktree contains changes
- no credentials or environment secrets are present
- the external script exactly matches the preserved base copy
- Memo's maintained runtime does not import or invoke the external script
- the preserved duplicate is clearly labeled backup/reference, not maintained
- no complete RL environment ran locally

- [ ] **Step 5: Write the parity report**

Include:

- fixture provenance
- module-by-module maximum absolute, relative, and ULP differences
- complete-step and short-scan results
- configuration compatibility counts
- logging compatibility result
- memory measurements and recommended EC2/Batch resources
- explicitly unsupported branches

Do not claim completion unless every acceptance criterion has fresh evidence.
