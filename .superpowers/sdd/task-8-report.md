# Task 8 Report: Historical Initialization and Online State Machine

## Status

Implemented the first complete strict numerical integration:

- canonical frozen `RTRRLComponentConfig` was extended in compatibility rather
  than duplicated;
- `RTRRLComponents`, explicit initialization keys, complete `RTRRLState`, and
  separate train/debug metric pytrees were added;
- `make_init_fn(components, config, env)` reproduces historical pre-sampling;
- `make_step_fn(components, config, env, debug)` composes the verified head, LRU,
  unbatched credit, pure rules, optimizer, slow target, and mock environment;
- no scan/JIT orchestration, program builder, or legacy façade delegation was
  added.

The returned init closure accepts only `root_key`. The returned step closure
accepts only `(state, key)`. Components, configuration, optimizer construction,
and environment are static closure values.

## Strict TDD Evidence

### Initialization RED

`test_init_parity.py` was created before either requested production module.
The focused local collection run failed for the missing state machine:

```text
ModuleNotFoundError:
No module named 'memorax.algorithms.rtrrl.state_machine'
```

The tests require exact historical split keys, full parameter and optimizer
state trees, pre-sampled action/value, advanced hidden state, zero persisted
credit leaves, zero eligibility traces, emphasis/average reward, and disabled
running-statistics leaves.

### Initialization GREEN

Authorized Batch job `bbd0d280-fd90-49c1-a7af-07764106d91f` passed both
initialization tests after the implementation preserved legacy Threefry mode
and the pinned pre-sampling phase:

```text
2 passed
```

Earlier GREEN attempts usefully exposed:

- missing `git` in the worker setup;
- an incomplete old Task 6 overlay;
- changed JAX 0.10 partitionable Threefry defaults; and
- the fact that pinned initialization advances hidden state but leaves online
  sensitivity leaves at zero.

No expected value was changed to accommodate these failures.

### Complete Step RED

`test_step_parity.py` was then written before `make_step_fn`. Its focused local
collection run failed exactly because that interface was absent:

```text
ImportError: cannot import name 'make_step_fn'
```

The first complete Batch implementation runs subsequently exposed a
FrozenDict custom-VJP boundary mismatch and then the pinned sensitivity
persistence behavior. Production code was corrected; expected oracle leaves
were not constructed or altered in Memorax.

### Complete Step GREEN

Authorized Batch job `0cdd64d7-08cc-4898-a1ef-a6891c91b073` passed the initial
and complete-step suite:

```text
6 passed
```

The suite compares the first full step, terminal transition, three-step
feedback phase, two-environment reduction order, metric schemas, debug cap, and
closure captures.

## Oracle Fixture Provenance

The fixture was extended by invoking actual pinned AAAI25 model, trace, and
optimizer code. It was regenerated in a separate clean runtime:

- final Batch job: `5a3f850b-e1b2-4802-b92a-ad8714527059`;
- resources: 4 vCPU, 8192 MiB on `rtrrl-cpu2-queue`;
- source: clean `RTRRL-AAAI25` checkout;
- source commit: `4301943c349171d828d0fcf3e40944c286451415`;
- runtime: Python 3.12.13, JAX/JAXLIB 0.4.38, Flax 0.10.2, CPU;
- seed: 7;
- NPZ size: 212964 bytes;
- manifest size: 126157 bytes;
- NPZ SHA-256:
  `080ef79d07e15e8c1afffe41530484153ac24a9305e16c739e6a228cb2283a27`.

The capture uses no Memorax function to derive expected values. The only
non-oracle code is the deterministic mock transition protocol and stable tree
flattening/storage.

## Key Order and Historical Initialization

The root key is split once, under legacy non-partitionable Threefry semantics:

```text
[unused, model, step, carry, environment, outer]
```

The model key initializes the complete historical Flax scope and is also the
explicit key used to pre-sample the first action. The recurrent carry is
initialized with the carry key. The environment reset consumes the environment
key. Each online transition then splits its explicit input key as:

```text
[next, action, dropout]
```

The dropout key remains consumed in the historical position even though this
strict fixture has dropout disabled.

## Mock Protocol and Phase Order

The deterministic adapter has one scalar observation, one continuous
two-vector action, and three transitions. Rewards are `0.625`, `0.5`, and
`0.375`; transition two is terminal. It records the exact action passed to the
environment.

Every step preserves this order:

1. split the dynamic step key;
2. pass the previously persisted action to `env.step`;
3. update the current environment state;
4. reset recurrent state for terminal environments;
5. construct current input from observation plus terminal-masked persisted
   action/reward;
6. compute per-environment hidden, sampled next action, value, gradients, and
   direct entropy gradients;
7. form per-environment TD errors before reduction;
8. combine incoming traces and direct gradients, update optimizer/fast/slow
   parameters, emphasis or average reward;
9. carry fresh traces and persist the sampled action/value/recurrent state.

The three-step test checks both the terminal-zeroed second input and the
third-step feedback action, preventing one-step phase drift.

## Full Leaf Comparison and Tolerances

Initialization compares key arrays, parameters, optimizer state, action, value,
model input, recurrent state, traces, emphasis, and average reward exactly.
Boolean/integer leaves, shapes, dtypes, and tree paths are exact.

For each of three transitions the tests compare:

- environment action and complete environment terminal feedback;
- model input and sampled next action;
- value target, TD error, value, actor loss, and entropy;
- full TD gradients and preweighted direct gradients;
- full incoming and carried traces;
- combined mean directions and optimizer updates;
- full fast parameters, slow parameters, optimizer state, recurrent state;
- emphasis and average reward.

Cross-runtime float/complex comparisons use `rtol=2e-6`, `atol=2e-7`. Exact
comparisons remain for initialization, keys, terminal masks, zeroed terminal
feedback, and scalar state branches. The only observed accumulated
environment-action difference was `2.98023224e-08` at step three, within this
previously reviewed narrow CPU policy.

The two-environment fixture stores heterogeneous deltas and their
delta-before-mean direction. Its test explicitly rejects multiplying separate
delta and trace means.

## Metrics and Debug Bound

Production mode returns `TrainStepMetrics` containing only scalar reward,
terminal event, mean TD error, mean value/target, entropy, and actor loss.
Debug mode returns all full trees only while the incoming `step_count < 3`;
later eager calls return the production schema.

## Final Batch and Static Evidence

Final authorized Batch job `0c8f221a-366a-46d2-9731-5721d506bd55` used 4 vCPU
and 8192 MiB:

```text
81 passed, 5 skipped
ruff: All checks passed!
pyright: 0 errors, 0 warnings, 0 informations
compileall: exit 0
```

The five skips are the pre-existing opt-in Task 6 directional finite-difference
tests. All oracle-backed head, forward, credit, rule, initialization, and step
tests executed. The earlier broad ruff attempt reported only three pre-existing
ambiguous-name findings in untouched `legacy.py`; the final changed-file static
scope is clean.

No complete RL environment, heavy eager parity, or JIT compilation ran on the
local controller. Local work was limited to collection RED, ruff/compile
checks, IDE diagnostics, and diff checks.

## Concerns

- Pinned `OnlineLRUCell` computes updated sensitivity memories in the custom-VJP
  residual but returns incoming sensitivity leaves from the primal unless
  `force_trace_compute=True`. The actual full state-machine fixture therefore
  persists zero sensitivity leaves while the Task 6 unbatched custom VJP still
  supplies the correct recurrent gradients. This surprising upstream behavior
  is explicit in production comments and full-state tests.
- The state machine vmaps Task 6 only over individual unbatched environments;
  it does not add batched credit semantics.
- The leading environment axis is preserved on every gradient, direct-gradient,
  and trace leaf until `combine_update_directions` performs its final mean.
- This task intentionally provides eager composition only. Fixed-schema
  scan/JIT orchestration and legacy façade delegation remain Task 9 scope.
- Running statistics are `None`, exactly matching this pinned fixture with both
  normalization flags disabled. Normalized strict parity requires a separately
  captured fixture before enabling those branches.
