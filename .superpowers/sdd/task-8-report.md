# Task 8 Report: Historical Initialization and Online State Machine

> **Authority notice (2026-07-20):** The original implementation narrative
> through `## Concerns` is retained only as historical TDD context and is
> superseded by `## Review-Blocker Correction` and
> `## Final Coverage Resolution` below. Earlier fixture checksums and test
> totals are not authoritative. The only authoritative fixture checksum is in
> `### Corrected Fixture Provenance`; the only authoritative verification
> result is in `### Final Verification` at the end of this report.

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

## Review-Blocker Correction

### Corrected Oracle Independence

The original Task 8 state-machine fixture was rejected because its capture
function independently reassembled the online equations. That implementation
has been removed. The replacement in `oracle_complete_path.py` executes the
actual nested `step_fn` body from pinned `RTRRL-AAAI25/rtrrl.py`.

Identity and control flow are guarded as follows:

- the clean oracle checkout is still required at commit
  `4301943c349171d828d0fcf3e40944c286451415`;
- the exact `rtrrl.py` bytes must have SHA-256
  `082914b2dbe95481e30c738945b58f7948d4065eb917ef1554f8127227ad0edf`;
- six exact, single-occurrence source anchors are checked before execution;
- the transformation removes only the nested `step_fn` `@jax.jit` decorator,
  inserts read-only local-variable callbacks at initialization, key split,
  pre-optimizer, pre-trace, and final-carry boundaries, and leaves every
  original statement and return unchanged;
- only `jax.lax.scan` is replaced by a Python eager driver, so the same nested
  function receives the same carry and step values;
- the production mock environment is substituted through the existing
  `make_env` seam; no update, trace, gradient, TD, optimizer, or phase equation
  exists in the corrected capture path.

The source instrumentation mode, source hash, and exact transformations are
stored in the manifest and asserted by `test_public_api.py`.

### Corrected Fixture Provenance

The corrected fixture was generated only on authorized Batch:

- successful generation job:
  `f07bc5be-a3c4-451d-b87d-ee75fde4ed5c`;
- queue/job definition: `rtrrl-cpu2-queue`, `rtrrl-cpu-job:14`;
- resources: 4 vCPU, 8192 MiB;
- clean source commit:
  `4301943c349171d828d0fcf3e40944c286451415`;
- oracle runtime: Python 3.12, JAX/JAXLIB 0.4.38, Flax 0.10.2;
- NPZ SHA-256:
  `9fe7b0b278fc9d163299ea2a07a10c367fb34ef1ce366c216fed6ead5f4535e7`;
- manifest SHA-256:
  `a1b7cde140898378d45ebb4ea32ccddcf630b89660a47416cfda8c6893ae98cd`;
- NPZ size: 370.8 KiB;
- manifest size: 221.0 KiB.

The generation job executes three one-environment transitions and one real
two-environment transition through the pinned complete body. It also invokes
the pinned `optimizers.make_optimizer` on five nondefault update inputs.

### Corrected TDD Evidence

The correction began with focused RED tests:

```text
4 failed
- missing complete grouped optimizer config
- invalid debug bounds accepted
- missing complete-path fixture provenance
```

The nondefault optimizer characterization was separately observed RED because
the old fixture had no `optimizer_characterization` leaves. Full-state and
two-environment tests were then observed RED against the corrected fixture
until the generic comparator represented and exactly compared `None` leaves.

GREEN evidence:

```text
Batch f7967e3c-68f8-4a77-a7a1-d163362534b4
9 passed
```

This focused job covers all initialization and eager state-machine tests,
including the complete one- and two-environment calls.

### Complete State and Observable Coverage

Initialization and every transition now compare one canonical full-state tree.
The comparator requires identical paths, shapes, dtypes, and values for:

- parameters and slow parameters;
- full optimizer state;
- every environment-state field (`obs`, `reward`, `done`, `phase`,
  `last_action`);
- persisted action, value, recurrent state, initial recurrent state, and
  traces;
- average reward, emphasis, model input, and step count;
- observation and reward statistics, represented by an exact string sentinel
  when the pinned value is `None`.

Because the expected state is a complete prefixed fixture tree, any future
persisted field—array or `None`—changes the path set and fails the test.
Initialization also compares every exposed split key (`root`, `model`, `step`,
`carry`, `environment`, and `outer`). Each step compares input and output keys.

The debug fixture separately compares environment action, model input, sampled
next action, target, delta, value, actor loss, entropy, full TD/direct
gradients, incoming/carried traces, mean directions, optimizer updates, and
the complete resulting state.

### Real Two-Environment Integration

The corrected two-environment test constructs a two-row environment state and
calls production `make_init_fn` and `make_step_fn`. The pinned complete path and
production path both receive heterogeneous observations, rewards, terminal
masks, previous values, and deltas. The test compares the entire returned state
and all gradient, trace, update-direction, and key leaves against the oracle.
It additionally asserts a leading axis of size two on every gradient and trace
leaf, proving that Task 6 credit is vmapped per unbatched environment and that
the reduction occurs only after heterogeneous per-environment deltas are
applied.

### Optimizer Fidelity

`RTRRLComponentConfig` now stores the canonical frozen
`LegacyOptimizerConfig` separately for TD and recurrent groups. Conversion
preserves optimizer name, learning rate, optimizer kwargs (including moments
and epsilon), schedule type/kwargs, weight decay, gradient clipping, and
multistep accumulation.

The production transform follows pinned `make_optimizer` ordering:

```text
weight decay -> global-norm clipping -> optimizer -> optional MultiSteps
```

Maximum-direction signs and schedule values are preserved. The oracle-backed
characterization uses nondefault Adam moments/epsilon, exponential staircase
decay, clipping, weight decay, and two-step accumulation, and compares every
update, parameter, and optimizer-state leaf over five calls. A separate pinned
`optax.incremental_update` fixture verifies post-update fast/previous-slow
argument order.

### Stable Debug and Closure Contracts

`RTRRLComponentConfig` rejects `debug_max_steps` outside `[0, 3]`. A debug
closure always returns `DebugStepMetrics`; once its bound is exhausted it raises
before key splitting, environment stepping, or any numerical calculation.
Production closures continue to return only scalar/event metrics.

Closure tests now assert exact nonlocal names, explicit `(state, key)`
signatures, absence of captured JAX arrays/state/keys, object identity for only
the permitted static component/config/environment values, and byte-identical
results from repeated calls with the same explicit state and key. This rules
out retained mutable training state and hidden RNG progression.

### Corrected Full Verification

Final authorized job `11d95d07-eb8e-40be-85d5-716ced87fd83` used 4 vCPU and
8192 MiB:

```text
87 passed, 5 skipped
ruff: All checks passed!
pyright: 0 errors, 0 warnings, 0 informations
compileall: exit 0
```

Its test scope is all of `tests/rtrrl_parity`; static scope includes every
changed production, capture, and test file. The five skips are the pre-existing
opt-in Task 6 finite-difference cases. Local execution remained limited to
configuration/provenance RED checks, the small optimizer characterization,
ruff, pyright, compileall, and diff checks—no full RL environment was run.

## Final Coverage Resolution

### Focused RED/GREEN

The remaining review began with three focused tests and no production change.
The RED run failed exactly on the intended omissions:

```text
3 failed
- non-None observation/reward statistics were replaced by the None sentinel
- a future dataclass field was absent from the canonical path set
- DebugStepMetrics lacked value, actor_loss, and entropy
```

The comparator now recursively canonicalizes every dataclass field, mapping,
sequence, and nested pytree leaf. `None` alone becomes the exact `<none>`
sentinel; a non-None statistics tree retains all of its paths and values.
Named-tuple classes are reconstructed rather than converted to plain tuples,
so semantic optimizer-state paths are preserved. A synthetic future field is
therefore visible to the generic path comparison and cannot silently escape.
The focused local GREEN run for these three contract tests passed.

### Complete Observable Coverage

The stable eager `DebugStepMetrics` schema now includes per-environment
`value`, `actor_loss`, and `entropy` in addition to the previously exposed
fields. Production `TrainStepMetrics` remains scalar/event-only.

For every one-environment step, tests independently compare environment
action, model input, sampled action, value target, value, actor loss, entropy,
TD error, TD gradients, direct gradients, incoming traces, carried traces,
mean directions, optimizer updates, fast parameters, slow parameters, and the
complete resulting state.

The real two-environment production call independently compares that same
observable/intermediate set against the complete-path fixture. It also retains
the heterogeneous done/reward/delta and leading environment-axis assertions;
complete state equality is an additional check, not a substitute for any
observable assertion.

### Final Verification

Authorized Batch job `102f1826-2d7d-4ba3-8f8b-a65b55361be3` used 4 vCPU and
8192 MiB and completed successfully:

```text
focused init/step: 12 passed
full rtrrl_parity: 90 passed, 5 skipped
ruff: All checks passed!
pyright: 0 errors, 0 warnings, 0 informations
compileall: exit 0
git diff --check: exit 0
```

The five skips remain the pre-existing opt-in Task 6 directional
finite-difference cases. The oracle fixture was not changed by this final
coverage-only correction; its sole authoritative provenance and checksum are
the values in `### Corrected Fixture Provenance`.
