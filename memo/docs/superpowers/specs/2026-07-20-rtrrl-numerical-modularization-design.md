# RTRRL Numerical-Parity Modularization Design

## Goal

Refactor the existing Memorax RTRRL implementation into composable functional
modules while preserving, within calibrated floating-point tolerance, the
AAAI25 LRU-RTRRL algorithm represented by:

`/home/ubuntu/trainer/RTRRL-AAAI25/rtrrl.py`

Memorax becomes the single long-term implementation. The AAAI25 script is an
executable oracle during migration, not a second implementation to maintain.
After parity is established, the legacy-compatible entry point calls the
Memorax implementation.

The resulting framework must support controlled RTRRL ablations and component
replacement without changing the strict baseline accidentally.

## Scope

### Strict baseline

The strict profile covers the AAAI25 LRU path only:

- shared recurrent LRU torso
- continuous or discrete actor head
- value head
- meta-RL input
- exact historical initialization and PRNG consumption
- environment-first online step ordering
- LRU online credit state and custom-gradient behavior
- actor, critic, and recurrent eligibility traces
- episodic emphasis or average-reward update
- entropy and optional direct regularization gradients
- grouped optimizer update
- slow recurrent target
- evaluation behavior

CTRNN, `rnn_model=None`, CTRNN wiring, and CTRNN RFLO/RTRL branches are not
implemented in the strict profile.

### Memorax extensions retained

The modular implementation retains the new branches already introduced around
the Memorax RTRRL implementation:

- LRU or RTU backbone
- encoded or raw observation/action/reward input
- configurable LRU output width
- bounded or unbounded actor
- environment-facing action clipping
- frozen LRU input gain
- incoming or fresh eligibility-trace timing
- summed or mean action log-probability reduction
- auxiliary observation/reward prediction head
- shared or independent actor/critic recurrent paths
- observation and reward normalization profiles

Extensions are explicit component selections. They do not add runtime boolean
branches inside the strict numerical kernel.

## Development Strategy

Use an incremental replacement strategy inside Memorax.

This is not a direct decomposition of the legacy monolithic script and not an
independent greenfield reimplementation. The legacy script first generates
characterization data. Candidate Memorax modules then replace one behavior at a
time only after parity tests pass.

The migration sequence is:

1. Capture deterministic oracle inputs and outputs from AAAI25.
2. Implement or adapt one Memorax module.
3. Compare the old inline operation and candidate module on identical data.
4. Keep the module only after the red-to-green parity cycle succeeds.
5. Compose validated modules into initialization and a complete online step.
6. Redirect the legacy-compatible entry point to the composed Memorax program.
7. Add ablation branches through the same component interfaces.

There must be no permanent duplicate `oracle_rtrrl` mathematical core outside
Memorax.

## Functional Modules and Closure Composition

Algorithm components are predefined pure functions or immutable callable
objects. A construction function selects components from static configuration
and returns closed-over lifecycle functions:

- `init_fn(key) -> state`
- `step_fn(state, key) -> (state, metrics)`
- `train_epoch_fn(key, state, num_steps) -> (state, summary)`
- `evaluate_fn(key, state, num_steps) -> evaluation`

Closures capture only static configuration and callable components. Dynamic
state remains explicit:

- parameters and slow parameters
- optimizer state
- environment state
- recurrent carry and online credit state
- eligibility traces
- previous action and value
- running statistics
- episodic emphasis or average reward
- PRNG keys

No closure may retain mutable training state or consume hidden randomness.
The fully assembled closure is the JIT boundary, so algorithm selection occurs
at construction time rather than inside compiled steps.

## Module Boundaries

### Configuration compatibility

Consumes existing YAML, CLI, nested optimizer configuration, and environment
configuration. Produces a frozen normalized runtime configuration.

Responsibilities:

- preserve accepted scientific-notation strings
- preserve existing defaults
- map absent or explicit `rnn_model: lru` to the strict LRU component
- reject explicit CTRNN or `None` selections with a clear unsupported error
- retain legacy no-op fields as accepted, diagnosed fields
- reject unknown fields rather than silently dropping them
- preserve HPO dotted paths

It does not implement algorithm formulas.

### Network and online credit

Consumes parameters, inputs, carry, and online credit state. Produces policy
distribution, value, next carry, and next credit state.

The strict LRU component owns:

- AAAI25 parameter names, shapes, dtypes, and initializers
- complex recurrence and output projection
- custom-gradient/online-credit behavior
- reset behavior

RTU and other Memorax backbones implement the same external protocol but may
own different state layouts.

This layer does not access the environment, optimizer, or logger.

### Update rules

Pure modules implement:

- TD target and TD error
- accumulated and Dutch critic traces
- actor and recurrent accumulated traces
- incoming/fresh trace selection
- direct entropy, slow-state, action-magnitude, and prediction gradients
- grouped optimizer transformation
- slow-target update
- episodic emphasis or average-reward update

Every rule accepts and returns explicit values. It does not sample actions or
step environments.

### Online step state machine

The strict state machine preserves the AAAI25 phase:

1. Apply the previously persisted action to the environment.
2. Update running statistics according to the historical profile.
3. Reset recurrent and trace state at boundaries.
4. Construct the current `[observation, previous action, reward]` input.
5. Run the recurrent actor-critic and sample the next action.
6. Compute the TD error using the persisted previous value.
7. Compute recurrent and head gradients.
8. Apply the selected incoming/fresh traces and direct gradients.
9. Apply optimizer and slow-target updates.
10. Persist next action, value, carry, traces, emphasis, and RNG state.

PRNG splitting and initialization pre-sampling are part of this state machine.
Environment adapters may translate APIs but may not reorder these operations.

### Program and training orchestration

The program layer performs scan/JIT composition, epoch aggregation, evaluation,
early stopping, and lifecycle schema declaration.

Training returns only scalar summaries and required episode information.
One-to-three-step debug programs may return full gradients, traces, parameters,
and intermediate states. Production training must not stack those large trees
across an epoch.

### Logging compatibility

The compatibility layer preserves previously recorded keys and aggregation:

- `steps`
- `mean_reward`
- `num_episodes`
- `mean_delta`
- `mean_r_bar`
- `mean_v`
- `total_td_loss`
- `actor_loss`
- `critic_loss`
- `entropy`
- `v_targ`
- optional `magnitude_loss`
- `lr/td`
- `lr/rnn`
- `norms/*`
- `eval/rewards`
- `eval/best_eval_reward`
- `env/video`

Logger step semantics remain compatible. Existing configuration fields that
historically had no effect, such as unused model saving, remain accepted but
must not claim a new effect silently.

New diagnostic metrics may be added, but historical metrics may not be renamed,
removed, or change aggregation without a versioned schema.

## Profiles

### `aaai25_strict_lru`

This is the numerical baseline and permanent regression gate. It fixes all
historical semantics required for parity, including network initialization,
policy reduction, initialization pre-sampling, key order, environment-first
phase, trace order, normalization profile, and optimizer behavior.

Only calibrated backend floating-point tolerance is allowed.

### `memo_experimental`

This profile selects Memorax extensions through explicit components. Every
extension records its effective component choices in configuration and logs.

Experimental components are tested against algebraic invariants and their
declared behavior. They are not required to match AAAI25 when an ablation
intentionally changes the algorithm.

## Oracle and Test Data

AAAI25 and Memorax use different JAX/JAXLIB and environment versions. They are
not imported into one Python process.

The AAAI25 environment generates versioned NPZ fixtures containing:

- manifest with source commit and package versions
- fixed PRNG seeds and split protocol
- explicit parameter trees
- mock observations, actions, rewards, done flags, and carry
- forward distributions and values
- online credit state
- gradients
- traces
- optimizer states and updates
- initialization state
- complete one-step and short-scan states

Memorax consumes these fixtures in its own environment. Exact comparison is
used for discrete values, shapes, dtypes, tree paths, and values known to be
stable. Float32 comparisons start narrow and may use a documented ULP
whitelist only after measuring backend-specific differences.

Golden snapshots generated solely from Memorax are regression tests, not proof
of AAAI25 parity.

## TDD Order

Tests are implemented and made green in this order:

1. Configuration translation and rejection behavior.
2. Dense/MLP parameters, initialization, forward, and VJP.
3. Actor and critic outputs, sampling, log-probability, and entropy.
4. LRU recurrence and projection.
5. LRU two-step online credit and finite-difference direction.
6. Historical initialization and PRNG splitting.
7. Mock environment phase and terminal reset.
8. TD, emphasis, and average-reward rules.
9. Accumulated and Dutch traces.
10. Direct gradients and parameter-domain routing.
11. Optimizer schedules, clipping, moments, signs, and slow targets.
12. Eager complete online step.
13. JIT complete online step with calibrated ULP policy.
14. Two-environment reduction order.
15. Three-step scan with terminal transition.
16. Evaluation behavior.
17. Legacy YAML and logging contract.
18. RTU and other Memorax extension invariants.

Each production change follows red, green, then refactor. A test must fail for
the intended semantic difference before implementation changes are accepted.

## Resource Strategy

The local host is not expanded. Its 911 MiB RAM was insufficient for reliable
JAX teardown in the existing small RTRRL snapshot test.

Run locally only:

- static analysis
- configuration tests
- pure NumPy oracle helpers
- tests demonstrated to remain safely below the host limit

Use an interactive EC2 instance with at least 4 vCPUs and 8 GiB RAM for:

- JAX eager and JIT parity
- finite differences
- initialization and one-to-three-step full-state comparisons
- compilation and peak-memory measurement

Use AWS Batch only after module and complete-step parity:

- Brax integration smoke tests
- multi-seed behavior checks
- long training
- HPO and interruption/resume validation

Complete reinforcement-learning environments are not run on the local host.

## Failure Handling

- Parameter path, shape, dtype, discrete, or tree mismatches fail immediately.
- Unexplained systematic float differences are algorithm failures, not widened
  tolerances.
- ULP bounds must name exact leaves and record JAX/JAXLIB/backend versions.
- Non-finite carry, credit, trace, gradient, or update values fail the test.
- Unknown legacy configuration fields fail with the field path.
- Unsupported removed branches fail during construction, before JIT.
- EC2 or Batch resource exhaustion stops the test and produces a resource
  recommendation; it does not trigger local expansion.

## Acceptance Criteria

The refactor is complete when:

- the strict LRU profile matches AAAI25 fixtures for forward, online credit,
  initialization, complete eager step, JIT step, and short scan
- all differences are within reviewed exact, relative/absolute, or leaf-specific
  ULP bounds
- existing YAML files load without edits
- explicit removed branches fail clearly
- historical training and evaluation metrics are still recorded with unchanged
  aggregation and step semantics
- the legacy-compatible entry point delegates to Memorax
- no duplicate maintained AAAI25 mathematical core remains outside Memorax
- production epoch scans do not stack full debug state
- existing Memorax RTRRL extension tests remain green
- RTU, encoder, actor, trace timing, prediction, and independent-path branches
  compose through documented interfaces
- local resource constraints and no-local-full-RL requirements are respected

## Non-Goals

- Bitwise equality across arbitrary JAX, JAXLIB, CPU, and GPU combinations.
- Reimplementation of AAAI25 CTRNN or no-RNN branches.
- Silent correction of historical normalization or logging bugs inside the
  strict profile.
- Running long Brax training before module and complete-step parity.
- Committing, pushing, or replacing existing experiment artifacts as part of
  the design phase.
