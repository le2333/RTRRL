# R2D2 with Full BPTT and TBPTT

## Goal

Replace the proposed minimal DRQN with a current-platform R2D2 algorithm that
uses the same LRU or RTU recurrent cells as RTRRL and supports two explicit
sequence-gradient modes:

- standard R2D2-style burn-in followed by a fixed TBPTT learning unroll; and
- a full-episode BPTT extension with no gradient boundary inside the episode.

This work is confined to recurrent Q-learning. It does not change RTRL,
recurrent sensitivity implementations, StreamAC, or RTRRL.

## References and migration policy

The implementation has three references with different jobs:

1. Hausknecht and Stone's DRQN paper defines recurrent DQN, sequence replay,
   and the distinction between episode-sequential and random-window replay.
2. DeepMind Acme's JAX R2D2 learner is the primary executable reference for
   sequence layout, burn-in, online/target unroll, Double-Q selection, n-step
   TD loss, importance weighting, sequence priorities, optimizer order, and
   target updates.
3. Google SEED RL's R2D2 learner is a second implementation used to check
   recurrent input alignment, shifted current/next Q values, burn-in, and
   priority calculation.

The existing `memorax/algorithms/r2d2.py` is an API migration source, not the
numerical oracle. It contains useful Flax, Flashbax, and environment wiring but
uses the retired algorithm/runtime contract and must not be preserved merely
because it already runs.

The port will retain the algorithmic invariants of the references while using
Memorax's existing environment, network, buffer, parameter, observation, and
runtime contracts. Reference behavior will be captured in tests before the old
implementation is replaced.

References:

- <https://arxiv.org/abs/1507.06527>
- <https://openreview.net/forum?id=r1lyTjAqYX>
- <https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/r2d2/learning.py>
- <https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/r2d2/builder.py>
- <https://github.com/google-research/seed_rl/blob/master/agents/r2d2/learner.py>

## Algorithm identity

The public algorithm and entry are named `r2d2`, not `drqn`. The learner keeps
the R2D2 mechanisms that distinguish it from a minimal DRQN:

- recurrent Q network;
- Double Q-learning;
- n-step targets;
- online and target networks;
- prioritized sequence replay;
- importance-sampling correction;
- stored actor recurrent state and burn-in for the TBPTT branch;
- sequence priority formed from a weighted maximum and mean absolute TD error;
- optional dueling Q head; and
- optional signed-hyperbolic value transformation.

The current worker and vectorized environments supply execution parallelism.
The algorithm will not create a separate distributed actor system or import
Acme, Reverb, TensorFlow, or RLax.

## Boundaries

### Algorithm graph

`memorax/algorithms/r2d2.py` owns the complete R2D2 graph:

- parameter declarations;
- recurrent input construction;
- Q-network topology;
- acting carry;
- epsilon-greedy action selection;
- environment interaction;
- replay insertion and sampling;
- burn-in;
- Full BPTT and TBPTT loss graphs;
- Double-Q n-step targets;
- importance-weighted optimization;
- target-network updates;
- replay-priority updates;
- training and evaluation loops; and
- R2D2-specific observations and metrics.

The Full BPTT and TBPTT update paths remain private functions or methods in
this file for this development round. No common temporal learner or generic
differentiation framework is introduced.

### Reused components

R2D2 reuses:

- `LRUCell` through `Memoroid` and `RTUCell` through `RNN` via the existing
  pure `backbone()` constructor;
- `Sequence`, `FFN`, normalization/activation components, and `Readout`;
- the existing scalar discrete Q head for the non-dueling branch;
- existing initializers and Optax-backed optimizer components;
- the prioritized episode buffer where its behavior satisfies the new tests;
- environment streams, interaction termination semantics, `StepMetrics`,
  `ObservationSchema`, assembly, runtime, reporter, and entry helpers.

R2D2 does not use `BACKBONE_FAMILY`, `RecurrentDifferentiation`, or the current
class named `TruncatedBPTT`. Those APIs describe online step differentiation
and cannot express a replay sequence or an arbitrary chunk length.

### Private semantic components

Components needed only to express the R2D2 graph stay beside the algorithm:

- the recurrent input encoder, which combines the current observation with
  the previous action, previous reward, and episode boundary;
- the dueling Q readout; and
- signed-hyperbolic transform and inverse-transform operations.

They are not registered as cross-algorithm public components in this round.

## Network graph

Each sampled or acting timestep is represented as:

```text
(observation_t, previous_action_t, previous_reward_t, episode_start_t)
```

The previous discrete action is one-hot encoded. The episode-start flag resets
the recurrent carry before the timestep is consumed. The graph is:

```text
R2D2 input encoder
  -> feature projection
  -> normalization
  -> activation
  -> selected LRU or RTU
  -> selected linear or dueling Q readout
```

LRU and RTU expose the same ordinary differentiable forward contract. They do
not declare a gradient method. One R2D2 sequence-update implementation serves
both backbones.

The environment action space must be finite and discrete. This is a graph
precondition rather than a runtime fallback.

## Replay record and sequence alignment

Replay stores the executed history, not a counterfactual history under the
current policy. Each transition includes at least:

```text
observation_t
previous_action_t
previous_reward_t
episode_start_t
action_t
reward_t
next_observation_t
done_t
terminal_t
actor_recurrent_state_t
```

`done` ends the environment episode and resets recurrence. `terminal` alone
removes bootstrap value. A time-limit truncation is therefore
`done=True, terminal=False`.

A learner sample contains `T + 1` recurrent inputs. Online and target networks
unroll over the same aligned history:

```text
online Q for loss:       online_q[0:T]
online selector action:  argmax(online_q[1:T+1])
target bootstrap value:  target_q[1:T+1, selector_action]
executed replay action:  action[0:T]
reward/terminal:         reward[0:T], terminal[0:T]
```

The target network is never initialized by independently unrolling only the
`next_observation` sequence. That loses the history preceding each next state.
Targets are stopped before the online loss is differentiated.

## Standard TBPTT branch

The structural selection is `learning.kind: tbptt`. A replay item has fixed
length:

```text
burn_in_length + unroll_length + 1
```

matching the Acme R2D2 sequence convention. The replay item carries the actor
recurrent state at its first step. Learning proceeds as follows:

1. initialize online and target carry from the stored actor state;
2. unroll both networks across the burn-in prefix;
3. stop gradients through both warmed carries;
4. unroll the remaining sequence with online and target parameters;
5. form Double-Q n-step targets on the aligned outputs;
6. compute the importance-weighted sequence loss;
7. perform one optimizer update; and
8. update target parameters and sampled sequence priorities.

The burn-in prefix changes the numerical recurrent state but contributes no TD
loss and receives no gradient. `unroll_length` is the maximum recurrent credit
horizon of this branch.

## Full BPTT branch

The structural selection is `learning.kind: full_bptt`. Sampling begins at an
episode boundary. Each sample is padded to the environment's static
`episode_length`, plus the final recurrent input needed for next-state Q
values. A valid mask covers the real transitions through the first `done`.

The recurrent carry begins from the network's initial state. Stored actor carry
and burn-in are not used. Online parameters are unrolled once across the full
valid episode inside one differentiated loss. There is no `stop_gradient` on
an online recurrent carry before the episode ends. Padding and steps after the
first ending contribute neither loss nor priority. Near an episode or padded
sequence boundary, the final `n_step - 1` valid starts use progressively
shorter returns, as in the SEED R2D2 reference, rather than being silently
discarded.

The Full BPTT branch retains the same Double-Q target, n-step return, value
transform, importance weights, optimizer, target update, and max/mean sequence
priority as the TBPTT branch. Full BPTT is an explicit extension to R2D2, not a
claim about the standard published R2D2 training schedule.

## TD loss and priority

For each valid starting step, the online network selects the replayed action's
value. The current online network selects the bootstrap action and the target
network evaluates it. The n-step target stops on `terminal`, not every `done`.
The target is transformed when the configured value transform is enabled.

The per-step loss is half squared transformed TD error. A sampled sequence's
loss is reduced across valid time and weighted by its normalized replay
importance weight. Its new priority is:

```text
max_priority_weight * max(abs(td_error))
+ (1 - max_priority_weight) * mean(abs(td_error))
```

where both reductions ignore invalid padded positions. A small positive replay
epsilon is added only when storing the resulting priority.

## State and execution

`R2D2State` contains:

```text
environment step and learner update counters
current timestep and environment state
acting recurrent carry
online parameters
target parameters
optimizer state
prioritized replay state
```

Training interaction uses epsilon-greedy actions and writes replay. Evaluation
starts from a fresh environment and fresh recurrent carry, uses the evaluation
epsilon, and does not change parameters, optimizer state, replay, priorities,
or learner counters.

The algorithm exposes the current `Program(init, train, evaluate)` contract.
`train` returns `(state, StepMetrics)` and `evaluate` returns evaluation
observations in the same form expected by `Runtime`. Logging remains outside
the algorithm; the graph only returns declared observations.

## Parameter structure

Structure is fixed within an experiment's trials. The initial public shape is:

```yaml
backbone:
  kind: [lru]                 # lru | rtu
  lru:
    feature_dim: [128]
    hidden_dim: [128]

head:
  kind: [dueling]             # dueling | linear

learning:
  kind: [tbptt]               # tbptt | full_bptt
  tbptt:
    burn_in_length: [40]
    unroll_length: [80]

optimizer:
  kind: [adam]
  adam:
    lr: [0.001]

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
    kind: [signed_hyperbolic] # signed_hyperbolic | identity

gamma: [0.997]

exploration:
  epsilon_start: [0.4]
  epsilon_end: [0.01]
  epsilon_decay_steps: [1000000]
  evaluation_epsilon: [0.001]
```

RTU declares the same feature and hidden dimensions under its own branch. The
Full BPTT branch has no burn-in or unroll parameter subtree; its static length
comes from the configured environment episode length. Parameters are processed
by the graph or component that uses them.

## Tests and evidence

Tests are organized by the behavior they establish rather than by every helper
function.

### Reference-contract tests

- reproduce Acme/SEED's `T + 1` current/next Q alignment on hand-written data;
- reproduce an independently calculated Double-Q n-step target;
- reproduce max/mean sequence priority and importance weighting;
- prove burn-in changes carry but contributes no gradient; and
- prove terminal cuts bootstrap while time-limit truncation does not.

### Gradient tests

- Full BPTT gradient equals `jax.grad` of a direct explicit full unroll;
- TBPTT carry values and Q outputs equal an untruncated forward unroll;
- TBPTT blocks gradient across its configured boundary;
- when the TBPTT horizon covers the complete valid episode and burn-in is zero,
  its gradient equals Full BPTT; and
- padding has zero loss, zero priority contribution, and zero gradient.

### Component and graph tests

- LRU and RTU both build under the pure recurrent forward contract;
- linear and dueling heads produce one Q value per discrete action;
- LRU/RTU times Full/TBPTT all initialize and complete an update;
- recurrent parameters receive finite gradients; and
- the R2D2 graph rejects no valid catalog-expanded parameter tree.

### Algorithm and platform tests

- a deterministic tiny partially observable discrete environment supplies a
  real end-to-end learning smoke;
- replay sampling begins at a valid boundary for Full BPTT and obeys the fixed
  burn-in/unroll layout for TBPTT;
- Full BPTT samples only completed episodes, never a partially collected
  episode that merely occupies enough buffer positions;
- target parameters update at the declared learner period;
- evaluation leaves all training state unchanged;
- entry declarations, catalog output, experiment overrides, assembly, runtime,
  and worker execution agree; and
- focused tests, the complete local suite, and Ruff pass before remote image
  verification is requested.

The full local suite is run only on the development checkout, in accordance
with `AGENTS.md`.

## Migration sequence

1. Freeze the reference math and sequence layout in tests independent of the
   legacy class.
2. Establish a pure LRU/RTU R2D2 Q-network and its two readouts.
3. Establish prioritized sequence replay records and the two sample layouts.
4. Port the canonical TBPTT R2D2 update against the reference tests.
5. Add the Full BPTT update and its direct-gradient oracle.
6. Connect acting, training, evaluation, observations, and metrics.
7. Register parameters, add the entry and experiment template, and verify the
   complete platform path.
8. Remove retired R2D2 exports or compatibility code only when no current
   consumer or test depends on them.

## Non-goals

This development does not:

- modify RTRL, RFLO, StreamAC, or RTRRL;
- make BPTT a kernel-level differentiation component;
- create a generic online/sequence learner interface;
- add a new distributed actor service;
- introduce Acme, Reverb, TensorFlow, or RLax dependencies;
- support continuous actions; or
- claim that the new Full BPTT branch is part of published standard R2D2.

No defensive fallback or broad exception handling is added. Invalid structural
configuration is rejected during normal parameter expansion or graph assembly;
numerical failures remain visible to the worker and runtime.
