# Issue 27: Bounded sampled-episode reporting

## Goal

Fix RTRRL reporting so a long reporting interval does not retain an
interval-sized JAX scan output or random-key array. A configured sample point
must produce exactly one complete training episode for the corresponding
environment stream, even when the episode crosses scan chunks or the final
training budget.

This change is confined to execution and observability. RTRRL's objectives,
eligibility traces, optimizer groups, update order, defaults, and AAAI numerical
semantics do not change.

## Existing failure

`Runtime` currently calls `program.train(key, state, epoch_steps)` once per
reporting epoch. RTRRL creates one random key per scan row and returns the
stacked `StepMetrics` tree for that entire call. Runtime only cuts episodes and
Rerun only filters them after the device has materialized the full epoch.

Consequently, the peak allocation grows with `epoch_steps`. Chunk boundaries
also have episode semantics even though they are only scheduling boundaries:
an episode prefix from a previous chunk is unavailable to a later sample.

## Definitions

- An **environment step** is one transition from one environment stream.
- A vectorized scan row advances every stream once and therefore consumes
  `num_envs` environment steps.
- A **sample point** `t` is a boundary in the global environment-step axis.
  It selects stream `t % num_envs`.
- The selected trajectory is the episode containing that boundary. If the
  preceding transition ended an episode at the boundary, the selected episode
  is the next episode after reset.
- A **chunk** is only a bounded transport batch with shape
  `[scan_time, num_envs, ...]`. It is not an episode boundary.

For a sample `t`, Runtime emits the trajectory from the selected stream's most
recent reset at or before the selected episode through that episode's first
`done`. A sample at a `done` boundary selects the next episode, as agreed for
Issue 27.

The periodic schedule starts at the configured interval, not at zero. A 50M
run with an interval of 10M therefore requests 10M, 20M, 30M, 40M, and 50M.

## Chosen architecture

### 1. Bounded training chunks

Runtime retains `epoch_steps` as the evaluation and reporting schedule, but it
does not use it as one train-call allocation size. It divides each epoch into
train calls no larger than:

```text
maximum_episode_length * num_envs
```

The final call in an epoch may be shorter. The algorithm still performs the
same train steps in the same order. Only the grouping of those steps into JIT
invocations changes.

RTRRL may continue to implement a train call with `jax.lax.scan`; its key array
and `StepMetrics` output are now bounded by maximum episode length rather than
reporting interval length.

### 2. Runtime-owned episode tracker

Runtime consumes each bounded chunk in chronological order. It owns one current
episode buffer per environment stream. Each buffer contains the transitions
since that stream's most recent reset, along with zero or more sample points
that selected that episode.

For every transition Runtime:

1. applies any sample point at the boundary before that transition;
2. appends the transition to the corresponding stream buffer;
3. finalizes the buffer when the transition is `done`;
4. emits ordinary episode statistics;
5. emits a sampled trajectory for each sample point attached to the episode;
6. clears and reuses the same stream buffer for the next episode.

An unfinished buffer survives the end of a chunk. One environment may complete
multiple episodes inside one chunk; its slot is finalized and reused each time.
Chunk boundaries never truncate or fabricate an episode.

The tracker stores episode data once. If multiple sample points select the same
episode, they reference the same completed data and produce one sampled record
per requested point. Runtime does not allocate a maximum-sized trajectory for
every sample before training starts.

The retained state is bounded by the current chunk, one current episode per
stream, the maximum episode length, and the sample identifiers that can fall
within those current episodes. It does not grow with `epoch_steps`, total
training steps, or the number of already-emitted samples.

### 3. Budget-end continuation

Earlier sampled episodes complete naturally while later training chunks run.
At the final training boundary, Runtime applies the final sample point. If its
selected episode is incomplete, Runtime forks the final algorithm state and
uses a fourth program operation:

```text
interact(key, state) -> (advanced_state, one_step_metrics)
```

`interact` performs one vectorized behavior-policy interaction from the current
environment and recurrent state. It may advance environment state, the current
timestep, and recurrent carry. It does not update network parameters,
optimizer state, eligibility traces, normalization statistics, update counters,
or the training-step budget.

Runtime calls this operation only until the selected episode reaches `done`.
The original final training state remains unchanged and evaluation does not
participate. Transitions produced by this continuation are marked
`post_budget=true`; transitions produced by normal training are marked false.

`interact` belongs to the algorithm program contract because the algorithm owns
behavior-policy action selection and recurrent-state advancement. Runtime owns
when it is scheduled. RTRRL and StreamAC provide the same operation so the
runtime contract remains algorithm-independent.

### 4. Separate scalar episodes from sampled trajectories

Runtime exposes two reporting events:

- a completed episode, used to compute scalar statistics for Aim and the
  mandatory JSONL metrics artifact;
- a sampled trajectory, used by Rerun and containing the requested sample point
  plus a per-transition post-budget mask.

Conceptually:

```text
bounded StepMetrics chunk
          |
          v
Runtime episode tracker
          |-- every completed episode --> statistics --> Aim + metrics.jsonl
          |
          `-- selected complete episode --> sampled trajectory --> Rerun RRD
```

Rerun no longer infers sampling from episode spans. It serializes only the
sampled trajectories Runtime sends it. The RRD metadata includes the sample
point, stream, episode span, and run identity. Its sequence data includes the
post-budget mask.

Evaluation remains a separate branch derived from current trained parameters,
with a fresh environment and recurrent state. At each configured epoch boundary
Runtime cuts complete evaluation episodes and reports
`eval/episode/return` and the existing evaluation statistics to Aim. Evaluation
does not fill a sampled training episode.

### 5. Recording projection

The built algorithm's observation schema distinguishes lightweight fields
needed for episode statistics from full trajectory fields. Reward, `done`,
termination, and configured diagnostic series remain available for scalar
reporting. Observation, next observation, and action are requested when sampled
trajectory reporting is enabled.

Entry projects the run document into both concerns:

- `rerun.every_steps` becomes the explicit sample-point tuple in
  `RuntimeConfig`;
- the environment episode limit becomes Runtime's chunk bound;
- Rerun-enabled builds request trajectory fields;
- runs without Rerun have no sample points and create no RRD artifacts.

Runtime and Memorax do not infer whether a run is an HPO run. Infra already
controls this through the run configuration: fixed reference configurations
enable Rerun, while HPO configurations omit it.

## Contracts

The implementation will extend the existing contracts without introducing a
logger dependency into an algorithm:

- `RuntimeConfig` gains the maximum episode length and an explicit ordered tuple
  of sample points.
- `Program` gains the one-step, no-learning `interact` operation.
- the Runtime reporter destination gains a sampled-trajectory event separate
  from ordinary completed episodes.
- a sampled-trajectory value contains a completed `Episode`, its sample point,
  and a boolean post-budget value for every transition.
- `RerunSink` consumes sampled trajectories and performs serialization only.

The exact helper and file names may follow existing runtime module conventions,
but these semantic boundaries are fixed.

## Failure behavior

The implementation does not add recovery or fallback paths. Invalid schedules,
ragged vectorized step budgets, or an episode exceeding the configured maximum
raise directly. An unfinished final sample that cannot reach `done` within the
environment's declared maximum is an error rather than a truncated artifact.

No callback performs Python or Rerun I/O from inside a JAX scan. No logger is
called by an algorithm graph.

## Testing

Tests are organized by contract and behavior:

1. A deterministic multi-stream runtime program contains multiple episode ends
   per chunk, an episode spanning chunks, and a sample exactly at a `done`
   boundary. It proves that every sample starts at the correct reset, contains
   the selected boundary, ends at the first subsequent `done`, and selects the
   next episode at a done boundary.
2. A final-budget test proves that Runtime switches to `interact`, performs no
   additional train update, leaves the final training state unchanged, and
   marks only the continuation tail post-budget.
3. A behavioral memory-shape test uses a fake program that refuses train calls
   larger than the configured chunk bound. A long epoch must complete through
   multiple bounded calls. This proves the scheduling shape without inspecting
   source text.
4. Reporter and Rerun tests prove that scalar episodes and sampled trajectories
   take separate paths and that sample/post-budget metadata is serialized.
5. Entry tests prove exact periodic sample expansion, no step-zero sample, and
   no Rerun artifacts when the run configuration omits Rerun.
6. Existing RTRRL behavior, numerical parity, paper parity, and assembly tests
   remain green.

After local verification, the paid remote acceptance remains a separate,
explicitly authorized action: a masked-Hopper long-interval CPU smoke and a
fixed run producing five RRD trajectories and five evaluation records.

## Rejected alternatives

### Device-side trajectory ring buffers

A JAX scan could carry fixed trajectory buffers and conditional sample state.
This would couple generic observability semantics to every algorithm graph,
complicate dynamic episode completion, and still require host-side RRD I/O.
The bounded host tracker gives the required memory bound with clearer ownership.

### One host dispatch per training step

Calling a compiled train step from Python for every vectorized row avoids large
outputs but creates millions of dispatches in long runs. Bounded scans retain
useful compilation and device batching without unbounded reporting memory.

### Rerun callbacks inside the scan

JAX callbacks would introduce ordered host side effects into the training graph,
block device execution, and make cross-chunk prefix ownership obscure. Rerun is
therefore strictly downstream of Runtime's completed sampled trajectory.
