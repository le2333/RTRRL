# Memo-First Script Registration Design

## Status and Relationship to Existing Specifications

This specification narrows the initial script-integration scope in
`2026-07-20-training-observability-sdk-design.md`. It does not change the
control-plane, Aim, Rerun, or AWS boundaries.

The first release does not register any script from the legacy `rtrrl` project.
Those scripts remain future work because their current training APIs do not
truthfully expose all mandatory training-episode statistics.

## Script Identity

Environment is experiment configuration, not script identity. The memo image
therefore exposes exactly two facility launchers:

- `memo_stream_ac`, initially restricted to the `rtu_rtrl` topology;
- `memo_rtrrl`, initially restricted to the `shared` topology.

One launcher may support multiple environments without creating one descriptor
per environment. The experiment YAML still lists explicit groups, and every
group remains an independent study regardless of whether another group uses the
same launcher or environment.

QRC, TBPTT, independent RTRRL, examples, random-policy evaluation, and legacy
`rtrrl` scripts are not registered in the first release.

## Environment Configuration

The current control-plane `EnvironmentSpec` is too specific to the legacy
RTRRL Hopper configuration. It must become an immutable generic value:

```yaml
environment:
  name: memory_chain
  options:
    length: 75
    max_episode_steps: 1000
```

`name` selects a launcher-owned environment builder. `options` is a JSON object
copied into the concrete configuration and run context. The facility validates
that the value is JSON-compatible; the launcher validates environment-specific
combinations. Environment remains fixed within one explicit group in v1. A
user who wants several environments writes several groups, which are traversed
independently. Shared sampling spaces remain future work.

The two launchers use explicit maps from environment names to existing memo
builders. They do not use dynamic imports from user input. Adding another
qualified environment extends a launcher's map and descriptor; it does not
create a new script identity.

## Image Boundary

Both launchers run only in the memo image and use memo's Python, JAX, Brax, and
lock file. They are never copied into the legacy RTRRL image.

The memo CPU and GPU Dockerfiles must:

- build from a lock-consistent memo environment;
- contain the facility SDK and memo descriptors;
- require the deterministic script-catalog build argument;
- expose the catalog under `org.rtrrl.trainer.scripts.v1`.

The memo GitHub workflow and local image build path generate and pass that
catalog. The existing GPU JAX pin must be reconciled with the memo lock before
the image is accepted.

## Observability Data Flow

### Mandatory training summaries

The existing memo `RecordEpisodeStatistics` path is the source of truth for
training summaries. At each host logging boundary, the launcher reports only
completed training episodes using:

- `returned_episode_returns`;
- `returned_episode_lengths`;
- `returned_episode`;
- the actual algorithm `state.step` as native environment steps.

The derived `(epoch + 1) * num_steps` value is not accepted as the facility
step because it can disagree with actual interactions for non-divisible
configurations. Evaluation metrics remain under `eval/*` and are never
relabeled as training data.

### Complete Rerun episodes

Rerun receives complete evaluation episodes, not transitions from the policy
update scan. Existing evaluation functions are extended to return a fixed-shape
trace alongside their existing scalar result:

- N+1 observations;
- N actions, rewards, terminal flags, and truncation flags;
- optional environment states;
- a valid transition count.

The host truncates the fixed-shape arrays to the first terminal or truncation,
constructs `Episode`, and submits it to the SDK. The SDK remains responsible for
the recording interval and artifact publication.

Environment adapters must preserve termination and truncation separately.
Gymnasium adapters use their native pair. Brax adapters use the environment's
truncation information when present. An environment that cannot distinguish
the two is not added to the descriptor until its adapter has an explicit,
tested rule; a combined `done` value is not silently labeled terminal.

SDK calls and NumPy conversion remain outside JIT. Evaluation trace collection
may add outputs to the evaluation JIT, but must not alter update equations,
training PRNG splits, optimizer state, or training return values.

## Child-Process Bootstrap

The worker gives each child a run-context path through a facility environment
variable. A backend-neutral SDK bootstrap:

1. reads and validates that context;
2. opens the stable Aim run;
3. opens the durable event spool under the artifact directory;
4. creates the Rerun adapter from the context logging policy;
5. installs one process-local `TrainingRun`.

With no facility context variable, bootstrap is a no-op so existing local memo
commands retain their behavior. A present but invalid context is a hard error.
Repeated bootstrap calls in one process are idempotent only for the same context
and fail for conflicting contexts.

## Acceptance

The two launchers are registered only after all of the following pass:

- descriptor tests prove environment is configuration rather than script
  identity and unsupported topologies are rejected;
- host-level tests prove summaries originate from completed training episodes
  and use `state.step`;
- complete-trace tests prove N+1/N lengths and separate final
  terminal/truncation flags;
- bootstrap tests prove facility and local behavior;
- existing memo online-AC golden, parity, evaluation-parity, and JIT-contract
  tests remain unchanged;
- short CLI smoke tests run both launchers;
- memo CPU image smoke loads the embedded catalog and writes Aim/spool output;
- the GPU image is not accepted until its JAX dependency matches the lock.

All other memo algorithms and legacy RTRRL scripts remain absent from the
catalog rather than being registered with partial observability.
