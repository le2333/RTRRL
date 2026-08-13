# Semantic Framework Refactor Plan

**Status:** Planning complete for the current repository. Algorithm and facility
boundaries, wire ownership, metrics/artifacts, failure behaviour, and test
migration are specified below. The real Batch round executor is not present in
the current source and is therefore a separately identified implementation
capability, not an assumed prerequisite of the semantic refactor.

**Goal:** Make every computation-graph layer and every facility block answer to
one explicit domain meaning, while preserving StreamAC and RTRRL numerical
behaviour and making the default test loop smaller and faster.

**Execution:** Tasks 1-11 are implemented locally as of 2026-08-13. StreamAC and RTRRL share only
the proven interaction components and expose closed Programs to Runtime;
algorithm topology and private subgraphs remain algorithm-owned. Observability
owns metric naming, episode reduction, fan-out, and local backend artifacts.
The version-8 deployment contract separates Catalog, Worker-envelope, and Entry
projections. Worker now owns only serial process supervision, isolated scratch,
artifact upload, and completion results; scoring policy, metric reduction, and
HPO feedback belong to Infra. Local and Batch round executors are implemented;
the pushed image and real Batch acceptance remain the final remote gates.

Before Task 4, the current version-7 Worker path was smoke-tested on 2026-08-12
with a real StreamAC RTU+RTRL child, Moto S3, local Aim, metrics scoring, and a
four-step Brax episode. The recurrent critic eligibility-trace norm reached
both Aim and the Worker score. Task 8 has since replaced that legacy scoring
path with version-8 artifact transport.

## Governing rules

- Split algorithm code by algorithmic meaning, not file size.
- Distinguish algorithm-private subgraphs, proven shared components, and private
  implementation helpers.
- A graph component must own a parameter domain, state transition, mathematical
  objective, temporal boundary, or coupling/routing invariant.
- Use the replacement test: a parent should replace a semantic child by changing
  construction, without reading or synchronising with the child's internals.
- The algorithm owns its graph topology. Entry is a composition root and owns no
  algorithm semantics.
- Domain ownership and process placement are separate. Code can execute beside
  Worker data without becoming Worker domain logic.
- Refactors that promise numerical preservation require numerical tests, not
  source inspection.

## Agreed target responsibilities

### Shared components

Reusable mathematical and interaction capabilities such as backbones, heads,
credit assignment, update rules, normalizers, and environment adapters. A
component declares and consumes its own parameter subtree.

### Core

The complete per-transition agent computation: role-specific objectives,
actor/critic/torso coupling, gradient routing, traces, update grouping, and
parameter-state transitions. Core exposes the single-transition operations the
algorithm loop needs and hides all child internals.

### Algorithm

`StreamAC` or `RTRRL` owns Core, environment interaction, normalization, reset
order, train/evaluation step semantics, concrete scan loops, algorithm graph
construction, the composed parameter tree, and the available reading schema.
The stable Runtime-facing surface is initialization, training for a requested
number of steps, and evaluation for a requested number of steps.

### Runtime

Owns run scheduling rather than RL semantics: total and epoch budgets,
train/evaluation invocation, random-key scheduling, compilation, episode
cutting, and delivery of observation events to logging.

### Logging/observability

Owns scalar, episode, and trajectory event contracts; aggregation that is
intrinsic to those observation events; backend adapters; and resource
lifecycle. Backends implement only the capabilities they consume.

### Entry

The composition root for one executable entry. It reads a resolved run
configuration, obtains the algorithm from its algorithm-side builder, builds
configured loggers, builds Runtime, and starts the run. It may re-export
algorithm metadata for catalog discovery but may not derive that metadata or
define graph topology.

### Worker

Owns execution supervision: selecting an entry command, starting runs, managing
scratch space, heartbeat and failure state, and collecting/uploading declared
artifacts. It does not interpret algorithm structure or HPO meaning.

### Infra/HPO and scoring

Infra owns experiment parsing, catalog masking, parameter sampling, trial
ranking, and HPO feedback. Scoring is a distinct domain service that turns a
metric artifact plus a score policy into one objective; its exact execution
location and artifact contract are pending the facility audit below.

## Confirmed problems in the current implementation

### Algorithm and entry boundaries

- `entries/stream_ac.py` declares the composed algorithm parameter tree and
  builds the actor/critic `backbone + head` topology.
- Entry parses optimizer, credit, head, backbone, and normalization component
  subtrees that should be consumed by the algorithm-side graph builder.
- Entry imports `PLACES` and derives training metric names from graph-internal
  position groups.
- The new RTRRL implementation is not the public `RTRRL` export and has no entry,
  catalog item, or algorithm-side parameter builder.
- The public algorithm exports still advertise `EvalSummary`, which no longer
  exists, and this breaks `IndependentRTRRL` imports.

### Shared and private graph boundaries

- StreamAC and RTRRL contain identical private copies of `Environment` and
  `Normalization`; their shared semantics are already proven by two users.
- StreamAC `Network` is a valid semantic online parameter block, but its name
  hides that it owns recurrence, credit, traces, rule state, and a complete
  online update.
- RTRRL `Network` fails the replacement test: Torso reads `block.credit`, roles
  read `block.module`, and Core reads role `.block` internals. Its forward modes,
  trace recurrence, and metric grouping do not form one closed contract.
- `RTRRL.Core.update_parameters` is long but owns one coherent coupling
  operation. It should be decomposed into private functions, not additional
  graph layers.

### Runtime, logging, Worker, and scoring

- Runtime has a meaningful scheduling role, but `drive()` is a test-only wrapper
  around `AgentProgram + Runtime` and adds no semantic layer.
- `AgentProgram` is defined under the algorithm contract although Runtime owns
  and consumes this execution contract.
- Reporter currently loads deployment configuration, derives episode
  statistics, fans out events, and constructs sinks.
- One wide Sink protocol forces scalar-only and episode-only backends to carry
  empty methods.
- Rerun logging uploads through `worker.objects`, coupling logging to Worker/S3.
- Worker computes the HPO score, including window, reduction, non-finite policy,
  and optimization direction, even though these are experiment/HPO meanings.
- Deployment contracts such as `RunConfig` and `Catalog` live under Worker even
  though Worker is only one consumer.

### Tests

- `memo/tests` currently contains 29 Python files, 6,165 lines, and 243 test
  functions, with parameterization expanding the actual case count.
- About 46 test sites initialize, train, evaluate, or step a complete algorithm.
- `test_blocks.py`, `test_algorithms.py`, and `test_loop.py` mix several domain
  owners in single files.
- Component declaration tests import `entries.stream_ac`, coupling fast component
  tests to deployment assembly.
- Template validation builds a real Brax environment.
- Memo and Infra round-trip tests cross-import source despite the two deployed
  sides using different environments.
- Moto, Aim, Optuna, Brax, and external reference requirements are mixed into
  the same default test collection.

## Known refactor sequence

The order below is agreed at architectural level. Facility phases must not be
implemented until the pending audit refines their exact contracts.

1. Freeze public imports, parameter/metric declarations, local numerical
   references, and explicit external parity requirements.
2. Establish lightweight test support and separate fast, integration, local
   parity, external parity, and container suites.
3. Extract the proven shared Environment/Normalization interaction components.
4. Move StreamAC parameter composition and graph building into the algorithm;
   reduce its entry to composition and metadata re-export.
5. Replace RTRRL's leaky `Network` boundary, close Torso/Actor/Critic contracts,
   and split Core implementation only through private helpers.
6. Finalize the Runtime/Program boundary and remove semantic-free wrappers.
7. Separate logging capabilities and make backend artifacts local outputs.
8. Separate deployment contracts from Worker.
9. Move HPO scoring ownership out of Worker according to the audited artifact
   and feedback flow.
10. Add the RTRRL entry/catalog path, remove obsolete dual paths, and run local
    integration, external parity, and container smoke suites.

## Agreed test organization

```text
memo/tests/
  support/
  unit/
    components/
    algorithms/stream_ac/
    algorithms/rtrrl/
    runtime/
    observability/
    deployment/
    worker/
  integration/
    algorithms/
    entries/
    catalog/
    worker/
  parity/
    local/
    external/
```

Infra mirrors the domains it owns: parameter resolution, HPO, scoring,
deployment contracts, and completed-trial integration.

Test rules:

- Component and Core tests use small arrays and do not JIT unless compilation is
  the behaviour under test.
- Full algorithm tests use one tiny deterministic environment and the minimum
  streams/steps needed for the asserted transition.
- Real Brax belongs to container smoke, not template validation.
- One algorithm keeps one complete leaf-wise parity oracle; other tests target a
  local invariant or one wiring boundary.
- External reference tests are selected explicitly and never silently stand in
  for the default suite.
- Heavy fixtures live in their owning subtree, not the root `conftest.py`.
- Cross-process producer/consumer contracts are tested through serialized
  artifacts, not source imports across deployment environments.
- Delete an old test only after its semantic assertion is mapped to and covered
  by a new owner-specific test.

## Facility audit results

### The current end-to-end path is incomplete

The checked-in source currently implements the following path:

```text
experiment YAML + image catalog
              |
              v
ExperimentRunner -> HPO.ask -> resolved run dictionaries
              |
              +-> CLI prints the first round as JSON
              |
              +-> ExperimentRunner.run(external round_executor callback)

S3 manifest -> Worker -> entry subprocess -> Runtime -> Reporter/sinks
                   |                              |
                   +-- compute score.json <--- metrics.jsonl
```

There is no concrete round executor in `infra/src`: nothing there currently
writes the run dictionaries and manifests to object storage, submits Batch,
waits for jobs, collects trial artifacts, or invokes `ExperimentRunner.run()`
with a real backend. The CLI calls `next_round()` once and prints it. Historical
manuals describe a fuller control plane, but that description is not evidence
of functionality in this checkout.

The semantic refactor can proceed using a fake/local executor contract. A real
Batch executor is a later Infra feature and must receive its own acceptance
test, remote dispatch, and approval for AWS-mutating verification.

### Current responsibility leaks found by the audit

- `worker.contract` mixes a deployment wire schema with environment meaning,
  graph width, Runtime budget validation, logger configuration, and HPO scoring.
- Infra treats five blocks as opaque pass-through data but also depends on
  `score.direction` to construct Optuna and derives sink-specific S3 fields.
- The Infra range resolver rejects unknown override names but does not check
  that an override is within the catalog node's `valid` domain.
- Worker blocks in `subprocess.call`; no code implements the heartbeat described
  by comments or old documentation.
- Worker computes an HPO objective and therefore must understand metric names,
  score windows, reductions, non-finite policy, and optimization direction.
- Worker deletes every scratch directory in `finally`, including a failed run's
  only local diagnostics.
- Reporter reads process environment, constructs sinks, aggregates episodes,
  and fans out events. These are three different reasons to change.
- A single wide sink protocol forces scalar-only and episode-only backends to
  implement meaningless methods.
- Rerun writes an RRD and uploads it through `worker.objects` from inside the
  training subprocess. Logger configuration is consequently coupled to the
  worker's storage implementation and an upload error appears as an algorithm
  process failure.
- Aim and Rerun receive the complete `RunConfig` although each consumes only a
  small metadata/configuration projection.
- `MetricsSink` claims to be a heartbeat in its documentation, but Worker never
  observes it while the subprocess is alive. It is currently only a scalar
  JSONL artifact.
- `runner.catalog` is an image-build deployment adapter but imports its catalog
  types and version from Worker.
- Catalog round-trip tests import production source across the Infra/image
  boundary even though the deployed environments exchange JSON, not Python
  objects.

## Field ownership after the audit

The experiment file is an Infra input, not the Entry's run configuration. A
field belongs to the layer that gives it meaning; placement in one serialized
document does not transfer that ownership.

| Field | Defining owner | Runtime consumer | Target placement |
|---|---|---|---|
| `name`, `description` | Infra/HPO | Optuna/control plane | experiment only |
| `image`, `compute`, `storage` | Infra/deployment | executor | experiment only |
| `hpo.*` | HPO | Optuna | experiment only |
| `space` | Infra parameter resolver | HPO sampler | experiment only |
| `score.metric/window_steps/reduce/non_finite/direction` | Infra scoring/HPO | Scorer and Optuna | experiment only |
| catalog `entry.command` | image deployment adapter | Worker | catalog artifact |
| catalog `entry.parameters` | algorithm declaration | Infra resolver/HPO | catalog artifact |
| catalog `entry.metrics` | algorithm observation declaration | Infra score validation | catalog artifact |
| `contract` | deployment protocol | Infra, Worker, Entry | catalog and run envelope |
| `run_id`, `experiment`, `launch_id`, `trial`, `digest` | Infra coordination | Worker and logger metadata | run identity |
| `entry` | Infra selection | Worker command lookup | run envelope |
| `environment.id/backend/observed/episode_length` | Algorithm | algorithm builder/environment | algorithm config |
| `training.num_envs` | Algorithm graph | algorithm builder/state shapes | algorithm config |
| sampled `params` | Algorithm/components | algorithm builder | algorithm config |
| `environment.seed` | Runtime scheduling | Runtime key creation | runtime config |
| `training.total_steps/epoch_steps` | Runtime scheduling | Runtime | runtime config |
| `evaluation.steps` | Runtime scheduling | Runtime | runtime config |
| Aim address | observability composition | Aim sink | logging config |
| Rerun enablement and sampling interval | observability policy | Rerun sink | logging config |
| artifact root | Infra coordination | Worker uploader | run envelope |

Two validations follow from this table:

1. Infra must check `score.metric` against the selected catalog entry's declared
   metrics before starting a round.
2. Infra must validate every experiment override against the declaration's
   `valid` domain, not only against its name and shape. Until structural sweeps
   are deliberately supported, every reachable `kind` override must resolve to
   exactly one choice for all trials in one experiment.

## Target run and catalog contracts

Do not reintroduce an SDK package. The cross-environment interface remains
versioned JSON. Each receiver owns a small local parser for the projection it
actually consumes, and serialized contract fixtures test the projections
together.

The next breaking contract version should replace the flat `RunConfig` with a
semantically nested run specification:

```yaml
contract: 8
identity:
  run_id: ...
  experiment: ...
  launch_id: ...
  trial: 0
  digest: sha256:...
entry: stream_ac
artifacts:
  root: s3://.../experiment/launch/run
algorithm:
  environment:
    id: brax::hopper
    backend: spring
    observed: [0, 2, 4, 6, 8, 10]
    episode_length: 1000
  num_envs: 16
  parameters: {...}
runtime:
  seed: 0
  total_steps: 2000000
  epoch_steps: 10000
  evaluation_steps: 1000
logging:
  aim:
    url: aim://host:port
  rerun:
    every_steps: 20000
```

Worker validates only `contract`, `identity`, `entry`, and `artifacts`, retaining
the remaining payload as opaque JSON for the child. Entry validates the whole
shape at its composition boundary, then passes `algorithm`, `runtime`, and
`logging` projections to their respective owners. Domain validators remain
with those owners; the Worker must not validate that an epoch is divisible by
the graph width.

`score` is deliberately absent. It cannot affect a run and must not enter the
training image. Infra retains the score policy beside the trial identity.
Sink-specific destinations such as `score.s3` and `logging.rerun_s3` are also
absent. One artifact root is enough for Worker to upload all local artifacts.

The catalog remains one build-time JSON artifact:

```text
entry name -> command + parameter declaration + metric declaration
```

The entry may re-export the algorithm's declarations for discovery. The catalog
scanner may import entry modules at image-build time, but it must not recreate,
expand, or reinterpret an algorithm's parameter tree. Move its ownership from
the generic `runner` name to an explicit deployment/image adapter; no runtime
registry is introduced.

## Target in-process contracts

### Algorithm and Runtime

Runtime owns the protocol it calls. In conceptual form:

```python
class Program(Protocol):
    def init(key): ...
    def train(state, key, steps): ...
    def evaluate(state, key, steps): ...

@dataclass(frozen=True)
class BuiltAlgorithm:
    program: Program
    observations: ObservationSchema
```

`StreamAC.build()` and `RTRRL.build()` return this closed result. Their private
Core and subgraphs remain invisible. `ObservationSchema` names where Runtime can
read transition boundaries, environment reward, and optional algorithm series;
Entry no longer imports graph-internal groups such as `PLACES` to derive metric
names.

Remove compatibility paths once both algorithms implement this protocol:

- move `AgentProgram` from algorithm code to Runtime;
- remove test-only `drive()`;
- remove `program_of()` and evaluation-shape compatibility branches;
- replace `Runtime.from_config()` duck typing with explicit `RuntimeConfig`.

Runtime keeps compilation, keys, budgets, train/evaluation calls, and complete
episode cutting. Episode cutting is scheduling/rollout semantics because it
knows chunk boundaries, phase, streams, stride, and global step. Algorithm only
produces the canonical observation schema.

### Observability

Use capability-specific protocols:

```python
class ScalarSink(Protocol):
    def report(self, values: Mapping[str, float], step: int) -> None: ...

class EpisodeSink(Protocol):
    def log_episode(self, episode: Episode) -> None: ...
```

An episode aggregator receives a completed `Episode`, derives the agreed scalar
statistics, and sends the original episode and derived scalars to the sinks that
support each capability. Logger lifecycle is explicit and failures propagate;
there is no silent best-effort path for a configured backend.

Ownership is:

- Runtime/rollout: `Episode` identity and complete-episode cutting;
- observability: metric naming, episode statistics, fan-out, scalar JSONL, Aim,
  Rerun sampling, and backend lifecycle;
- Entry: construct configured sinks and the mandatory local metric artifact;
- Worker: upload produced local artifacts.

Aim receives only run metadata plus its endpoint. Rerun receives only its
sampling configuration, stream/step metadata, and output path. Rerun writes a
local `.rrd`; it has no S3 import. Metrics JSONL is a required local artifact
because scoring consumes it, not a heartbeat implementation.

### Local artifact boundary

Entry and loggers write only below:

```text
scratch/artifacts/
  metrics.jsonl
  rerun/*.rrd
```

After a successful child exit, Worker recursively uploads that tree below the
run's single artifact root, preserving relative paths. It then writes a small
`result.json` containing run/trial identity, success status, and the uploaded
relative artifact names. Infra's executor reads `result.json`, downloads
`metrics.jsonl`, and runs Scorer locally. `score.json` may be written by Infra as
an archival derivative but is no longer a Worker output or run input.

This convention avoids a Python artifact-manifest API across the subprocess
boundary and does not require every logger to know object storage.

## Target process and feedback flow

```text
                  image build
algorithm declarations -> entry re-export -> catalog.json/image label
                                             |
                                             v
experiment YAML -> Infra validate/mask -> HPO.ask
                                             |
                                      RunSpec + manifest
                                             |
                                             v
Worker -> config file/env -> Entry -> Algorithm.build -> BuiltAlgorithm
  |                              |             |
  |                              +-> Runtime --+-> complete Episode
  |                                             |
  |                                      observability sinks
  |                                             |
  +<-- child exit <--- scratch/artifacts/{metrics.jsonl, *.rrd}
  |
  +-> upload artifact tree + result.json
                         |
                         v
Infra executor -> Scorer(metrics, ScoreSpec) -> HPO.tell -> next round
```

Scoring is a pure Infra domain service, not a separately deployed service at
this stage. Moving it to another process would add transport without changing
its meaning. The first HPO round remains valid without prior metrics; later
rounds are requested only after the preceding round has been scored and told.

## Failure and scheduling policy

The policy is correctness-first and explicit:

- Runs inside one manifest remain serial and use a fresh scratch directory.
- A non-zero entry exit, required logger close failure, artifact upload failure,
  missing score metric/window, or contract mismatch fails the manifest.
- A configured Aim or Rerun backend is required. Its failure is not caught and
  hidden.
- A non-finite metric with the declared `non_finite` policy is a scored bad
  trial, not an execution failure. A missing metric/window is an experiment
  failure.
- Worker never interprets score policy and never converts training output into
  an HPO value.
- Completed artifacts may survive a later run's failure, but a failed round is
  not partially fed into Optuna. Infra marks the asked trials failed, starts no
  next round, and terminates still-running jobs when a real executor exists.
- There is no heartbeat requirement in the current implementation. Remove the
  false claim. If future queue supervision requires one, design it as a Worker
  process-liveness capability with its own timeout contract.
- Cleanup happens only after artifacts/result state have been handled. Do not
  add broad exception suppression; failure diagnostics must remain visible.

The current source has no executor capable of enforcing cross-job termination,
so that last part is a required acceptance criterion for the future Batch
executor rather than a claim about present behaviour.

## Test migration map

| Existing tests | New semantic owner(s) | Main change |
|---|---|---|
| `test_parameters.py`, Infra `test_adapter.py`/`test_parameters.py` | Memorax declaration contract; Infra resolver/sampler | retain declaration tests; add override-within-`valid` and fixed-`kind` tests |
| `test_backbones.py`, `test_heads.py`, `test_credit.py`, `test_initialization.py`, `test_sequence.py` | `unit/components` | small-array capability tests; no entry imports |
| `test_environments.py`, `test_normalization.py`, `test_observation_selection.py` | shared interaction components | one shared contract plus algorithm wiring tests |
| `test_blocks.py` | StreamAC Core/online block and RTRRL role/Torso/Core suites | split by algorithm meaning; do not create one-file-per-helper tests |
| `test_algorithms.py`, `test_rtrrl.py` | per-algorithm interface and semantic functionality | tiny fake environment; assert init/train/eval and readings |
| `test_layered_parity.py`, `test_paper_parity.py`, `test_rtrrl_parity.py` | local/external parity | keep one complete leaf-wise oracle per promised equivalence; explicit selection |
| `test_loop.py`, `test_rollout.py`, `test_episode.py` | Runtime rollout plus observability aggregation | fake Program for scheduling; exact episode-to-scalar tests separately |
| `test_reporter.py` | observability aggregation/fan-out and Entry composition | remove environment/config parsing from Reporter tests |
| `test_metrics_sink.py`, `test_aim_sink.py`, `test_rerun_sink.py` | backend unit/integration suites | narrow protocols; Rerun asserts local RRD only, no S3/moto |
| `test_score.py` | Infra Scorer | move unchanged numerical policy cases to Infra; Worker has no scoring tests |
| `test_worker.py` | Worker supervision/artifact integration | fake subprocess writes local artifact tree; test serial order, isolation, stop-on-failure, upload, result |
| `test_template.py` | deployment/catalog contract | schema/declaration checks only; no real Brax graph |
| Memo/Infra `test_round_trip.py` | serialized deployment integration | exchange JSON fixtures/files; remove cross-source imports |

The cross-process acceptance chain is intentionally narrow:

1. Infra emits a versioned serialized RunSpec and manifest from a tiny catalog.
2. Worker consumes it and launches a tiny test entry.
3. The entry writes one metrics artifact through the real observability path.
4. Worker uploads/returns the artifact inventory through a fake object store.
5. Infra Scorer consumes the serialized metric artifact and HPO records the
   value.

Real Brax, Aim server, external reference repositories, object storage, Batch,
and Docker are separate opt-in gates rather than default unit dependencies.

Suggested local gates after the migration:

```text
uv run --project memo ruff check memo
uv run --project memo pytest memo/tests/unit
uv run --project memo pytest memo/tests/integration
uv run --project infra ruff check infra
uv run --project infra pytest infra/tests
```

Parity, Docker, and remote Batch commands remain explicit gates and must not be
silently included in the fast suite.

## Executable refactor tasks

Each task ends with its owner-specific tests green before the next begins. Do
not combine tasks merely because two files are nearby.

### Task 1: Freeze behaviour and build the test skeleton (completed)

- Record current public imports, StreamAC/RTRRL declarations, and promised
  numerical equivalences.
- Create the target test directories and lightweight fake environment/program,
  sink, object-store, and serialized-contract fixtures.
- Mark external parity and service/container tests explicitly.
- No production move in this task.

### Task 2: Extract proven shared interaction components (completed)

- Move the duplicated Environment and Normalization semantics to shared
  components.
- Rewire StreamAC and RTRRL without altering their graph topology.
- Run component invariants, each algorithm's wiring test, and numerical
  equivalence before deleting the copies.

### Task 3: Close StreamAC's algorithm boundary (completed)

- Introduce one algorithm-neutral assembly service. It creates the environment,
  provides build context, delegates branch routing to component families, and
  closes the result over the Runtime-owned program contract. It must not name
  StreamAC, RTRRL, or any algorithm-specific graph relation.
- Move each `kind` table and its leaf construction behind the corresponding
  component family. A family owns branch selection; a leaf owns only its own
  parameters and construction.
- Move StreamAC's parameter and observation declarations beside its graph. The
  graph alone declares instance count, sharing, connection, state flow, and
  gradient semantics. In particular, StreamAC requests two independent
  backbones even when both use the same resolved component configuration.
- Remove environment creation and component parameter parsing from
  `entries/stream_ac.py`; do not replace them with an algorithm-private
  assembler.
- Return one Runtime-owned `AgentProgram` now; Task 5 renames/relocates the
  final `BuiltAlgorithm` contract after both algorithms use this path.
- Reduce Entry to config projection, logger construction, Runtime construction,
  and invocation; metadata is re-export only.
- Gate first with algorithm-neutral assembly and component-family contract
  tests, then declaration, StreamAC graph topology, tiny train/eval, local
  numerical, and entry composition tests.

### Task 4: Repair RTRRL's graph boundaries (completed)

- Replace the leaky RTRRL `Network` abstraction with semantic private subgraphs
  whose modules, credit state, and forward modes do not escape their contracts.
- Keep shared-backbone coupling in one Core; do not introduce two Agents.
- Extract only private helpers from `Core.update_parameters`.
- Establish the same `BuiltAlgorithm` and declaration surface as StreamAC.
- Gate every changed leaf/update quantity numerically, then one complete parity
  run.

Completed on 2026-08-13. The behaviour plus published-reference suite passes
29/29, the combined declaration/assembly/numerical gate passes 32/32, and the
default repository suite passes 320 tests. A direct catalog-parameter-to-graph
Brax run trained for 256 steps with distinct torso and heads Adam rates and
finite TD-error and recurrent-trace readings.

### Task 5: Finalize Runtime

- Move Program/BuiltAlgorithm execution contracts into Runtime.
- Add explicit `RuntimeConfig` and remove full-config duck typing.
- Consume the algorithm-provided observation schema for episode cutting.
- Delete `drive()`, `program_of()`, and compatibility result-shape branches after
  both algorithms use the final contract.
- Test scheduling with a fake Program and rollout cutting independently.

Completed on 2026-08-13. Runtime now owns `Program`, `BuiltAlgorithm`,
`ObservationSchema`, and `RuntimeConfig`. Generic assembly closes StreamAC and
RTRRL directly over that contract; Entry explicitly projects the deployment
document into assembly and Runtime inputs. Episode cutting reads only the
algorithm-provided schema. The old `AgentProgram`, `program_of()`, `drive()`,
full-config duck typing, and evaluation result-shape compatibility paths are
removed. The focused Runtime/rollout/assembly gate passes 78 tests and the
default repository suite passes 321 tests.
The final Entry-to-Runtime path also completed a 256-step Brax Hopper run,
producing four training episodes and one evaluation episode.

### Task 6: Refactor observability (completed)

- Move Episode statistics and metric naming to observability while retaining
  rollout identity/cutting under Runtime.
- Introduce narrow scalar and episode sink protocols plus aggregation/fan-out.
- Make Aim depend on endpoint + run metadata only.
- Make Rerun produce local RRD artifacts only.
- Make Metrics JSONL the mandatory local scalar artifact and remove heartbeat
  claims.
- Keep backend failures visible; add no silent catch-all paths.

Completed on 2026-08-12. The pure observability and Runtime boundary tests,
local Aim/Rerun integration tests, full default suite, and the real StreamAC
Worker smoke all pass. Rerun writes local RRD files only; uploading the local
artifact tree remains Task 8.

### Task 7: Introduce deployment contract version 8

- Create image-side Worker-envelope and Entry run-spec parsers with their
  consumer-specific validation.
- Change Infra assembly to the nested RunSpec and remove `score` and per-sink S3
  destinations.
- Add `artifact.root`, score-metric/catalog validation, valid-domain override
  validation, and fixed-structure validation.
- Move catalog types/version out of Worker and rename `runner.catalog` to the
  deployment/image adapter.
- Replace source-import round trips with serialized fixture tests.

Completed on 2026-08-13. The shared wire shape is now contract 8 with nested
identity, artifact, algorithm, runtime, and logging projections. Infra emits
that shape without score policy or sink-specific object-store destinations;
the Worker envelope and Entry parser independently consume the same serialized
fixture. Catalog ownership moved to `deployment`, and Entry projects the full
RunSpec to assembly, Runtime, and observability. Infra now refuses undeclared
score metrics, overrides outside declared valid domains, and any reachable
structure choice that is not fixed for the experiment. The real StreamAC
template was completed with its two reachable head-initialization choices.
Infra passes 35 tests, Memo's default suite passes 324 tests, and the real
catalog-to-StreamAC round trip assembles and steps. Task 8 subsequently
activated the Worker envelope and removed its legacy flat score path.

### Task 8: Reduce Worker to supervision and artifact transport (completed)

- Remove score imports and score computation from Worker.
- Run each manifest item serially in isolated scratch.
- Upload the local artifact tree after child success and emit `result.json`.
- Preserve fail-fast semantics and visible diagnostics; align cleanup with the
  declared failure policy.
- Test with a fake entry process and fake object store; keep moto/S3 separate.

Completed on 2026-08-13. Worker now consumes only its version-8 envelope,
starts manifest items serially in unique scratch directories, recursively
uploads `scratch/artifacts/`, and writes `result.json` as the successful commit
marker. A child or upload failure stops the manifest and preserves that run's
scratch; fully published runs are cleaned. Worker no longer imports scoring or
emits `score.json`. Seven fake-store/fake-child supervision tests are part of
the default suite, which passes 331 tests. A separate Moto/local-Aim StreamAC
smoke confirms the critic trace metric reaches both Aim and the uploaded
`metrics.jsonl` before Worker publishes the result.

### Task 9: Move scoring into Infra (completed)

- Move the pure scoring implementation and tests to Infra.
- Let the round executor collect metrics artifacts, apply the experiment's
  ScoreSpec, and return trial values to `HPO.tell()`.
- Define failed-round behaviour in HPO so asked trials do not remain silently
  running in Optuna.
- Keep `ExperimentRunner.run(round_executor)` as the backend seam and test it
  with a deterministic fake executor.

Completed on 2026-08-13. `ScoreSpec`, the scorer, and all numerical policy tests
now live in Infra; Memo and Worker contain no scoring model or implementation.
`ExperimentRunner` passes the experiment-owned score policy to the round
executor, validates a complete one-result-per-trial return, and correlates
out-of-order results before HPO feedback. HPO marks every still-running trial
in a failed round as `FAIL`, re-raises the original error, and asks no next
round. The deterministic fake executor scores local metrics artifacts through
the real Scorer. Infra passes 63 tests and no longer imports Memo or carries
Pydantic solely for cross-source tests.

### Task 10: Add the missing concrete execution capability (implementation complete)

- First implement a local/mock round executor that writes real config/manifest
  files and exercises Worker as a subprocess.
- Separately implement or restore the Batch executor: upload inputs, pack runs
  into manifests, submit jobs, wait, terminate survivors on failure, collect
  result/artifact files, and score the completed round.
- Extend CLI from first-round JSON printing to the complete selected backend.
- Do not claim Batch completion until a pushed remote commit passes the remote
  acceptance run.

The local half completed on 2026-08-13. `LocalRoundExecutor` writes serialized
run documents and manifests, starts Worker as an independent command, collects
`result.json` plus `metrics.jsonl`, and feeds the Infra Scorer. Worker supports
the same artifact contract over local `file://` storage as over S3. The CLI now
runs every configured HPO round and prints the completed Optuna study rather
than stopping after `ask()`. Infra's 65 default tests and Memo's 311 default
tests pass; an explicit cross-process service test also passes through Infra,
the real Worker module, a fake Entry, local artifact transport, Scorer, and
Optuna.

The Batch implementation completed offline on 2026-08-13. It packs each round
according to `hpo.parallel_jobs`, uploads v8 configs/manifests to S3, routes the
pinned digest and instance profile to the run/dev queue plus job definition,
submits all siblings, polls to terminal state, terminates nonterminal siblings
on the first failure, exposes the CloudWatch tail, and collects/scorers results
only after full success. CLI supports both backends, the shipped template now
declares `parallel_jobs`, and 69 default Infra tests pass. Fake S3/Batch/Logs
cover the complete AWS API flow without mutating AWS. Task 10 is not remotely
accepted until this pushed commit runs successfully on real Batch; that final
gate still requires explicit approval before AWS mutation.

### Task 11: Complete public entry/catalog migration and cleanup (completed locally)

- Add RTRRL's real Entry and catalog item using the same composition contract.
- Remove obsolete RTRRL exports/dual paths including the broken `EvalSummary`
  export after compatibility decisions are tested.
- Regenerate the catalog and update the experiment template to contract 8.
- Remove old tests only after every assertion is mapped above.
- Run unit, integration, local parity, external parity, Docker, and finally
  remote Batch gates in that order.

Completed locally on 2026-08-13. The public `RTRRL` export, the new `rtrrl`
Entry, and catalog discovery now select the semantic RTRRL implementation. The
obsolete older RTRRL and broken IndependentRTRRL/EvalSummary public paths are
removed. RTRRL trace observations now reflect their actual graph boundaries:
the shared torso is split into before/recurrence/after, while each readout owns
one whole actor or critic trace norm. Every norm and its variance reaches both
Metrics JSONL and Aim in the Worker service smoke. The contract-8 HalfCheetah
smoke resolves through the real catalog and Infra HPO before assembling and
stepping. Ruff passes, Memo's 314 default tests pass, and the seven explicit
external parity tests pass. Docker and real Batch acceptance follow the pushed
commit and are intentionally not claimed by this local result.

## Deliberately unresolved implementation details

The analysis closes ownership but leaves three details to the task that has the
necessary evidence:

- Exact AWS Batch packing, polling, and cancellation APIs: no implementation is
  present to refactor, and AWS mutation requires explicit approval.
- The final RTRRL private subgraph names: choose them while performing Task 4
  against the actual equations; the boundary criteria are fixed, the names are
  not.
- Whether a future live Rerun relay is needed: current scope is a valid local
  `.rrd` artifact. Online discovery/relay remains a separate Infra capability.

These do not block Tasks 1-9 and must not be filled with speculative wrapper
layers.
