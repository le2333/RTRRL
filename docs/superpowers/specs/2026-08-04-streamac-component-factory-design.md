# StreamAC Algorithm Factory Boundary Design

## Purpose

Establish and verify the recursive component-factory boundary at the StreamAC
algorithm and entry layers before applying the model to backbones, heads, credit,
optimizers, normalizers, or other algorithms.

This is a contract-first structural refactor. Destructive intermediate changes
are allowed, and StreamAC need not run end to end until every required recursive
layer has been migrated.

## Phase-one scope

Phase one changes only the StreamAC algorithm and entry boundary:

- introduce the standard graph-facing `param()` and `build()` contract;
- express StreamAC as a composite factory whose children are replaceable factory
  dependencies;
- move ownership of StreamAC parameter composition and graph assembly out of
  `memo/entries/stream_ac.py` and behind the factory;
- make the entry delegate parameter export and construction to the factory;
- define each behavior through tests before implementation, observe the expected
  failure, and implement the minimum needed to make this layer pass;
- add implementation-independent contract tests reusable by later component
  factories.

Phase one does not recursively convert the inner components. Existing leaves may
be represented by test doubles, explicit incomplete bindings, or temporary legacy
adapters. The user will review the algorithm/entry structure before any inner
component receives the same `param/build` API.

Only tests for this recursive layer must be green. Existing end-to-end StreamAC
tests may fail because deeper layers have not adopted the contract. Such failures
must be reported as expected consequences of the incomplete migration rather than
hidden with compatibility logic.

## Deferred work

The following remains outside phase one:

- conversion of backbone, head, credit, optimizer, initializer, and normalizer
  declarations and implementations;
- conversion of existing branch maps and their Training SDK parameter metadata;
- nested public YAML and nested runtime parameter trees;
- removal of inactive-branch placeholders;
- generic HPO traversal and final nested trial manifests;
- observability, `PARTS`, `TRAINING_METRICS`, `METRICS`, and `RECORD`;
- RTRRL and all other algorithms;
- structural HPO or structural experiment scanning.

The factory protocol and generic contract tests must not name Training SDK,
Optuna, JAX, Flax, or a particular StreamAC leaf. A temporary entry binding may be
incomplete until its child layers are migrated.

## Factory contract

Every opaque graph layer follows the same two-operation protocol:

```python
class GraphFactory(Protocol[T]):
    def param(self, structure: Mapping[str, object]) -> ParameterTree: ...

    def build(
        self,
        params: Mapping[str, object],
        context: BuildContext,
    ) -> T: ...
```

`param()` receives the complete structure subtree for the current graph layer. It
validates fields owned by that layer, sends each child structure subtree downward,
and combines the returned child parameter trees. It declares parameter domains
but does not perform HPO sampling.

`build()` receives the fully resolved parameter subtree plus neutral construction
context. It sends each child's parameter subtree downward and combines returned
child graphs upward. The return type is component-specific.

The protocol describes behavior and does not require inheritance. A concrete
factory may be a frozen object holding child factories, algorithm-local valid
branch mappings, and composition functions.

## StreamAC composite layer

The top-level StreamAC factory owns this graph shape:

```text
StreamACFactory
|-- shared StreamAC parameters
|-- actor child factory
|-- critic child factory
|-- normalization child factory
|-- top-down structure/parameter delegation
|-- bottom-up child graph composition
`-- final StreamAC program construction
```

The precise contents of actor, critic, and normalization are opaque to this layer.
Generic fake factories prove recursion without relying on their implementation. A
real child is not considered migrated merely because a compatibility adapter makes
an end-to-end test pass.

Private StreamAC-only subgraphs may live beside the algorithm and do not require
registration. A discovery registry can locate components, but the factory's
explicit local mapping remains the authority over what StreamAC permits.

## Entry boundary

The entry remains the executable/infrastructure binding. Its intended shape is:

```python
STREAM_AC = StreamACFactory(...child factories...)

PARAMETERS = adapter.parameters(STREAM_AC)
build = adapter.build(STREAM_AC)
```

The entry may temporarily retain `PARTS`, `TRAINING_METRICS`, `METRICS`, `RECORD`,
`run()`, and `main()` because observability is a separate design decision. It may
also be temporarily non-runnable while child bindings are incomplete. It must not
retain a second legacy graph-construction path merely to preserve end-to-end
behavior.

No helper module is added under `memo/entries`, so automatic scanning cannot
mistake it for another executable algorithm.

## Dependency rule

The protocol, StreamAC composite factory, and generic tests are SDK-neutral.
Training SDK translation belongs to the entry adapter. New direct Optuna or runner
dependencies in the algorithm layer are forbidden.

The later inner-component phase removes existing Training SDK imports from
`memo/memorax`. Phase one must not spread those imports or encode SDK model types
in the factory protocol.

## Layer acceptance

Phase one is accepted when the StreamAC composite layer independently proves:

- top-down structure and parameter-subtree delegation;
- bottom-up child graph composition;
- ownership of shared StreamAC parameters;
- rejection of missing, unknown, or unconsumed fields at this layer;
- no dependency on a particular child implementation;
- entry delegation through one factory binding with no legacy assembly path.

End-to-end training, catalog generation, old flat manifests, and numerical parity
are not phase-one acceptance criteria. Their temporary failure is allowed and must
be reported explicitly.

## Error handling

- `param()` rejects unknown or missing structure fields owned by StreamAC and
  reports their logical nested paths.
- `build()` rejects missing or unconsumed resolved fields owned by StreamAC.
- Child errors propagate with their child path attached.
- The entry does not catch algorithm construction errors.
- No fallback build path remains in the entry.

## Test-first development

Tests precede every production change. Each behavior follows this cycle:

1. add one focused test for the target contract or behavior;
2. run it and confirm it fails for the expected missing behavior;
3. implement the minimum production change;
4. rerun it and confirm it passes;
5. refactor only while the focused test remains green.

The red result is part of the evidence. A test that unexpectedly passes does not
authorize implementation; it must first be corrected so that it detects the
missing behavior.

## Reusable contract tests

Implementation-independent tests use tiny fake factories and a reusable contract
suite. They verify that any factory:

- exposes callable `param(structure)` and `build(params, context)` operations;
- returns a parameter tree from `param()`;
- consumes only its owned subtree;
- rejects unknown leftovers;
- builds deterministically from resolved parameters and context;
- does not rely on Training SDK, Optuna, JAX, Flax, or StreamAC types.

Composite-factory tests use spies to verify top-down call order and exact child
subtrees, followed by bottom-up assembly from child return values. StreamAC tests
verify its shared fields, actor/critic/normalization boundaries, and entry
delegation. Static import tests verify that neutral layers import neither Training
SDK nor Optuna.

These tests assert the protocol and recursive behavior rather than concrete RTU,
MLP, optimizer, or normalizer fields. The same contract suite will be applied to
each inner factory in later phases.

Existing end-to-end tests are diagnostic during phase one. They run after the
focused suite to inventory incomplete deeper layers, but phase one does not add
compatibility code solely to make them green.

## Implementation sequence

1. Write reusable factory contract tests and confirm the missing protocol is red.
2. Add the minimal SDK-neutral protocol and make only those tests green.
3. Write composite-recursion tests with fake children and confirm they are red.
4. Implement the StreamAC composite factory until recursion tests are green.
5. Write entry-delegation tests, remove the legacy assembly path, and make the
   entry-layer tests green.
6. Run focused tests and static checks, then run broader tests only to inventory
   expected failures caused by unmigrated child layers.
7. Present the code structure, red-green evidence, passing layer tests, and known
   end-to-end gaps to the user.
8. Do not migrate an inner component until the user approves this boundary.

After approval, migrate one child family at a time through the same contract and
red-green cycle. Once the recursive graph is complete, connect the nested HPO
adapter and restore end-to-end StreamAC execution.
