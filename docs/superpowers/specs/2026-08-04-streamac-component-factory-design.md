# StreamAC Algorithm Factory Boundary Design

## Purpose

Establish and verify the component-factory boundary at the StreamAC algorithm and
entry layers before applying the model recursively to backbones, heads, credit,
optimizers, normalizers, or other algorithms.

This is deliberately a behavior-preserving structural refactor. It gives the
next component migration a concrete boundary to target without changing HPO,
experiment syntax, numerical behavior, or observability at the same time.

## Phase-one scope

Phase one changes only the StreamAC algorithm and entry boundary:

- introduce the standard algorithm-facing `param()` and `build()` contract;
- create one StreamAC factory object implementing that contract;
- move ownership of StreamAC parameter composition and complete graph assembly
  out of `memo/entries/stream_ac.py` and behind the factory;
- make the entry delegate parameter export and construction to the factory;
- preserve the current Training SDK catalog, flat runtime manifest, experiment
  files, HPO behavior, metrics, and numerical results;
- add boundary and parity tests proving that the new indirection changes no
  behavior.

Phase one does not recursively convert the inner components. The user will review
the resulting algorithm/entry structure before any inner component receives the
same `param/build` API.

## Deferred work

The following remains unchanged until the phase-one result is reviewed:

- backbone, head, credit, optimizer, initializer, and normalizer declarations;
- the existing component branch maps and their Training SDK parameter metadata;
- nested public YAML and nested runtime parameter trees;
- removal of inactive-branch placeholders;
- SDK-independent core parameter nodes and generic HPO traversal;
- observability, `PARTS`, `TRAINING_METRICS`, `METRICS`, and `RECORD`;
- RTRRL and all other algorithms;
- structural HPO or structural experiment scanning.

This deferral is intentional. It means phase one may still pass the existing
Training SDK-compatible parameter declaration through the factory. The factory
protocol itself must not name Training SDK or Optuna types, so its implementation
can later change without changing the entry-facing contract.

## Factory contract

The algorithm-facing protocol contains two standard operations:

```python
class AlgorithmFactory(Protocol[T]):
    def param(self) -> Mapping[str, object]: ...

    def build(
        self,
        params: Mapping[str, object],
        environment: object,
        training: object,
    ) -> T: ...
```

`param()` returns the complete parameter declaration accepted by the current
algorithm boundary. In phase one this declaration is compatible with the current
catalog and HPO path. Later it will become the recursively produced neutral
parameter tree without changing the entry call site.

`build()` accepts the fully resolved parameters plus environment and training
context, and returns the complete StreamAC program/agent. The entry may not parse
algorithm parameters, choose branches, assemble networks, or construct optimizer
and estimator components.

The protocol describes behavior and does not require inheritance. The concrete
factory may be a frozen object holding the algorithm-local branch mappings and
private helper functions.

## StreamAC ownership after phase one

The factory owns all of the logic currently needed to turn a manifest into a
StreamAC instance:

```text
StreamACFactory
|-- parameter declaration/composition
|-- environment-derived dimensions
|-- actor network assembly
|-- critic network assembly
|-- credit selection
|-- actor optimizer selection
|-- critic optimizer selection
|-- observation estimator selection
|-- reward estimator selection
`-- StreamAC construction
```

The inner implementations remain the existing functions, classes, and branch
maps. They are opaque legacy leaves for phase one. Moving them behind the factory
does not yet claim that they conform to the final recursive component contract.

Private StreamAC-only helpers may live with the factory and do not need
registration. Existing reusable component registries remain discovery mechanisms;
the factory's explicit branch mappings continue to determine which components are
actually legal for StreamAC.

## Entry after phase one

The entry remains the Training SDK executable binding. Its algorithm-facing shape
becomes equivalent to:

```python
STREAM_AC = StreamACFactory()

PARAMETERS = STREAM_AC.param()
build = STREAM_AC.build
```

The entry temporarily retains `PARTS`, `TRAINING_METRICS`, `METRICS`, `RECORD`,
`run()`, and `main()` because the observability adapter is a separate design
decision. `run()` must obtain the agent through `STREAM_AC.build()` and must not
retain a second construction path.

Catalog discovery remains unchanged: the module still exports `PARAMETERS`,
`METRICS`, and `main`. No new module is added under `memo/entries`, so automatic
entry scanning cannot mistake a helper for an executable algorithm.

## Dependency rule

The protocol module is SDK-neutral. The phase-one concrete factory is allowed to
bridge the current parameter declarations as a temporary migration seam because
the inner components still use Training SDK metadata. New direct Optuna or runner
dependencies are forbidden.

The follow-up inner-component phase must remove the remaining Training SDK imports
from `memo/memorax`. Phase one must not spread those imports to additional core
modules or encode SDK model classes in the protocol signature.

## Validation and behavior

Phase one deliberately keeps existing validation and representation:

- structures remain fixed exactly as the current control plane requires;
- parameters remain flat at the current runtime boundary;
- inactive branches keep their current placeholder behavior;
- current templates and archived run manifests remain readable;
- every existing parameter must arrive at the same inner implementation with the
  same value as before.

The final rules already agreed for nested active-only trees remain the target for
the next phase, but implementing them now would make it impossible to evaluate the
factory boundary independently.

## Error handling

- `param()` must fail during import/catalog creation if the composed declaration
  is invalid, as the current entry does.
- `build()` must preserve current missing-parameter and invalid-branch failures.
- The entry must not catch or translate algorithm construction errors.
- No fallback build path may remain in the entry.

## Tests

Phase-one tests prove boundary placement and behavioral equivalence:

- the factory satisfies the two-operation protocol;
- `entry.PARAMETERS` is the declaration returned by `STREAM_AC.param()`;
- `entry.build` delegates to the factory rather than reassembling the graph;
- a watched manifest confirms the factory reads the same complete set of
  parameters as the current entry implementation;
- pinned actor, critic, backbone, credit, optimizer, and normalizer selections
  reach the same constructed objects as before;
- existing StreamAC golden, parity, entry-contract, template, and short training
  tests remain green;
- catalog discovery still finds exactly one StreamAC entry;
- static import checks confirm that the neutral factory protocol imports neither
  Training SDK nor Optuna.

No tests for nested parameters, neutral leaf declarations, or conditional HPO are
added in phase one because those behaviors are explicitly deferred.

## Implementation sequence

1. Add a minimal SDK-neutral algorithm-factory protocol and its contract test.
2. Add the StreamAC factory around the current declaration and construction logic.
3. Move the current `build()` helpers and assembly behind the factory without
   rewriting their inner component behavior.
4. Replace the entry's declaration and build implementation with delegation.
5. Run focused entry/factory tests, then the existing StreamAC and static suites.
6. Present the resulting code structure and verification evidence to the user.
7. Do not migrate an inner component until the user approves this boundary.

After approval, the next design/implementation slice will replace the opaque
legacy leaves one family at a time with recursive `param(overrides)` and
`build(resolved_params, context)` factories, followed by the nested HPO adapter.
