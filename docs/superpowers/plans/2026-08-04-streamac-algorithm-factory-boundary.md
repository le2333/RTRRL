# StreamAC Algorithm Factory Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish one production `StreamACFactory` implementation with recursive `param/build` behavior, reusable implementation-independent contract tests, and a StreamAC entry that delegates to that same implementation.

**Architecture:** Add an SDK-neutral graph-factory contract and neutral parameter declarations under `memorax`, then implement StreamAC as a composite factory with injected child factories. Apply the reusable contract suite directly to the production `StreamACFactory`; fake children only observe top-down and bottom-up recursion. Bind the exact exported `STREAM_AC` object through a small runner adapter and remove the entry's legacy parameter parsing and graph assembly, accepting an intentionally incomplete end-to-end StreamAC until inner factories are migrated.

**Tech Stack:** Python 3.12, dataclasses, typing protocols, pytest, Ruff, existing `uv` workspace.

## Global Constraints

- Work only on the StreamAC algorithm/entry recursive layer; do not migrate backbone, head, credit, optimizer, initializer, normalizer, RTRRL, HPO, or observability internals.
- The protocol, production `StreamACFactory`, and generic contract tests must import neither Training SDK nor Optuna; keep them independent of JAX and Flax as well.
- Generic contract tests, StreamAC-specific tests, and entry tests must exercise the same production `StreamACFactory` class; the entry must bind the exported production `STREAM_AC` instance.
- Fake factories are permitted only as injected child spies. Do not create a test-only StreamAC implementation or a second entry-side graph implementation.
- Follow strict red-green-refactor for every behavior: add one focused test, run it and observe the expected failure, implement minimally, rerun to green, then refactor.
- Only this recursive layer's focused tests must pass. Run broader StreamAC tests diagnostically and report failures caused by unmigrated children; do not add compatibility paths to hide them.
- Destructive changes to `memo/entries/stream_ac.py` are authorized. Its current uncommitted changes only add three Chinese comments inside blocks this plan removes; do not stage or alter any other pre-existing deletion or modification.
- Keep virtual environments and caches outside the repository. In PowerShell, set:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path $env:TEMP "streaming-rtrrl-memo-venv"
$env:UV_CACHE_DIR = Join-Path $env:TEMP "streaming-rtrrl-uv-cache"
```

- This is a development checkout, so focused pytest and Ruff runs are allowed. Do not mutate AWS or dispatch paid remote work without owner approval.

---

## File map

- Create `memo/memorax/factory.py`: neutral parameter declarations, `GraphFactory` protocol, exact subtree validation, and path-aware child error propagation.
- Create `memo/tests/factory_contract.py`: reusable `FactoryCase` and `assert_factory_contract()` support applied to production factories.
- Create `memo/tests/test_factory_contract.py`: red-green tests for the neutral protocol and validation primitives.
- Create `memo/memorax/algorithms/stream_ac_factory.py`: production `StreamACFactory`, explicit boundary children, production `STREAM_AC`, and `STREAM_AC_STRUCTURE`.
- Create `memo/tests/test_stream_ac_factory.py`: apply the generic suite to production `STREAM_AC` and verify recursive delegation with spy children.
- Create `memo/runner/entry.py`: thin `TrainingEntry` adapter that only translates the call shape.
- Create `memo/tests/test_runner_entry.py`: prove the adapter uses the same factory object and does not reimplement either operation.
- Modify `memo/entries/stream_ac.py`: remove algorithm parameter declarations and graph assembly; bind `STREAM_AC` once through `TrainingEntry`; retain the existing observability block and runner wiring.
- Create `memo/tests/test_stream_ac_entry_boundary.py`: prove the real entry binds the exported production object and contains no legacy assembly helpers.

---

### Task 1: Neutral graph-factory contract and reusable contract suite

**Files:**
- Create: `memo/memorax/factory.py`
- Create: `memo/tests/factory_contract.py`
- Create: `memo/tests/test_factory_contract.py`

**Interfaces:**
- Consumes: only Python standard-library `collections.abc`, `dataclasses`, and `typing`.
- Produces: `Scalar`, `ParameterTree`, `BuildContext`, `ChoiceParameter`, `FloatParameter`, `FactoryError`, `GraphFactory[T]`, `as_tree()`, `exact_fields()`, and `under()`.
- Produces for later tests: `FactoryCase[T]` and `assert_factory_contract(case)`.

- [ ] **Step 1: Write the failing neutral-contract tests**

Create `memo/tests/factory_contract.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import pytest

from memorax.factory import FactoryError, GraphFactory

T = TypeVar("T")


@dataclass(frozen=True)
class FactoryCase(Generic[T]):
    factory: GraphFactory[T]
    structure: dict[str, object]
    expected_parameters: dict[str, object]
    resolved: dict[str, object]
    context: dict[str, object]
    expected_built: T
    invalid_structure: dict[str, object]
    invalid_resolved: dict[str, object]


def assert_factory_contract(case: FactoryCase[Any]) -> None:
    assert isinstance(case.factory, GraphFactory)
    assert case.factory.param(case.structure) == case.expected_parameters
    assert case.factory.build(case.resolved, case.context) == case.expected_built
    assert case.factory.build(case.resolved, case.context) == case.expected_built

    with pytest.raises(FactoryError):
        case.factory.param(case.invalid_structure)
    with pytest.raises(FactoryError):
        case.factory.build(case.invalid_resolved, case.context)
```

Create `memo/tests/test_factory_contract.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from factory_contract import FactoryCase, assert_factory_contract
from memorax.factory import (
    ChoiceParameter,
    FactoryError,
    as_tree,
    exact_fields,
)


@dataclass(frozen=True)
class EchoFactory:
    declaration: ChoiceParameter = ChoiceParameter(
        valid=(False, True), search=(False, True), placeholder=False
    )

    def param(self, structure):
        exact_fields(structure, (), path="echo.structure")
        return {"enabled": self.declaration}

    def build(self, params, context):
        values = exact_fields(params, ("enabled",), path="echo.params")
        return values["enabled"], context["token"]


def test_a_production_factory_can_share_one_implementation_independent_contract():
    case = FactoryCase(
        factory=EchoFactory(),
        structure={},
        expected_parameters={"enabled": EchoFactory().declaration},
        resolved={"enabled": True},
        context={"token": "context"},
        expected_built=(True, "context"),
        invalid_structure={"unknown": True},
        invalid_resolved={"enabled": True, "unknown": 1},
    )

    assert_factory_contract(case)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"one": 1}, "missing fields: two"),
        ({"one": 1, "two": 2, "extra": 3}, "unknown fields: extra"),
    ],
)
def test_exact_fields_rejects_incomplete_or_unconsumed_input(value, message):
    with pytest.raises(FactoryError, match=message):
        exact_fields(value, ("one", "two"), path="graph")


def test_as_tree_rejects_a_non_mapping_at_its_logical_path():
    with pytest.raises(FactoryError, match="stream_ac.actor must be a mapping"):
        as_tree(3, path="stream_ac.actor")


def test_the_neutral_contract_has_no_framework_or_infrastructure_imports():
    source = (
        Path(__file__).parents[1] / "memorax" / "factory.py"
    ).read_text(encoding="utf-8")
    forbidden = ("training_sdk", "optuna", "jax", "flax")
    assert not [name for name in forbidden if name in source]
```

- [ ] **Step 2: Run the tests and verify the expected red result**

Run from `memo/`:

```powershell
uv run pytest tests/test_factory_contract.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'memorax.factory'`. This is the required red result; do not create the module before observing it.

- [ ] **Step 3: Implement the minimal neutral contract**

Create `memo/memorax/factory.py`:

```python
"""SDK-neutral contracts shared by recursively built computation graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

Scalar: TypeAlias = bool | int | float | str
ParameterTree: TypeAlias = dict[str, object]
BuildContext: TypeAlias = Mapping[str, object]

T_co = TypeVar("T_co", covariant=True)
R = TypeVar("R")


@dataclass(frozen=True)
class ChoiceParameter:
    valid: tuple[Scalar, ...]
    search: tuple[Scalar, ...]
    placeholder: Scalar


@dataclass(frozen=True)
class FloatParameter:
    valid: tuple[float | None, float | None]
    search: tuple[float, float]
    placeholder: float
    log: bool = False


class FactoryError(ValueError):
    """A graph factory rejected its local structure or parameter subtree."""


@runtime_checkable
class GraphFactory(Protocol[T_co]):
    def param(self, structure: Mapping[str, object]) -> ParameterTree: ...

    def build(
        self,
        params: Mapping[str, object],
        context: BuildContext,
    ) -> T_co: ...


def as_tree(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FactoryError(f"{path} must be a mapping")
    return value


def exact_fields(
    tree: Mapping[str, object],
    expected: tuple[str, ...],
    *,
    path: str,
) -> dict[str, object]:
    names = set(tree)
    required = set(expected)
    missing = sorted(required - names)
    unknown = sorted(names - required)
    problems = []
    if missing:
        problems.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown fields: {', '.join(unknown)}")
    if problems:
        raise FactoryError(f"{path}: {'; '.join(problems)}")
    return {name: tree[name] for name in expected}


def under(path: str, operation: Callable[[], R]) -> R:
    try:
        return operation()
    except FactoryError as error:
        raise FactoryError(f"{path}: {error}") from error
```

- [ ] **Step 4: Run the focused test and verify green**

Run:

```powershell
uv run pytest tests/test_factory_contract.py -v
```

Expected: all tests in `test_factory_contract.py` pass.

- [ ] **Step 5: Run Ruff on the new unit**

Run:

```powershell
uv run ruff check memorax/factory.py tests/factory_contract.py tests/test_factory_contract.py
```

Expected: `All checks passed!` Fix only formatting/import issues in these three files and rerun the focused pytest command after any edit.

- [ ] **Step 6: Commit the neutral contract**

```powershell
git add -- memo/memorax/factory.py memo/tests/factory_contract.py memo/tests/test_factory_contract.py
git commit -m "feat(memo): define recursive graph factory contract"
```

---

### Task 2: Production StreamAC composite factory

**Files:**
- Create: `memo/memorax/algorithms/stream_ac_factory.py`
- Create: `memo/tests/test_stream_ac_factory.py`

**Interfaces:**
- Consumes: `GraphFactory`, `ParameterTree`, `BuildContext`, `ChoiceParameter`, `FloatParameter`, `FactoryError`, `as_tree()`, `exact_fields()`, and `under()` from Task 1.
- Produces: `OpaqueGraph`, `BoundaryFactory`, `StreamACAssembly`, `StreamACFactory`, `STREAM_AC_SHARED`, `STREAM_AC_STRUCTURE`, and the production singleton `STREAM_AC`.
- Contract: `STREAM_AC.param(STREAM_AC_STRUCTURE)` returns the active top-level parameter tree; `STREAM_AC.build(resolved, context)` returns a deterministic `StreamACAssembly` until children are migrated.

- [ ] **Step 1: Write the failing production-factory tests**

Create `memo/tests/test_stream_ac_factory.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from factory_contract import FactoryCase, assert_factory_contract
from memorax.algorithms.stream_ac_factory import (
    STREAM_AC,
    STREAM_AC_SHARED,
    STREAM_AC_STRUCTURE,
    BoundaryFactory,
    OpaqueGraph,
    StreamACAssembly,
    StreamACFactory,
)
from memorax.factory import FactoryError, exact_fields


def resolved():
    return {
        "shared": {
            "gamma": 0.99,
            "trace_lambda": 0.9,
            "entropy_coefficient": 1e-4,
            "meta_rl": False,
        },
        "actor": {},
        "critic": {},
        "normalization": {},
    }


def expected_assembly(context):
    return StreamACAssembly(
        shared=resolved()["shared"],
        actor=OpaqueGraph("actor", {}, context),
        critic=OpaqueGraph("critic", {}, context),
        normalization=OpaqueGraph("normalization", {}, context),
        context=context,
    )


def test_the_production_stream_ac_factory_passes_the_generic_contract():
    context = {"environment": "env", "training": "training"}
    case = FactoryCase(
        factory=STREAM_AC,
        structure=STREAM_AC_STRUCTURE,
        expected_parameters={
            "shared": STREAM_AC_SHARED,
            "actor": {},
            "critic": {},
            "normalization": {},
        },
        resolved=resolved(),
        context=context,
        expected_built=expected_assembly(context),
        invalid_structure={**STREAM_AC_STRUCTURE, "extra": {}},
        invalid_resolved={**resolved(), "extra": {}},
    )

    assert_factory_contract(case)


@dataclass
class SpyFactory:
    name: str
    calls: list[tuple[str, object, object]] = field(default_factory=list)

    def param(self, structure):
        self.calls.append(("param", structure, None))
        exact_fields(structure, ("kind",), path=f"{self.name}.structure")
        return {"declared_by": self.name}

    def build(self, params, context):
        self.calls.append(("build", params, context))
        exact_fields(params, ("value",), path=f"{self.name}.params")
        return self.name, params["value"]


def test_stream_ac_delegates_down_and_assembles_up_on_the_same_factory():
    actor = SpyFactory("actor")
    critic = SpyFactory("critic")
    normalization = SpyFactory("normalization")
    assembled = []

    def compose(*, shared, actor, critic, normalization, context):
        value = shared, actor, critic, normalization, context
        assembled.append(value)
        return value

    factory = StreamACFactory(
        shared={"gamma": STREAM_AC_SHARED["gamma"]},
        actor=actor,
        critic=critic,
        normalization=normalization,
        compose=compose,
    )
    structure = {
        "actor": {"kind": "actor-kind"},
        "critic": {"kind": "critic-kind"},
        "normalization": {"kind": "normalization-kind"},
    }

    assert factory.param(structure) == {
        "shared": {"gamma": STREAM_AC_SHARED["gamma"]},
        "actor": {"declared_by": "actor"},
        "critic": {"declared_by": "critic"},
        "normalization": {"declared_by": "normalization"},
    }

    context = {"shape": "context"}
    params = {
        "shared": {"gamma": 0.95},
        "actor": {"value": 1},
        "critic": {"value": 2},
        "normalization": {"value": 3},
    }
    expected = (
        {"gamma": 0.95},
        ("actor", 1),
        ("critic", 2),
        ("normalization", 3),
        context,
    )
    assert factory.build(params, context) == expected
    assert assembled == [expected]
    assert actor.calls == [
        ("param", {"kind": "actor-kind"}, None),
        ("build", {"value": 1}, context),
    ]


def test_child_errors_are_reported_under_the_child_path():
    factory = StreamACFactory(
        shared={},
        actor=BoundaryFactory("actor"),
        critic=BoundaryFactory("critic"),
        normalization=BoundaryFactory("normalization"),
        compose=lambda **parts: parts,
    )
    with pytest.raises(FactoryError, match="stream_ac.actor"):
        factory.param(
            {
                "actor": {"unknown": True},
                "critic": {},
                "normalization": {},
            }
        )


def test_stream_ac_factory_is_framework_and_sdk_neutral():
    source = (
        Path(__file__).parents[1]
        / "memorax"
        / "algorithms"
        / "stream_ac_factory.py"
    ).read_text(encoding="utf-8")
    forbidden = ("training_sdk", "optuna", "jax", "flax")
    assert not [name for name in forbidden if name in source]
```

- [ ] **Step 2: Run the StreamAC factory tests and verify red**

Run:

```powershell
uv run pytest tests/test_stream_ac_factory.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'memorax.algorithms.stream_ac_factory'`.

- [ ] **Step 3: Implement the production StreamAC factory**

Create `memo/memorax/algorithms/stream_ac_factory.py`:

```python
"""The recursive, SDK-neutral graph definition of StreamAC."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from memorax.factory import (
    BuildContext,
    ChoiceParameter,
    FloatParameter,
    GraphFactory,
    ParameterTree,
    as_tree,
    exact_fields,
    under,
)

T = TypeVar("T")

CHILDREN = ("actor", "critic", "normalization")
PARAMETER_GROUPS = ("shared", *CHILDREN)


@dataclass(frozen=True)
class OpaqueGraph:
    name: str
    params: Mapping[str, object]
    context: BuildContext


@dataclass(frozen=True)
class BoundaryFactory:
    """An explicit unmigrated child boundary, not a second StreamAC graph."""

    name: str

    def param(self, structure: Mapping[str, object]) -> ParameterTree:
        exact_fields(structure, (), path=f"{self.name}.structure")
        return {}

    def build(
        self,
        params: Mapping[str, object],
        context: BuildContext,
    ) -> OpaqueGraph:
        values = exact_fields(params, (), path=f"{self.name}.params")
        return OpaqueGraph(self.name, values, context)


@dataclass(frozen=True)
class StreamACAssembly:
    shared: Mapping[str, object]
    actor: object
    critic: object
    normalization: object
    context: BuildContext


Compose = Callable[..., T]


@dataclass(frozen=True)
class StreamACFactory(Generic[T]):
    shared: Mapping[str, object]
    actor: GraphFactory[object]
    critic: GraphFactory[object]
    normalization: GraphFactory[object]
    compose: Compose[T]

    def param(self, structure: Mapping[str, object]) -> ParameterTree:
        children = exact_fields(
            structure, CHILDREN, path="stream_ac.structure"
        )
        return {
            "shared": dict(self.shared),
            **{
                name: under(
                    f"stream_ac.{name}",
                    lambda name=name: getattr(self, name).param(
                        as_tree(
                            children[name],
                            path=f"stream_ac.{name}.structure",
                        )
                    ),
                )
                for name in CHILDREN
            },
        }

    def build(
        self,
        params: Mapping[str, object],
        context: BuildContext,
    ) -> T:
        groups = exact_fields(params, PARAMETER_GROUPS, path="stream_ac.params")
        shared = exact_fields(
            as_tree(groups["shared"], path="stream_ac.shared.params"),
            tuple(self.shared),
            path="stream_ac.shared.params",
        )
        built = {
            name: under(
                f"stream_ac.{name}",
                lambda name=name: getattr(self, name).build(
                    as_tree(groups[name], path=f"stream_ac.{name}.params"),
                    context,
                ),
            )
            for name in CHILDREN
        }
        return self.compose(shared=shared, context=context, **built)


STREAM_AC_SHARED: ParameterTree = {
    "gamma": FloatParameter(
        valid=(0.5, 0.9999), search=(0.9, 0.9999), placeholder=0.99
    ),
    "trace_lambda": FloatParameter(
        valid=(0.0, 1.0), search=(0.0, 1.0), placeholder=0.9
    ),
    "entropy_coefficient": FloatParameter(
        valid=(1e-8, 1.0),
        search=(1e-8, 1e-2),
        placeholder=1e-4,
        log=True,
    ),
    "meta_rl": ChoiceParameter(
        valid=(False, True), search=(False, True), placeholder=False
    ),
}

STREAM_AC_STRUCTURE = {name: {} for name in CHILDREN}


def _assemble(**parts: Any) -> StreamACAssembly:
    return StreamACAssembly(**parts)


STREAM_AC = StreamACFactory(
    shared=STREAM_AC_SHARED,
    actor=BoundaryFactory("actor"),
    critic=BoundaryFactory("critic"),
    normalization=BoundaryFactory("normalization"),
    compose=_assemble,
)
```

- [ ] **Step 4: Run the production and generic contract tests**

Run:

```powershell
uv run pytest tests/test_factory_contract.py tests/test_stream_ac_factory.py -v
```

Expected: all tests pass. If the call-order assertion fails, correct production delegation; do not weaken the spy assertions.

- [ ] **Step 5: Run Ruff on the StreamAC factory unit**

Run:

```powershell
uv run ruff check memorax/factory.py memorax/algorithms/stream_ac_factory.py tests/factory_contract.py tests/test_factory_contract.py tests/test_stream_ac_factory.py
```

Expected: `All checks passed!` Then rerun the two focused pytest files.

- [ ] **Step 6: Commit the production composite factory**

```powershell
git add -- memo/memorax/algorithms/stream_ac_factory.py memo/tests/test_stream_ac_factory.py
git commit -m "feat(memo): define StreamAC as a recursive factory"
```

---

### Task 3: Runner adapter over the same production factory

**Files:**
- Create: `memo/runner/entry.py`
- Create: `memo/tests/test_runner_entry.py`

**Interfaces:**
- Consumes: `GraphFactory[T]`, `ParameterTree`, and the production `STREAM_AC`/`STREAM_AC_STRUCTURE` from Task 2.
- Produces: `TrainingEntry[T]` with `param(structure)` and `build(params, environment, training)`.
- Contract: `TrainingEntry.factory` is the exact object it delegates to; the adapter only packages environment/training into a neutral context mapping.

- [ ] **Step 1: Write the failing adapter tests against production `STREAM_AC`**

Create `memo/tests/test_runner_entry.py`:

```python
from types import SimpleNamespace

from memorax.algorithms.stream_ac_factory import (
    STREAM_AC,
    STREAM_AC_STRUCTURE,
    StreamACAssembly,
)
from runner.entry import TrainingEntry


def resolved():
    return {
        "shared": {
            "gamma": 0.99,
            "trace_lambda": 0.9,
            "entropy_coefficient": 1e-4,
            "meta_rl": False,
        },
        "actor": {},
        "critic": {},
        "normalization": {},
    }


def test_the_adapter_delegates_to_the_exact_factory_object():
    entry = TrainingEntry(STREAM_AC)
    assert entry.factory is STREAM_AC
    assert entry.param(STREAM_AC_STRUCTURE) == STREAM_AC.param(
        STREAM_AC_STRUCTURE
    )


def test_the_adapter_only_packages_the_build_context():
    entry = TrainingEntry(STREAM_AC)
    environment = SimpleNamespace(name="environment")
    training = SimpleNamespace(name="training")

    built = entry.build(resolved(), environment, training)

    assert isinstance(built, StreamACAssembly)
    assert built == STREAM_AC.build(
        resolved(),
        {"environment": environment, "training": training},
    )
```

- [ ] **Step 2: Run the adapter tests and verify red**

Run:

```powershell
uv run pytest tests/test_runner_entry.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'runner.entry'`.

- [ ] **Step 3: Implement the minimal delegating adapter**

Create `memo/runner/entry.py`:

```python
"""Bind an SDK-neutral graph factory to the current runner call shape."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from memorax.factory import GraphFactory, ParameterTree

T = TypeVar("T")


@dataclass(frozen=True)
class TrainingEntry(Generic[T]):
    factory: GraphFactory[T]

    def param(self, structure: Mapping[str, object]) -> ParameterTree:
        return self.factory.param(structure)

    def build(
        self,
        params: Mapping[str, object],
        environment: object,
        training: object,
    ) -> T:
        return self.factory.build(
            params,
            {"environment": environment, "training": training},
        )
```

- [ ] **Step 4: Run the adapter tests and all layer tests**

Run:

```powershell
uv run pytest tests/test_runner_entry.py tests/test_factory_contract.py tests/test_stream_ac_factory.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run Ruff and commit**

Run:

```powershell
uv run ruff check runner/entry.py tests/test_runner_entry.py
```

Expected: `All checks passed!`

Commit only this task's files:

```powershell
git add -- memo/runner/entry.py memo/tests/test_runner_entry.py
git commit -m "feat(memo): add thin graph factory entry adapter"
```

---

### Task 4: Bind the real StreamAC entry and remove its second implementation

**Files:**
- Modify: `memo/entries/stream_ac.py`
- Create: `memo/tests/test_stream_ac_entry_boundary.py`

**Interfaces:**
- Consumes: the exact exported `STREAM_AC` and `STREAM_AC_STRUCTURE` from Task 2, and `TrainingEntry` from Task 3.
- Produces: module-level `ENTRY`, `param`, and `build`, all delegating to the same production `STREAM_AC` object.
- Preserves temporarily: the existing `PARTS`, `TRAINING_METRICS`, `METRICS`, `RECORD`, `run`, and `main` observability/runner code.
- Removes: `StreamACParameters`, `BACKBONE_BRANCHES`, `CREDIT_BRANCHES`, `PARAMETERS`, `_optimizer`, `_estimator`, local network construction, and direct `StreamAC(...)` construction.

- [ ] **Step 1: Write the failing entry-boundary tests**

Create `memo/tests/test_stream_ac_entry_boundary.py`:

```python
from pathlib import Path
from types import SimpleNamespace

from entries import stream_ac
from memorax.algorithms.stream_ac_factory import (
    STREAM_AC,
    STREAM_AC_STRUCTURE,
    StreamACAssembly,
)


def resolved():
    return {
        "shared": {
            "gamma": 0.99,
            "trace_lambda": 0.9,
            "entropy_coefficient": 1e-4,
            "meta_rl": False,
        },
        "actor": {},
        "critic": {},
        "normalization": {},
    }


def test_the_entry_binds_the_exact_production_factory():
    assert stream_ac.ENTRY.factory is STREAM_AC
    assert stream_ac.param(STREAM_AC_STRUCTURE) == STREAM_AC.param(
        STREAM_AC_STRUCTURE
    )


def test_the_entry_build_delegates_to_that_same_factory():
    environment = SimpleNamespace(name="environment")
    training = SimpleNamespace(name="training")

    built = stream_ac.build(resolved(), environment, training)

    assert isinstance(built, StreamACAssembly)
    assert built == STREAM_AC.build(
        resolved(),
        {"environment": environment, "training": training},
    )


def test_the_entry_contains_no_algorithm_assembly_path():
    source = (
        Path(__file__).parents[1] / "entries" / "stream_ac.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "class StreamACParameters",
        "def _optimizer",
        "def _estimator",
        "def network",
        "read_branch(",
        "return StreamAC(",
    )
    assert not [name for name in forbidden if name in source]
```

- [ ] **Step 2: Run the boundary tests and verify red against the legacy entry**

Run:

```powershell
uv run pytest tests/test_stream_ac_entry_boundary.py -v
```

Expected: failures show that `entries.stream_ac` has no `ENTRY`/`param` binding and still contains the legacy construction symbols.

- [ ] **Step 3: Replace algorithm declarations and assembly with one binding**

In `memo/entries/stream_ac.py`, replace the module docstring and imports with:

```python
"""Training entry bound to the SDK-neutral StreamAC graph factory."""

from __future__ import annotations

from training_sdk.episode import metric_names
from training_sdk.reporter import Reporter

from memorax.algorithms.stream_ac_factory import STREAM_AC
from memorax.networks.sequence import PLACES
from runner.entry import TrainingEntry
from runner.loop import EPISODE_FIELDS, drive

ENTRY = TrainingEntry(STREAM_AC)
param = ENTRY.param
build = ENTRY.build
```

Delete `StreamACParameters`, all branch maps and `PARAMETERS`, `_optimizer`, `_estimator`, and the local `build()` implementation. Keep the metric block with this exact content:

```python
PARTS: tuple[str, ...] = PLACES

TRAINING_METRICS: tuple[str, ...] = (
    "update.td_error",
    "update.actor_step_size",
    "update.critic_step_size",
    "forward.value",
    "forward.log_prob",
    "forward.entropy",
    *(
        f"update.{domain}_{reading}_norm.{part}"
        for domain in ("actor", "critic")
        for reading in ("grad", "trace")
        for part in PARTS
    ),
)

METRICS: tuple[str, ...] = metric_names("train", TRAINING_METRICS) + metric_names(
    "eval"
)

RECORD = frozenset(EPISODE_FIELDS) | set(TRAINING_METRICS)
```

Keep the existing `run()` and `main()` definitions, but ensure `run()` calls the bound module-level `build`. It is acceptable that `run()` cannot train the returned `StreamACAssembly` until child factories are migrated; do not add a legacy fallback.

- [ ] **Step 4: Run the entry boundary tests and verify green**

Run:

```powershell
uv run pytest tests/test_stream_ac_entry_boundary.py tests/test_runner_entry.py tests/test_stream_ac_factory.py tests/test_factory_contract.py -v
```

Expected: all focused layer tests pass.

- [ ] **Step 5: Run Ruff on every changed production and focused-test file**

Run:

```powershell
uv run ruff check memorax/factory.py memorax/algorithms/stream_ac_factory.py runner/entry.py entries/stream_ac.py tests/factory_contract.py tests/test_factory_contract.py tests/test_stream_ac_factory.py tests/test_runner_entry.py tests/test_stream_ac_entry_boundary.py
```

Expected: `All checks passed!` Rerun the focused pytest command after any Ruff-driven edit.

- [ ] **Step 6: Inspect the overlap with the user's pre-existing entry edits**

Run from the repository root:

```powershell
git diff -- memo/entries/stream_ac.py
git status --short
```

Expected: the entry diff is the intentional destructive factory refactor. The user's three comment-only edits disappear because their declaration/metric-adjacent blocks were replaced or normalized. All unrelated deleted experiment/config files remain unstaged and unchanged.

- [ ] **Step 7: Commit only the entry boundary**

```powershell
git add -- memo/entries/stream_ac.py memo/tests/test_stream_ac_entry_boundary.py
git commit -m "refactor(memo): bind StreamAC entry to its factory"
```

---

### Task 5: Verify the completed layer and inventory expected integration gaps

**Files:**
- No production file changes.

**Interfaces:**
- Consumes: all focused test files and production units from Tasks 1-4.
- Produces: verification evidence for user review; no inner component migration and no compatibility patches.

- [ ] **Step 1: Run the complete focused layer suite**

Run from `memo/`:

```powershell
uv run pytest tests/test_factory_contract.py tests/test_stream_ac_factory.py tests/test_runner_entry.py tests/test_stream_ac_entry_boundary.py -v
```

Expected: all focused tests pass.

- [ ] **Step 2: Run focused static verification**

Run:

```powershell
uv run ruff check memorax/factory.py memorax/algorithms/stream_ac_factory.py runner/entry.py entries/stream_ac.py tests/factory_contract.py tests/test_factory_contract.py tests/test_stream_ac_factory.py tests/test_runner_entry.py tests/test_stream_ac_entry_boundary.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Run the first broader diagnostic without fixing deferred layers**

Run:

```powershell
uv run pytest tests/test_entries.py tests/test_template.py -v
```

Expected: failure during entry discovery/template collection because the destructive intermediate entry no longer exports the old SDK `PARAMETERS`, or failure when old runtime code expects an executable agent instead of `StreamACAssembly`. Record the exact failing tests and messages for the user; do not restore `PARAMETERS` or the old build path.

- [ ] **Step 4: Run StreamAC-specific diagnostics**

Run:

```powershell
uv run pytest tests/test_stream_ac_golden.py tests/test_heads.py tests/test_initialization.py -v
```

Expected: kernel-level tests that import `memorax.algorithms.stream_ac` may still pass; entry/declaration tests may fail because inner factories and SDK translation are deferred. Report the exact split. Do not change inner component files in response.

- [ ] **Step 5: Review the final diff and commit set**

Run from the repository root:

```powershell
git log -5 --oneline
git diff HEAD~4..HEAD --check
git status --short
```

Expected: four implementation commits after the plan commit, no whitespace errors, and only the user's pre-existing unrelated deletions/modifications left unstaged. If a task required a small correction commit, include it in the log rather than rewriting unrelated history.

- [ ] **Step 6: Stop at the user review gate**

Report:

- the exact production `STREAM_AC` object used by generic, algorithm, adapter, and entry tests;
- the focused red-green commands and final passing counts;
- the intentionally opaque child boundaries;
- each broader diagnostic failure caused by an unmigrated recursive layer;
- confirmation that no inner component was migrated.

Do not begin actor, critic, backbone, optimizer, normalizer, HPO, or observability work until the user approves this boundary.
