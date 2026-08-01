# Parameter Catalog and Conditional Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace flat hand-written `SPACE` dictionaries with dataclass-backed parameter declarations, export catalog `parameters`, validate `valid/search/placeholder`, and sample structure-dependent parameters conditionally. One entry, `stream_ac`, is carried all the way through as the template the rest are written against later.

**Architecture:** The shared contract gains parameter-tree nodes and `EntryDescriptor.parameters`; experiment YAML keeps top-level `space` as an override document. A lightweight `training_sdk.parameters` helper turns dataclass field metadata into the contract model, so entries keep declarations beside the code that reads the resolved flat `params`. The control plane resolves overrides against the tree, asks Optuna for an empty trial, recursively calls `trial.suggest_*`, and fills inactive branches with `placeholder`.

**Tech Stack:** Python 3.12, dataclasses, pydantic v2, Optuna, uv, GitHub Actions.

> **Executed. The spec is the authority, not this file.** Tasks 1 to 3 landed as written
> in mechanism; the parameter names in them were superseded while the work ran and the
> code is what shipped. Task 4 was rewritten down to one entry: this batch delivers
> `stream_ac` as the standard template and nothing else. Read
> `2026-07-30-configuration-surface-design.md` before extending any of it.

## Global Constraints

- Complete `2026-07-31-training-evaluation-sections.md` before starting this plan. This plan assumes `CONTRACT_VERSION == 5`, `EnvironmentConfig.seed`, `TrainingConfig`, and `EvaluationConfig`.
- Never run pytest or docker on this machine. Tests are written here and executed in GitHub Actions.
- Static checks are allowed: run `uv run ruff check` for the changed packages named in each task.
- Commit and push before triggering remote tests; `workflow_dispatch` runs against the remote ref.
- Work on `feature/rtrrl-lru-paper-parity`; do not commit to `main`.
- Stage explicit paths only. Do not use `git add -A` or `git add .`.
- Do not add rationale comments to code or configuration files.
- `CONTRACT_VERSION` becomes `6` in this plan.
- The YAML top-level key remains `space`; only the catalog entry field changes from `space` to `parameters`.
- This plan does not implement OBGD `bound/base` decomposition or the metrics changes.
- Only `stream_ac` is in scope. `upstream_stream_ac`, `rtrrl`, `rtrrl_aaai` and the experiment YAML migration are out of this batch; memo's catalog does not build and its suite stays red until they follow.
- Tests run locally in WSL with virtualenvs and caches outside the repository. After changing `training-sdk`, reinstall it into memo's environment (`uv sync --group development --reinstall-package training-sdk`) or memo keeps importing the old copy.

---

## File Structure

- `training-sdk/src/training_sdk/contract.py`: pydantic wire models for `ParameterSpec`, `StructureSpec`, and `EntryDescriptor.parameters`.
- `training-sdk/src/training_sdk/parameters.py`: dataclass declaration helper used by entry modules. It depends only on the SDK contract, not on JAX, Brax, Flax, or Optuna.
- `memo/runner/catalog.py`, `rtrrl/scripts/build_catalog.py`, `rtrrl/infra/mock-trainer/scripts/build_catalog.py`: discover entries with `PARAMETERS`, not `SPACE`.
- `rtrrl/infra/control-plane/src/trainer_infra/space.py`: resolve YAML overrides, validate bounds, flatten active parameters, and sample with Optuna.
- `rtrrl/infra/control-plane/src/trainer_infra/study.py`: ask empty trials and reject unsupported grid searches based on the resolved parameter tree.
- `rtrrl/infra/control-plane/src/trainer_infra/preflight.py`, `launch.py`, `loop.py`: carry resolved parameters instead of resolved flat space.
- `memo/entries/*.py`, `rtrrl/entries/rtrrl_aaai.py`, `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/space.py`: declare `PARAMETERS`.

---

### Task 1: Contract Models for Parameter Trees

**Files:**
- Modify: `training-sdk/src/training_sdk/contract.py`
- Create: `training-sdk/src/training_sdk/parameters.py`
- Modify: `training-sdk/tests/test_contract.py`
- Create: `training-sdk/tests/test_parameters.py`

**Interfaces:**
- Consumes: `CONTRACT_VERSION == 5` from the preceding plan.
- Produces: `CONTRACT_VERSION == 6`; `EntryDescriptor(command, metrics, parameters)`; `ParameterSpec(value_type, valid, search, placeholder, log=False, step=1)`; `StructureSpec(placeholder, search, branches)`.

- [x] **Step 1: Write failing contract tests**

In `training-sdk/tests/test_contract.py`, replace `test_contract_version_is_five` with:

```python
def test_contract_version_is_six() -> None:
    assert CONTRACT_VERSION == 6
```

Replace `test_catalog_parses_float_int_and_choice_entries` with:

```python
def test_catalog_parses_parameter_and_structure_entries() -> None:
    catalog = Catalog.model_validate(
        {
            "contract": 6,
            "entries": {
                "agent": {
                    "command": ["python", "-m", "agent"],
                    "metrics": ["eval/episode_return"],
                    "parameters": {
                        "learning_rate": {
                            "kind": "param",
                            "value_type": "float",
                            "valid": {"type": "float", "low": 1e-9, "high": 10.0},
                            "search": {
                                "type": "float",
                                "low": 1e-4,
                                "high": 1e-2,
                                "log": True,
                            },
                            "placeholder": 0.001,
                        },
                        "optimizer": {
                            "kind": "structure",
                            "placeholder": "adam",
                            "search": ["adam", "obgd"],
                            "branches": {
                                "adam": {
                                    "b1": {
                                        "kind": "param",
                                        "value_type": "float",
                                        "valid": {"type": "float", "low": 0.0, "high": 1.0},
                                        "placeholder": 0.9,
                                    }
                                },
                                "obgd": {},
                            },
                        },
                    },
                }
            },
        }
    )

    entry = catalog.entries["agent"]
    assert entry.parameters["learning_rate"].placeholder == 0.001
    assert entry.parameters["optimizer"].branches["adam"]["b1"].placeholder == 0.9
```

Add:

```python
def test_entry_descriptor_rejects_legacy_space_field() -> None:
    with pytest.raises(ValidationError):
        EntryDescriptor.model_validate(
            {"command": ["run"], "metrics": ["m"], "space": {"x": [1]}}
        )
```

- [x] **Step 2: Write failing helper tests**

Create `training-sdk/tests/test_parameters.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pytest

from training_sdk.parameters import describe_parameters, param, structure


@dataclass(frozen=True)
class Adam:
    b1: float = param(valid=(0.0, 1.0), placeholder=0.9)


@dataclass(frozen=True)
class Agent:
    learning_rate: float = param(
        valid=(1e-9, 10.0), search=(1e-4, 1e-2), placeholder=0.001, log=True
    )
    optimizer: str = structure(
        placeholder="adam",
        search=["adam", "obgd"],
        branches={"adam": Adam, "obgd": ()},
    )


def test_dataclass_metadata_exports_parameter_tree() -> None:
    tree = describe_parameters(Agent)

    assert tree["learning_rate"].search.low == 1e-4
    assert tree["optimizer"].branches["adam"]["b1"].placeholder == 0.9


def test_search_must_be_inside_valid() -> None:
    @dataclass(frozen=True)
    class Bad:
        x: float = param(valid=(0.0, 1.0), search=(2.0, 3.0), placeholder=0.5)

    with pytest.raises(ValueError, match="x"):
        describe_parameters(Bad)


def test_placeholder_must_be_inside_valid() -> None:
    @dataclass(frozen=True)
    class Bad:
        x: int = param(valid=(1, 5), placeholder=0)

    with pytest.raises(ValueError, match="x"):
        describe_parameters(Bad)
```

- [x] **Step 3: Commit and run the remote red check**

```bash
git add training-sdk/tests/test_contract.py training-sdk/tests/test_parameters.py
git commit -m "test(contract): require parameter tree catalog entries"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: training-sdk tests fail because `training_sdk.parameters` and the new contract models do not exist.

- [x] **Step 4: Implement contract models**

In `training-sdk/src/training_sdk/contract.py`, set:

```python
CONTRACT_VERSION = 6
```

Keep `FloatSpec`, `IntSpec`, and `ChoiceSpec` for searchable domains, then add valid-domain models that permit one open side:

```python
class FloatValidSpec(_Frozen):
    type: Literal["float"]
    low: float | None = None
    high: float | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "FloatValidSpec":
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("float valid low must not exceed high")
        return self


class IntValidSpec(_Frozen):
    type: Literal["int"]
    low: int | None = None
    high: int | None = None
    step: int = 1

    @model_validator(mode="after")
    def _ordered(self) -> "IntValidSpec":
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("int valid low must not exceed high")
        if self.step < 1:
            raise ValueError("int valid step must be positive")
        return self


ValidSpec: TypeAlias = Annotated[
    FloatValidSpec | IntValidSpec | ChoiceSpec, Field(union_mode="left_to_right")
]


class ParameterSpec(_Frozen):
    kind: Literal["param"] = "param"
    value_type: Literal["float", "int", "str", "bool"]
    valid: ValidSpec
    search: SpaceEntry | None = None
    placeholder: Scalar


ParameterTree: TypeAlias = dict[str, "ParameterNode"]


class StructureSpec(_Frozen):
    kind: Literal["structure"] = "structure"
    placeholder: Scalar
    search: tuple[Scalar, ...] | None = None
    branches: dict[str, ParameterTree]

    @model_validator(mode="before")
    @classmethod
    def _from_lists(cls, value: object) -> object:
        if isinstance(value, dict) and isinstance(value.get("search"), list):
            value = dict(value)
            value["search"] = tuple(value["search"])
        return value


ParameterNode: TypeAlias = Annotated[
    ParameterSpec | StructureSpec, Field(discriminator="kind")
]
```

Change `EntryDescriptor` to:

```python
class EntryDescriptor(_Frozen):
    command: tuple[str, ...]
    metrics: tuple[str, ...]
    parameters: dict[str, ParameterNode]
```

and update its validator to keep the command/metrics checks.

- [x] **Step 5: Implement declaration helper**

Create `training-sdk/src/training_sdk/parameters.py`:

```python
from __future__ import annotations

from dataclasses import Field as DataclassField, field, fields, is_dataclass

from training_sdk.contract import (
    ChoiceSpec,
    FloatSpec,
    FloatValidSpec,
    IntSpec,
    IntValidSpec,
    ParameterNode,
    ParameterSpec,
    Scalar,
    SpaceEntry,
    StructureSpec,
    ValidSpec,
)


def param(*, valid, placeholder: Scalar, search=None, log: bool = False, step: int = 1):
    return field(
        default=placeholder,
        metadata={
            "trainer_kind": "param",
            "valid": valid,
            "search": search,
            "placeholder": placeholder,
            "log": log,
            "step": step,
        },
    )


def structure(*, placeholder: Scalar, branches: dict[str, object], search=None):
    return field(
        default=placeholder,
        metadata={
            "trainer_kind": "structure",
            "placeholder": placeholder,
            "branches": branches,
            "search": search,
        },
    )


def describe_parameters(model: type) -> dict[str, ParameterNode]:
    if not is_dataclass(model):
        raise TypeError(f"{model.__name__} must be a dataclass")
    described: dict[str, ParameterNode] = {}
    for item in fields(model):
        kind = item.metadata.get("trainer_kind")
        if kind == "param":
            described[item.name] = _parameter(item)
        elif kind == "structure":
            described[item.name] = _structure(item)
        else:
            raise ValueError(f"{model.__name__}.{item.name} is not a trainer parameter")
    return described


def _parameter(item: DataclassField) -> ParameterSpec:
    valid = _valid_domain(item.metadata["valid"], step=int(item.metadata["step"]))
    search = item.metadata["search"]
    search_spec = None if search is None else _search_domain(
        search, log=bool(item.metadata["log"]), step=int(item.metadata["step"])
    )
    placeholder = item.metadata["placeholder"]
    _check_value(item.name, "placeholder", valid, placeholder)
    if search_spec is not None:
        _check_search(item.name, valid, search_spec)
    return ParameterSpec(
        value_type=_value_type(item.type),
        valid=valid,
        search=search_spec,
        placeholder=placeholder,
    )


def _structure(item: DataclassField) -> StructureSpec:
    branches = {}
    for name, branch in item.metadata["branches"].items():
        if branch in (None, (), {}):
            branches[name] = {}
        else:
            branches[name] = describe_parameters(branch)
    search = item.metadata["search"]
    if search is not None:
        missing = set(search) - set(branches)
        if missing:
            raise ValueError(f"{item.name} search names unknown branches: {sorted(missing)}")
    placeholder = item.metadata["placeholder"]
    if placeholder not in branches:
        raise ValueError(f"{item.name} placeholder {placeholder!r} is not a branch")
    return StructureSpec(
        placeholder=placeholder,
        search=tuple(search) if search is not None else None,
        branches=branches,
    )


def _valid_domain(value, *, step: int) -> ValidSpec:
    if isinstance(value, list):
        return ChoiceSpec.model_validate(value)
    if isinstance(value, tuple) and len(value) == 2:
        low, high = value
        if all(isinstance(side, int) or side is None for side in value):
            return IntValidSpec(type="int", low=low, high=high, step=step)
        return FloatValidSpec(type="float", low=low, high=high)
    raise TypeError(f"unsupported valid domain {value!r}")


def _search_domain(value, *, log: bool, step: int) -> SpaceEntry:
    if isinstance(value, list):
        return ChoiceSpec.model_validate(value)
    if isinstance(value, tuple) and len(value) == 2:
        low, high = value
        if low is None or high is None:
            raise ValueError("search domains must be closed")
        if all(isinstance(side, int) for side in value):
            return IntSpec(type="int", low=low, high=high, step=step, log=log)
        return FloatSpec(type="float", low=low, high=high, log=log)
    raise TypeError(f"unsupported search domain {value!r}")


def _value_type(annotation) -> str:
    if annotation in (float, "float"):
        return "float"
    if annotation in (int, "int"):
        return "int"
    if annotation in (str, "str"):
        return "str"
    if annotation in (bool, "bool"):
        return "bool"
    raise TypeError(f"unsupported parameter type {annotation!r}")


def _check_value(name: str, label: str, valid: ValidSpec, value: Scalar) -> None:
    if isinstance(valid, ChoiceSpec):
        if value not in valid.choices:
            raise ValueError(f"{name} {label} {value!r} is outside valid choices")
        return
    if isinstance(valid, IntValidSpec):
        if type(value) is not int:
            raise ValueError(f"{name} {label} must be int")
        if valid.low is not None and value < valid.low:
            raise ValueError(f"{name} {label} {value!r} is below valid low {valid.low}")
        if valid.high is not None and value > valid.high:
            raise ValueError(f"{name} {label} {value!r} is above valid high {valid.high}")
        return
    if type(value) not in (int, float):
        raise ValueError(f"{name} {label} must be numeric")
    numeric = float(value)
    if valid.low is not None and numeric < valid.low:
        raise ValueError(f"{name} {label} {value!r} is below valid low {valid.low}")
    if valid.high is not None and numeric > valid.high:
        raise ValueError(f"{name} {label} {value!r} is above valid high {valid.high}")


def _check_search(name: str, valid: ValidSpec, search: SpaceEntry) -> None:
    if isinstance(search, ChoiceSpec):
        for choice in search.choices:
            _check_value(name, "search choice", valid, choice)
        return
    if search.log and search.low <= 0:
        raise ValueError(f"{name} log search low must be positive")
    _check_value(name, "search low", valid, search.low)
    _check_value(name, "search high", valid, search.high)
```

- [x] **Step 6: Run static checks and commit**

```bash
uv run ruff check training-sdk
git add training-sdk/src/training_sdk/contract.py training-sdk/src/training_sdk/parameters.py training-sdk/tests/test_contract.py training-sdk/tests/test_parameters.py
git commit -m "feat(contract): describe entry parameters as trees"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: training-sdk tests pass. Other jobs fail until their catalog builders are migrated.

---

### Task 2: Control Plane Resolves and Samples Parameter Trees

**Files:**
- Rewrite: `rtrrl/infra/control-plane/src/trainer_infra/space.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/experiment.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/study.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/preflight.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/launch.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/loop.py`
- Modify: `rtrrl/infra/control-plane/tests/test_space.py`
- Modify: `rtrrl/infra/control-plane/tests/test_study.py`
- Modify: `rtrrl/infra/control-plane/tests/test_experiments.py`
- Modify: `rtrrl/infra/control-plane/tests/helpers.py`

**Interfaces:**
- Consumes: `EntryDescriptor.parameters`.
- Produces: `ResolvedParameters`, `resolve_parameters(entry, overrides)`, `sample_parameters(trial, resolved) -> dict[str, Scalar]`, `grid_distributions(resolved) -> dict[str, CategoricalDistribution]`, and `has_unpinned_structure(resolved) -> bool`.

- [x] **Step 1: Write failing resolver and sampler tests**

Replace `rtrrl/infra/control-plane/tests/test_space.py` with tests for the new API:

```python
import optuna
import pytest
from training_sdk.contract import ChoiceSpec, EntryDescriptor

from trainer_infra.space import (
    SpaceError,
    grid_distributions,
    resolve_parameters,
    sample_parameters,
)


def make_entry(parameters: dict) -> EntryDescriptor:
    return EntryDescriptor.model_validate(
        {"command": ["run"], "metrics": ["episode_return"], "parameters": parameters}
    )


def _lr():
    return {
        "kind": "param",
        "value_type": "float",
        "valid": {"type": "float", "low": 1e-9, "high": 10.0},
        "search": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "placeholder": 0.001,
    }


def test_unknown_override_key_is_rejected() -> None:
    entry = make_entry({"learning_rate": _lr()})
    with pytest.raises(SpaceError, match="learnign_rate"):
        resolve_parameters(entry, {"learnign_rate": ChoiceSpec.model_validate([0.1])})


def test_override_outside_valid_is_rejected() -> None:
    entry = make_entry({"learning_rate": _lr()})
    with pytest.raises(SpaceError, match="learning_rate"):
        resolve_parameters(entry, {"learning_rate": ChoiceSpec.model_validate([20.0])})


def test_unsearched_parameter_uses_placeholder() -> None:
    entry = make_entry(
        {
            "learning_rate": {
                "kind": "param",
                "value_type": "float",
                "valid": {"type": "float", "low": 1e-9, "high": 10.0},
                "placeholder": 0.001,
            }
        }
    )
    resolved = resolve_parameters(entry, {})
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()

    assert sample_parameters(trial, resolved) == {"learning_rate": 0.001}


def test_inactive_branch_collapses_to_placeholders() -> None:
    entry = make_entry(
        {
            "optimizer": {
                "kind": "structure",
                "placeholder": "adam",
                "search": ["adam", "obgd"],
                "branches": {
                    "adam": {
                        "b1": {
                            "kind": "param",
                            "value_type": "float",
                            "valid": {"type": "float", "low": 0.0, "high": 1.0},
                            "search": {"type": "float", "low": 0.8, "high": 0.99},
                            "placeholder": 0.9,
                        }
                    },
                    "obgd": {
                        "kappa": {
                            "kind": "param",
                            "value_type": "float",
                            "valid": {"type": "float", "low": 0.0, "high": 100.0},
                            "search": {"type": "float", "low": 0.5, "high": 10.0},
                            "placeholder": 1.0,
                        }
                    },
                },
            }
        }
    )
    resolved = resolve_parameters(entry, {"optimizer": ChoiceSpec.model_validate(["obgd"])})
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()

    params = sample_parameters(trial, resolved)

    assert params["optimizer"] == "obgd"
    assert params["b1"] == 0.9
    assert 0.5 <= params["kappa"] <= 10.0
```

- [x] **Step 2: Write failing study tests**

In `rtrrl/infra/control-plane/tests/test_study.py`, update `ask_round` expectations:

```python
def test_ask_round_returns_empty_trials_for_manual_suggestion(tmp_path: Path) -> None:
    study = make(tmp_path)
    trials = ask_round(study, 3)

    assert len(trials) == 3
    assert all(trial.params == {} for trial in trials)
```

Add:

```python
def test_grid_rejects_unpinned_structures() -> None:
    with pytest.raises(ValueError, match="structure"):
        check_sampler("grid", has_unpinned_structure=True, grid_space={})
```

- [x] **Step 3: Write failing scalar override test**

In `rtrrl/infra/control-plane/tests/test_experiments.py`, add:

```python
from training_sdk.contract import ChoiceSpec

from tests.helpers import _document
from trainer_infra.experiment import Experiment


def test_scalar_space_value_pins_one_candidate() -> None:
    document = _document()
    document["space"] = {"learning_rate": 0.001, "normalize_reward": True}

    experiment = Experiment.model_validate(document)

    assert isinstance(experiment.space["learning_rate"], ChoiceSpec)
    assert experiment.space["learning_rate"].choices == (0.001,)
    assert experiment.space["normalize_reward"].choices == (True,)
```

- [x] **Step 4: Commit and run the remote red check**

```bash
git add rtrrl/infra/control-plane/tests/test_space.py rtrrl/infra/control-plane/tests/test_study.py rtrrl/infra/control-plane/tests/test_experiments.py
git commit -m "test(control-plane): resolve and sample parameter trees"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: control-plane tests fail because the old resolver only knows `EntryDescriptor.space`, scalar YAML values are not normalized to one-choice overrides, and `ask_round` still requires distributions.

- [x] **Step 5: Implement resolver and sampler**

In `rtrrl/infra/control-plane/src/trainer_infra/space.py`, replace the old flat functions with:

```python
from __future__ import annotations

from dataclasses import dataclass

from optuna.distributions import CategoricalDistribution
from optuna.trial import Trial
from training_sdk.contract import (
    ChoiceSpec,
    EntryDescriptor,
    FloatSpec,
    FloatValidSpec,
    IntSpec,
    IntValidSpec,
    ParameterNode,
    ParameterSpec,
    Scalar,
    SpaceEntry,
    StructureSpec,
    ValidSpec,
)


class SpaceError(ValueError):
    """The resolved search space is not usable."""


@dataclass(frozen=True)
class ResolvedParameters:
    tree: dict[str, ParameterNode]
    overrides: dict[str, SpaceEntry]


def resolve_parameters(
    entry: EntryDescriptor, overrides: dict[str, SpaceEntry]
) -> ResolvedParameters:
    declared = _flatten_names(entry.parameters)
    unknown = sorted(set(overrides) - declared)
    if unknown:
        raise SpaceError(
            "experiment declares parameters the entry does not accept: "
            f"{', '.join(unknown)}; entry declares: {', '.join(sorted(declared))}"
        )
    for name, override in overrides.items():
        node = declared[name]
        if isinstance(node, StructureSpec):
            _validate_structure_override(name, node, override)
        else:
            _validate_override(name, node, override)
    return ResolvedParameters(tree=entry.parameters, overrides=dict(overrides))


def sample_parameters(trial: Trial, resolved: ResolvedParameters) -> dict[str, Scalar]:
    params: dict[str, Scalar] = {}
    _sample_tree(trial, resolved.tree, resolved.overrides, params, active=True)
    return params

def has_unpinned_structure(resolved: ResolvedParameters) -> bool:
    return _has_unpinned_structure(resolved.tree, resolved.overrides, active=True)


def grid_distributions(resolved: ResolvedParameters) -> dict[str, CategoricalDistribution]:
    search_space: dict[str, CategoricalDistribution] = {}
    _grid_tree(resolved.tree, resolved.overrides, search_space, active=True)
    return search_space


def _flatten_names(tree: dict[str, ParameterNode]) -> dict[str, ParameterNode]:
    found: dict[str, ParameterNode] = {}

    def visit(nodes: dict[str, ParameterNode], trail: tuple[str, ...]) -> None:
        for name, node in nodes.items():
            if name in found:
                location = ".".join((*trail, name))
                raise SpaceError(f"parameter {name!r} is declared more than once at {location}")
            found[name] = node
            if isinstance(node, StructureSpec):
                for branch_name, branch in node.branches.items():
                    visit(branch, (*trail, name, branch_name))

    visit(tree, ())
    return found


def _sample_tree(
    trial: Trial,
    tree: dict[str, ParameterNode],
    overrides: dict[str, SpaceEntry],
    params: dict[str, Scalar],
    *,
    active: bool,
) -> None:
    for name, node in tree.items():
        if isinstance(node, StructureSpec):
            if active:
                choice = _structure_choice(trial, name, node, overrides)
                params[name] = choice
                for branch_name, branch in node.branches.items():
                    if branch_name == choice:
                        _sample_tree(trial, branch, overrides, params, active=True)
                    else:
                        _placeholder_tree(branch, params)
            else:
                params[name] = node.placeholder
                for branch in node.branches.values():
                    _placeholder_tree(branch, params)
        else:
            params[name] = _sample_param(trial, name, node, overrides, active=active)


def _structure_choice(
    trial: Trial,
    name: str,
    node: StructureSpec,
    overrides: dict[str, SpaceEntry],
) -> str:
    override = overrides.get(name)
    if override is not None:
        if not isinstance(override, ChoiceSpec):
            raise SpaceError(f"{name} structure override must be a choice")
        return str(_suggest(trial, name, override))
    if node.search is None:
        return str(node.placeholder)
    return str(trial.suggest_categorical(name, list(node.search)))


def _sample_param(
    trial: Trial,
    name: str,
    node: ParameterSpec,
    overrides: dict[str, SpaceEntry],
    *,
    active: bool,
) -> Scalar:
    if not active:
        return node.placeholder
    spec = overrides.get(name)
    if spec is None:
        spec = node.search
    if spec is None:
        return node.placeholder
    return _suggest(trial, name, spec)


def _suggest(trial: Trial, name: str, spec: SpaceEntry) -> Scalar:
    if isinstance(spec, FloatSpec):
        return trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    if isinstance(spec, IntSpec):
        return trial.suggest_int(name, spec.low, spec.high, step=spec.step, log=spec.log)
    if isinstance(spec, ChoiceSpec):
        return trial.suggest_categorical(name, list(spec.choices))
    raise TypeError(f"unsupported space entry for {name}: {spec!r}")


def _validate_structure_override(
    name: str, node: StructureSpec, override: SpaceEntry
) -> None:
    if not isinstance(override, ChoiceSpec):
        raise SpaceError(f"{name} structure override must be a choice")
    unknown = sorted(set(override.choices) - set(node.branches))
    if unknown:
        raise SpaceError(f"{name} chooses unknown branches: {unknown}")


def _validate_override(name: str, node: ParameterSpec, override: SpaceEntry) -> None:
    if isinstance(override, ChoiceSpec):
        for choice in override.choices:
            _inside_valid(name, node.valid, choice)
        return
    if node.value_type == "int" and isinstance(override, FloatSpec):
        raise SpaceError(f"{name} is int but override is float")
    if override.log and override.low <= 0:
        raise SpaceError(f"{name} log search low must be positive")
    _inside_valid(name, node.valid, override.low)
    _inside_valid(name, node.valid, override.high)


def _inside_valid(name: str, valid: ValidSpec, value: Scalar) -> None:
    if isinstance(valid, ChoiceSpec):
        if value not in valid.choices:
            raise SpaceError(f"{name} value {value!r} is outside valid choices")
        return
    if isinstance(valid, IntValidSpec):
        if type(value) is not int:
            raise SpaceError(f"{name} value {value!r} is not an int")
        if valid.low is not None and value < valid.low:
            raise SpaceError(f"{name} value {value!r} is below valid low {valid.low}")
        if valid.high is not None and value > valid.high:
            raise SpaceError(f"{name} value {value!r} is above valid high {valid.high}")
        return
    if not isinstance(valid, FloatValidSpec):
        raise TypeError(f"unsupported valid spec for {name}: {valid!r}")
    if type(value) not in (int, float):
        raise SpaceError(f"{name} value {value!r} is not numeric")
    numeric = float(value)
    if valid.low is not None and numeric < valid.low:
        raise SpaceError(f"{name} value {value!r} is below valid low {valid.low}")
    if valid.high is not None and numeric > valid.high:
        raise SpaceError(f"{name} value {value!r} is above valid high {valid.high}")


def _placeholder_tree(tree: dict[str, ParameterNode], params: dict[str, Scalar]) -> None:
    for name, node in tree.items():
        if isinstance(node, StructureSpec):
            params[name] = node.placeholder
            for branch in node.branches.values():
                _placeholder_tree(branch, params)
        else:
            params[name] = node.placeholder


def _has_unpinned_structure(
    tree: dict[str, ParameterNode],
    overrides: dict[str, SpaceEntry],
    *,
    active: bool,
) -> bool:
    if not active:
        return False
    for name, node in tree.items():
        if not isinstance(node, StructureSpec):
            continue
        values = _structure_values(name, node, overrides)
        if len(values) > 1:
            return True
        chosen = values[0]
        if _has_unpinned_structure(node.branches[str(chosen)], overrides, active=True):
            return True
    return False


def _grid_tree(
    tree: dict[str, ParameterNode],
    overrides: dict[str, SpaceEntry],
    search_space: dict[str, CategoricalDistribution],
    *,
    active: bool,
) -> None:
    if not active:
        return
    for name, node in tree.items():
        if isinstance(node, StructureSpec):
            values = _structure_values(name, node, overrides)
            if len(values) > 1:
                raise SpaceError("the grid sampler cannot enumerate an unpinned structure")
            if name in overrides or node.search is not None:
                search_space[name] = CategoricalDistribution(values)
            _grid_tree(node.branches[str(values[0])], overrides, search_space, active=True)
            continue
        spec = overrides.get(name)
        if spec is None:
            spec = node.search
        if spec is None:
            continue
        if not isinstance(spec, ChoiceSpec):
            raise SpaceError(f"grid sampler requires categorical search for {name}")
        search_space[name] = CategoricalDistribution(list(spec.choices))


def _structure_values(
    name: str,
    node: StructureSpec,
    overrides: dict[str, SpaceEntry],
) -> list[Scalar]:
    override = overrides.get(name)
    if override is not None:
        if not isinstance(override, ChoiceSpec):
            raise SpaceError(f"{name} structure override must be a choice")
        return list(override.choices)
    if node.search is not None:
        return list(node.search)
    return [node.placeholder]
```

- [x] **Step 6: Normalize scalar experiment overrides and update study orchestration**

In `rtrrl/infra/control-plane/src/trainer_infra/experiment.py`, add `field_validator` to the pydantic imports and add this validator to `Experiment` before `_space_is_only_algorithm`:

```python
    @field_validator("space", mode="before")
    @classmethod
    def _space_scalar_pins(cls, value: object) -> object:
        if value is None:
            return {}
        if not isinstance(value, dict):
            return value
        return {key: _space_entry(raw) for key, raw in value.items()}
```

Add this helper near `load_experiment`:

```python
def _space_entry(value: object) -> object:
    if type(value) in (str, int, float, bool):
        return [value]
    return value
```

In `rtrrl/infra/control-plane/src/trainer_infra/study.py`, change:

```python
def check_sampler(
    name: str,
    *,
    has_unpinned_structure: bool,
    grid_space: Mapping[str, CategoricalDistribution],
) -> None:
```

For `grid`, raise:

```python
    if has_unpinned_structure:
        raise ValueError("the grid sampler cannot enumerate an unpinned structure")
```

Keep the existing unsupported-sampler check.

Change `_sampler` and `create_study` so `grid` receives `grid_space`, while TPE and random do not require distributions.

Change `ask_round` to:

```python
def ask_round(study: optuna.Study, count: int) -> list[optuna.trial.Trial]:
    return [study.ask() for _ in range(count)]
```

- [x] **Step 7: Update preflight, launch, and loop**

In `preflight.py`, replace `resolve_space` with `resolve_parameters`, and return `ResolvedParameters` from `check_offline`.

In `LaunchPlan`, rename `space` to `parameters` and update every construction.

In `launch.py`, archive a `space.json` that contains the experiment override document, not the entire catalog tree:

```python
    space_payload = {
        key: (list(spec.choices) if hasattr(spec, "choices") else spec.model_dump())
        for key, spec in experiment.space.items()
    }
```

In `loop.py`, replace:

```python
built = distributions(launch.plan.space)
trials = ask_round(study, built, experiment.hpo.trials_per_round)
configs = [build_run_config(launch, t.number, t.params) for t in trials]
```

with:

```python
trials = ask_round(study, experiment.hpo.trials_per_round)
params = [sample_parameters(t, launch.plan.parameters) for t in trials]
configs = [
    build_run_config(launch, trial.number, chosen)
    for trial, chosen in zip(trials, params, strict=True)
]
```

and record/report `chosen` instead of `trial.params`.

- [x] **Step 8: Run static checks and commit**

```bash
uv run ruff check rtrrl/infra/control-plane
git add rtrrl/infra/control-plane/src/trainer_infra/space.py rtrrl/infra/control-plane/src/trainer_infra/experiment.py rtrrl/infra/control-plane/src/trainer_infra/study.py rtrrl/infra/control-plane/src/trainer_infra/preflight.py rtrrl/infra/control-plane/src/trainer_infra/launch.py rtrrl/infra/control-plane/src/trainer_infra/loop.py rtrrl/infra/control-plane/tests/test_space.py rtrrl/infra/control-plane/tests/test_study.py rtrrl/infra/control-plane/tests/test_experiments.py rtrrl/infra/control-plane/tests/helpers.py
git commit -m "feat(control-plane): sample resolved parameter trees"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: control-plane tests pass. Catalog-related tests still fail until Task 3 migrates builders and entries.

---

### Task 3: Catalog Builders Export `PARAMETERS`

**Files:**
- Modify: `memo/runner/catalog.py`
- Modify: `rtrrl/scripts/build_catalog.py`
- Modify: `rtrrl/infra/mock-trainer/scripts/build_catalog.py`
- Modify: `memo/tests/test_loop.py`
- Modify: `rtrrl/tests/test_catalog.py`
- Modify: `rtrrl/infra/mock-trainer/tests/test_catalog.py`

**Interfaces:**
- Consumes: entry modules declaring `PARAMETERS`.
- Produces: catalog JSON with entry field `parameters`.

- [x] **Step 1: Write failing catalog-builder tests**

In each catalog test, change assertions from `.space` to `.parameters`. For example in `rtrrl/tests/test_catalog.py`:

```python
def test_catalog_contains_the_entry_parameters() -> None:
    catalog = load_catalog()
    entry = catalog.entries["rtrrl_aaai"]

    assert set(entry.parameters) == set(rtrrl_aaai.PARAMETERS)
    assert entry.metrics == tuple(rtrrl_aaai.METRICS)
```

In `memo/tests/test_loop.py`, change:

```python
assert set(entry.parameters) == set(modules[name].PARAMETERS)
```

- [x] **Step 2: Commit and run the remote red checks**

```bash
git add memo/tests/test_loop.py rtrrl/tests/test_catalog.py rtrrl/infra/mock-trainer/tests/test_catalog.py
git commit -m "test(catalog): entries export parameter trees"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
gh workflow run build-aaai-image.yml --ref "$(git branch --show-current)"
```

Expected: tests fail because builders still require `SPACE` and entries do not define `PARAMETERS`.

- [x] **Step 3: Update builders**

In `memo/runner/catalog.py`, change discovery to require:

```python
for name in ("PARAMETERS", "METRICS", "main")
```

and emit:

```python
                    "parameters": dict(module.PARAMETERS),
```

Apply the same change in `rtrrl/scripts/build_catalog.py`. In `rtrrl/infra/mock-trainer/scripts/build_catalog.py`, emit the acceptance entry's `PARAMETERS`.

Remove imports of `SpaceEntry` if they become unused.

- [x] **Step 4: Run static checks and commit**

```bash
uv run ruff check memo/runner rtrrl/scripts rtrrl/infra/mock-trainer/scripts
git add memo/runner/catalog.py rtrrl/scripts/build_catalog.py rtrrl/infra/mock-trainer/scripts/build_catalog.py memo/tests/test_loop.py rtrrl/tests/test_catalog.py rtrrl/infra/mock-trainer/tests/test_catalog.py
git commit -m "feat(catalog): export entry parameter trees"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Expected: tests still fail because entry modules have not yet declared `PARAMETERS`. That failure closes in Task 4.

---

### Task 4: `stream_ac` Declares Its Parameters

**Files:**
- Modify: `memo/entries/stream_ac.py`
- Modify: `memo/memorax/rl/updates.py`
- Modify: `memo/memorax/rl/normalization.py`
- Modify: `memo/memorax/networks/torso.py`

**Interfaces:**
- Consumes: `param`, `structure`, `describe_parameters`, `read_branch` from Task 1.
- Produces: `stream_ac.PARAMETERS`, and a `build` that reads a component out of the branch in force.

This batch delivers **one entry only**. `stream_ac` is the template the other three
are written against later; `upstream_stream_ac`, `rtrrl` and `rtrrl_aaai` are out of
scope here and keep their `SPACE`, so `memo`'s catalog does not build and its suite
stays red. That is the intended state: the contract is what this batch must get right,
not an end-to-end run.

- [x] **Step 1: Declare the components beside their implementations**

The bound and base rules in `memorax/rl/updates.py`, the running and discounted
normalisers in `memorax/rl/normalization.py`, the backbones in
`memorax/networks/torso.py`. A component holds declarations and no methods.

- [x] **Step 2: Declare the entry**

`stream_ac` names structures and the parameters no structure holds. An optimiser per
role, a normaliser per stream. Nothing a component already declares is restated.

- [x] **Step 3: Read components rather than fields**

`build` uses `read_branch` for each structure. What remains hand-written is only what
this kernel cannot hold, and each such case raises naming the phase that removes it.

---

## Self-Review

**Spec coverage.** This plan covers §3 `valid/search/placeholder`, YAML scalar pins, catalog `parameters`, manifest flat full params, and no Python-side `params.get` defaults for declared parameters. It covers §4 structure trees, inactive-branch placeholder collapse, conditional Optuna suggestions, grid rejection for unpinned structures, and the decision not to add a general cross-parameter constraint system. It covers the StreamAC `eps` split and the AAAI "no eps" decision. It does not cover OBGD `bound/base` decomposition or metric logging.

**Placeholder scan.** The word `placeholder` appears only as the domain term from the spec. Every step names concrete files, commands, expected failures, and expected passing states.

**Type consistency.** The plan consistently uses `PARAMETERS`, `EntryDescriptor.parameters`, `ResolvedParameters`, `resolve_parameters`, `sample_parameters`, `optimizer_eps`, and `normalization_eps`.
