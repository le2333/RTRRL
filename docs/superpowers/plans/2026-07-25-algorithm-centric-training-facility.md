# Algorithm-Centric Training Facility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `trainerctl run experiment.yaml` executes one complete Optuna study for
one algorithm on AWS Batch from the local machine, in the foreground, and stops
the whole launch on the first abnormal exit.

**Architecture:** The shared contract (catalog schema, run configuration schema)
lives in `training-sdk`, which is the only package installed into images and
provides both the algorithm-facing reporter and the container entry point
(`python -m training_sdk.worker`). The control plane (`trainer_infra`) resolves
configuration, asks Optuna for parameters, uploads run configurations to S3,
submits one Batch job per group of serially-executed runs, waits, reads the
score objects the workers uploaded, and tells Optuna. The control plane never
reads Aim.

**Tech Stack:** Python 3.12, uv, pydantic 2, Optuna 4.9, aim 3.28, rerun-sdk
0.34, boto3, pytest, ruff, AWS Batch, Amazon S3, Amazon ECR.

**Spec:** `docs/superpowers/specs/2026-07-25-algorithm-centric-training-facility-design.md`

## Global Constraints

- Python `>=3.12,<3.13`; ruff `line-length = 100`; tests are pytest.
- **The development machine never builds or runs a container image.** Images are
  built only by `.github/workflows/build-infra-acceptance-image.yml`, and
  containers run only as AWS Batch jobs. The machine has 2 cores, 2.2 GiB of
  usable memory, and under 5 GiB of free disk: a single GPU image does not fit.
  No task may add a `docker build`, `docker run`, or image pull step.
- **Anything heavy runs on the `dev-*` Batch queues, not locally.** That is what
  those queues exist for. Local tests may start real Aim, Rerun, moto, and
  short-lived worker subprocesses — these are ordinary Python processes with a
  peak well under 1 GiB — but never a real training framework, a real GPU
  workload, or a container. If a local test needs more than about 1 GiB or more
  than a minute, it belongs on `dev-*` instead.
- `dev-*` queues are for infrastructure development only. Delivered
  `trainerctl run` workflows use `run-*` queues; selecting `dev-*` requires the
  explicit `--queues dev` flag added in Task 16.
- Both packages are managed with uv. Run commands with
  `uv run --project <dir> ...` from the repository root.
- `CONTRACT_VERSION = 2`, defined once in `training_sdk.contract` and imported
  everywhere. No second definition.
- No changes under `memo/`. No changes to `infra/submit.sh` or any historical
  Batch entry point.
- No new Batch queues, compute environments, or instance types. Only the four
  `run-*` queues are used by delivered commands; `dev-*` queues are never
  selected by `trainer_infra`.
- No retries anywhere: no Batch attempts above one, no boto3 retry configuration
  beyond the SDK default, no resampling, no resume.
- No hand-written test double for Aim, Optuna, or Rerun. Those are exercised
  against the real packages. Object storage is exercised against a local
  S3-compatible server, never a fake. Hand-written doubles are allowed only for
  the AWS control-plane clients whose behaviour is a request/response contract
  rather than stored state: Batch, ECR, CloudWatch Logs, and `head_bucket`.
- The local S3 server fixture is defined once, in
  `training-sdk/src/training_sdk/testing.py`, and both packages load it with
  `pytest_plugins = ["training_sdk.testing"]`. Do not copy it into a second
  `conftest.py`.
- Old modules are deleted only in Task 19, after the new path works end to end.
- Several tests import fixtures from sibling test modules (`tests.helpers`,
  `tests.test_reporter`). Neither package has a `tests/__init__.py` today, so
  Task 1 creates one in each package and puts the project root on the test path:
  `training-sdk` gains `[tool.pytest.ini_options] pythonpath = ["."]`, and the
  control plane's existing `pythonpath = ["src"]` becomes `["src", "."]`. Without
  this the imports fail with `ModuleNotFoundError: No module named 'tests'`.
- Every task ends with a commit.

---

## File Structure

**`training-sdk/src/training_sdk/`** — installed into images, imported by
algorithms, and run as the container entry point.

| File | Responsibility |
| --- | --- |
| `contract.py` | `CONTRACT_VERSION`, catalog and run configuration schemas. Shared with the control plane. |
| `episode.py` | The `Episode` value type handed to the Rerun sink (moved from `types.py` unchanged). |
| `reporter.py` | Algorithm-facing API. Fans one report out to the sinks. |
| `sinks/aim.py` | Aim RPC sink. |
| `sinks/rerun.py` | Episode recording, upload, local delete. |
| `sinks/metrics.py` | Append-only JSONL metrics file; doubles as the heartbeat. |
| `score.py` | Window selection and reduction over a metrics file. |
| `objects.py` | Minimal S3 get/put/delete used by the worker and the Rerun sink. |
| `worker.py` | Manifest loop, child process, heartbeat watch, score upload. |

Deleted in Task 19: `spool.py`, `storage.py`, `bootstrap.py`, `context.py`,
`execution.py`, `run.py`, `aim_adapter.py`, `rerun_adapter.py`, `types.py`.

**`rtrrl/infra/control-plane/src/trainer_infra/`**

| File | Responsibility |
| --- | --- |
| `experiment.py` | Experiment file schema and loader. |
| `space.py` | Space merge, validation, Optuna distribution construction. |
| `queues.py` | `instance_type` to queue and job definition table. |
| `images.py` | Tag to digest resolution and catalog label decoding. |
| `preflight.py` | All pre-spend checks; produces a `LaunchPlan`. |
| `launch.py` | Launch id, archive directory, `launch.json`, trial to run configuration. |
| `packing.py` | Round to jobs, manifests, S3 upload. |
| `backends/base.py` | Execution backend protocol. |
| `backends/batch.py` | AWS Batch backend: submit, poll, terminate, log tail. |
| `backends/local.py` | Local process backend used by the end-to-end gate. |
| `study.py` | Optuna study creation, ask, tell. |
| `loop.py` | The control loop. |
| `report.py` | Round summary and final report. |
| `cli.py` | `validate` and `run`. |

Deleted in Task 19: `aim_reader.py`, `aim_scratch.py`, `aim_process_gate.py`,
`sampling.py`, `materialize.py`, `resolve.py`, `controller.py`, `execution.py`,
`models.py`, `loaders.py`, `aws_profiles.py`.

Kept unchanged: `heavy_tests.py`, `heavy_test_cli.py`, `facility_control.py`,
`ecr.py`, `image_catalog.py`, `adapters/s3.py`, `scripts/deploy_facility.py`
(except the log group change in Task 17).

---

## Phase 1: Contract and Configuration

No network, no AWS, no Optuna. Ends with `trainerctl validate` working against a
catalog file on disk.

### Task 1: Contract schemas

**Files:**
- Create: `training-sdk/src/training_sdk/contract.py`
- Create: `training-sdk/tests/__init__.py` (empty)
- Create: `rtrrl/infra/control-plane/tests/__init__.py` (empty)
- Modify: `training-sdk/pyproject.toml` (add `[tool.pytest.ini_options] pythonpath = ["."]`)
- Modify: `rtrrl/infra/control-plane/pyproject.toml` (`pythonpath = ["src"]` becomes `["src", "."]`)
- Test: `training-sdk/tests/test_contract.py`

**Interfaces:**
- Produces: `CONTRACT_VERSION: int`; `FloatSpec`, `IntSpec`, `ChoiceSpec`,
  `SpaceEntry`, `EntryDescriptor`, `Catalog`, `LoggingConfig`, `ScoreConfig`,
  `RunConfig` (all pydantic `BaseModel` except `SpaceEntry`, which is a union
  alias); `Scalar = int | float | str | bool`.

- [ ] **Step 1: Write the failing test**

```python
# training-sdk/tests/test_contract.py
import pytest
from pydantic import ValidationError

from training_sdk.contract import (
    CONTRACT_VERSION,
    Catalog,
    ChoiceSpec,
    FloatSpec,
    RunConfig,
)


def test_contract_version_is_two() -> None:
    assert CONTRACT_VERSION == 2


def test_catalog_parses_float_and_choice_entries() -> None:
    catalog = Catalog.model_validate(
        {
            "contract": 2,
            "entries": {
                "brax_ppo": {
                    "command": ["python", "-m", "brax_ppo.train"],
                    "source_hash": "sha256:41b0",
                    "metrics": ["episode_return"],
                    "space": {
                        "total_steps": [128],
                        "learning_rate": {
                            "type": "float",
                            "low": 1e-6,
                            "high": 1e-2,
                            "log": True,
                        },
                    },
                }
            },
        }
    )
    entry = catalog.entries["brax_ppo"]
    assert isinstance(entry.space["total_steps"], ChoiceSpec)
    assert entry.space["total_steps"].choices == (128,)
    assert isinstance(entry.space["learning_rate"], FloatSpec)
    assert entry.space["learning_rate"].log is True


def test_choice_spec_rejects_non_scalar_choices() -> None:
    with pytest.raises(ValidationError):
        Catalog.model_validate(
            {
                "contract": 2,
                "entries": {
                    "e": {
                        "command": ["run"],
                        "source_hash": "sha256:0",
                        "metrics": ["m"],
                        "space": {"hidden_sizes": [[256, 256]]},
                    }
                },
            }
        )


def test_run_config_round_trips() -> None:
    payload = {
        "contract": 2,
        "run_id": "sweep-20260725-051400-t7",
        "experiment": "locomotion",
        "name": "sweep",
        "launch_id": "20260725-051400",
        "trial": 7,
        "entry": "brax_ppo",
        "params": {"total_steps": 128, "learning_rate": 0.0003},
        "logging": {"aim": "aim://127.0.0.1:53801", "every_steps": 1},
        "score": {
            "metric": "episode_return",
            "window_steps": [0, 128],
            "reduce": "mean",
            "direction": "maximize",
            "non_finite": "worst",
            "s3": "s3://bucket/score.json",
        },
    }
    config = RunConfig.model_validate(payload)
    assert RunConfig.model_validate(config.model_dump(mode="json")) == config
    assert config.model_dump(mode="json", exclude_none=True) == payload


def test_inverted_bounds_are_rejected() -> None:
    for spec in ({"type": "float", "low": 2.0, "high": 1.0},
                 {"type": "int", "low": 2, "high": 1},
                 {"type": "int", "low": 1, "high": 2, "step": 0}):
        with pytest.raises(ValidationError):
            ChoiceSpec.model_validate([])  # empty choice list
        with pytest.raises(ValidationError):
            Catalog.model_validate(
                {"contract": 2,
                 "entries": {"e": {"command": ["run"], "source_hash": "sha256:0",
                                   "metrics": ["m"], "space": {"total_steps": spec}}}}
            )


def test_empty_command_metrics_and_window_order_are_rejected() -> None:
    ...
```

The round-trip assertion is split deliberately: the property that matters is
that a configuration survives being written to S3 and read back by the worker,
and the compact wire form omits unset optional fields. A single
`model_dump(mode="json") == payload` would instead force the schema to suppress
`None` fields, which on pydantic 2.8 has no supported spelling.

Write the two rejection tests out in full rather than leaving the ellipsis: one
`pytest.raises(ValidationError)` per invariant the module declares — inverted
`FloatSpec` bounds, inverted `IntSpec` bounds, non-positive `IntSpec.step`, empty
`ChoiceSpec`, empty `EntryDescriptor.command`, empty `EntryDescriptor.metrics`,
and inverted `ScoreConfig.window_steps`. A validator with no failing-path test is
a claim, not a guarantee. Also exercise `IntSpec` in the catalog parsing test so
all three arms of the union are covered.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project training-sdk pytest tests/test_contract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training_sdk.contract'`

- [ ] **Step 3: Write the implementation**

```python
# training-sdk/src/training_sdk/contract.py
from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = 2

Scalar: TypeAlias = int | float | str | bool


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FloatSpec(_Frozen):
    type: Literal["float"]
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "FloatSpec":
        if self.low > self.high:
            raise ValueError("float low must not exceed high")
        return self


class IntSpec(_Frozen):
    type: Literal["int"]
    low: int
    high: int
    step: int = 1
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "IntSpec":
        if self.low > self.high:
            raise ValueError("int low must not exceed high")
        if self.step < 1:
            raise ValueError("int step must be positive")
        return self


class ChoiceSpec(_Frozen):
    choices: tuple[Scalar, ...]

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value: object) -> object:
        if isinstance(value, list):
            return {"choices": value}
        return value

    @model_validator(mode="after")
    def _non_empty(self) -> "ChoiceSpec":
        if not self.choices:
            raise ValueError("choice list must not be empty")
        return self


SpaceEntry: TypeAlias = Annotated[
    FloatSpec | IntSpec | ChoiceSpec, Field(union_mode="left_to_right")
]


class EntryDescriptor(_Frozen):
    command: tuple[str, ...]
    source_hash: str
    metrics: tuple[str, ...]
    space: dict[str, SpaceEntry]

    @model_validator(mode="after")
    def _non_empty(self) -> "EntryDescriptor":
        if not self.command:
            raise ValueError("command must not be empty")
        if not self.metrics:
            raise ValueError("metrics must not be empty")
        return self


class Catalog(_Frozen):
    contract: int
    entries: dict[str, EntryDescriptor]


class LoggingConfig(_Frozen):
    aim: str
    every_steps: int
    rerun_s3: str | None = None
    rerun_every_episodes: int | None = None


class ScoreConfig(_Frozen):
    metric: str
    window_steps: tuple[int, int]
    reduce: Literal["mean", "median", "min", "max", "last"]
    direction: Literal["maximize", "minimize"]
    non_finite: Literal["worst"] | float
    s3: str

    @model_validator(mode="after")
    def _ordered(self) -> "ScoreConfig":
        if self.window_steps[0] > self.window_steps[1]:
            raise ValueError("window_steps must be ordered")
        return self


class RunConfig(_Frozen):
    contract: int
    run_id: str
    experiment: str
    name: str
    launch_id: str
    trial: int
    entry: str
    digest: str
    source_hash: str
    params: dict[str, Scalar]
    logging: LoggingConfig
    score: ScoreConfig
```

`digest` and `source_hash` travel with every run configuration because the Aim
sink writes them onto the run: an archived run that cannot be traced back to the
image it ran in and the algorithm source it ran is not usable as a record. The
control plane knows both at launch time — the digest from image resolution, the
source hash from the catalog entry — so they are filled in once by
`build_run_config` and never derived inside the container.

`ChoiceSpec` must be listed last in the union so that a mapping with `type` is
never coerced into a choice list.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project training-sdk pytest tests/test_contract.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add training-sdk/src/training_sdk/contract.py training-sdk/tests/test_contract.py \
        training-sdk/tests/__init__.py training-sdk/pyproject.toml \
        rtrrl/infra/control-plane/tests/__init__.py rtrrl/infra/control-plane/pyproject.toml
git commit -m "feat(sdk): add contract v2 catalog and run configuration schemas"
```

---

### Task 2: Space resolution

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/space.py`
- Test: `rtrrl/infra/control-plane/tests/test_space.py`

**Interfaces:**
- Consumes: `training_sdk.contract.{EntryDescriptor, SpaceEntry, ChoiceSpec, FloatSpec, IntSpec}`
- Produces:
  - `resolve_space(entry: EntryDescriptor, overrides: dict[str, SpaceEntry]) -> dict[str, SpaceEntry]`
  - `distributions(space: dict[str, SpaceEntry]) -> dict[str, optuna.distributions.BaseDistribution]`
  - `minimum_total_steps(space: dict[str, SpaceEntry]) -> int`
  - `SpaceError(ValueError)`

- [ ] **Step 1: Write the failing test**

```python
# rtrrl/infra/control-plane/tests/test_space.py
import optuna
import pytest
from training_sdk.contract import EntryDescriptor

from trainer_infra.space import (
    SpaceError,
    distributions,
    minimum_total_steps,
    resolve_space,
)


def make_entry(space: dict) -> EntryDescriptor:
    return EntryDescriptor.model_validate(
        {
            "command": ["run"],
            "source_hash": "sha256:0",
            "metrics": ["episode_return"],
            "space": space,
        }
    )


def test_override_replaces_entry_by_key() -> None:
    entry = make_entry(
        {
            "total_steps": {"type": "int", "low": 1, "high": 1000},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2},
        }
    )
    resolved = resolve_space(entry, {"total_steps": [128]})
    assert resolved["total_steps"].choices == (128,)
    assert resolved["learning_rate"].high == 1e-2


def test_unknown_override_key_is_rejected() -> None:
    entry = make_entry({"total_steps": [128]})
    with pytest.raises(SpaceError, match="learnign_rate"):
        resolve_space(entry, {"learnign_rate": [0.1]})


def test_space_without_total_steps_is_rejected() -> None:
    entry = make_entry({"learning_rate": [0.1]})
    with pytest.raises(SpaceError, match="total_steps"):
        resolve_space(entry, {})


def test_distributions_cover_every_key() -> None:
    entry = make_entry(
        {
            "total_steps": [128],
            "env": ["walker2d", "ant"],
            "num_envs": {"type": "int", "low": 256, "high": 1024, "step": 256},
            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        }
    )
    built = distributions(resolve_space(entry, {}))
    assert set(built) == {"total_steps", "env", "num_envs", "learning_rate"}
    assert isinstance(built["total_steps"], optuna.distributions.CategoricalDistribution)
    assert isinstance(built["num_envs"], optuna.distributions.IntDistribution)
    assert built["learning_rate"].log is True


def test_minimum_total_steps_uses_smallest_producible_value() -> None:
    entry = make_entry({"total_steps": {"type": "int", "low": 100, "high": 900}})
    assert minimum_total_steps(resolve_space(entry, {})) == 100
    entry = make_entry({"total_steps": [900, 300]})
    assert minimum_total_steps(resolve_space(entry, {})) == 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project rtrrl/infra/control-plane pytest tests/test_space.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainer_infra.space'`

- [ ] **Step 3: Write the implementation**

```python
# rtrrl/infra/control-plane/src/trainer_infra/space.py
from __future__ import annotations

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from training_sdk.contract import ChoiceSpec, EntryDescriptor, FloatSpec, IntSpec, SpaceEntry

TOTAL_STEPS = "total_steps"


class SpaceError(ValueError):
    """The resolved search space is not usable."""


def resolve_space(
    entry: EntryDescriptor, overrides: dict[str, SpaceEntry]
) -> dict[str, SpaceEntry]:
    unknown = sorted(set(overrides) - set(entry.space))
    if unknown:
        declared = ", ".join(sorted(entry.space))
        raise SpaceError(
            f"experiment declares parameters the entry does not accept: "
            f"{', '.join(unknown)}; entry declares: {declared}"
        )
    resolved = dict(entry.space) | dict(overrides)
    if TOTAL_STEPS not in resolved:
        raise SpaceError(f"entry must declare the reserved parameter {TOTAL_STEPS}")
    return resolved


def distributions(space: dict[str, SpaceEntry]) -> dict[str, BaseDistribution]:
    built: dict[str, BaseDistribution] = {}
    for key, spec in space.items():
        if isinstance(spec, ChoiceSpec):
            built[key] = CategoricalDistribution(choices=list(spec.choices))
        elif isinstance(spec, IntSpec):
            built[key] = IntDistribution(
                low=spec.low, high=spec.high, step=spec.step, log=spec.log
            )
        elif isinstance(spec, FloatSpec):
            built[key] = FloatDistribution(low=spec.low, high=spec.high, log=spec.log)
        else:  # pragma: no cover - the union is closed
            raise SpaceError(f"unsupported space entry for {key}")
    return built


def minimum_total_steps(space: dict[str, SpaceEntry]) -> int:
    spec = space[TOTAL_STEPS]
    if isinstance(spec, ChoiceSpec):
        values = [value for value in spec.choices if isinstance(value, int)]
        if len(values) != len(spec.choices):
            raise SpaceError(f"{TOTAL_STEPS} choices must all be integers")
        return min(values)
    if isinstance(spec, IntSpec):
        return spec.low
    raise SpaceError(f"{TOTAL_STEPS} must be an integer range or an integer choice list")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --project rtrrl/infra/control-plane pytest tests/test_space.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/space.py \
        rtrrl/infra/control-plane/tests/test_space.py
git commit -m "feat(control-plane): resolve search space from catalog and overrides"
```

---

### Task 3: Experiment file

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/experiment.py`
- Create: `rtrrl/infra/control-plane/examples/experiment-acceptance.yaml`
- Test: `rtrrl/infra/control-plane/tests/test_experiment.py`

**Interfaces:**
- Produces: `Experiment` (pydantic model with fields `experiment`, `name`,
  `description`, `image`, `entry`, `storage`, `compute`, `hpo`, `space`,
  `score`, `logging`), `Compute`, `Hpo`, `ScoreSpec`, `LoggingSpec`,
  `load_experiment(path: Path) -> Experiment`.
- `ScoreSpec` carries every `training_sdk.contract.ScoreConfig` field except
  `s3`, which is generated per run.
- `LoggingSpec` carries `aim`, `every_steps`, `rerun_every_episodes | None`.

- [ ] **Step 1: Write the failing test**

```python
# rtrrl/infra/control-plane/tests/test_experiment.py
from pathlib import Path

import pytest
from pydantic import ValidationError

from trainer_infra.experiment import load_experiment

EXAMPLE = Path("examples/experiment-acceptance.yaml")


def test_example_file_loads() -> None:
    experiment = load_experiment(EXAMPLE)
    assert experiment.experiment == "infra-acceptance"
    assert experiment.entry == "brax_ppo_acceptance"
    assert experiment.compute.instance_type == "c7a.medium"
    assert experiment.hpo.trials_per_round >= experiment.hpo.parallel_jobs
    assert experiment.space["total_steps"].choices == (128,)


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(EXAMPLE.read_text() + "\ngroups: {}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="groups"):
        load_experiment(path)


def test_parallel_jobs_may_not_exceed_trials_per_round(tmp_path: Path) -> None:
    text = EXAMPLE.read_text().replace("parallel_jobs: 2", "parallel_jobs: 99")
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError, match="parallel_jobs"):
        load_experiment(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project rtrrl/infra/control-plane pytest tests/test_experiment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainer_infra.experiment'`

- [ ] **Step 3: Write the example file**

```yaml
# rtrrl/infra/control-plane/examples/experiment-acceptance.yaml
experiment: infra-acceptance
name: brax-ppo-smoke
description: Infrastructure-owned CPU acceptance sweep

image: 007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl:infra-acceptance-brax-ppo-cpu
entry: brax_ppo_acceptance
storage: s3://rtrrl-training-data

compute:
  instance_type: c7a.medium
  timeout_minutes: 60
  startup_minutes: 10
  stall_factor: 10

hpo:
  sampler: tpe
  rounds: 2
  trials_per_round: 2
  parallel_jobs: 2

space:
  env: [inverted_pendulum]
  backend: [generalized]
  total_steps: [128]
  seed: [0]
  learning_rate: {type: float, low: 1.0e-4, high: 1.0e-3, log: true}

score:
  metric: episode_return
  window_steps: [0, 128]
  reduce: mean
  direction: maximize
  non_finite: worst

logging:
  aim: aim://127.0.0.1:53801
  every_steps: 1
  rerun_every_episodes: 1
```

- [ ] **Step 4: Write the implementation**

```python
# rtrrl/infra/control-plane/src/trainer_infra/experiment.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from training_sdk.contract import SpaceEntry


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Compute(_Frozen):
    instance_type: str
    timeout_minutes: int
    startup_minutes: int
    stall_factor: int

    @model_validator(mode="after")
    def _positive(self) -> "Compute":
        if min(self.timeout_minutes, self.startup_minutes, self.stall_factor) < 1:
            raise ValueError("compute durations and stall_factor must be positive")
        return self


class Hpo(_Frozen):
    sampler: Literal["tpe", "random", "grid"]
    rounds: int
    trials_per_round: int
    parallel_jobs: int

    @model_validator(mode="after")
    def _consistent(self) -> "Hpo":
        if min(self.rounds, self.trials_per_round, self.parallel_jobs) < 1:
            raise ValueError("rounds, trials_per_round and parallel_jobs must be positive")
        if self.parallel_jobs > self.trials_per_round:
            raise ValueError("parallel_jobs must not exceed trials_per_round")
        return self


class ScoreSpec(_Frozen):
    metric: str
    window_steps: tuple[int, int]
    reduce: Literal["mean", "median", "min", "max", "last"]
    direction: Literal["maximize", "minimize"]
    non_finite: Literal["worst"] | float

    @model_validator(mode="after")
    def _ordered(self) -> "ScoreSpec":
        if self.window_steps[0] > self.window_steps[1]:
            raise ValueError("window_steps must be ordered")
        return self


class LoggingSpec(_Frozen):
    aim: str
    every_steps: int
    rerun_every_episodes: int | None = None


class Experiment(_Frozen):
    experiment: str
    name: str
    description: str
    image: str
    entry: str
    storage: str
    compute: Compute
    hpo: Hpo
    space: dict[str, SpaceEntry]
    score: ScoreSpec
    logging: LoggingSpec


def load_experiment(path: Path) -> Experiment:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Experiment.model_validate(document)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_experiment.py -v`
Expected: PASS, 3 tests

- [ ] **Step 6: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/experiment.py \
        rtrrl/infra/control-plane/examples/experiment-acceptance.yaml \
        rtrrl/infra/control-plane/tests/test_experiment.py
git commit -m "feat(control-plane): add flat experiment file schema and example"
```

---

### Task 4: Offline preflight and `trainerctl validate`

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/preflight.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/cli.py`
- Test: `rtrrl/infra/control-plane/tests/test_preflight_offline.py`

**Interfaces:**
- Produces:
  - `LaunchPlan` dataclass: `experiment: Experiment`, `entry_name: str`,
    `entry: EntryDescriptor`, `space: dict[str, SpaceEntry]`, `digest: str`,
    `queue: str`, `job_definition: str`.
  - `check_offline(experiment: Experiment, catalog: Catalog) -> dict[str, SpaceEntry]`
    raising `PreflightError`.
  - `format_space(space: dict[str, SpaceEntry]) -> str`.
  - `PreflightError(ValueError)`.
- The AWS-dependent half of preflight is added in Task 15; `LaunchPlan` fields
  `digest`, `queue`, and `job_definition` are populated there.

- [ ] **Step 1: Write the failing test**

```python
# rtrrl/infra/control-plane/tests/test_preflight_offline.py
from pathlib import Path

import pytest
import yaml
from training_sdk.contract import Catalog

from trainer_infra.experiment import Experiment, load_experiment
from trainer_infra.preflight import PreflightError, check_offline, format_space

EXAMPLE = Path("examples/experiment-acceptance.yaml")

CATALOG = Catalog.model_validate(
    {
        "contract": 2,
        "entries": {
            "brax_ppo_acceptance": {
                "command": ["python", "-m", "brax_ppo_acceptance"],
                "source_hash": "sha256:0",
                "metrics": ["episode_return", "episode_length"],
                "space": {
                    "env": ["inverted_pendulum"],
                    "backend": ["generalized"],
                    "total_steps": {"type": "int", "low": 1, "high": 100000},
                    "seed": {"type": "int", "low": 0, "high": 1000},
                    "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2},
                },
            }
        },
    }
)


def modified(tmp_path: Path, old: str, new: str) -> Experiment:
    text = EXAMPLE.read_text()
    assert old in text, f"fixture no longer contains {old!r}"
    path = tmp_path / "experiment.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return load_experiment(path)


def test_example_passes_offline_checks() -> None:
    space = check_offline(load_experiment(EXAMPLE), CATALOG)
    assert space["total_steps"].choices == (128,)


def test_unknown_entry_is_rejected(tmp_path: Path) -> None:
    experiment = modified(tmp_path, "entry: brax_ppo_acceptance", "entry: missing")
    with pytest.raises(PreflightError, match="missing"):
        check_offline(experiment, CATALOG)


def test_unsupported_contract_is_rejected() -> None:
    catalog = Catalog.model_validate(CATALOG.model_dump() | {"contract": 99})
    with pytest.raises(PreflightError, match="contract"):
        check_offline(load_experiment(EXAMPLE), catalog)


def test_metric_not_reported_by_entry_is_rejected(tmp_path: Path) -> None:
    experiment = modified(tmp_path, "metric: episode_return", "metric: reward")
    with pytest.raises(PreflightError, match="reward"):
        check_offline(experiment, CATALOG)


def test_window_beyond_smallest_total_steps_is_rejected(tmp_path: Path) -> None:
    experiment = modified(tmp_path, "window_steps: [0, 128]", "window_steps: [0, 129]")
    with pytest.raises(PreflightError, match="window"):
        check_offline(experiment, CATALOG)


def test_format_space_lists_every_key() -> None:
    space = check_offline(load_experiment(EXAMPLE), CATALOG)
    text = format_space(space)
    for key in space:
        assert key in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_preflight_offline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainer_infra.preflight'`

- [ ] **Step 3: Write the implementation**

```python
# rtrrl/infra/control-plane/src/trainer_infra/preflight.py
from __future__ import annotations

from dataclasses import dataclass

from training_sdk.contract import CONTRACT_VERSION, Catalog, ChoiceSpec, EntryDescriptor
from training_sdk.contract import SpaceEntry

from trainer_infra.experiment import Experiment
from trainer_infra.space import SpaceError, minimum_total_steps, resolve_space


class PreflightError(ValueError):
    """A launch precondition failed before anything was spent."""


@dataclass(frozen=True)
class LaunchPlan:
    experiment: Experiment
    entry_name: str
    entry: EntryDescriptor
    space: dict[str, SpaceEntry]
    digest: str
    queue: str
    job_definition: str


def check_offline(experiment: Experiment, catalog: Catalog) -> dict[str, SpaceEntry]:
    if catalog.contract != CONTRACT_VERSION:
        raise PreflightError(
            f"image declares contract {catalog.contract}; "
            f"this control plane implements contract {CONTRACT_VERSION}"
        )
    entry = catalog.entries.get(experiment.entry)
    if entry is None:
        available = ", ".join(sorted(catalog.entries))
        raise PreflightError(
            f"image does not declare entry {experiment.entry!r}; available: {available}"
        )
    if experiment.score.metric not in entry.metrics:
        reported = ", ".join(entry.metrics)
        raise PreflightError(
            f"entry {experiment.entry} does not report metric "
            f"{experiment.score.metric!r}; it reports: {reported}"
        )
    try:
        space = resolve_space(entry, experiment.space)
        budget = minimum_total_steps(space)
    except SpaceError as error:
        raise PreflightError(str(error)) from error
    if experiment.score.window_steps[1] > budget:
        raise PreflightError(
            f"score window upper bound {experiment.score.window_steps[1]} exceeds the "
            f"smallest total_steps the space can produce ({budget})"
        )
    return space


def format_space(space: dict[str, SpaceEntry]) -> str:
    lines = []
    for key in sorted(space):
        spec = space[key]
        if isinstance(spec, ChoiceSpec):
            rendered = " | ".join(repr(choice) for choice in spec.choices)
        else:
            rendered = spec.model_dump_json()
        lines.append(f"  {key}: {rendered}")
    return "resolved search space:\n" + "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_preflight_offline.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Add the `validate` command**

Replace the body of `main` in `cli.py` with a subcommand parser. Keep the
existing module importable while the old commands still exist; only add the new
`validate` path here.

```python
# rtrrl/infra/control-plane/src/trainer_infra/cli.py  (new command only)
import argparse
import json
import sys
from pathlib import Path

from training_sdk.contract import Catalog

from trainer_infra.experiment import load_experiment
from trainer_infra.preflight import PreflightError, check_offline, format_space


def validate_command(experiment_path: Path, catalog_path: Path) -> int:
    experiment = load_experiment(experiment_path)
    catalog = Catalog.model_validate(json.loads(catalog_path.read_text(encoding="utf-8")))
    try:
        space = check_offline(experiment, catalog)
    except PreflightError as error:
        print(f"preflight failed: {error}", file=sys.stderr)
        return 1
    print(format_space(space))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainerctl")
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate", help="check an experiment file")
    validate.add_argument("experiment", type=Path)
    validate.add_argument("--catalog", type=Path, required=True)
    return parser
```

Wire `main` so that `validate` dispatches to `validate_command`.

- [ ] **Step 6: Verify the command end to end**

Write the catalog fixture used by the test to
`rtrrl/infra/control-plane/tests/data/acceptance-catalog.json`, then run:

```bash
cd rtrrl/infra/control-plane
uv run trainerctl validate examples/experiment-acceptance.yaml \
  --catalog tests/data/acceptance-catalog.json
```

Expected: exit 0 and a printed space listing `backend`, `env`, `learning_rate`,
`seed`, `total_steps`.

- [ ] **Step 7: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/preflight.py \
        rtrrl/infra/control-plane/src/trainer_infra/cli.py \
        rtrrl/infra/control-plane/tests/test_preflight_offline.py \
        rtrrl/infra/control-plane/tests/data/acceptance-catalog.json
git commit -m "feat(control-plane): add offline preflight and trainerctl validate"
```

---

## Phase 2: SDK and Worker

Real Aim, real Rerun, real S3 server. No AWS, no Optuna. Ends with a worker that
runs a manifest of child processes and uploads their scores.

### Task 5: Metrics sink and reporter

**Files:**
- Create: `training-sdk/src/training_sdk/episode.py` (copy of the existing
  `types.py` `Episode` dataclass, unchanged)
- Create: `training-sdk/src/training_sdk/sinks/__init__.py` (empty)
- Create: `training-sdk/src/training_sdk/sinks/metrics.py`
- Create: `training-sdk/src/training_sdk/reporter.py`
- Test: `training-sdk/tests/test_metrics_sink.py`
- Test: `training-sdk/tests/test_reporter.py`

**Interfaces:**
- Produces:
  - `MetricsSink(path: Path)` with `report(step: int, metrics: Mapping[str, float]) -> None`
    and `close() -> None`.
  - `Reporter(config: RunConfig, scratch: Path, sinks: Sequence[Sink])` with
    `report(step, metrics)`, `log_episode(episode: Episode)`, `close()`, and
    context-manager support.
  - `Reporter.from_env() -> Reporter` reading the run configuration path from
    `TRAINER_RUN_CONFIG` and the scratch directory from `TRAINER_SCRATCH`.
  - `METRICS_FILENAME = "metrics.jsonl"`.
- A `Sink` is any object with `report`, `log_episode`, and `close`; sinks that do
  not care about episodes implement `log_episode` as a no-op.

- [ ] **Step 1: Write the failing tests**

```python
# training-sdk/tests/test_metrics_sink.py
import json
from pathlib import Path

from training_sdk.sinks.metrics import MetricsSink


def test_report_appends_one_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(path)
    sink.report(10, {"episode_return": 1.5})
    sink.report(20, {"episode_return": 2.5, "episode_length": 7})
    sink.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [
        {"step": 10, "metrics": {"episode_return": 1.5}},
        {"step": 20, "metrics": {"episode_return": 2.5, "episode_length": 7.0}},
    ]


def test_report_updates_modification_time(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(path)
    sink.report(1, {"m": 1.0})
    first = path.stat().st_mtime_ns
    sink.report(2, {"m": 2.0})
    assert path.stat().st_mtime_ns >= first
    sink.close()
```

```python
# training-sdk/tests/test_reporter.py
import json
from pathlib import Path

from training_sdk.contract import RunConfig
from training_sdk.reporter import METRICS_FILENAME, Reporter


def make_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "contract": 2,
            "run_id": "smoke-20260725-000000-t0",
            "experiment": "infra-acceptance",
            "name": "smoke",
            "launch_id": "20260725-000000",
            "trial": 0,
            "entry": "e",
            "params": {"total_steps": 4},
            "logging": {"aim": "aim://127.0.0.1:1", "every_steps": 1},
            "score": {
                "metric": "episode_return",
                "window_steps": [0, 4],
                "reduce": "mean",
                "direction": "maximize",
                "non_finite": "worst",
                "s3": "s3://bucket/score.json",
            },
        }
    )


class RecordingSink:
    def __init__(self) -> None:
        self.reports: list[tuple[int, dict[str, float]]] = []
        self.closed = False

    def report(self, step: int, metrics: dict[str, float]) -> None:
        self.reports.append((step, dict(metrics)))

    def log_episode(self, episode: object) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_reporter_fans_out_and_writes_metrics_file(tmp_path: Path) -> None:
    sink = RecordingSink()
    with Reporter(make_config(), tmp_path, sinks=[sink]) as reporter:
        reporter.report(1, {"episode_return": 3.0})
    assert sink.reports == [(1, {"episode_return": 3.0})]
    assert sink.closed is True
    written = json.loads((tmp_path / METRICS_FILENAME).read_text().strip())
    assert written == {"step": 1, "metrics": {"episode_return": 3.0}}


def test_reporter_closes_every_sink_even_when_one_raises(tmp_path: Path) -> None:
    class Failing(RecordingSink):
        def close(self) -> None:
            raise RuntimeError("sink failed to close")

    failing, healthy = Failing(), RecordingSink()
    reporter = Reporter(make_config(), tmp_path, sinks=[failing, healthy])
    try:
        reporter.close()
    except RuntimeError:
        pass
    else:  # pragma: no cover - the test asserts the raise happens
        raise AssertionError("close must propagate the failure")
    assert healthy.closed is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd training-sdk && uv run pytest tests/test_metrics_sink.py tests/test_reporter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training_sdk.sinks'`

- [ ] **Step 3: Write the implementation**

```python
# training-sdk/src/training_sdk/sinks/metrics.py
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import IO


class MetricsSink:
    """Append-only record of every report; also the worker's heartbeat."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] = self._path.open("a", encoding="utf-8")

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        line = json.dumps(
            {"step": int(step), "metrics": {str(k): float(v) for k, v in metrics.items()}},
            sort_keys=True,
        )
        self._handle.write(line + "\n")
        self._handle.flush()

    def log_episode(self, episode: object) -> None:
        return None

    def close(self) -> None:
        self._handle.close()
```

```python
# training-sdk/src/training_sdk/reporter.py
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from training_sdk.contract import RunConfig
from training_sdk.episode import Episode
from training_sdk.sinks.metrics import MetricsSink

METRICS_FILENAME = "metrics.jsonl"


class Sink(Protocol):
    def report(self, step: int, metrics: Mapping[str, float]) -> None: ...
    def log_episode(self, episode: Episode) -> None: ...
    def close(self) -> None: ...


class Reporter:
    def __init__(
        self,
        config: RunConfig,
        scratch: Path,
        sinks: Sequence[Sink] | None = None,
    ) -> None:
        self.config = config
        self.scratch = Path(scratch)
        metrics = MetricsSink(self.scratch / METRICS_FILENAME)
        self._sinks: tuple[Sink, ...] = (metrics, *(sinks or ()))

    @classmethod
    def from_env(cls) -> "Reporter":
        config_path = Path(os.environ["TRAINER_RUN_CONFIG"])
        scratch = Path(os.environ["TRAINER_SCRATCH"])
        config = RunConfig.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
        return cls(config, scratch, sinks=build_default_sinks(config, scratch))

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        for sink in self._sinks:
            sink.report(step, metrics)

    def log_episode(self, episode: Episode) -> None:
        for sink in self._sinks:
            sink.log_episode(episode)

    def close(self) -> None:
        failure: BaseException | None = None
        for sink in self._sinks:
            try:
                sink.close()
            except BaseException as error:  # noqa: BLE001 - every sink must be closed
                failure = failure or error
        if failure is not None:
            raise failure

    def __enter__(self) -> "Reporter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_default_sinks(config: RunConfig, scratch: Path) -> tuple[Sink, ...]:
    """Populated in Task 6 (Aim) and Task 7 (Rerun)."""
    return ()
```

Copy `Episode` from `types.py` into `episode.py` verbatim; `types.py` stays in
place until Task 19.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd training-sdk && uv run pytest tests/test_metrics_sink.py tests/test_reporter.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add training-sdk/src/training_sdk/episode.py \
        training-sdk/src/training_sdk/sinks/ \
        training-sdk/src/training_sdk/reporter.py \
        training-sdk/tests/test_metrics_sink.py training-sdk/tests/test_reporter.py
git commit -m "feat(sdk): add metrics sink and reporter fan-out"
```

---

### Task 6: Aim sink

**Files:**
- Create: `training-sdk/src/training_sdk/sinks/aim.py`
- Modify: `training-sdk/src/training_sdk/reporter.py` (`build_default_sinks`)
- Test: `training-sdk/tests/test_aim_sink.py`

**Interfaces:**
- Produces: `AimSink(config: RunConfig, repo: str)` with the `Sink` methods.
- The repository argument is the Aim endpoint from `config.logging.aim` in
  production and a temporary directory path in tests. Aim accepts both.

- [ ] **Step 1: Write the failing test**

This test uses a real `aim.Repo` on disk. It is the regression gate for the
read-only `Run.close()` incompatibility that reached a paid AWS run.

```python
# training-sdk/tests/test_aim_sink.py
from pathlib import Path

from aim import Repo

from training_sdk.contract import RunConfig
from training_sdk.sinks.aim import AimSink
from tests.test_reporter import make_config


def test_run_is_named_by_run_id_and_carries_launch_fields(tmp_path: Path) -> None:
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init_if_undefined=True)
    config = make_config()

    sink = AimSink(config, repo=repo_path)
    sink.report(1, {"episode_return": 2.0})
    sink.report(2, {"episode_return": 4.0})
    sink.close()

    repo = Repo.from_path(repo_path)
    runs = list(repo.iter_runs())
    assert len(runs) == 1
    run = runs[0]
    assert run.name == config.run_id
    assert run["launch_id"] == config.launch_id
    assert run["trial"] == config.trial
    assert run["params"]["total_steps"] == 4
    values = list(run.metrics())
    assert values, "the metric sequence must exist"
    run.close()


def test_reading_a_finished_run_and_closing_it_does_not_raise(tmp_path: Path) -> None:
    repo_path = str(tmp_path / "aim")
    Repo.from_path(repo_path, init_if_undefined=True)
    sink = AimSink(make_config(), repo=repo_path)
    sink.report(1, {"episode_return": 1.0})
    sink.close()

    repo = Repo.from_path(repo_path, read_only=True)
    for run in repo.iter_runs():
        run.close()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd training-sdk && uv run pytest tests/test_aim_sink.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training_sdk.sinks.aim'`

- [ ] **Step 3: Write the implementation**

```python
# training-sdk/src/training_sdk/sinks/aim.py
from __future__ import annotations

from collections.abc import Mapping

from aim import Run

from training_sdk.contract import RunConfig


class AimSink:
    def __init__(self, config: RunConfig, repo: str) -> None:
        self._run = Run(repo=repo, experiment=config.experiment)
        self._run.name = config.run_id
        self._run["launch_id"] = config.launch_id
        self._run["trial"] = config.trial
        self._run["entry"] = config.entry
        self._run["digest"] = config.digest
        self._run["source_hash"] = config.source_hash
        self._run["params"] = dict(config.params)
        self._every = config.logging.every_steps
        self._last: int | None = None

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        if self._last is not None and step - self._last < self._every:
            return
        self._last = step
        for name, value in metrics.items():
            self._run.track(float(value), name=str(name), step=int(step))

    def log_episode(self, episode: object) -> None:
        return None

    def close(self) -> None:
        self._run.close()
```

`every_steps` throttles Aim only. The first report always reaches Aim, and after
that a report is forwarded once at least `every_steps` steps have passed since the
last one that was. The comparison is against the elapsed step count rather than
`step % every_steps`, because an algorithm reporting at 1024-step intervals with
`every_steps: 1000` would otherwise never match and the run would appear empty.

The metrics file is never throttled: the worker reads it to compute the trial's
score and watches its modification time as a heartbeat, so it needs every report.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd training-sdk && uv run pytest tests/test_aim_sink.py -v`
Expected: PASS, 2 tests

- [ ] **Step 5: Wire it into `build_default_sinks`**

```python
# training-sdk/src/training_sdk/reporter.py
def build_default_sinks(config: RunConfig, scratch: Path) -> tuple[Sink, ...]:
    from training_sdk.sinks.aim import AimSink

    return (AimSink(config, repo=config.logging.aim),)
```

- [ ] **Step 6: Commit**

```bash
git add training-sdk/src/training_sdk/sinks/aim.py \
        training-sdk/src/training_sdk/reporter.py \
        training-sdk/tests/test_aim_sink.py
git commit -m "feat(sdk): add Aim sink with a real-repository regression test"
```

---

### Task 7: Object store helper and Rerun sink

**Files:**
- Create: `training-sdk/src/training_sdk/objects.py`
- Create: `training-sdk/src/training_sdk/sinks/rerun.py`
- Create: `training-sdk/src/training_sdk/testing.py` (local S3 server fixture)
- Modify: `training-sdk/pyproject.toml` (add `boto3>=1.35,<2`; add the `testing`
  extra with `moto[server]>=5,<6`)
- Modify: `rtrrl/infra/control-plane/pyproject.toml` (dev depends on `training-sdk[testing]`)
- Modify: `training-sdk/src/training_sdk/reporter.py` (`build_default_sinks`)
- Create: `training-sdk/tests/conftest.py` (one line loading the plugin)
- Test: `training-sdk/tests/test_objects.py`
- Test: `training-sdk/tests/test_rerun_sink.py`

**Interfaces:**
- Produces:
  - `split_uri(uri: str) -> tuple[str, str]` returning bucket and key.
  - `get_bytes(uri) -> bytes`, `put_bytes(uri, payload: bytes) -> None`,
    `put_file(uri, path: Path) -> None`, `exists(uri) -> bool`.
  - `client()` honouring `TRAINER_S3_ENDPOINT` when set, so the same code runs
    against AWS and against a local S3 server.
  - `RerunSink(config: RunConfig, scratch: Path)` with the `Sink` methods.
- The `s3_server` fixture yields a base URI such as `s3://trainer-test` with the
  bucket already created and `TRAINER_S3_ENDPOINT` exported.

- [ ] **Step 1: Write the failing tests**

```python
# training-sdk/src/training_sdk/testing.py
"""Pytest fixtures shared by both packages; loaded with pytest_plugins."""
import os
from collections.abc import Iterator

import boto3
import pytest
from moto.server import ThreadedMotoServer


@pytest.fixture(scope="session")
def s3_endpoint() -> Iterator[str]:
    server = ThreadedMotoServer(port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    previous = os.environ.get("TRAINER_S3_ENDPOINT")
    os.environ["TRAINER_S3_ENDPOINT"] = endpoint
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    os.environ.setdefault("AWS_DEFAULT_REGION", "eu-north-1")
    yield endpoint
    if previous is None:
        del os.environ["TRAINER_S3_ENDPOINT"]
    else:
        os.environ["TRAINER_S3_ENDPOINT"] = previous
    server.stop()


@pytest.fixture
def s3_base(s3_endpoint: str) -> str:
    bucket = "trainer-test"
    boto3.client("s3", endpoint_url=s3_endpoint).create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-north-1"},
    )
    return f"s3://{bucket}"
```

Creating the same bucket twice raises; make the fixture tolerate
`BucketAlreadyOwnedByYou` by catching `ClientError` and continuing.

Then load it from both packages with a one-line `conftest.py`:

```python
# training-sdk/tests/conftest.py
# rtrrl/infra/control-plane/tests/conftest.py  (same line, plus that package's own fixtures)
pytest_plugins = ["training_sdk.testing"]
```

`moto[server]` therefore has to be a runtime extra rather than a dev dependency
of `training-sdk`, because the control plane's test suite imports it through this
module. Declare it as `[project.optional-dependencies] testing = ["moto[server]>=5,<6"]`
and have both packages depend on `training-sdk[testing]` in their dev group.

```python
# training-sdk/tests/test_objects.py
from pathlib import Path

from training_sdk import objects


def test_put_and_get_round_trip(s3_base: str) -> None:
    uri = f"{s3_base}/round/trip.json"
    objects.put_bytes(uri, b'{"value": 1}')
    assert objects.get_bytes(uri) == b'{"value": 1}'
    assert objects.exists(uri) is True


def test_missing_object_is_not_reported_as_present(s3_base: str) -> None:
    assert objects.exists(f"{s3_base}/absent") is False


def test_put_file_uploads_contents(s3_base: str, tmp_path: Path) -> None:
    source = tmp_path / "episode.rrd"
    source.write_bytes(b"rrd-bytes")
    objects.put_file(f"{s3_base}/episodes/episode.rrd", source)
    assert objects.get_bytes(f"{s3_base}/episodes/episode.rrd") == b"rrd-bytes"
```

```python
# training-sdk/tests/test_rerun_sink.py
from pathlib import Path

import rerun as rr

from training_sdk import objects
from training_sdk.episode import Episode
from training_sdk.sinks.rerun import RerunSink
from tests.test_reporter import make_config


def make_episode(number: int) -> Episode:
    return Episode(
        number=number,
        phase="evaluation",
        start_env_steps=0,
        end_env_steps=2,
        observations=[[0.0], [1.0], [2.0]],
        actions=[[0.0], [1.0]],
        rewards=[1.0, 2.0],
        terminals=[False, True],
        truncations=[False, False],
    )


def test_episode_is_uploaded_and_local_copy_removed(s3_base: str, tmp_path: Path) -> None:
    config = make_config().model_copy(
        update={
            "logging": make_config().logging.model_copy(
                update={"rerun_s3": f"{s3_base}/episodes/", "rerun_every_episodes": 1}
            )
        }
    )
    sink = RerunSink(config, tmp_path)
    sink.log_episode(make_episode(1))
    sink.close()

    uri = f"{s3_base}/episodes/episode-000001.rrd"
    payload = objects.get_bytes(uri)
    assert payload[:3] == b"RRF" or len(payload) > 0
    assert list(tmp_path.glob("*.rrd")) == []


def test_only_every_nth_episode_is_recorded(s3_base: str, tmp_path: Path) -> None:
    config = make_config().model_copy(
        update={
            "logging": make_config().logging.model_copy(
                update={"rerun_s3": f"{s3_base}/every2/", "rerun_every_episodes": 2}
            )
        }
    )
    sink = RerunSink(config, tmp_path)
    sink.log_episode(make_episode(1))
    sink.log_episode(make_episode(2))
    sink.close()
    assert objects.exists(f"{s3_base}/every2/episode-000001.rrd") is False
    assert objects.exists(f"{s3_base}/every2/episode-000002.rrd") is True


def test_recording_is_readable_by_rerun(s3_base: str, tmp_path: Path) -> None:
    config = make_config().model_copy(
        update={
            "logging": make_config().logging.model_copy(
                update={"rerun_s3": f"{s3_base}/readable/", "rerun_every_episodes": 1}
            )
        }
    )
    sink = RerunSink(config, tmp_path)
    sink.log_episode(make_episode(1))
    sink.close()
    downloaded = tmp_path / "downloaded.rrd"
    downloaded.write_bytes(objects.get_bytes(f"{s3_base}/readable/episode-000001.rrd"))
    recording = rr.dataframe.load_recording(downloaded)
    assert recording.num_rows() > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd training-sdk && uv run pytest tests/test_objects.py tests/test_rerun_sink.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training_sdk.objects'`

- [ ] **Step 3: Write the implementation**

```python
# training-sdk/src/training_sdk/objects.py
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


@lru_cache(maxsize=1)
def client():
    return boto3.client("s3", endpoint_url=os.environ.get("TRAINER_S3_ENDPOINT") or None)


def split_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"not an s3 uri: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def get_bytes(uri: str) -> bytes:
    bucket, key = split_uri(uri)
    return client().get_object(Bucket=bucket, Key=key)["Body"].read()


def put_bytes(uri: str, payload: bytes) -> None:
    bucket, key = split_uri(uri)
    client().put_object(Bucket=bucket, Key=key, Body=payload)


def put_file(uri: str, path: Path) -> None:
    bucket, key = split_uri(uri)
    client().upload_file(str(path), bucket, key)


def exists(uri: str) -> bool:
    bucket, key = split_uri(uri)
    try:
        client().head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
            return False
        raise
    return True
```

```python
# training-sdk/src/training_sdk/sinks/rerun.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import rerun as rr

from training_sdk import objects
from training_sdk.contract import RunConfig
from training_sdk.episode import Episode


class RerunSink:
    def __init__(self, config: RunConfig, scratch: Path) -> None:
        if config.logging.rerun_s3 is None or config.logging.rerun_every_episodes is None:
            raise ValueError("rerun sink requires rerun_s3 and rerun_every_episodes")
        self._prefix = config.logging.rerun_s3.rstrip("/")
        self._every = config.logging.rerun_every_episodes
        self._scratch = Path(scratch)
        self._config = config

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        return None

    def log_episode(self, episode: Episode) -> None:
        if episode.number % self._every:
            return
        name = f"episode-{episode.number:06d}.rrd"
        path = self._scratch / name
        self._write(path, episode)
        objects.put_file(f"{self._prefix}/{name}", path)
        path.unlink()

    def _write(self, path: Path, episode: Episode) -> None:
        stream = rr.RecordingStream("training_sdk", recording_id=path.stem)
        stream.save(path)
        stream.log(
            "episode/metadata",
            rr.AnyValues(
                run_id=self._config.run_id,
                launch_id=self._config.launch_id,
                trial=self._config.trial,
                episode=episode.number,
                phase=episode.phase,
                start_env_steps=episode.start_env_steps,
                end_env_steps=episode.end_env_steps,
            ),
            static=True,
        )
        series: dict[str, Sequence[object]] = {
            "observations": episode.observations,
            "actions": episode.actions,
            "rewards": episode.rewards,
            "terminals": episode.terminals,
            "truncations": episode.truncations,
        }
        for entity, values in series.items():
            for index, value in enumerate(values):
                stream.set_time("episode_step", sequence=index)
                stream.log(f"episode/{entity}", rr.Tensor(np.asarray(value, dtype=np.float64)))
        stream.flush()
        stream.disconnect()

    def close(self) -> None:
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd training-sdk && uv run pytest tests/test_objects.py tests/test_rerun_sink.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Wire it into `build_default_sinks`**

```python
# training-sdk/src/training_sdk/reporter.py
def build_default_sinks(config: RunConfig, scratch: Path) -> tuple[Sink, ...]:
    from training_sdk.sinks.aim import AimSink
    from training_sdk.sinks.rerun import RerunSink

    sinks: list[Sink] = [AimSink(config, repo=config.logging.aim)]
    if config.logging.rerun_s3 and config.logging.rerun_every_episodes:
        sinks.append(RerunSink(config, scratch))
    return tuple(sinks)
```

- [ ] **Step 6: Commit**

```bash
git add training-sdk/pyproject.toml training-sdk/uv.lock \
        rtrrl/infra/control-plane/pyproject.toml \
        training-sdk/src/training_sdk/objects.py \
        training-sdk/src/training_sdk/sinks/rerun.py \
        training-sdk/src/training_sdk/testing.py \
        training-sdk/src/training_sdk/reporter.py \
        training-sdk/tests/conftest.py training-sdk/tests/test_objects.py \
        training-sdk/tests/test_rerun_sink.py
git commit -m "feat(sdk): add S3 helper and Rerun episode sink"
```

---

### Task 8: Score computation

**Files:**
- Create: `training-sdk/src/training_sdk/score.py`
- Test: `training-sdk/tests/test_score.py`

**Interfaces:**
- Produces:
  - `compute_score(metrics_path: Path, spec: ScoreConfig) -> float`
  - `ScoreError(ValueError)` raised when the window holds no point for the metric.
  - `WORST_MAGNITUDE = 1e30`.

- [ ] **Step 1: Write the failing test**

```python
# training-sdk/tests/test_score.py
import json
import math
from pathlib import Path

import pytest

from training_sdk.contract import ScoreConfig
from training_sdk.score import WORST_MAGNITUDE, ScoreError, compute_score


def write_metrics(path: Path, rows: list[tuple[int, float]]) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"step": step, "metrics": {"episode_return": value}})
            for step, value in rows
        ),
        encoding="utf-8",
    )


def spec(**overrides: object) -> ScoreConfig:
    payload = {
        "metric": "episode_return",
        "window_steps": [10, 20],
        "reduce": "mean",
        "direction": "maximize",
        "non_finite": "worst",
        "s3": "s3://bucket/score.json",
    }
    payload.update(overrides)
    return ScoreConfig.model_validate(payload)


def test_window_is_inclusive_and_mean_is_used(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(5, 100.0), (10, 1.0), (15, 2.0), (20, 3.0), (25, 100.0)])
    assert compute_score(path, spec()) == 2.0


def test_last_reduction_takes_the_highest_step(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(20, 3.0), (10, 1.0), (15, 2.0)])
    assert compute_score(path, spec(reduce="last")) == 3.0


def test_empty_window_raises_naming_metric_and_window(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(1, 1.0)])
    with pytest.raises(ScoreError, match="episode_return.*10.*20"):
        compute_score(path, spec())


def test_non_finite_becomes_worst_for_each_direction(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, math.nan)])
    assert compute_score(path, spec()) == -WORST_MAGNITUDE
    assert compute_score(path, spec(direction="minimize")) == WORST_MAGNITUDE


def test_declared_numeric_substitute_is_used_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [(10, math.inf)])
    assert compute_score(path, spec(non_finite=-5.0)) == -5.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd training-sdk && uv run pytest tests/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training_sdk.score'`

- [ ] **Step 3: Write the implementation**

```python
# training-sdk/src/training_sdk/score.py
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from training_sdk.contract import ScoreConfig

WORST_MAGNITUDE = 1e30


class ScoreError(ValueError):
    """The metrics file does not contain a usable value for the score window."""


def compute_score(metrics_path: Path, spec: ScoreConfig) -> float:
    low, high = spec.window_steps
    selected: list[tuple[int, float]] = []
    for line in Path(metrics_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["step"])
        if low <= step <= high and spec.metric in row["metrics"]:
            selected.append((step, float(row["metrics"][spec.metric])))
    if not selected:
        raise ScoreError(
            f"no reported value for metric {spec.metric!r} with step in "
            f"[{low}, {high}]; the run finished without covering the score window"
        )
    selected.sort()
    values = [value for _, value in selected]
    if not all(math.isfinite(value) for value in values):
        if spec.non_finite == "worst":
            return -WORST_MAGNITUDE if spec.direction == "maximize" else WORST_MAGNITUDE
        return float(spec.non_finite)
    return float(
        {
            "mean": lambda: statistics.fmean(values),
            "median": lambda: statistics.median(values),
            "min": lambda: min(values),
            "max": lambda: max(values),
            "last": lambda: values[-1],
        }[spec.reduce]()
    )
```

The non-finite check happens **before** the reduction, and a single non-finite
value anywhere in the window decides the whole score. Reducing first and
inspecting the result afterwards is wrong in two ways that both corrupt the
study silently:

A window of `[10.0, 20.0, inf]` reduces under `median` to `20.0` and under `min`
to `10.0`, so a run that diverged would be recorded as an ordinary result and
Optuna would explore toward the configuration that produced it.

Worse, `min`, `max` and `median` over a list containing `NaN` return different
answers depending on where in the list the `NaN` sits — `min([nan, 10, 20])` is
`nan` while `min([10, nan, 20])` is `10`. The same run would score differently
according to when it diverged, which is not a defensible number to hand an
optimiser.

Treating any non-finite value as decisive avoids both. It matches the intent of
`non_finite`: a run that produced a non-finite number diverged, and divergence is
a real result the study should record as such rather than a reading to be
averaged away.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd training-sdk && uv run pytest tests/test_score.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add training-sdk/src/training_sdk/score.py training-sdk/tests/test_score.py
git commit -m "feat(sdk): compute run scores from the metrics file"
```

---

### Task 9: Worker

**Files:**
- Create: `training-sdk/src/training_sdk/worker.py`
- Create: `training-sdk/src/training_sdk/__main__.py` (dispatch to the worker)
- Test: `training-sdk/tests/test_worker.py`

**Interfaces:**
- Consumes: `objects`, `contract`, `score`, `reporter.METRICS_FILENAME`.
- Produces:
  - `CATALOG_PATH = Path("/opt/trainer/catalog.json")` (overridable with
    `TRAINER_CATALOG`).
  - `run_manifest(manifest_uri: str, workspace: Path, *, startup_seconds: float,
    stall_factor: int, poll_seconds: float = 5.0, minimum_stall_seconds: float = 60.0) -> None`
    raising `WorkerError` on any failure. The `startup_seconds` allowance applies
    until the first report; after that the limit is the median reporting interval
    times `stall_factor`, floored at `minimum_stall_seconds`.
  - `main(argv: Sequence[str] | None = None) -> int` reading `TRAINER_MANIFEST`,
    `TRAINER_STARTUP_SECONDS`, `TRAINER_STALL_FACTOR`.
- Manifest format: `{"runs": ["s3://.../trials/t0/config.json", ...]}`.
- The child receives `TRAINER_RUN_CONFIG` and `TRAINER_SCRATCH` in its
  environment.

- [ ] **Step 1: Write the failing test**

```python
# training-sdk/tests/test_worker.py
import json
import sys
from pathlib import Path

import pytest

from training_sdk import objects
from training_sdk.worker import WorkerError, run_manifest
from tests.test_reporter import make_config

CHILD = """
import json, os, sys, time
config = json.loads(open(os.environ["TRAINER_RUN_CONFIG"]).read())
scratch = os.environ["TRAINER_SCRATCH"]
mode = os.environ.get("CHILD_MODE", "ok")
if mode == "crash":
    sys.exit(3)
with open(os.path.join(scratch, "metrics.jsonl"), "a") as handle:
    for step in (0, 4):
        handle.write(json.dumps({"step": step, "metrics": {"episode_return": 2.0}}) + "\\n")
        handle.flush()
if mode == "hang":
    time.sleep(600)
"""


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    child = tmp_path / "child.py"
    child.write_text(CHILD, encoding="utf-8")
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "contract": 2,
                "entries": {
                    "e": {
                        "command": [sys.executable, str(child)],
                        "source_hash": "sha256:0",
                        "metrics": ["episode_return"],
                        "space": {"total_steps": [4]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINER_CATALOG", str(path))
    return path


def publish(s3_base: str, trial: int) -> str:
    config = make_config().model_copy(
        update={
            "trial": trial,
            "run_id": f"smoke-20260725-000000-t{trial}",
            "score": make_config().score.model_copy(
                update={"s3": f"{s3_base}/trials/t{trial}/score.json"}
            ),
        }
    )
    uri = f"{s3_base}/trials/t{trial}/config.json"
    objects.put_bytes(uri, config.model_dump_json().encode())
    return uri


def write_manifest(s3_base: str, uris: list[str]) -> str:
    manifest = f"{s3_base}/rounds/round-000/job-0.json"
    objects.put_bytes(manifest, json.dumps({"runs": uris}).encode())
    return manifest


def test_every_run_is_executed_and_scored(s3_base: str, tmp_path: Path, catalog: Path) -> None:
    manifest = write_manifest(s3_base, [publish(s3_base, 0), publish(s3_base, 1)])
    run_manifest(manifest, tmp_path, startup_seconds=60, stall_factor=10)
    for trial in (0, 1):
        payload = json.loads(objects.get_bytes(f"{s3_base}/trials/t{trial}/score.json"))
        assert payload["value"] == 2.0
        assert payload["run_id"] == f"smoke-20260725-000000-t{trial}"


def test_scratch_is_removed_between_runs(s3_base: str, tmp_path: Path, catalog: Path) -> None:
    manifest = write_manifest(s3_base, [publish(s3_base, 0), publish(s3_base, 1)])
    run_manifest(manifest, tmp_path, startup_seconds=60, stall_factor=10)
    assert list(tmp_path.glob("*/metrics.jsonl")) == []


def test_crashing_run_stops_the_manifest(
    s3_base: str, tmp_path: Path, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHILD_MODE", "crash")
    manifest = write_manifest(s3_base, [publish(s3_base, 0), publish(s3_base, 1)])
    with pytest.raises(WorkerError, match="exit code 3"):
        run_manifest(manifest, tmp_path, startup_seconds=60, stall_factor=10)
    assert objects.exists(f"{s3_base}/trials/t1/score.json") is False


def test_stalled_run_is_killed(
    s3_base: str, tmp_path: Path, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHILD_MODE", "hang")
    manifest = write_manifest(s3_base, [publish(s3_base, 0)])
    with pytest.raises(WorkerError, match="stalled"):
        run_manifest(manifest, tmp_path, startup_seconds=30, stall_factor=1, poll_seconds=0.05,
                     minimum_stall_seconds=1.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd training-sdk && uv run pytest tests/test_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training_sdk.worker'`

- [ ] **Step 3: Write the implementation**

```python
# training-sdk/src/training_sdk/worker.py
from __future__ import annotations

import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from training_sdk import objects
from training_sdk.contract import CONTRACT_VERSION, Catalog, RunConfig
from training_sdk.reporter import METRICS_FILENAME
from training_sdk.score import compute_score

CATALOG_PATH = Path("/opt/trainer/catalog.json")
TERMINATE_GRACE_SECONDS = 10.0


class WorkerError(RuntimeError):
    """A run in this manifest did not complete."""


def load_catalog() -> Catalog:
    path = Path(os.environ.get("TRAINER_CATALOG", CATALOG_PATH))
    catalog = Catalog.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if catalog.contract != CONTRACT_VERSION:
        raise WorkerError(
            f"image catalog declares contract {catalog.contract}; "
            f"this worker implements contract {CONTRACT_VERSION}"
        )
    return catalog


def run_manifest(
    manifest_uri: str,
    workspace: Path,
    *,
    startup_seconds: float,
    stall_factor: int,
    poll_seconds: float = 5.0,
    minimum_stall_seconds: float = 60.0,
) -> None:
    catalog = load_catalog()
    manifest = json.loads(objects.get_bytes(manifest_uri))
    for config_uri in manifest["runs"]:
        config = RunConfig.model_validate(json.loads(objects.get_bytes(config_uri)))
        if config.contract != CONTRACT_VERSION:
            raise WorkerError(
                f"run configuration declares contract {config.contract}; "
                f"this worker implements contract {CONTRACT_VERSION}"
            )
        scratch = Path(workspace) / config.run_id
        scratch.mkdir(parents=True, exist_ok=True)
        try:
            _execute(config, catalog, scratch, startup_seconds, stall_factor,
                     poll_seconds, minimum_stall_seconds)
            value = compute_score(scratch / METRICS_FILENAME, config.score)
            objects.put_bytes(
                config.score.s3,
                json.dumps({"run_id": config.run_id, "trial": config.trial,
                            "value": value}).encode(),
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def _execute(
    config: RunConfig,
    catalog: Catalog,
    scratch: Path,
    startup_seconds: float,
    stall_factor: int,
    poll_seconds: float,
    minimum_stall_seconds: float,
) -> None:
    entry = catalog.entries.get(config.entry)
    if entry is None:
        raise WorkerError(f"image catalog does not declare entry {config.entry!r}")
    config_path = scratch / "config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    environment = dict(os.environ)
    environment["TRAINER_RUN_CONFIG"] = str(config_path)
    environment["TRAINER_SCRATCH"] = str(scratch)
    heartbeat = scratch / METRICS_FILENAME

    process = subprocess.Popen(list(entry.command), env=environment)
    watcher = _Heartbeat(heartbeat, startup_seconds, stall_factor, minimum_stall_seconds)
    while True:
        code = process.poll()
        if code is not None:
            break
        if watcher.stalled():
            _kill(process)
            raise WorkerError(
                f"run {config.run_id} stalled: no report for "
                f"{watcher.silence():.0f}s (limit {watcher.limit():.0f}s)"
            )
        time.sleep(poll_seconds)
    if code != 0:
        raise WorkerError(f"run {config.run_id} exited with exit code {code}")


class _Heartbeat:
    def __init__(
        self, path: Path, startup_seconds: float, stall_factor: int, minimum: float
    ) -> None:
        self._path = path
        self._startup = startup_seconds
        self._factor = stall_factor
        self._minimum = minimum
        self._started = time.monotonic()
        self._intervals: list[float] = []
        self._last_mtime: float | None = None
        self._last_seen = self._started

    def _poll(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            return
        if self._last_mtime is None or mtime > self._last_mtime:
            now = time.monotonic()
            if self._last_mtime is not None:
                self._intervals.append(now - self._last_seen)
            self._last_mtime = mtime
            self._last_seen = now

    def limit(self) -> float:
        if not self._intervals:
            return self._startup
        return max(statistics.median(self._intervals) * self._factor, self._minimum)

    def silence(self) -> float:
        return time.monotonic() - (self._last_seen if self._last_mtime else self._started)

    # The startup grace applies until an *interval* has been observed, not merely
    # until the first report arrives. Until two reports have been seen the worker
    # has no idea what this algorithm's normal reporting interval is, and falling
    # back to `minimum` there kills healthy runs: an algorithm that finishes JIT
    # compilation, reports once, and then reports every five minutes would be
    # declared stalled sixty seconds later. Under the no-retry policy that
    # destroys the whole experiment, so the tolerant direction is the safe one
    # while the interval is still unknown.

    def stalled(self) -> bool:
        self._poll()
        return self.silence() > self.limit()


def _kill(process: subprocess.Popen[bytes]) -> None:
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main(argv: Sequence[str] | None = None) -> int:
    manifest = os.environ["TRAINER_MANIFEST"]
    workspace = Path(os.environ.get("TRAINER_WORKSPACE", "/tmp/trainer"))
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        run_manifest(
            manifest,
            workspace,
            startup_seconds=float(os.environ.get("TRAINER_STARTUP_SECONDS", "900")),
            stall_factor=int(os.environ.get("TRAINER_STALL_FACTOR", "10")),
        )
    except Exception as error:  # noqa: BLE001 - the exit code is the only signal
        print(f"worker failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd training-sdk && uv run pytest tests/test_worker.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Run the whole SDK suite and the linter**

Run: `cd training-sdk && uv run pytest -q && uv run ruff check src tests`
Expected: all tests pass, ruff reports no findings.

- [ ] **Step 6: Commit**

```bash
git add training-sdk/src/training_sdk/worker.py \
        training-sdk/src/training_sdk/__main__.py \
        training-sdk/tests/test_worker.py
git commit -m "feat(sdk): add serial worker with heartbeat and score upload"
```

---

## Phase 3: Control Loop Without AWS

Real Optuna, real Aim server, real S3 server, real worker, and a trainer stub of
a few lines, with jobs executed as local processes instead of Batch jobs. Ends
with `trainerctl run --backend local` completing a two-round study.

This phase is deliberately light: no container, no training framework, nothing
that exceeds the machine's 2.2 GiB. It is the fast gate that runs on every edit
and catches contract and lifecycle bugs — the kind that reached a paid run last
time. It does not prove the code works inside the image; that is what the
`dev-*` queue smoke test at the end of Task 18 is for, and neither gate
substitutes for the other.

### Task 10: Optuna study

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/study.py`
- Test: `rtrrl/infra/control-plane/tests/test_study.py`

**Interfaces:**
- Produces:
  - `create_study(name: str, storage_path: Path, sampler: str, direction: str, user_attrs: Mapping[str, object]) -> optuna.Study`
  - `ask_round(study: optuna.Study, distributions: Mapping[str, BaseDistribution], count: int) -> list[optuna.trial.Trial]`
  - `tell_value(study: optuna.Study, trial: optuna.trial.Trial, value: float) -> None`

- [ ] **Step 1: Write the failing test**

```python
# rtrrl/infra/control-plane/tests/test_study.py
from pathlib import Path

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution

from trainer_infra.study import ask_round, create_study, tell_value

DISTRIBUTIONS = {
    "total_steps": CategoricalDistribution(choices=[128]),
    "learning_rate": FloatDistribution(low=1e-4, high=1e-3, log=True),
}


def make(tmp_path: Path) -> optuna.Study:
    return create_study(
        "sweep-20260725-000000",
        tmp_path / "study.db",
        sampler="tpe",
        direction="maximize",
        user_attrs={"launch_id": "20260725-000000", "digest": "sha256:0"},
    )


def test_every_trial_carries_every_parameter(tmp_path: Path) -> None:
    study = make(tmp_path)
    trials = ask_round(study, DISTRIBUTIONS, 3)
    assert len(trials) == 3
    for trial in trials:
        assert set(trial.params) == {"total_steps", "learning_rate"}
        assert trial.params["total_steps"] == 128


def test_told_values_persist_in_the_sqlite_file(tmp_path: Path) -> None:
    study = make(tmp_path)
    for index, trial in enumerate(ask_round(study, DISTRIBUTIONS, 2)):
        tell_value(study, trial, float(index))
    reopened = optuna.load_study(
        study_name="sweep-20260725-000000",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
    )
    assert sorted(t.value for t in reopened.trials) == [0.0, 1.0]
    assert reopened.user_attrs["digest"] == "sha256:0"


def test_unknown_sampler_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sampler"):
        create_study("s", tmp_path / "s.db", sampler="cma", direction="maximize",
                     user_attrs={})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_study.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainer_infra.study'`

- [ ] **Step 3: Write the implementation**

```python
# rtrrl/infra/control-plane/src/trainer_infra/study.py
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import optuna
from optuna.distributions import BaseDistribution

optuna.logging.set_verbosity(optuna.logging.WARNING)


SAMPLERS = ("tpe", "random", "grid")


def check_sampler(name: str, space: Mapping[str, BaseDistribution]) -> None:
    """Reject a sampler the space cannot be searched with. Called by preflight."""
    if name not in SAMPLERS:
        raise ValueError(f"unsupported sampler {name!r}; use {', '.join(SAMPLERS)}")
    if name != "grid":
        return
    continuous = sorted(
        key for key, dist in space.items() if not isinstance(dist, CategoricalDistribution)
    )
    if continuous:
        raise ValueError(
            f"the grid sampler needs every parameter to be a fixed list of values, "
            f"but these are ranges: {', '.join(continuous)}; either pin them to lists "
            f"or use the tpe or random sampler"
        )


def _sampler(name: str, space: Mapping[str, BaseDistribution]):
    check_sampler(name, space)
    if name == "tpe":
        return optuna.samplers.TPESampler()
    if name == "random":
        return optuna.samplers.RandomSampler()
    return optuna.samplers.GridSampler(
        {key: list(dist.choices) for key, dist in space.items()}  # type: ignore[attr-defined]
    )


def create_study(
    name: str,
    storage_path: Path,
    sampler: str,
    direction: str,
    user_attrs: Mapping[str, object],
    space: Mapping[str, BaseDistribution],
) -> optuna.Study:
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{storage_path}",
        sampler=_sampler(sampler, space),
        direction=direction,
    )
    for key, value in user_attrs.items():
        study.set_user_attr(key, value)
    return study


def ask_round(
    study: optuna.Study, distributions: Mapping[str, BaseDistribution], count: int
) -> list[optuna.trial.Trial]:
    return [study.ask(dict(distributions)) for _ in range(count)]


def tell_value(study: optuna.Study, trial: optuna.trial.Trial, value: float) -> None:
    study.tell(trial, value)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_study.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/study.py \
        rtrrl/infra/control-plane/tests/test_study.py
git commit -m "feat(control-plane): add per-launch Optuna study"
```

---

### Task 11: Launch identity and run configuration

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/launch.py`
- Create: `rtrrl/infra/control-plane/tests/helpers.py`
- Modify: `rtrrl/infra/control-plane/tests/conftest.py`
- Test: `rtrrl/infra/control-plane/tests/test_launch.py`

**Interfaces:**
- Produces:
  - `Launch` frozen dataclass: `plan: LaunchPlan`, `launch_id: str`,
    `archive: Path`, `prefix: str`.
  - `create_launch(plan: LaunchPlan, archive_root: Path, source: Path, now: datetime) -> Launch`
    writing `experiment.yaml`, `space.json`, `launch.json` to both the archive
    directory and `prefix`.
  - `build_run_config(launch: Launch, trial: int, params: Mapping[str, Scalar]) -> RunConfig`
  - `config_uri(launch: Launch, trial: int) -> str`

- [ ] **Step 1: Write the failing test**

```python
# rtrrl/infra/control-plane/tests/test_launch.py
import json
from datetime import UTC, datetime
from pathlib import Path

from training_sdk import objects

from trainer_infra.launch import build_run_config, config_uri, create_launch
from tests.helpers import make_plan  # defined in Step 3

WHEN = datetime(2026, 7, 25, 5, 14, 0, tzinfo=UTC)


def test_launch_id_is_a_utc_timestamp(s3_base: str, tmp_path: Path) -> None:
    launch = create_launch(make_plan(s3_base), tmp_path, Path("examples/experiment-acceptance.yaml"), WHEN)
    assert launch.launch_id == "20260725-051400"
    assert launch.prefix == f"{s3_base}/infra-acceptance/brax-ppo-smoke/20260725-051400"


def test_launch_metadata_is_written_to_archive_and_s3(s3_base: str, tmp_path: Path) -> None:
    launch = create_launch(make_plan(s3_base), tmp_path, Path("examples/experiment-acceptance.yaml"), WHEN)
    archived = json.loads((launch.archive / "launch.json").read_text())
    assert archived["digest"] == "sha256:" + "a" * 64
    assert archived["source_hash"] == "sha256:0"
    assert archived["contract"] == 2
    remote = json.loads(objects.get_bytes(f"{launch.prefix}/launch.json"))
    assert remote == archived
    assert json.loads(objects.get_bytes(f"{launch.prefix}/space.json"))["total_steps"] == [128]


def test_run_config_uses_trial_params_verbatim(s3_base: str, tmp_path: Path) -> None:
    launch = create_launch(make_plan(s3_base), tmp_path, Path("examples/experiment-acceptance.yaml"), WHEN)
    config = build_run_config(launch, 7, {"total_steps": 128, "learning_rate": 0.0003})
    assert config.run_id == "brax-ppo-smoke-20260725-051400-t7"
    assert config.params == {"total_steps": 128, "learning_rate": 0.0003}
    assert config.digest == "sha256:" + "a" * 64
    assert config.source_hash == "sha256:0"
    assert config.score.s3 == f"{launch.prefix}/trials/t7/score.json"
    assert config.logging.rerun_s3 == f"{launch.prefix}/trials/t7/episodes/"
    assert config_uri(launch, 7) == f"{launch.prefix}/trials/t7/config.json"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_launch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainer_infra.launch'`

- [ ] **Step 3: Write the test helper**

```python
# rtrrl/infra/control-plane/tests/helpers.py
from pathlib import Path

from trainer_infra.experiment import load_experiment
from trainer_infra.preflight import LaunchPlan, check_offline
from tests.test_preflight_offline import CATALOG

EXAMPLE = Path("examples/experiment-acceptance.yaml")


def make_plan(s3_base: str) -> LaunchPlan:
    experiment = load_experiment(EXAMPLE)
    experiment = experiment.model_copy(update={"storage": s3_base})
    return LaunchPlan(
        experiment=experiment,
        entry_name=experiment.entry,
        entry=CATALOG.entries[experiment.entry],
        space=check_offline(experiment, CATALOG),
        digest="sha256:" + "a" * 64,
        queue="run-cpu-c7am-queue",
        job_definition="trainer-c7am-" + "a" * 64,
    )
```

Create `rtrrl/infra/control-plane/tests/conftest.py` with
`pytest_plugins = ["training_sdk.testing"]` so this package gets the same
`s3_endpoint` and `s3_base` fixtures. Do not redefine them.

- [ ] **Step 4: Write the implementation**

```python
# rtrrl/infra/control-plane/src/trainer_infra/launch.py
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from training_sdk import objects
from training_sdk.contract import (
    CONTRACT_VERSION,
    LoggingConfig,
    RunConfig,
    Scalar,
    ScoreConfig,
)

from trainer_infra.preflight import LaunchPlan


@dataclass(frozen=True)
class Launch:
    plan: LaunchPlan
    launch_id: str
    archive: Path
    prefix: str


def create_launch(
    plan: LaunchPlan, archive_root: Path, source: Path, now: datetime
) -> Launch:
    experiment = plan.experiment
    launch_id = now.strftime("%Y%m%d-%H%M%S")
    archive = (
        Path(archive_root) / experiment.experiment / experiment.name / launch_id
    )
    archive.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"{experiment.storage.rstrip('/')}/{experiment.experiment}"
        f"/{experiment.name}/{launch_id}"
    )

    space_payload = {
        key: (list(spec.choices) if hasattr(spec, "choices") else spec.model_dump())
        for key, spec in plan.space.items()
    }
    launch_payload = {
        "contract": CONTRACT_VERSION,
        "experiment": experiment.experiment,
        "name": experiment.name,
        "description": experiment.description,
        "launch_id": launch_id,
        "entry": plan.entry_name,
        "digest": plan.digest,
        "source_hash": plan.entry.source_hash,
        "queue": plan.queue,
        "job_definition": plan.job_definition,
        "sampler": experiment.hpo.sampler,
        "rounds": experiment.hpo.rounds,
        "trials_per_round": experiment.hpo.trials_per_round,
        "parallel_jobs": experiment.hpo.parallel_jobs,
    }
    documents = {
        "experiment.yaml": Path(source).read_bytes(),
        "space.json": json.dumps(space_payload, sort_keys=True).encode(),
        "launch.json": json.dumps(launch_payload, sort_keys=True).encode(),
    }
    for name, payload in documents.items():
        (archive / name).write_bytes(payload)
        objects.put_bytes(f"{prefix}/{name}", payload)
    return Launch(plan=plan, launch_id=launch_id, archive=archive, prefix=prefix)


def config_uri(launch: Launch, trial: int) -> str:
    return f"{launch.prefix}/trials/t{trial}/config.json"


def build_run_config(
    launch: Launch, trial: int, params: Mapping[str, Scalar]
) -> RunConfig:
    experiment = launch.plan.experiment
    trial_prefix = f"{launch.prefix}/trials/t{trial}"
    rerun_s3 = (
        f"{trial_prefix}/episodes/"
        if experiment.logging.rerun_every_episodes
        else None
    )
    return RunConfig(
        contract=CONTRACT_VERSION,
        run_id=f"{experiment.name}-{launch.launch_id}-t{trial}",
        experiment=experiment.experiment,
        name=experiment.name,
        launch_id=launch.launch_id,
        trial=trial,
        entry=launch.plan.entry_name,
        digest=launch.plan.digest,
        source_hash=launch.plan.entry.source_hash,
        params=dict(params),
        logging=LoggingConfig(
            aim=experiment.logging.aim,
            every_steps=experiment.logging.every_steps,
            rerun_s3=rerun_s3,
            rerun_every_episodes=experiment.logging.rerun_every_episodes,
        ),
        score=ScoreConfig(
            metric=experiment.score.metric,
            window_steps=experiment.score.window_steps,
            reduce=experiment.score.reduce,
            direction=experiment.score.direction,
            non_finite=experiment.score.non_finite,
            s3=f"{trial_prefix}/score.json",
        ),
    )
```

`build_run_config` and `config_uri` live in `launch.py` because they depend on
nothing but `Launch`; the design document lists a separate `runconfig.py` and
should be corrected when this task lands.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_launch.py -v`
Expected: PASS, 3 tests

- [ ] **Step 6: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/launch.py \
        rtrrl/infra/control-plane/tests/test_launch.py \
        rtrrl/infra/control-plane/tests/helpers.py \
        rtrrl/infra/control-plane/tests/conftest.py
git commit -m "feat(control-plane): create launches and generate run configurations"
```

---

### Task 12: Round packing and upload

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/packing.py`
- Test: `rtrrl/infra/control-plane/tests/test_packing.py`

**Interfaces:**
- Produces:
  - `split(count: int, jobs: int) -> list[int]` returning group sizes with the
    remainder on the earliest groups.
  - `JobPlan` frozen dataclass: `manifest_uri: str`, `trials: tuple[int, ...]`.
  - `publish_round(launch: Launch, round_index: int, configs: Sequence[RunConfig], jobs: int) -> list[JobPlan]`
    uploading every configuration and every manifest, then returning one
    `JobPlan` per job in submission order. The `trials` tuple is what lets the
    control loop record which job ran which trial.

- [ ] **Step 1: Write the failing test**

```python
# rtrrl/infra/control-plane/tests/test_packing.py
import json
from datetime import UTC, datetime
from pathlib import Path

from training_sdk import objects

from trainer_infra.launch import build_run_config, create_launch
from trainer_infra.packing import publish_round, split
from tests.helpers import EXAMPLE, make_plan

WHEN = datetime(2026, 7, 25, 5, 14, 0, tzinfo=UTC)


def test_remainder_goes_to_the_earliest_jobs() -> None:
    assert split(8, 3) == [3, 3, 2]
    assert split(8, 2) == [4, 4]
    assert split(2, 2) == [1, 1]


def test_configs_and_manifests_are_uploaded(s3_base: str, tmp_path: Path) -> None:
    launch = create_launch(make_plan(s3_base), tmp_path, EXAMPLE, WHEN)
    configs = [
        build_run_config(launch, trial, {"total_steps": 128, "learning_rate": 0.0003})
        for trial in range(3)
    ]
    plans = publish_round(launch, 0, configs, jobs=2)
    assert [plan.manifest_uri for plan in plans] == [
        f"{launch.prefix}/rounds/round-000/job-0.json",
        f"{launch.prefix}/rounds/round-000/job-1.json",
    ]
    assert [plan.trials for plan in plans] == [(0, 1), (2,)]
    first = json.loads(objects.get_bytes(plans[0].manifest_uri))
    second = json.loads(objects.get_bytes(plans[1].manifest_uri))
    assert len(first["runs"]) == 2 and len(second["runs"]) == 1
    for uri in first["runs"] + second["runs"]:
        assert json.loads(objects.get_bytes(uri))["contract"] == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_packing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainer_infra.packing'`

- [ ] **Step 3: Write the implementation**

```python
# rtrrl/infra/control-plane/src/trainer_infra/packing.py
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from training_sdk import objects
from training_sdk.contract import RunConfig

from trainer_infra.launch import Launch, config_uri


@dataclass(frozen=True)
class JobPlan:
    manifest_uri: str
    trials: tuple[int, ...]


def split(count: int, jobs: int) -> list[int]:
    if jobs < 1 or count < jobs:
        raise ValueError("jobs must be between one and the number of trials")
    base, remainder = divmod(count, jobs)
    return [base + (1 if index < remainder else 0) for index in range(jobs)]


def publish_round(
    launch: Launch, round_index: int, configs: Sequence[RunConfig], jobs: int
) -> list[JobPlan]:
    uris: list[str] = []
    for config in configs:
        uri = config_uri(launch, config.trial)
        objects.put_bytes(uri, config.model_dump_json().encode())
        uris.append(uri)

    plans: list[JobPlan] = []
    offset = 0
    for job_index, size in enumerate(split(len(configs), jobs)):
        group = slice(offset, offset + size)
        offset += size
        manifest_uri = f"{launch.prefix}/rounds/round-{round_index:03d}/job-{job_index}.json"
        objects.put_bytes(manifest_uri, json.dumps({"runs": uris[group]}).encode())
        plans.append(
            JobPlan(
                manifest_uri=manifest_uri,
                trials=tuple(config.trial for config in configs[group]),
            )
        )
    return plans
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_packing.py -v`
Expected: PASS, 2 tests

- [ ] **Step 5: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/packing.py \
        rtrrl/infra/control-plane/tests/test_packing.py
git commit -m "feat(control-plane): pack rounds into job manifests and upload them"
```

---

### Task 13: Execution backend protocol and local backend

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/backends/__init__.py` (empty)
- Create: `rtrrl/infra/control-plane/src/trainer_infra/backends/base.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/backends/local.py`
- Test: `rtrrl/infra/control-plane/tests/test_local_backend.py`

**Interfaces:**
- Produces:
  - `JobResult` frozen dataclass: `job_id: str`, `name: str`,
    `succeeded: bool`, `log_stream: str | None`, `reason: str | None`.
    `reason` explains a failure and is `None` on success, in both backends, so
    the control loop cannot mistake a populated `reason` for a failed job.
  - `Backend` protocol: `submit(launch, manifest_uri, name) -> str`,
    `wait(job_ids: Sequence[str]) -> list[JobResult]` returning once every job
    has finished **or** as soon as any job has failed, whichever comes first, so
    the control loop can stop paying for the doomed round's siblings. On the
    failure path the returned list covers only the jobs that reached a terminal
    state, so it may be shorter than `job_ids`; the loop must check for failure
    before pairing results with job plans.
    `terminate` is then what stops the survivors, and must tolerate ids that
    have already finished,
    `terminate(job_ids: Sequence[str]) -> None`,
    `log_tail(result: JobResult, lines: int) -> str`.
  - `LocalBackend(workspace: Path, catalog_path: Path, python: str = sys.executable)`.
- The local backend runs `python -m training_sdk.worker` with `TRAINER_MANIFEST`,
  `TRAINER_WORKSPACE`, `TRAINER_CATALOG`, `TRAINER_STARTUP_SECONDS`, and
  `TRAINER_STALL_FACTOR` set, capturing combined output into
  `<workspace>/<name>.log`, which is also its log stream.

- [ ] **Step 1: Write the failing test**

```python
# rtrrl/infra/control-plane/tests/test_local_backend.py
import json
import sys
from pathlib import Path

from trainer_infra.backends.local import LocalBackend


def write_catalog(tmp_path: Path, body: str) -> Path:
    child = tmp_path / "child.py"
    child.write_text(body, encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "contract": 2,
                "entries": {
                    "e": {
                        "command": [sys.executable, str(child)],
                        "source_hash": "sha256:0",
                        "metrics": ["m"],
                        "space": {"total_steps": [1]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return catalog


def publish_sleeping_manifest(s3_base: str, tmp_path: Path) -> str:
    """A manifest the worker can actually start, whose child never finishes."""
    launch = create_launch(make_plan(s3_base), tmp_path / "archive", EXAMPLE,
                           datetime(2026, 7, 25, 5, 14, tzinfo=UTC))
    config = build_run_config(launch, 0, {"total_steps": 1, "learning_rate": 1e-4})
    objects.put_bytes(config_uri(launch, 0), config.model_dump_json().encode())
    manifest = f"{launch.prefix}/rounds/round-000/job-0.json"
    objects.put_bytes(manifest, json.dumps({"runs": [config_uri(launch, 0)]}).encode())
    return manifest


def test_failed_worker_is_reported_with_a_readable_log(tmp_path: Path, s3_base: str) -> None:
    backend = LocalBackend(tmp_path, write_catalog(tmp_path, "import sys; sys.exit(3)"))
    job_id = backend.submit_raw(f"{s3_base}/missing-manifest.json", "job-0")
    results = backend.wait([job_id])
    assert len(results) == 1 and results[0].succeeded is False
    assert "worker failed" in backend.log_tail(results[0], 50)


def test_terminate_stops_a_running_job(tmp_path: Path, s3_base: str) -> None:
    backend = LocalBackend(tmp_path, write_catalog(tmp_path, "import time; time.sleep(600)"))
    job_id = backend.submit_raw(publish_sleeping_manifest(s3_base, tmp_path), "job-0")
    backend.terminate([job_id])
    results = backend.wait([job_id])
    assert results[0].succeeded is False
```

`submit_raw` is the backend method that takes a manifest URI directly; `submit`
wraps it with launch-derived naming and is exercised in Task 14. Import
`json`, `datetime`, `UTC`, `objects`, `create_launch`, `build_run_config`,
`config_uri`, and `tests.helpers.{EXAMPLE, make_plan}` at the top of the file.

`terminate` kills the worker, not its child, so the child outlives the worker
until the test's process group is torn down. Make `LocalBackend.terminate` kill
the whole process group: start the worker with `start_new_session=True` and send
`SIGKILL` to `os.getpgid(process.pid)`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_local_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainer_infra.backends'`

- [ ] **Step 3: Write the implementation**

```python
# rtrrl/infra/control-plane/src/trainer_infra/backends/base.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from trainer_infra.launch import Launch


@dataclass(frozen=True)
class JobResult:
    job_id: str
    name: str
    succeeded: bool
    log_stream: str | None = None
    reason: str | None = None


class Backend(Protocol):
    def submit(self, launch: Launch, manifest_uri: str, name: str) -> str: ...
    def wait(self, job_ids: Sequence[str]) -> list[JobResult]: ...
    def terminate(self, job_ids: Sequence[str]) -> None: ...
    def log_tail(self, result: JobResult, lines: int) -> str: ...
```

```python
# rtrrl/infra/control-plane/src/trainer_infra/backends/local.py
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from trainer_infra.backends.base import JobResult
from trainer_infra.launch import Launch


class LocalBackend:
    """Runs the real worker as a local process instead of a Batch job."""

    def __init__(
        self,
        workspace: Path,
        catalog_path: Path,
        python: str = sys.executable,
        startup_seconds: float = 120.0,
        stall_factor: int = 10,
    ) -> None:
        self._workspace = Path(workspace)
        self._catalog = Path(catalog_path)
        self._python = python
        self._startup = startup_seconds
        self._stall_factor = stall_factor
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._names: dict[str, str] = {}
        self._logs: dict[str, Path] = {}

    def submit(self, launch: Launch, manifest_uri: str, name: str) -> str:
        del launch
        return self.submit_raw(manifest_uri, name)

    def submit_raw(self, manifest_uri: str, name: str) -> str:
        directory = self._workspace / name
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "worker.log"
        environment = dict(os.environ)
        environment.update(
            {
                "TRAINER_MANIFEST": manifest_uri,
                "TRAINER_WORKSPACE": str(directory),
                "TRAINER_CATALOG": str(self._catalog),
                "TRAINER_STARTUP_SECONDS": str(self._startup),
                "TRAINER_STALL_FACTOR": str(self._stall_factor),
            }
        )
        handle = log_path.open("wb")
        process = subprocess.Popen(
            [self._python, "-m", "training_sdk.worker"],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job_id = f"local-{name}-{process.pid}"
        self._processes[job_id] = process
        self._names[job_id] = name
        self._logs[job_id] = log_path
        return job_id

    def _result(self, job_id: str) -> JobResult:
        code = self._processes[job_id].returncode
        return JobResult(
            job_id=job_id,
            name=self._names[job_id],
            succeeded=code == 0,
            log_stream=str(self._logs[job_id]),
            reason=None if code == 0 else f"exit code {code}",
        )

    def wait(self, job_ids: Sequence[str]) -> list[JobResult]:
        while True:
            done = [job_id for job_id in job_ids if self._processes[job_id].poll() is not None]
            results = [self._result(job_id) for job_id in done]
            if len(done) == len(job_ids) or any(not result.succeeded for result in results):
                return results
            time.sleep(0.2)

    def terminate(self, job_ids: Sequence[str]) -> None:
        for job_id in job_ids:
            process = self._processes.get(job_id)
            if process is None or process.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                continue

    def log_tail(self, result: JobResult, lines: int) -> str:
        if result.log_stream is None:
            return ""
        text = Path(result.log_stream).read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_local_backend.py -v`
Expected: PASS, 2 tests

- [ ] **Step 5: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/backends/ \
        rtrrl/infra/control-plane/tests/test_local_backend.py
git commit -m "feat(control-plane): add execution backend protocol and local backend"
```

---

### Task 14: Control loop, report, and the local end-to-end gate

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/report.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/loop.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/cli.py`
- Create: `rtrrl/infra/control-plane/tests/test_end_to_end_local.py`
- Modify: `rtrrl/infra/control-plane/tests/conftest.py` (real Aim server fixture)

**Interfaces:**
- Produces:
  - `TrialRecord` frozen dataclass: `trial: int`, `params: dict`,
    `value: float | None`, `job_id: str | None`, `log_stream: str | None`.
  - `Report` frozen dataclass: `launch_id`, `status` (`"succeeded"` or
    `"failed"`), `trials: list[TrialRecord]`, `best: TrialRecord | None`,
    `elapsed_seconds: float`, `failure: str | None`; `write(archive: Path, prefix: str)`.
  - `LaunchFailed(RuntimeError)`.
  - `run_launch(launch: Launch, backend: Backend, *, printer=print) -> Report`.
- `run_launch` raises `LaunchFailed` after writing the failed report, so the CLI
  can exit non-zero with the message already printed.

- [ ] **Step 1: Write the failing end-to-end test**

```python
# rtrrl/infra/control-plane/tests/test_end_to_end_local.py
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from aim import Repo
from training_sdk import objects

from trainer_infra.backends.local import LocalBackend
from trainer_infra.launch import create_launch
from trainer_infra.loop import LaunchFailed, run_launch
from tests.conftest import AimServer
from tests.helpers import EXAMPLE, make_plan


def plan_using(s3_base: str, aim_server: AimServer):
    plan = make_plan(s3_base)
    experiment = plan.experiment.model_copy(
        update={
            "logging": plan.experiment.logging.model_copy(update={"aim": aim_server.uri})
        }
    )
    return dataclasses.replace(plan, experiment=experiment)


def test_two_round_study_completes_and_reports(
    s3_base: str, tmp_path: Path, aim_endpoint: AimServer, acceptance_catalog: Path
) -> None:
    launch = create_launch(
        plan_using(s3_base, aim_endpoint), tmp_path / "archive", EXAMPLE, datetime.now(UTC)
    )
    backend = LocalBackend(tmp_path / "jobs", acceptance_catalog)

    report = run_launch(launch, backend)

    assert report.status == "succeeded"
    assert len(report.trials) == 4  # 2 rounds x 2 trials per round
    assert report.best is not None and report.best.value is not None
    for record in report.trials:
        assert objects.exists(f"{launch.prefix}/trials/t{record.trial}/score.json")
        assert record.job_id is not None
    archived = json.loads((launch.archive / "report.json").read_text())
    assert archived["status"] == "succeeded"

    runs = list(Repo.from_path(aim_endpoint.path).iter_runs())
    assert {run.name for run in runs} == {
        f"brax-ppo-smoke-{launch.launch_id}-t{index}" for index in range(4)
    }


def test_failing_run_stops_the_launch_and_prints_the_log(
    s3_base: str, tmp_path: Path, aim_endpoint: AimServer, failing_catalog: Path
) -> None:
    launch = create_launch(
        plan_using(s3_base, aim_endpoint), tmp_path / "archive", EXAMPLE, datetime.now(UTC)
    )
    backend = LocalBackend(tmp_path / "jobs", failing_catalog)
    printed: list[str] = []

    with pytest.raises(LaunchFailed):
        run_launch(launch, backend, printer=printed.append)

    assert any("worker failed" in line for line in printed)
    archived = json.loads((launch.archive / "report.json").read_text())
    assert archived["status"] == "failed"
    assert archived["trials"] == []
```

`aim_endpoint` yields an `AimServer` with `uri` for the run configuration and
`path` for reading the repository directly; both are added in the next step.

- [ ] **Step 2: Add the fixtures**

```python
# rtrrl/infra/control-plane/tests/conftest.py  (additions)
import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from aim import Repo


@dataclass(frozen=True)
class AimServer:
    uri: str
    path: str


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def aim_endpoint(tmp_path_factory: pytest.TempPathFactory) -> AimServer:
    repo_path = tmp_path_factory.mktemp("aim-repo")
    Repo.from_path(str(repo_path), init_if_undefined=True)
    port = _free_port()
    process = subprocess.Popen(
        ["aim", "server", "--repo", str(repo_path), "--port", str(port),
         "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                break
        if process.poll() is not None:
            raise RuntimeError("aim server exited before accepting connections")
        time.sleep(0.2)
    else:
        process.kill()
        raise RuntimeError("aim server did not start within 60s")
    yield AimServer(uri=f"aim://127.0.0.1:{port}", path=str(repo_path))
    process.terminate()
    process.wait(timeout=30)


TRAINER = """
import json, os
from training_sdk.reporter import Reporter
config_path = os.environ["TRAINER_RUN_CONFIG"]
config = json.loads(open(config_path).read())
total = int(config["params"]["total_steps"])
rate = float(config["params"]["learning_rate"])
with Reporter.from_env() as reporter:
    for step in range(0, total + 1, max(total // 4, 1)):
        reporter.report(step, {"episode_return": rate * 1000 + step})
"""


@pytest.fixture
def acceptance_catalog(tmp_path: Path) -> Path:
    trainer = tmp_path / "trainer.py"
    trainer.write_text(TRAINER, encoding="utf-8")
    return _catalog(tmp_path, [sys.executable, str(trainer)])


@pytest.fixture
def failing_catalog(tmp_path: Path) -> Path:
    child = tmp_path / "boom.py"
    child.write_text("import sys; sys.exit(7)", encoding="utf-8")
    return _catalog(tmp_path, [sys.executable, str(child)])


def _catalog(tmp_path: Path, command: list[str]) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "contract": 2,
                "entries": {
                    "brax_ppo_acceptance": {
                        "command": command,
                        "source_hash": "sha256:0",
                        "metrics": ["episode_return", "episode_length"],
                        "space": {
                            "env": ["inverted_pendulum"],
                            "backend": ["generalized"],
                            "total_steps": {"type": "int", "low": 1, "high": 100000},
                            "seed": {"type": "int", "low": 0, "high": 1000},
                            "learning_rate": {"type": "float", "low": 1e-6, "high": 1e-2},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_end_to_end_local.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainer_infra.loop'`

- [ ] **Step 4: Write the report module**

```python
# rtrrl/infra/control-plane/src/trainer_infra/report.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from training_sdk import objects


@dataclass(frozen=True)
class TrialRecord:
    trial: int
    params: dict[str, object]
    value: float | None = None
    job_id: str | None = None
    log_stream: str | None = None


@dataclass(frozen=True)
class Report:
    launch_id: str
    status: str
    trials: list[TrialRecord] = field(default_factory=list)
    best: TrialRecord | None = None
    elapsed_seconds: float = 0.0
    failure: str | None = None

    def payload(self) -> dict[str, object]:
        return {
            "launch_id": self.launch_id,
            "status": self.status,
            "trials": [asdict(record) for record in self.trials],
            "best": asdict(self.best) if self.best else None,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "failure": self.failure,
        }

    def write(self, archive: Path, prefix: str) -> None:
        payload = json.dumps(self.payload(), sort_keys=True, indent=2).encode()
        (Path(archive) / "report.json").write_bytes(payload)
        objects.put_bytes(f"{prefix}/report.json", payload)
```

- [ ] **Step 5: Write the control loop**

```python
# rtrrl/infra/control-plane/src/trainer_infra/loop.py
from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence

from training_sdk import objects

from trainer_infra.backends.base import Backend, JobResult
from trainer_infra.launch import Launch, build_run_config
from trainer_infra.packing import publish_round
from trainer_infra.report import Report, TrialRecord
from trainer_infra.space import distributions
from trainer_infra.study import ask_round, create_study, tell_value

LOG_TAIL_LINES = 200


class LaunchFailed(RuntimeError):
    """The launch stopped because something exited abnormally."""


def run_launch(
    launch: Launch, backend: Backend, printer: Callable[[str], None] = print
) -> Report:
    experiment = launch.plan.experiment
    started = time.monotonic()
    built = distributions(launch.plan.space)
    study = create_study(
        name=f"{experiment.name}-{launch.launch_id}",
        storage_path=launch.archive / "study.db",
        sampler=experiment.hpo.sampler,
        direction=experiment.score.direction,
        user_attrs={
            "experiment": experiment.experiment,
            "name": experiment.name,
            "launch_id": launch.launch_id,
            "entry": launch.plan.entry_name,
            "digest": launch.plan.digest,
            "source_hash": launch.plan.entry.source_hash,
        },
        space=built,
    )

    records: list[TrialRecord] = []
    submitted: list[str] = []
    try:
        for round_index in range(experiment.hpo.rounds):
            trials = ask_round(study, built, experiment.hpo.trials_per_round)
            configs = [build_run_config(launch, t.number, t.params) for t in trials]
            plans = publish_round(
                launch, round_index, configs, jobs=experiment.hpo.parallel_jobs
            )
            submitted = [
                backend.submit(
                    launch, plan.manifest_uri, f"round-{round_index:03d}-job-{index}"
                )
                for index, plan in enumerate(plans)
            ]
            results = backend.wait(submitted)
            failed = [result for result in results if not result.succeeded]
            if failed:
                # wait returned early, so siblings may still be burning instance time.
                backend.terminate(submitted)
                submitted = []
                for result in failed:
                    printer(f"job {result.name} failed: {result.reason}")
                    printer(backend.log_tail(result, LOG_TAIL_LINES))
                raise LaunchFailed(
                    f"round {round_index} had {len(failed)} failed job(s)"
                )
            owner = {
                trial_number: result
                for plan, result in zip(plans, results, strict=True)
                for trial_number in plan.trials
            }
            submitted = []
            for trial, config in zip(trials, configs, strict=True):
                value = _read_score(config.score.s3)
                tell_value(study, trial, value)
                result = owner[trial.number]
                records.append(
                    TrialRecord(
                        trial=trial.number,
                        params=dict(trial.params),
                        value=value,
                        job_id=result.job_id,
                        log_stream=result.log_stream,
                    )
                )
                printer(f"trial {trial.number}: {trial.params} -> {value}")
    except BaseException as failure:
        # Every abnormal end leaves the same evidence behind: an unexpected
        # exception on a paid run is exactly when an archived report matters, and
        # a Ctrl-C must not leave Batch jobs running.
        backend.terminate(submitted)
        report = Report(
            launch_id=launch.launch_id,
            status="failed",
            trials=records,
            elapsed_seconds=time.monotonic() - started,
            failure=f"{type(failure).__name__}: {failure}",
        )
        report.write(launch.archive, launch.prefix)
        raise

    best = max(
        (record for record in records if record.value is not None),
        key=lambda record: record.value
        if experiment.score.direction == "maximize"
        else -record.value,
        default=None,
    )
    report = Report(
        launch_id=launch.launch_id,
        status="succeeded",
        trials=records,
        best=best,
        elapsed_seconds=time.monotonic() - started,
    )
    report.write(launch.archive, launch.prefix)
    printer(f"best trial {best.trial} scored {best.value}" if best else "no trials")
    return report


def _read_score(uri: str) -> float:
    # A worker that could not upload its score exits non-zero, so reaching this
    # with a missing or malformed object means something unmodelled happened;
    # name the object rather than surfacing a botocore or KeyError trace.
    try:
        return float(json.loads(objects.get_bytes(uri))["value"])
    except Exception as error:
        raise LaunchFailed(f"could not read the score at {uri}: {error}") from error
```

Drop the unused `JobResult` and `Sequence` imports if ruff reports them.

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_end_to_end_local.py -v`
Expected: PASS, 2 tests

- [ ] **Step 7: Add `trainerctl run --backend local`**

Extend the parser from Task 4 with a `run` subcommand taking the experiment
file, `--catalog`, `--archive-dir`, and `--backend {local,batch}`. For
`local` it builds a `LaunchPlan` from the catalog file with
`digest="local"`, `queue="local"`, `job_definition="local"`, creates the launch,
and calls `run_launch`. Exit 1 on `LaunchFailed` or `PreflightError`.

- [ ] **Step 8: Run the whole control-plane suite**

Run: `cd rtrrl/infra/control-plane && uv run pytest -q && uv run ruff check src tests`
Expected: all tests pass, ruff reports no findings.

- [ ] **Step 9: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/report.py \
        rtrrl/infra/control-plane/src/trainer_infra/loop.py \
        rtrrl/infra/control-plane/src/trainer_infra/cli.py \
        rtrrl/infra/control-plane/src/trainer_infra/packing.py \
        rtrrl/infra/control-plane/tests/test_end_to_end_local.py \
        rtrrl/infra/control-plane/tests/conftest.py
git commit -m "feat(control-plane): add control loop, report and local end-to-end gate"
```

---

## Phase 4: AWS Batch

Two verification points on real infrastructure. Task 18 ends with a single-trial
smoke on the `dev` CPU queue — the first time the worker runs inside the real
image, and the task's own acceptance gate. Task 20 is the paid three-job run on
the `run` queues, which requires separate authorisation at the time it is
proposed.

### Task 15: Queue table, image resolution, and AWS preflight

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/queues.py`
- Create: `rtrrl/infra/control-plane/src/trainer_infra/images.py`
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/preflight.py`
- Test: `rtrrl/infra/control-plane/tests/test_queues.py`
- Test: `rtrrl/infra/control-plane/tests/test_preflight_aws.py`

**Interfaces:**
- Produces:
  - `QUEUES: dict[str, QueueBinding]` keyed by instance type, where
    `QueueBinding` has `profile: str`, `run_queue: str`, `dev_queue: str`,
    `max_vcpus: int`, `vcpus_per_job: int`, and `concurrency` computed as
    `max_vcpus // vcpus_per_job`.
  - `QueueBinding.queue(tier: str) -> str` returning `run_queue` for `"run"` and
    `dev_queue` for `"dev"`, raising `PreflightError` for anything else.
  - `binding(instance_type: str) -> QueueBinding` raising `PreflightError`.
  - `job_definition_name(binding: QueueBinding, digest: str) -> str` returning
    `f"trainer-{binding.profile}-{hex}"`, which is the naming scheme
    `scripts/deploy_facility.py` already registers. The digest is part of the
    name, so a job definition can never point at a different image than the one
    preflight resolved.
  - `CATALOG_LABEL = "org.rtrrl.trainer.catalog.v2"`,
    `encode_catalog(catalog: Catalog) -> str`, `decode_catalog(value: str) -> Catalog`
    (base64 of gzip of the canonical JSON, matching the existing v1 encoding).
  - `resolve_image(reference: str, ecr_client, read_url: Callable[[str], bytes]) -> ResolvedImage`
    with `digest: str` and `catalog: Catalog`. It calls `describe_images` for the
    digest, `batch_get_image` for the manifest, `get_download_url_for_layer` for
    the config blob, and `read_url` to fetch it, exactly as the existing
    `ecr.BotoEcrCatalogReader` does for the v1 label.
  - `check_offline` additionally calls `study.check_sampler(experiment.hpo.sampler,
    distributions(space))`, so a sampler the space cannot be searched with is
    caught by `trainerctl validate` rather than at the first `study.ask`. The
    reachable case is `sampler: grid` with any parameter left as a range: without
    the check, Optuna's `GridSampler` construction fails with
    `AttributeError: 'FloatDistribution' object has no attribute 'choices'`, which
    names neither the sampler nor the offending parameter, and only after money
    has been spent.
  - `check_aws(experiment, catalog, space, *, ecr_client, batch_client, s3_client, read_url, connect, tier: str = "run") -> LaunchPlan`.
    The tier selects which queue name is checked and recorded in the plan;
    everything else is identical, because both tiers share job definitions.
  - `connect(host: str, port: int) -> None` defaults to a TCP connect with a
    five-second timeout and raises `PreflightError` on failure.

- [ ] **Step 1: Write the failing tests**

```python
# rtrrl/infra/control-plane/tests/test_queues.py
import pytest

from trainer_infra.preflight import PreflightError
from trainer_infra.queues import QUEUES, binding, job_definition_name

DIGEST = "sha256:" + "1" * 64


def test_every_instance_type_maps_to_both_tiers() -> None:
    assert set(QUEUES) == {"c7a.medium", "c7a.large", "c7a.xlarge", "g6.xlarge"}
    for instance_type, entry in QUEUES.items():
        assert entry.queue("run").startswith("run-"), instance_type
        assert entry.queue("dev").startswith("dev-"), instance_type


def test_unknown_queue_tier_is_rejected() -> None:
    with pytest.raises(PreflightError, match="tier"):
        binding("c7a.medium").queue("prod")


def test_concurrency_follows_compute_environment_capacity() -> None:
    assert binding("g6.xlarge").concurrency == 8
    assert binding("c7a.medium").concurrency == 16
    assert binding("c7a.xlarge").concurrency == 4


def test_job_definition_name_embeds_the_digest() -> None:
    assert job_definition_name(binding("c7a.medium"), DIGEST) == f"trainer-c7am-{'1' * 64}"


def test_unknown_instance_type_is_rejected_with_the_available_list() -> None:
    with pytest.raises(PreflightError, match="c7a.medium"):
        binding("p5.48xlarge")
```

```python
# rtrrl/infra/control-plane/tests/test_preflight_aws.py
import json

import pytest

from trainer_infra.experiment import load_experiment
from trainer_infra.images import encode_catalog
from trainer_infra.preflight import PreflightError, check_aws, check_offline
from tests.helpers import EXAMPLE
from tests.test_preflight_offline import CATALOG

DIGEST = "sha256:1111111111111111111111111111111111111111111111111111111111111111"


CONFIG_BLOB = json.dumps(
    {"config": {"Labels": {"org.rtrrl.trainer.catalog.v2": encode_catalog(CATALOG)}}}
).encode()


class FakeEcr:
    def __init__(self, digest: str = DIGEST) -> None:
        self.digest = digest

    def describe_images(self, **kwargs: object) -> dict:
        return {"imageDetails": [{"imageDigest": self.digest}]}

    def batch_get_image(self, **kwargs: object) -> dict:
        manifest = json.dumps({"config": {"digest": "sha256:config"}})
        return {"images": [{"imageManifest": manifest}]}

    def get_download_url_for_layer(self, **kwargs: object) -> dict:
        return {"downloadUrl": "https://example.invalid/config"}


def read_url(url: str) -> bytes:
    assert url == "https://example.invalid/config"
    return CONFIG_BLOB


DEFINITION = f"trainer-c7am-{DIGEST.removeprefix('sha256:')}"


class FakeBatch:
    def __init__(self, queues=("run-cpu-c7am-queue",), definitions=(DEFINITION,)) -> None:
        self.queues, self.definitions = queues, definitions

    def describe_job_queues(self, **kwargs: object) -> dict:
        return {"jobQueues": [{"jobQueueName": name, "state": "ENABLED", "status": "VALID"}
                              for name in self.queues]}

    def describe_job_definitions(self, jobDefinitionName: str, **kwargs: object) -> dict:
        if jobDefinitionName not in self.definitions:
            return {"jobDefinitions": []}
        return {"jobDefinitions": [{"jobDefinitionName": jobDefinitionName,
                                    "revision": 1,
                                    "status": "ACTIVE"}]}


class FakeS3:
    def head_bucket(self, **kwargs: object) -> dict:
        return {}


def plan_arguments():
    experiment = load_experiment(EXAMPLE)
    return experiment, CATALOG, check_offline(experiment, CATALOG)


def check(ecr=None, batch=None, connect=lambda host, port: None):
    experiment, catalog, space = plan_arguments()
    return check_aws(
        experiment,
        catalog,
        space,
        ecr_client=ecr or FakeEcr(),
        batch_client=batch or FakeBatch(),
        s3_client=FakeS3(),
        read_url=read_url,
        connect=connect,
    )


def test_plan_carries_digest_queue_and_job_definition() -> None:
    plan = check()
    assert plan.digest == DIGEST
    assert plan.queue == "run-cpu-c7am-queue"
    assert plan.job_definition == DEFINITION


def test_dev_tier_selects_the_dev_queue() -> None:
    experiment, catalog, space = plan_arguments()
    plan = check_aws(
        experiment, catalog, space,
        ecr_client=FakeEcr(), batch_client=FakeBatch(queues=("dev-cpu-c7am-queue",)),
        s3_client=FakeS3(), read_url=read_url, connect=lambda host, port: None,
        tier="dev",
    )
    assert plan.queue == "dev-cpu-c7am-queue"


def test_unreachable_aim_endpoint_is_rejected() -> None:
    def refuse(host: str, port: int) -> None:
        raise OSError("connection refused")

    with pytest.raises(PreflightError, match="aim"):
        check(connect=refuse)


def test_missing_queue_is_rejected() -> None:
    with pytest.raises(PreflightError, match="run-cpu-c7am-queue"):
        check(batch=FakeBatch(queues=()))


def test_image_without_a_registered_job_definition_is_rejected() -> None:
    other = "sha256:" + "2" * 64
    with pytest.raises(PreflightError, match=f"trainer-c7am-{'2' * 64}"):
        check(ecr=FakeEcr(digest=other))
```

Adjust `FakeEcr.batch_get_image` to the shape the implementation actually reads;
the implementation must fetch the image config blob and read
`config.Labels[CATALOG_LABEL]`. Keep the fake's payload identical to a recorded
real response saved at `tests/data/ecr-batch-get-image.json`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_queues.py tests/test_preflight_aws.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainer_infra.queues'`

- [ ] **Step 3: Write `queues.py`**

```python
# rtrrl/infra/control-plane/src/trainer_infra/queues.py
from __future__ import annotations

from dataclasses import dataclass

from trainer_infra.preflight import PreflightError


@dataclass(frozen=True)
class QueueBinding:
    instance_type: str
    profile: str
    run_queue: str
    dev_queue: str
    max_vcpus: int
    vcpus_per_job: int
    gpus_per_job: int = 0

    @property
    def concurrency(self) -> int:
        return self.max_vcpus // self.vcpus_per_job

    def queue(self, tier: str) -> str:
        if tier == "run":
            return self.run_queue
        if tier == "dev":
            return self.dev_queue
        raise PreflightError(f"unknown queue tier {tier!r}; use run or dev")


QUEUES: dict[str, QueueBinding] = {
    "c7a.medium": QueueBinding(
        "c7a.medium", "c7am", "run-cpu-c7am-queue", "dev-cpu-c7am-queue", 16, 1
    ),
    "c7a.large": QueueBinding(
        "c7a.large", "c7al", "run-cpu-c7al-queue", "dev-cpu-c7al-queue", 32, 2
    ),
    "c7a.xlarge": QueueBinding(
        "c7a.xlarge", "c7ax", "run-cpu-c7ax-queue", "dev-cpu-c7ax-queue", 16, 4
    ),
    "g6.xlarge": QueueBinding(
        "g6.xlarge", "g6x", "run-gpu-queue", "dev-gpu-queue", 32, 4, 1
    ),
}


def binding(instance_type: str) -> QueueBinding:
    try:
        return QUEUES[instance_type]
    except KeyError:
        available = ", ".join(sorted(QUEUES))
        raise PreflightError(
            f"instance_type {instance_type!r} has no queue; available: {available}"
        ) from None


def job_definition_name(entry: QueueBinding, digest: str) -> str:
    return f"trainer-{entry.profile}-{digest.removeprefix('sha256:')}"
```

The queue table mirrors `scripts/deploy_facility.py` and the compute environment
capacities recorded in the design document. `concurrency` is informational for
now; nothing in this plan throttles submission, because `parallel_jobs` is
already bounded by `trials_per_round`.

- [ ] **Step 4: Write `images.py` and `check_aws`**

`images.py` mirrors the existing v1 encoding in `image_catalog.py`: canonical
JSON, gzip with `mtime=0`, base64. Only the schema and the label name change.

`check_aws` performs, in order: resolve the tag to a digest; read and decode the
label; confirm the decoded catalog equals the one already validated offline;
look up the queue binding for `compute.instance_type`; confirm the queue exists
and is `ENABLED` and `VALID`; confirm `job_definition_name(binding, digest)` is
registered and `ACTIVE`; `head_bucket` the bucket parsed out of
`experiment.storage`; and connect to the host and port parsed out of
`logging.aim`. Each failure raises `PreflightError` naming the value it rejected
and, for the job definition, the command that would register it. It returns a
fully populated `LaunchPlan`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_queues.py tests/test_preflight_aws.py -v`
Expected: PASS, 11 tests

- [ ] **Step 6: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/queues.py \
        rtrrl/infra/control-plane/src/trainer_infra/images.py \
        rtrrl/infra/control-plane/src/trainer_infra/preflight.py \
        rtrrl/infra/control-plane/tests/test_queues.py \
        rtrrl/infra/control-plane/tests/test_preflight_aws.py \
        rtrrl/infra/control-plane/tests/data/ecr-batch-get-image.json
git commit -m "feat(control-plane): resolve images and validate AWS preconditions"
```

---

### Task 16: AWS Batch backend

**Files:**
- Create: `rtrrl/infra/control-plane/src/trainer_infra/backends/batch.py`
- Test: `rtrrl/infra/control-plane/tests/test_batch_backend.py`
- Create: `rtrrl/infra/control-plane/tests/data/batch-describe-jobs.json`

**Interfaces:**
- Produces: `BatchBackend(batch_client, logs_client, poll_seconds: float = 20.0)`
  implementing the `Backend` protocol.
- `submit` sends `containerOverrides.environment` with `TRAINER_MANIFEST`,
  `TRAINER_WORKSPACE`, `TRAINER_STARTUP_SECONDS`, `TRAINER_STALL_FACTOR`, and
  `timeout.attemptDurationSeconds` from `compute.timeout_minutes`.
- The recorded fixture must be a real `describe_jobs` response captured from the
  2026-07-24 acceptance run, with account-specific values replaced but the shape
  untouched.

- [ ] **Step 1: Write the failing test**

```python
# rtrrl/infra/control-plane/tests/test_batch_backend.py
import json
from pathlib import Path

from trainer_infra.backends.batch import BatchBackend

RECORDED = json.loads(Path("tests/data/batch-describe-jobs.json").read_text())


class FakeBatch:
    def __init__(self, sequence: list[str]) -> None:
        self.sequence = sequence
        self.submitted: list[dict] = []
        self.terminated: list[str] = []

    def submit_job(self, **kwargs: object) -> dict:
        self.submitted.append(kwargs)
        return {"jobId": f"job-{len(self.submitted)}"}

    def describe_jobs(self, jobs: list[str]) -> dict:
        status = self.sequence.pop(0) if self.sequence else "SUCCEEDED"
        payload = json.loads(json.dumps(RECORDED))
        entry = payload["jobs"][0]
        entries = []
        for job_id in jobs:
            item = json.loads(json.dumps(entry))
            item["jobId"] = job_id
            item["status"] = status
            item["container"]["exitCode"] = 0 if status == "SUCCEEDED" else 3
            entries.append(item)
        return {"jobs": entries}

    def terminate_job(self, jobId: str, reason: str) -> dict:
        self.terminated.append(jobId)
        return {}


class FakeLogs:
    def get_log_events(self, **kwargs: object) -> dict:
        return {"events": [{"message": "Traceback (most recent call last):"},
                           {"message": "RuntimeError: boom"}]}


def test_submit_passes_manifest_and_timeout(launch_for_batch) -> None:
    batch = FakeBatch(["SUCCEEDED"])
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    backend.submit(launch_for_batch, "s3://bucket/manifest.json", "round-000-job-0")
    request = batch.submitted[0]
    environment = {item["name"]: item["value"]
                   for item in request["containerOverrides"]["environment"]}
    assert environment["TRAINER_MANIFEST"] == "s3://bucket/manifest.json"
    assert request["timeout"]["attemptDurationSeconds"] == 60 * 60
    assert request["jobQueue"] == "run-cpu-c7am-queue"


def test_wait_polls_until_every_job_is_terminal(launch_for_batch) -> None:
    batch = FakeBatch(["RUNNABLE", "RUNNING", "SUCCEEDED"])
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    job_id = backend.submit(launch_for_batch, "s3://bucket/m.json", "job-0")
    results = backend.wait([job_id])
    assert results[0].succeeded is True
    assert not batch.sequence


def test_failed_job_exposes_its_log_tail(launch_for_batch) -> None:
    batch = FakeBatch(["FAILED"])
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    job_id = backend.submit(launch_for_batch, "s3://bucket/m.json", "job-0")
    result = backend.wait([job_id])[0]
    assert result.succeeded is False
    assert "RuntimeError: boom" in backend.log_tail(result, 10)


def test_terminate_calls_batch_for_every_job(launch_for_batch) -> None:
    batch = FakeBatch(["SUCCEEDED"])
    backend = BatchBackend(batch, FakeLogs(), poll_seconds=0)
    job_id = backend.submit(launch_for_batch, "s3://bucket/m.json", "job-0")
    backend.terminate([job_id])
    assert batch.terminated == [job_id]
```

Add a `launch_for_batch` fixture to `conftest.py` returning the `Launch` built
from `tests.helpers.make_plan`, which already carries
`queue="run-cpu-c7am-queue"` and the digest-bound job definition name.

- [ ] **Step 2: Capture the recorded response**

```bash
aws batch describe-jobs --jobs 0ea76b3c-0000-0000-0000-000000000000 \
  > rtrrl/infra/control-plane/tests/data/batch-describe-jobs.json
```

If that job has aged out of Batch history, take the payload from
`docs/acceptance/2026-07-24-infra-only-training-acceptance-failed-run.md`, which
records the same response. Replace the account id with `000000000000`.

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_batch_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trainer_infra.backends.batch'`

- [ ] **Step 4: Write the implementation**

```python
# rtrrl/infra/control-plane/src/trainer_infra/backends/batch.py
from __future__ import annotations

import time
from collections.abc import Sequence

from trainer_infra.backends.base import JobResult
from trainer_infra.launch import Launch

TERMINAL = {"SUCCEEDED", "FAILED"}


class BatchBackend:
    def __init__(self, batch_client, logs_client, poll_seconds: float = 20.0) -> None:
        self._batch = batch_client
        self._logs = logs_client
        self._poll_seconds = poll_seconds
        self._names: dict[str, str] = {}

    def submit(self, launch: Launch, manifest_uri: str, name: str) -> str:
        compute = launch.plan.experiment.compute
        response = self._batch.submit_job(
            jobName=f"{launch.plan.experiment.name}-{launch.launch_id}-{name}",
            jobQueue=launch.plan.queue,
            jobDefinition=launch.plan.job_definition,
            timeout={"attemptDurationSeconds": compute.timeout_minutes * 60},
            containerOverrides={
                "environment": [
                    {"name": "TRAINER_MANIFEST", "value": manifest_uri},
                    {"name": "TRAINER_WORKSPACE", "value": "/tmp/trainer"},
                    {"name": "TRAINER_STARTUP_SECONDS",
                     "value": str(compute.startup_minutes * 60)},
                    {"name": "TRAINER_STALL_FACTOR", "value": str(compute.stall_factor)},
                ]
            },
        )
        job_id = response["jobId"]
        self._names[job_id] = name
        return job_id

    def wait(self, job_ids: Sequence[str]) -> list[JobResult]:
        pending = list(job_ids)
        finished: dict[str, dict] = {}
        while pending:
            described = self._batch.describe_jobs(jobs=pending)["jobs"]
            for job in described:
                if job["status"] in TERMINAL:
                    finished[job["jobId"]] = job
            pending = [job_id for job_id in pending if job_id not in finished]
            # Returning the moment one job fails is what lets the loop terminate
            # the survivors instead of paying for a round that is already doomed.
            if any(job["status"] == "FAILED" for job in finished.values()):
                break
            if pending:
                time.sleep(self._poll_seconds)
        job_ids = [job_id for job_id in job_ids if job_id in finished]
        return [
            JobResult(
                job_id=job_id,
                name=self._names.get(job_id, job_id),
                succeeded=finished[job_id]["status"] == "SUCCEEDED",
                log_stream=finished[job_id].get("container", {}).get("logStreamName"),
                reason=(
                    None
                    if finished[job_id]["status"] == "SUCCEEDED"
                    else finished[job_id].get("statusReason")
                    or f"exit code {finished[job_id].get('container', {}).get('exitCode')}"
                ),
            )
            for job_id in job_ids
        ]

    def terminate(self, job_ids: Sequence[str]) -> None:
        for job_id in job_ids:
            self._batch.terminate_job(jobId=job_id, reason="trainerctl stopped")

    def log_tail(self, result: JobResult, lines: int) -> str:
        if result.log_stream is None:
            return ""
        events = self._logs.get_log_events(
            logGroupName="/trainer/jobs",
            logStreamName=result.log_stream,
            limit=lines,
            startFromHead=False,
        )["events"]
        return "\n".join(event["message"] for event in events)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_batch_backend.py -v`
Expected: PASS, 4 tests

- [ ] **Step 6: Wire `--backend batch` into the CLI**

`run` with `--backend batch` builds boto3 clients, runs `check_aws`, creates the
launch, installs a `SIGINT` handler that calls `backend.terminate` on the jobs of
the current round, and calls `run_launch`.

Add `--queues {run,dev}` defaulting to `run`, passed straight through to
`check_aws` as `tier`. Its help text reads: "dev queues are for infrastructure
development only; delivered runs use run queues." Print a one-line warning to
stderr when `dev` is selected, so a dev-queue launch is never mistaken for a real
one in a scrollback.

Give `validate` the same `--backend` flag: with `--backend batch` it runs
`check_offline` followed by `check_aws` and submits nothing; with `--catalog` it
runs `check_offline` alone against a catalog file. Exactly one of the two must be
given. Add a CLI test asserting that `validate --backend batch` never calls
`submit_job`.

- [ ] **Step 7: Commit**

```bash
git add rtrrl/infra/control-plane/src/trainer_infra/backends/batch.py \
        rtrrl/infra/control-plane/src/trainer_infra/cli.py \
        rtrrl/infra/control-plane/tests/test_batch_backend.py \
        rtrrl/infra/control-plane/tests/data/batch-describe-jobs.json \
        rtrrl/infra/control-plane/tests/conftest.py
git commit -m "feat(control-plane): add AWS Batch backend with log tails"
```

---

### Task 17: Dedicated log group

**Files:**
- Modify: `rtrrl/infra/control-plane/scripts/deploy_facility.py`
- Test: `rtrrl/infra/control-plane/tests/test_facility_deployment.py`

- [ ] **Step 1: Extend the deployment test**

Add a test asserting that every registered job definition declares
`logConfiguration` with `logDriver: "awslogs"` and options
`{"awslogs-group": "/trainer/jobs", "awslogs-region": <region>}`, and that the
deployment creates the log group with a 30-day retention when it is absent.

- [ ] **Step 2: Run it to verify it fails**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_facility_deployment.py -v`
Expected: FAIL on the missing `awslogs-group` option.

- [ ] **Step 3: Implement**

In `deploy_facility.py`, call `logs.create_log_group(logGroupName="/trainer/jobs")`
tolerating `ResourceAlreadyExistsException`, then
`logs.put_retention_policy(logGroupName="/trainer/jobs", retentionInDays=30)`, and
set the job definitions' `logConfiguration` options accordingly.

- [ ] **Step 4: Run it to verify it passes**

Run: `cd rtrrl/infra/control-plane && uv run pytest tests/test_facility_deployment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add rtrrl/infra/control-plane/scripts/deploy_facility.py \
        rtrrl/infra/control-plane/tests/test_facility_deployment.py
git commit -m "feat(deploy): send Batch job logs to a retained /trainer/jobs group"
```

---

### Task 18: Acceptance image on contract 2

**Files:**
- Modify: `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/__main__.py`
- Modify: `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/train.py`
- Create: `rtrrl/infra/mock-trainer/catalog.json`
- Create: `rtrrl/infra/mock-trainer/scripts/build_catalog.py`
- Modify: `rtrrl/infra/mock-trainer/Dockerfile.cpu`, `Dockerfile.gpu`
- Modify: `.github/workflows/build-infra-acceptance-image.yml`
- Test: `rtrrl/infra/mock-trainer/tests/test_catalog.py`
- Test: `rtrrl/infra/mock-trainer/tests/test_train.py`

**Interfaces:**
- The trainer reads its configuration with `Reporter.from_env()` and reports
  `episode_return` and `episode_length` with the environment step count as the
  step, so the step unit equals `total_steps`.
- `build_catalog.py` computes `source_hash` as the SHA-256 of the sorted
  `(relative path, file bytes)` pairs under
  `rtrrl/infra/mock-trainer/src/brax_ppo_acceptance/`, writes `catalog.json`, and
  prints the encoded label value.
- The Dockerfile copies `catalog.json` to `/opt/trainer/catalog.json` and the
  workflow passes the encoded value as
  `--label org.rtrrl.trainer.catalog.v2=<value>`.

- [ ] **Step 1: Write the failing tests**

```python
# rtrrl/infra/mock-trainer/tests/test_catalog.py
import json
from pathlib import Path

from training_sdk.contract import CONTRACT_VERSION, Catalog

CATALOG = Path("catalog.json")


def test_catalog_declares_contract_two_and_the_reserved_parameter() -> None:
    catalog = Catalog.model_validate(json.loads(CATALOG.read_text()))
    assert catalog.contract == CONTRACT_VERSION
    entry = catalog.entries["brax_ppo_acceptance"]
    assert "total_steps" in entry.space
    assert set(entry.metrics) >= {"episode_return", "episode_length"}


def test_source_hash_matches_the_current_sources() -> None:
    from scripts.build_catalog import source_hash

    catalog = Catalog.model_validate(json.loads(CATALOG.read_text()))
    assert catalog.entries["brax_ppo_acceptance"].source_hash == source_hash()
```

```python
# rtrrl/infra/mock-trainer/tests/test_train.py  (addition)
def test_reported_step_matches_the_environment_step_budget(tmp_path, monkeypatch):
    """The score window is expressed in total_steps units, so the reported step
    must be the environment step count, not an iteration index."""
```

Implement that test by running the trainer with `total_steps=8` against a
temporary Aim repository and asserting the metrics file's largest step equals 8.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd rtrrl/infra/mock-trainer && uv run pytest tests/test_catalog.py -v`
Expected: FAIL with `FileNotFoundError: catalog.json`

- [ ] **Step 3: Implement**

Write `scripts/build_catalog.py`, generate `catalog.json`, convert the trainer to
`Reporter.from_env()`, and update both Dockerfiles and the workflow.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd rtrrl/infra/mock-trainer && uv run pytest -q`
Expected: PASS

- [ ] **Step 5: Build and push both images**

The development machine cannot build images; the workflow is the only builder.

```bash
gh workflow run build-infra-acceptance-image.yml -f push=true
gh run watch
```

Expected: both CPU and GPU builds succeed. Record both digests from the output.

- [ ] **Step 6: Register job definitions for the new digests**

```bash
cd rtrrl/infra/control-plane
uv run python scripts/deploy_facility.py --cpu-image <cpu-digest> --gpu-image <gpu-digest>
```

Expected: four active `trainer-<profile>-<digest>` job definitions and the
`/trainer/jobs` log group from Task 17.

- [ ] **Step 7: Smoke the image on the `dev` CPU queue**

This is the first time the worker runs inside the real image, on a real Batch
host, against real S3 and the real Aim server. Nothing before this proves the
container works; the local gate from Task 14 only proves the contract works.

Copy the example to `examples/experiment-dev-smoke.yaml` with `rounds: 1`,
`trials_per_round: 1`, `parallel_jobs: 1`, `total_steps: [128]`, and the pushed
CPU image digest. Then:

```bash
uv run trainerctl validate examples/experiment-dev-smoke.yaml --backend batch --queues dev
uv run trainerctl run examples/experiment-dev-smoke.yaml --backend batch --queues dev \
  > /tmp/dev-smoke.out 2> /tmp/dev-smoke.err
echo "exit=$?"
```

Do not pipe through `tee`; that hid a non-zero exit on 2026-07-24.

Expected: `exit=0`, one trial with a finite score, `score.json` and at least one
`.rrd` under the launch prefix in S3, and the run visible in Aim with its digest
and source hash. If the worker fails, read `/trainer/jobs` for the job's log
stream — the failure is in the image or the contract, and it must be fixed here
rather than discovered during the paid run.

- [ ] **Step 8: Commit**

```bash
git add rtrrl/infra/mock-trainer .github/workflows/build-infra-acceptance-image.yml \
        rtrrl/infra/control-plane/examples/experiment-dev-smoke.yaml
git commit -m "feat(acceptance): publish a contract v2 catalog and report env steps"
```

---

### Task 19: Remove the superseded implementation

**Files:**
- Delete (control plane): `aim_reader.py`, `aim_scratch.py`, `aim_process_gate.py`,
  `sampling.py`, `materialize.py`, `resolve.py`, `controller.py`, `execution.py`,
  `models.py`, `loaders.py`, `aws_profiles.py`, and their tests.
- Delete (SDK): `spool.py`, `storage.py`, `bootstrap.py`, `context.py`,
  `execution.py`, `run.py`, `aim_adapter.py`, `rerun_adapter.py`, `types.py`, and
  their tests.
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/adapters/aws_batch.py`,
  `heavy_tests.py`, `image_catalog.py`, `ecr.py`, `scripts/deploy_facility.py`
  wherever they import a deleted module.
- Modify: `rtrrl/infra/control-plane/pyproject.toml` (drop the `aim` dependency;
  the control plane no longer talks to Aim).

- [ ] **Step 1: Delete the modules and their tests**

```bash
cd rtrrl/infra/control-plane
git rm src/trainer_infra/{aim_reader,aim_scratch,aim_process_gate,sampling,materialize,resolve,controller,execution,models,loaders,aws_profiles}.py
git rm tests/test_{aim_reader,aim_scratch,aim_process_gate,sampling,materialize,resolve,controller,execution,models}.py
cd ../../../training-sdk
git rm src/training_sdk/{spool,storage,bootstrap,context,execution,run,aim_adapter,rerun_adapter,types}.py
git rm tests/test_{spool,storage,bootstrap,context,execution,aim_adapter,rerun_adapter,types}.py
```

- [ ] **Step 2: Run both suites to find every broken import**

Run: `cd training-sdk && uv run pytest -q; cd ../rtrrl/infra/control-plane && uv run pytest -q`
Expected: import errors listing exactly which surviving modules still depend on
the deleted ones.

- [ ] **Step 3: Repair the survivors**

`heavy_tests.py` and `adapters/aws_batch.py` are infrastructure-development
tooling that must keep working. Replace their `aws_profiles.PROFILES` use with
`queues.QUEUES`, their `models.ContractModel` use with plain pydantic models
local to the adapter, and their `execution.JobBundle` use with the fields they
actually read. Do not extend their behaviour.

- [ ] **Step 4: Run every suite and the linter**

```bash
cd training-sdk && uv run pytest -q && uv run ruff check src tests
cd ../rtrrl/infra/control-plane && uv run pytest -q && uv run ruff check src tests scripts
cd ../mock-trainer && uv run pytest -q
```

Expected: all green.

- [ ] **Step 5: Prove the merge boundary is intact**

```bash
git diff --stat $(git merge-base HEAD main) HEAD -- memo/ infra/submit.sh
```

Expected: empty output.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove the superseded control plane and SDK implementation"
```

---

### Task 20: Paid AWS acceptance run

**Prerequisite:** Tasks 1 to 19 complete, both suites green, and the local
end-to-end gate from Task 14 passing on the current commit. Ask for
authorisation before this task and do not run it otherwise. Agreement with this
plan is not authorisation for this task.

**Shape:** three jobs. Two CPU jobs on `c7a.medium` running a three-trial study
over two rounds, and one GPU job on `g6.xlarge` running a single fixed
configuration, each run limited to 128 environment steps.

- [ ] **Step 1: Confirm the images and job definitions from Task 18 are current**

```bash
cd rtrrl/infra/control-plane
git log --oneline -1 -- ../mock-trainer
aws batch describe-job-definitions --status ACTIVE \
  --query 'jobDefinitions[].jobDefinitionName' --output text
```

Expected: no commit to `mock-trainer` after the Task 18 build, and four active
`trainer-<profile>-<digest>` definitions whose digests match the images Task 18
pushed. If `mock-trainer` changed since, rebuild and re-register first — a stale
digest means the paid run tests code that no longer exists.

- [ ] **Step 2: Confirm the dev smoke passed on this image**

The dev-queue smoke from Task 18 Step 7 must have succeeded on exactly these
digests. If it did not, stop: the paid run is not the place to find out.

- [ ] **Step 3: Validate before spending**

```bash
uv run trainerctl validate examples/experiment-acceptance.yaml --backend batch
```

Expected: exit 0, the resolved space printed, no job submitted, and the queue
shown as a `run-*` queue.

- [ ] **Step 4: Run the CPU study**

```bash
uv run trainerctl run examples/experiment-acceptance.yaml --backend batch \
  > /tmp/acceptance-cpu.out 2> /tmp/acceptance-cpu.err
echo "exit=$?"
```

Do not pipe through `tee`; the 2026-07-24 run hid a non-zero exit that way.
Expected: `exit=0`, four trials reported, a best trial printed.

- [ ] **Step 5: Run the GPU configuration**

Copy the example to `examples/experiment-acceptance-gpu.yaml` with
`instance_type: g6.xlarge`, `rounds: 1`, `trials_per_round: 1`,
`parallel_jobs: 1`, and the GPU image, then run it the same way.

- [ ] **Step 6: Collect the evidence**

```bash
aws batch describe-jobs --jobs <ids> > /tmp/acceptance-jobs.json
aws s3 ls --recursive s3://<bucket>/<experiment>/ > /tmp/acceptance-s3.txt
```

Confirm every trial has `config.json`, `score.json`, and at least one `.rrd`, and
confirm the Aim interface shows the runs under this launch id with the digest and
source hash recorded.

- [ ] **Step 7: Write the acceptance record**

Create `docs/acceptance/<date>-algorithm-centric-facility-acceptance.md` with the
commands, the exit codes, the job ids, the digests, and the report contents.

- [ ] **Step 8: Commit**

```bash
git add docs/acceptance/
git commit -m "docs: record the algorithm-centric facility acceptance run"
```

---

## Self-Review

**Spec coverage.** Every section of the design maps to a task: experiment
configuration to Task 3; script metadata and `source_hash` to Tasks 1 and 18;
space resolution to Task 2; identity and layout to Task 11; preflight to Tasks 4
and 15; the HPO loop to Task 10; run configuration generation to Task 11; packing
and upload to Task 12; submit and wait to Tasks 13 and 16; the worker to Task 9;
SDK sinks to Tasks 5 to 7; score computation to Task 8; readback and reporting to
Task 14; the log group to Task 17; the failure policy to Tasks 9, 14, and 16; the
testing strategy to Tasks 6, 7, 9, 14, and 16; removals to Task 19.

**Known deviations from the design.** Two, both deliberate. The design describes
run configuration generation as its own module; Task 11 puts those two functions
in `launch.py` because they depend on nothing but `Launch`. The design does not
mention a local execution backend; Task 13 adds one because the development
machine cannot run containers at all, so the only way to exercise the control
loop during development is as local processes. It is never selected unless
`--backend local` is passed explicitly, and it does not replace the dev-queue
smoke in Task 18 — the local gate proves the contract, the dev-queue smoke proves
the image.

**Contract consistency.** `CONTRACT_VERSION` is defined once in Task 1 and read
in Tasks 4, 9, 11, 15, and 18. `Sink` has exactly `report`, `log_episode`, and
`close` in Tasks 5, 6, and 7. `Backend` has exactly `submit`, `wait`,
`terminate`, and `log_tail` in Tasks 13 and 16, and `LocalBackend` adds
`submit_raw` for tests only. `JobResult` fields are fixed in Task 13 and read in
Tasks 14 and 16. `publish_round` returns `JobPlan` objects in Task 12 and the
control loop in Task 14 consumes `manifest_uri` and `trials` from them. The score
object is `{"run_id", "trial", "value"}`, written in Task 9 and read in Task 14.
The manifest is `{"runs": [config_uri, ...]}`, written in Task 12 and read in
Task 9.
