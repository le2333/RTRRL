from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, replace
import hashlib
from itertools import product
from pathlib import Path
import time
from typing import Any, Literal
from uuid import uuid4

from optuna.trial import TrialState
from pydantic import ConfigDict

from trainer_infra.aws_profiles import profile
from trainer_infra.execution import (
    CompletionMarker,
    JobBundle,
    RunBundle,
    build_run_context,
)
from trainer_infra.identities import canonical_json
from trainer_infra.loaders import load_experiment
from trainer_infra.materialize import materialize_run
from trainer_infra.models import (
    ContractModel,
    ContinuousSearch,
    DiscreteSearch,
    ExperimentSpec,
    JsonScalar,
    ResolvedExperiment,
    ResolvedGroup,
    thaw_json,
)
from trainer_infra.resolve import resolve_experiment
from trainer_infra.sampling import (
    DuplicateConfigurationError,
    FiniteSpaceTracker,
    SpaceExhaustedError,
    create_study,
    sample_parameters,
)


class GroupValidation(ContractModel):
    name: str
    image_digest: str
    profile: str
    total_trials: int
    configs_per_batch: int
    runs_per_job: int
    estimated_jobs: int


class ValidationReport(ContractModel):
    status: Literal["valid"] = "valid"
    experiment_name: str
    groups: tuple[GroupValidation, ...]

    def to_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class ExperimentReport(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["succeeded", "failed"]
    experiment_id: str
    experiment_name: str
    submitted_job_ids: tuple[str, ...]
    completed_runs: int
    error: str | None = None

    def to_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class ExperimentRunError(RuntimeError):
    def __init__(
        self,
        report: ExperimentReport,
        *,
        original_cause: BaseException | None,
        persistence_errors: tuple[BaseException, ...] = (),
    ) -> None:
        message = (
            str(original_cause)
            if original_cause is not None
            else "final experiment persistence failed"
        )
        super().__init__(message)
        self.report = report
        self.original_cause = original_cause
        self.persistence_errors = persistence_errors


_MAX_DUPLICATE_ATTEMPTS = 32
_MAX_ENUMERATED_SPACE = 10_000


class SamplingExhaustedError(RuntimeError):
    """Sampling repeatedly collided and no safe fallback was available."""


@dataclass(frozen=True)
class _FiniteSpace:
    cardinality: int
    too_large: bool
    names: tuple[str, ...] = ()
    domains: tuple[Iterable[JsonScalar], ...] = ()


@dataclass
class _GroupLoop:
    group: ResolvedGroup
    study: Any
    tracker: FiniteSpaceTracker
    finite_candidates: Iterator[dict[str, JsonScalar]] | None
    finite_space_too_large: bool
    seen: set[str]
    terminal_trials: set[int]
    allocated: int = 0

    @property
    def complete(self) -> bool:
        return (
            self.allocated >= self.group.hpo.total_trials
            or self.tracker.exhausted
        )


def _estimated_jobs(group: ResolvedGroup) -> int:
    remaining = group.hpo.total_trials
    result = 0
    while remaining:
        round_configs = min(group.hpo.configs_per_batch, remaining)
        result += (
            round_configs + group.execution.runs_per_job - 1
        ) // group.execution.runs_per_job
        remaining -= round_configs
    return result


class ExperimentController:
    def __init__(
        self,
        *,
        catalog_reader: Any,
        preflight: Any,
        store_factory: Callable[..., Any],
        batch_factory: Callable[..., Any],
        aim_reader: Any | None = None,
        aim_reader_factory: Callable[[Any], Any] | None = None,
        study_factory: Callable[..., Any] = create_study,
        experiment_id_factory: Callable[[], str] = lambda: str(uuid4()),
        bucket: str,
        prefix: str = "experiments",
        poll_interval: float = 5.0,
        batch_timeout: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if prefix != "experiments":
            raise ValueError("control prefix must be exactly 'experiments'")
        if poll_interval <= 0 or batch_timeout <= 0:
            raise ValueError("poll and timeout values must be positive")
        if (aim_reader is None) == (aim_reader_factory is None):
            raise ValueError(
                "provide exactly one of aim_reader or aim_reader_factory"
            )
        self._catalog_reader = catalog_reader
        self._preflight = preflight
        self._store_factory = store_factory
        self._batch_factory = batch_factory
        self._aim_reader = aim_reader
        self._aim_reader_factory = aim_reader_factory
        self._study_factory = study_factory
        self._experiment_id_factory = experiment_id_factory
        self._bucket = bucket
        self._prefix = prefix
        self._poll_interval = poll_interval
        self._batch_timeout = batch_timeout
        self._clock = clock
        self._sleep = sleep

    def _resolve(self, path: Path) -> ResolvedExperiment:
        spec = load_experiment(Path(path))
        references = {spec.defaults.image}
        references.update(group.image for group in spec.groups.values() if group.image)
        references.update(
            str(group.overrides["image"])
            for group in spec.groups.values()
            if "image" in group.overrides
        )
        images = {
            reference: self._catalog_reader.resolve_and_fetch(reference)
            for reference in sorted(references)
        }
        payload = spec.model_dump(mode="json", exclude_none=True)
        payload["defaults"]["image"] = images[spec.defaults.image].reference
        for name, group in spec.groups.items():
            group_payload = payload["groups"][name]
            if group.image is not None:
                group_payload["image"] = images[group.image].reference
            if "image" in group.overrides:
                group_payload["overrides"]["image"] = images[
                    str(group.overrides["image"])
                ].reference
        digest_spec = ExperimentSpec.model_validate(payload)
        catalogs = {}
        for image in images.values():
            if image.catalog is None:
                raise ValueError(f"image {image.reference!r} has no script catalog")
            catalogs[image.reference] = image.catalog
        return resolve_experiment(digest_spec, catalogs)

    def _prepare(
        self, path: Path
    ) -> tuple[ResolvedExperiment, Mapping[str, Any], ValidationReport]:
        resolved = self._resolve(path)
        validated = self._preflight.validate(resolved)
        if isinstance(validated, Mapping):
            definitions = dict(validated)
        else:
            definitions = {item.resource_profile: item for item in validated}
        for group in resolved.groups:
            if group.resources.profile not in definitions:
                raise ValueError(
                    f"no validated job definition for profile {group.resources.profile!r}"
                )
            definition = definitions[group.resources.profile]
            if getattr(definition, "image_digest", None) != group.image:
                raise ValueError(
                    f"job definition for profile {group.resources.profile!r} "
                    f"is not bound to image {group.image!r}"
                )
        groups = tuple(
            GroupValidation(
                name=group.name,
                image_digest=group.image,
                profile=group.resources.profile,
                total_trials=group.hpo.total_trials,
                configs_per_batch=group.hpo.configs_per_batch,
                runs_per_job=group.execution.runs_per_job,
                estimated_jobs=_estimated_jobs(group),
            )
            for group in resolved.groups
        )
        return (
            resolved,
            definitions,
            ValidationReport(experiment_name=resolved.name, groups=groups),
        )

    def validate(self, path: str | Path) -> ValidationReport:
        return self._prepare(Path(path))[2]

    @staticmethod
    def _context_payload(context: Any) -> dict[str, Any]:
        payload = {
            field.name: thaw_json(getattr(context, field.name))
            for field in fields(context)
        }
        payload["artifact_directory"] = str(payload["artifact_directory"])
        return payload

    def _run_bundle(
        self,
        experiment_name: str,
        experiment_id: str,
        group: ResolvedGroup,
        concrete: Any,
    ) -> RunBundle:
        key_root = (
            f"{self._prefix}/{experiment_id}/groups/{group.name}/"
            f"runs/{concrete.run_id}/"
        )
        input_prefix = f"{key_root}input/"
        context = build_run_context(
            experiment_name,
            experiment_id,
            group.name,
            concrete,
            f"/artifacts/{experiment_id}/{group.name}/{concrete.run_number:04d}",
        )
        context_payload = self._context_payload(context)
        return RunBundle(
            run_id=concrete.run_id,
            argv=concrete.argv,
            image_digest=concrete.image,
            resource_profile=concrete.resources.profile,
            config_yaml=concrete.config_yaml,
            config_sha256=concrete.config_sha256,
            run_context=context_payload,
            run_context_sha256=hashlib.sha256(
                canonical_json(context_payload).encode()
            ).hexdigest(),
            artifact_prefix=input_prefix,
        )

    def _upload_run(self, store: Any, run: RunBundle) -> None:
        uri = f"s3://{self._bucket}/{run.artifact_prefix}"
        store.put_bytes(f"{uri}config.yaml", run.config_yaml.encode())
        store.put_bytes(
            f"{uri}run-context.json",
            canonical_json(run.run_context).encode(),
        )

    def _wait_jobs(self, batch: Any, job_ids: list[str]) -> None:
        deadline = self._clock() + self._batch_timeout
        while True:
            queries = batch.query(job_ids)
            failed = next((item for item in queries if item.status == "FAILED"), None)
            if failed is not None:
                raise RuntimeError(
                    f"Batch job {failed.job_id!r} FAILED: {failed.status_reason or ''}".rstrip()
                )
            if all(item.status == "SUCCEEDED" for item in queries):
                return
            now = self._clock()
            if now >= deadline:
                raise TimeoutError("timed out waiting for Batch jobs")
            self._sleep(min(self._poll_interval, deadline - now))

    def _collect_marker(self, store: Any, run: RunBundle) -> CompletionMarker:
        root = run.artifact_prefix.removesuffix("input/")
        uri = f"s3://{self._bucket}/{root}status/attempt-0.json"
        marker = CompletionMarker.model_validate(store.get_json(uri))
        if marker.run_id != run.run_id:
            raise ValueError(f"completion marker does not match run {run.run_id!r}")
        if marker.exit_code != 0 or marker.error is not None:
            raise RuntimeError(
                f"child run {run.run_id!r} failed with exit code {marker.exit_code}"
            )
        return marker

    @staticmethod
    def _tell_once(
        study: Any,
        trial: Any,
        terminal_trials: set[int],
        *,
        value: float | None = None,
        state: TrialState | None = None,
    ) -> None:
        if trial.number in terminal_trials:
            return
        study.tell(trial, values=value, state=state)
        terminal_trials.add(trial.number)

    @classmethod
    def _fail_pending(
        cls,
        study: Any,
        trials: list[Any],
        terminal_trials: set[int],
    ) -> tuple[BaseException, ...]:
        errors: list[BaseException] = []
        for trial in trials:
            try:
                cls._tell_once(
                    study,
                    trial,
                    terminal_trials,
                    state=TrialState.FAIL,
                )
            except BaseException as error:
                errors.append(error)
        return tuple(errors)

    @staticmethod
    def _finite_space(
        group: ResolvedGroup,
    ) -> _FiniteSpace | None:
        searchable = group.searchable_parameters()
        names: list[str] = []
        domains: list[Iterable[JsonScalar]] = []
        cardinality = 1
        for name, domain in searchable.items():
            if isinstance(domain, DiscreteSearch):
                size = len(domain.values)
                values: Iterable[JsonScalar] = domain.values
            elif isinstance(domain, ContinuousSearch) and domain.integer:
                step = int(domain.step or 1)
                size = (int(domain.high) - int(domain.low)) // step + 1
                values = range(
                    int(domain.low),
                    int(domain.high) + 1,
                    step,
                )
            else:
                return None
            if size <= 0:
                raise ValueError(f"search domain {name!r} is empty")
            if cardinality > _MAX_ENUMERATED_SPACE // size:
                return _FiniteSpace(
                    cardinality=_MAX_ENUMERATED_SPACE + 1,
                    too_large=True,
                )
            cardinality *= size
            names.append(name)
            domains.append(values)
        return _FiniteSpace(
            cardinality=cardinality,
            too_large=False,
            names=tuple(names),
            domains=tuple(domains),
        )

    @staticmethod
    def _iter_finite_candidates(
        finite_space: _FiniteSpace,
    ) -> Iterator[dict[str, JsonScalar]]:
        if finite_space.too_large:
            raise ValueError("too-large finite spaces cannot be enumerated")
        for values in product(*finite_space.domains):
            yield dict(
                zip(
                    finite_space.names,
                    values,
                    strict=True,
                )
            )

    @classmethod
    def _ask_unique(
        cls,
        study: Any,
        group: ResolvedGroup,
        tracker: FiniteSpaceTracker,
        seen: set[str],
        terminal_trials: set[int],
        finite_candidates: Iterator[dict[str, JsonScalar]] | None,
        *,
        finite_space_too_large: bool,
    ) -> tuple[Any, dict[str, JsonScalar]] | None:
        duplicate_attempts = 0
        while duplicate_attempts < _MAX_DUPLICATE_ATTEMPTS:
            trial = study.ask()
            try:
                sampled = sample_parameters(trial, group, tracker=tracker)
            except DuplicateConfigurationError:
                cls._tell_once(
                    study,
                    trial,
                    terminal_trials,
                    state=TrialState.PRUNED,
                )
                duplicate_attempts += 1
                if finite_candidates is None:
                    continue
                remaining = next(
                    (
                        candidate
                        for candidate in finite_candidates
                        if canonical_json(candidate) not in seen
                    ),
                    None,
                )
                if remaining is None:
                    return None
                study.enqueue_trial(remaining)
                fallback = study.ask()
                try:
                    sampled = sample_parameters(fallback, group, tracker=tracker)
                except (DuplicateConfigurationError, SpaceExhaustedError) as error:
                    cls._tell_once(
                        study,
                        fallback,
                        terminal_trials,
                        state=TrialState.FAIL,
                    )
                    raise RuntimeError(
                        "deterministic discrete fallback was not honored by the study"
                    ) from error
                except BaseException:
                    cls._tell_once(
                        study,
                        fallback,
                        terminal_trials,
                        state=TrialState.FAIL,
                    )
                    raise
                key = canonical_json(
                    {
                        name: sampled[name]
                        for name in group.searchable_parameters()
                    }
                )
                seen.add(key)
                return fallback, sampled
            except SpaceExhaustedError:
                cls._tell_once(
                    study,
                    trial,
                    terminal_trials,
                    state=TrialState.PRUNED,
                )
                return None
            except BaseException:
                cls._tell_once(
                    study,
                    trial,
                    terminal_trials,
                    state=TrialState.FAIL,
                )
                raise

            key = canonical_json(
                {name: sampled[name] for name in group.searchable_parameters()}
            )
            seen.add(key)
            return trial, sampled
        reason = (
            f"finite space exceeds {_MAX_ENUMERATED_SPACE}"
            if finite_space_too_large
            else "no enumerable finite fallback"
        )
        raise SamplingExhaustedError(
            f"duplicate sampling limit ({_MAX_DUPLICATE_ATTEMPTS}) reached "
            f"for group {group.name!r}: {reason}"
        )

    def _persist(
        self, store: Any, prefix: str, report: ExperimentReport
    ) -> tuple[BaseException, ...]:
        state = {
            "experiment_id": report.experiment_id,
            "status": report.status,
            "submitted_job_ids": list(report.submitted_job_ids),
            "completed_runs": report.completed_runs,
        }
        errors: list[BaseException] = []
        for uri, payload in (
            (f"{prefix}state/final.json", state),
            (f"{prefix}report.json", report.model_dump(mode="json")),
        ):
            try:
                store.put_json(uri, payload)
            except BaseException as error:
                errors.append(error)
        return tuple(errors)

    def run(self, path: str | Path) -> ExperimentReport:
        resolved, definitions, _ = self._prepare(Path(path))
        experiment_id = self._experiment_id_factory()
        experiment_prefix = (
            f"s3://{self._bucket}/{self._prefix}/{experiment_id}/"
        )
        store = self._store_factory(experiment_prefix)
        batch = self._batch_factory(experiment_prefix, store)
        aim_reader = (
            self._aim_reader_factory(store)
            if self._aim_reader_factory is not None
            else self._aim_reader
        )
        submitted_ids: list[str] = []
        completed_runs = 0

        try:
            loops: list[_GroupLoop] = []
            for original_group in resolved.groups:
                group = replace(
                    original_group,
                    study_key=f"{experiment_id}:{original_group.name}",
                )
                study = self._study_factory(
                    study_name=group.study_key,
                    direction=group.objective.direction,
                    load_if_exists=False,
                )
                tracker = FiniteSpaceTracker(group)
                finite_space = self._finite_space(group)
                loops.append(
                    _GroupLoop(
                        group=group,
                        study=study,
                        tracker=tracker,
                        finite_candidates=(
                            self._iter_finite_candidates(finite_space)
                            if finite_space is not None and not finite_space.too_large
                            else None
                        ),
                        finite_space_too_large=(
                            finite_space is not None and finite_space.too_large
                        ),
                        seen=set(),
                        terminal_trials=set(),
                    )
                )

            round_number = 0
            while any(not loop.complete for loop in loops):
                pending: list[tuple[_GroupLoop, list[Any]]] = []
                ready_by_group: list[
                    tuple[_GroupLoop, list[tuple[Any, RunBundle]]]
                ] = []
                try:
                    for loop in loops:
                        if loop.complete:
                            continue
                        group = loop.group
                        round_trials: list[Any] = []
                        trials_and_runs: list[tuple[Any, RunBundle]] = []
                        pending.append((loop, round_trials))
                        wanted = min(
                            group.hpo.configs_per_batch,
                            group.hpo.total_trials - loop.allocated,
                        )
                        while (
                            len(trials_and_runs) < wanted
                            and not loop.tracker.exhausted
                        ):
                            allocation = self._ask_unique(
                                loop.study,
                                group,
                                loop.tracker,
                                loop.seen,
                                loop.terminal_trials,
                                loop.finite_candidates,
                                finite_space_too_large=loop.finite_space_too_large,
                            )
                            if allocation is None:
                                break
                            trial, sampled = allocation
                            round_trials.append(trial)
                            loop.allocated += 1
                            concrete = materialize_run(
                                group,
                                trial,
                                sampled,
                                run_number=loop.allocated,
                            )
                            trials_and_runs.append(
                                (
                                    trial,
                                    self._run_bundle(
                                        resolved.name,
                                        experiment_id,
                                        group,
                                        concrete,
                                    ),
                                )
                            )
                        if trials_and_runs:
                            ready_by_group.append((loop, trials_and_runs))

                    if not ready_by_group:
                        break

                    round_number += 1
                    ordered: list[tuple[_GroupLoop, Any, RunBundle]] = []
                    max_group_runs = max(
                        len(items) for _, items in ready_by_group
                    )
                    for index in range(max_group_runs):
                        for loop, items in ready_by_group:
                            if index < len(items):
                                trial, run = items[index]
                                ordered.append((loop, trial, run))

                    partitions: dict[
                        tuple[str, str, int],
                        list[tuple[_GroupLoop, Any, RunBundle]],
                    ] = {}
                    for item in ordered:
                        loop, _, run = item
                        key = (
                            run.image_digest,
                            run.resource_profile,
                            loop.group.execution.runs_per_job,
                        )
                        partitions.setdefault(key, []).append(item)

                    round_jobs: list[
                        tuple[
                            Any,
                            list[tuple[_GroupLoop, Any, RunBundle]],
                        ]
                    ] = []
                    job_number = 0
                    for (
                        image_digest,
                        resource_profile,
                        runs_per_job,
                    ), partition in partitions.items():
                        for offset in range(0, len(partition), runs_per_job):
                            children = partition[offset : offset + runs_per_job]
                            for _, _, child in children:
                                self._upload_run(store, child)
                            job_number += 1
                            job = JobBundle(
                                job_id=(
                                    f"{experiment_id}-r{round_number}-j{job_number}"
                                ),
                                image_digest=image_digest,
                                resource_profile=resource_profile,
                                runs=tuple(child for _, _, child in children),
                            )
                            store.put_json(
                                f"{experiment_prefix}jobs/{job.job_id}/bundle.json",
                                job.model_dump(mode="json"),
                            )
                            submitted = batch.submit(
                                job,
                                profile(resource_profile),
                                definitions[resource_profile],
                            )
                            submitted_ids.append(submitted.job_id)
                            round_jobs.append((submitted, children))

                    self._wait_jobs(
                        batch,
                        [submitted.job_id for submitted, _ in round_jobs],
                    )
                    for _, children in round_jobs:
                        for loop, trial, run in children:
                            self._collect_marker(store, run)
                            value = aim_reader.wait_for_result(
                                run.run_id,
                                loop.group.objective.metric,
                                loop.group.execution.aim_result_timeout_seconds,
                            )
                            self._tell_once(
                                loop.study,
                                trial,
                                loop.terminal_trials,
                                value=value,
                            )
                            completed_runs += 1
                except BaseException as error:
                    lifecycle_errors: list[BaseException] = []
                    for loop, round_trials in pending:
                        lifecycle_errors.extend(
                            self._fail_pending(
                                loop.study,
                                round_trials,
                                loop.terminal_trials,
                            )
                        )
                    for lifecycle_error in lifecycle_errors:
                        error.add_note(
                            "failed to terminate pending Optuna trial: "
                            f"{type(lifecycle_error).__name__}: {lifecycle_error}"
                        )
                    raise
            report = ExperimentReport(
                status="succeeded",
                experiment_id=experiment_id,
                experiment_name=resolved.name,
                submitted_job_ids=tuple(submitted_ids),
                completed_runs=completed_runs,
            )
        except BaseException as error:
            report = ExperimentReport(
                status="failed",
                experiment_id=experiment_id,
                experiment_name=resolved.name,
                submitted_job_ids=tuple(submitted_ids),
                completed_runs=completed_runs,
                error=f"{type(error).__name__}: {error}",
            )
            persistence_errors = self._persist(store, experiment_prefix, report)
            raise ExperimentRunError(
                report,
                original_cause=error,
                persistence_errors=persistence_errors,
            ) from error

        persistence_errors = self._persist(store, experiment_prefix, report)
        if persistence_errors:
            failed_report = report.model_copy(
                update={
                    "status": "failed",
                    "error": "PersistenceError: final experiment persistence failed",
                }
            )
            raise ExperimentRunError(
                failed_report,
                original_cause=None,
                persistence_errors=persistence_errors,
            )
        return report
