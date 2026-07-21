from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, replace
import hashlib
from pathlib import Path
import time
from typing import Any, Literal
from uuid import uuid4

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
    ExperimentSpec,
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
    def __init__(self, report: ExperimentReport) -> None:
        super().__init__(report.error or "experiment failed")
        self.report = report


class ExperimentController:
    def __init__(
        self,
        *,
        catalog_reader: Any,
        preflight: Any,
        store_factory: Callable[..., Any],
        batch_factory: Callable[..., Any],
        aim_reader: Any,
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
        self._catalog_reader = catalog_reader
        self._preflight = preflight
        self._store_factory = store_factory
        self._batch_factory = batch_factory
        self._aim_reader = aim_reader
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
                estimated_jobs=(
                    group.hpo.total_trials + group.execution.runs_per_job - 1
                )
                // group.execution.runs_per_job,
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

    def _persist(self, store: Any, prefix: str, report: ExperimentReport) -> None:
        state = {
            "experiment_id": report.experiment_id,
            "status": report.status,
            "submitted_job_ids": list(report.submitted_job_ids),
            "completed_runs": report.completed_runs,
        }
        store.put_json(f"{prefix}state/final.json", state)
        store.put_json(f"{prefix}report.json", report.model_dump(mode="json"))

    def run(self, path: str | Path) -> ExperimentReport:
        resolved, definitions, _ = self._prepare(Path(path))
        experiment_id = self._experiment_id_factory()
        experiment_prefix = (
            f"s3://{self._bucket}/{self._prefix}/{experiment_id}/"
        )
        store = self._store_factory(experiment_prefix)
        batch = self._batch_factory(experiment_prefix, store)
        submitted_ids: list[str] = []
        completed_runs = 0

        try:
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
                allocated = 0
                round_number = 0
                while allocated < group.hpo.total_trials and not tracker.exhausted:
                    wanted = min(
                        group.hpo.configs_per_batch,
                        group.hpo.total_trials - allocated,
                    )
                    trials_and_runs = []
                    while len(trials_and_runs) < wanted and not tracker.exhausted:
                        trial = study.ask()
                        try:
                            sampled = sample_parameters(trial, group, tracker=tracker)
                        except DuplicateConfigurationError:
                            try:
                                from optuna.trial import TrialState

                                study.tell(trial, state=TrialState.PRUNED)
                            except TypeError:
                                study.tell(trial, float("nan"))
                            continue
                        except SpaceExhaustedError:
                            break
                        allocated += 1
                        concrete = materialize_run(
                            group,
                            trial,
                            sampled,
                            run_number=allocated,
                        )
                        trials_and_runs.append(
                            (trial, self._run_bundle(experiment_id, group, concrete))
                        )
                    if not trials_and_runs:
                        break
                    round_number += 1
                    round_jobs: list[tuple[Any, list[tuple[Any, RunBundle]]]] = []
                    runs_per_job = group.execution.runs_per_job
                    for offset in range(0, len(trials_and_runs), runs_per_job):
                        children = trials_and_runs[offset : offset + runs_per_job]
                        for _, child in children:
                            self._upload_run(store, child)
                        job = JobBundle(
                            job_id=(
                                f"{experiment_id}-{group.name}-r{round_number}-"
                                f"j{offset // runs_per_job + 1}"
                            ),
                            image_digest=group.image,
                            resource_profile=group.resources.profile,
                            runs=tuple(child for _, child in children),
                        )
                        store.put_json(
                            f"{experiment_prefix}jobs/{job.job_id}/bundle.json",
                            job.model_dump(mode="json"),
                        )
                        submitted = batch.submit(
                            job,
                            profile(group.resources.profile),
                            definitions[group.resources.profile],
                        )
                        submitted_ids.append(submitted.job_id)
                        round_jobs.append((submitted, children))
                    self._wait_jobs(
                        batch, [submitted.job_id for submitted, _ in round_jobs]
                    )
                    for _, children in round_jobs:
                        for trial, run in children:
                            self._collect_marker(store, run)
                            value = self._aim_reader.wait_for_result(
                                run.run_id,
                                group.objective.metric,
                                group.execution.aim_result_timeout_seconds,
                            )
                            study.tell(trial, value)
                            completed_runs += 1
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
            self._persist(store, experiment_prefix, report)
            raise ExperimentRunError(report) from error

        self._persist(store, experiment_prefix, report)
        return report
