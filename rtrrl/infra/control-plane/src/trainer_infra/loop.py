from __future__ import annotations

import json
import time
from collections.abc import Callable

from training_sdk import objects

from trainer_infra.backends.base import Backend
from trainer_infra.launch import Launch, build_run_config
from trainer_infra.packing import publish_round
from trainer_infra.report import Report, TrialRecord
from trainer_infra.space import grid_distributions, sample_parameters
from trainer_infra.study import ask_round, create_study, tell_value

LOG_TAIL_LINES = 200


class LaunchFailed(RuntimeError):
    """The launch stopped because something exited abnormally."""


def run_launch(
    launch: Launch, backend: Backend, printer: Callable[[str], None] = print
) -> Report:
    experiment = launch.plan.experiment
    started = time.monotonic()
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
        },
        round_size=experiment.hpo.trials_per_round,
        grid_space=(
            grid_distributions(
                launch.plan.parameters, points=experiment.hpo.points
            )
            if experiment.hpo.sampler == "grid"
            else None
        ),
        seed=experiment.hpo.seed,
    )

    records: list[TrialRecord] = []
    submitted: list[str] = []
    try:
        for round_index in range(experiment.hpo.rounds):
            trials = ask_round(study, experiment.hpo.trials_per_round)
            drawn = [
                sample_parameters(trial, launch.plan.parameters) for trial in trials
            ]
            configs = [
                build_run_config(launch, trial.number, chosen)
                for trial, chosen in zip(trials, drawn, strict=True)
            ]
            plans = publish_round(
                launch, round_index, configs, jobs=experiment.hpo.parallel_jobs
            )
            # Record each id as it is created rather than assigning the whole
            # list at the end: a Ctrl-C or an AWS error partway through would
            # otherwise leave already-running jobs invisible to the handler
            # below, and they would bill until their own timeout.
            submitted = []
            for index, plan in enumerate(plans):
                submitted.append(
                    backend.submit(
                        launch, plan.manifest_uri, f"round-{round_index:03d}-job-{index}"
                    )
                )
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
            exhausted = False
            for trial, config, chosen in zip(trials, configs, drawn, strict=True):
                value = _read_score(config.score.s3)
                exhausted |= tell_value(study, trial, value)
                result = owner[trial.number]
                records.append(
                    TrialRecord(
                        trial=trial.number,
                        params=dict(chosen),
                        value=value,
                        job_id=result.job_id,
                        log_stream=result.log_stream,
                    )
                )
                printer(f"trial {trial.number}: {chosen} -> {value}")
            if exhausted:
                # A grid with nothing left would spend the remaining rounds
                # re-running points it has already paid for.
                remaining = experiment.hpo.rounds - round_index - 1
                if remaining:
                    printer(
                        f"the search space is exhausted; skipping {remaining} "
                        f"remaining round(s)"
                    )
                break
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

    best = select_best(records, maximize=experiment.score.direction == "maximize")
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


def select_best(
    records: list[TrialRecord], *, maximize: bool
) -> TrialRecord | None:
    return max(
        (record for record in records if record.value is not None),
        key=lambda record: record.value if maximize else -record.value,
        default=None,
    )


def _read_score(uri: str) -> float:
    # A worker that could not upload its score exits non-zero, so reaching this
    # with a missing or malformed object means something unmodelled happened;
    # name the object rather than surfacing a botocore or KeyError trace.
    try:
        return float(json.loads(objects.get_bytes(uri))["value"])
    except Exception as error:
        raise LaunchFailed(f"could not read the score at {uri}: {error}") from error
