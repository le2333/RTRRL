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
