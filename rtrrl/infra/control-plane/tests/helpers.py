from pathlib import Path

from trainer_infra.experiment import load_experiment
from trainer_infra.preflight import LaunchPlan, check_offline
from tests.test_preflight_offline import CATALOG

EXAMPLE = Path("examples/experiment-acceptance.yaml")


def write_experiment(tmp_path: Path, s3_base: str, aim_uri: str) -> Path:
    content = EXAMPLE.read_text(encoding="utf-8")
    content = content.replace(
        "storage: s3://rtrrl-artifacts-007122174918/trainer", f"storage: {s3_base}"
    )
    content = content.replace("aim: aim://172.31.62.192:53801", f"aim: {aim_uri}")
    path = tmp_path / "experiment.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def make_plan(s3_base: str, *, rerun_enabled: bool = True) -> LaunchPlan:
    experiment = load_experiment(EXAMPLE)
    updates: dict[str, object] = {"storage": s3_base}
    if not rerun_enabled:
        updates["logging"] = experiment.logging.model_copy(
            update={"rerun_every_episodes": None}
        )
    experiment = experiment.model_copy(update=updates)
    return LaunchPlan(
        experiment=experiment,
        entry_name=experiment.entry,
        entry=CATALOG.entries[experiment.entry],
        space=check_offline(experiment, CATALOG),
        digest="sha256:" + "a" * 64,
        queue="run-cpu-c7am-queue",
        job_definition="trainer-c7am-" + "a" * 64,
    )
