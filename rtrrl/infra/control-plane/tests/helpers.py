from pathlib import Path

from training_sdk.contract import Catalog

from trainer_infra.experiment import load_experiment
from trainer_infra.preflight import LaunchPlan, check_offline

EXAMPLE = Path("tests/data/experiment.yaml")
"""The suite's own experiment, deliberately not one of the shipped examples.

The examples under `examples/` name real image digests and the real Aim host, and
they change whenever an image is rebuilt. Asserting against them made every image
push a test failure; `tests/test_examples.py` keeps them honest instead.
"""


CATALOG = Catalog.model_validate(
    {
        "contract": 5,
        "entries": {
            "brax_ppo_acceptance": {
                "command": ["python", "-m", "brax_ppo_acceptance"],
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


def _document() -> dict:
    return {
        "experiment": "demo",
        "name": "one",
        "image": "example.invalid/image@sha256:" + "0" * 64,
        "entry": "demo_entry",
        "storage": "s3://bucket/prefix",
        "environment": {
            "id": "brax::hopper",
            "backend": "spring",
            "seed": 0,
            "observed": [0, 1, 2, 3, 4],
        },
        "training": {"num_envs": 1, "total_steps": 2000, "epoch_steps": 1000},
        "evaluation": {"steps": 100, "num_envs": 1},
        "compute": {"instance_type": "c7a.medium", "timeout_minutes": 60},
        "hpo": {
            "sampler": "tpe",
            "rounds": 1,
            "trials_per_round": 1,
            "parallel_jobs": 1,
        },
        "space": {"learning_rate": [0.001]},
        "score": {
            "metric": "eval/episode_return",
            "window_steps": [0, 2000],
            "reduce": "max",
            "direction": "maximize",
            "non_finite": "worst",
        },
        "logging": {"aim": "aim://127.0.0.1:53801", "every_steps": 1},
    }


def replace_once(content: str, old: str, new: str) -> str:
    """Substitute `old`, refusing to silently leave the text unchanged."""
    if content.count(old) != 1:
        raise AssertionError(
            f"expected exactly one occurrence of {old!r}, found {content.count(old)}"
        )
    return content.replace(old, new)


def write_experiment(tmp_path: Path, s3_base: str, aim_uri: str) -> Path:
    content = EXAMPLE.read_text(encoding="utf-8")
    content = replace_once(
        content, "storage: s3://rtrrl-artifacts-007122174918/trainer", f"storage: {s3_base}"
    )
    content = replace_once(content, "aim: aim://10.0.0.1:53801", f"aim: {aim_uri}")
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
