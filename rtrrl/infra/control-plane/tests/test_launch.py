import json
from datetime import UTC, datetime
from pathlib import Path

from training_sdk import objects

from trainer_infra.launch import build_run_config, config_uri, create_launch
from tests.helpers import make_plan

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
