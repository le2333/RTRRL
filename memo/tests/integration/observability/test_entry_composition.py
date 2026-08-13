import json

import pytest
from aim import Repo

from entries._observability import build_reporter, load_run
from memorax.observability.sinks.metrics import METRICS_FILENAME
from tests.support.observability import completed_episode
from tests.support.run_config import make_run_config

pytestmark = [pytest.mark.integration, pytest.mark.service]


def config_with_local_aim(tmp_path, *, rerun=None):
    endpoint = str(tmp_path / "aim")
    Repo.from_path(endpoint, init=True)
    config = make_run_config()
    payload = config.model_dump(mode="json")
    payload["logging"] = {"aim": {"url": endpoint}, "rerun": rerun}
    return type(config).model_validate(payload)


def test_entry_loads_the_deployment_document_and_scratch(tmp_path, monkeypatch):
    config = make_run_config()
    config_path = tmp_path / "run-config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("TRAINER_RUN_CONFIG", str(config_path))
    monkeypatch.setenv("TRAINER_SCRATCH", str(scratch))

    loaded, loaded_scratch = load_run()

    assert loaded == config
    assert loaded_scratch == scratch


def test_entry_composes_mandatory_scalars_and_optional_local_episodes(tmp_path):
    scratch = tmp_path / "scratch"
    config = config_with_local_aim(
        tmp_path,
        rerun={"every_steps": 1},
    )

    with build_reporter(config, scratch) as reporter:
        reporter.log_episode(completed_episode())

    records = [
        json.loads(line)
        for line in (scratch / "artifacts" / METRICS_FILENAME).read_text().splitlines()
    ]
    assert records[0]["step"] == 8
    assert records[0]["metrics"]["train/episode/return"] == 4.0
    assert (scratch / "artifacts" / "rerun" / "train-000001.rrd").exists()
