import json
import sys
from pathlib import Path

import pytest

from tests.support.run_config import make_run_config
from deployment.contract import CONTRACT_VERSION
from worker import objects
from worker.worker import WorkerError, main, run_manifest

pytestmark = [pytest.mark.integration, pytest.mark.service]

make_config = make_run_config

CHILD = """
import json, os, sys
config = json.loads(open(os.environ["TRAINER_RUN_CONFIG"]).read())
scratch = os.environ["TRAINER_SCRATCH"]
mode = os.environ.get("CHILD_MODE", "ok")
if mode == "crash":
    sys.exit(3)
with open(os.path.join(scratch, "metrics.jsonl"), "a") as handle:
    for step in (0, 4):
        handle.write(json.dumps({"step": step, "metrics": {"episode_return": 2.0}}) + "\\n")
        handle.flush()
if mode == "empty_metrics":
    open(os.path.join(scratch, "metrics.jsonl"), "w").close()
"""


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    child = tmp_path / "child.py"
    child.write_text(CHILD, encoding="utf-8")
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "contract": CONTRACT_VERSION,
                "entries": {
                    "e": {
                        "command": [sys.executable, str(child)],
                        "metrics": ["episode_return"],
                        "parameters": {
                            "learning_rate": {
                                "kind": "param",
                                "value_type": "float",
                                "valid": {"type": "float", "low": 0.0, "high": 1.0},
                                "search": [0.001],
                                "placeholder": 0.001,
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINER_CATALOG", str(path))
    return path


def publish(s3_base: str, trial: int, *, entry: str = "e") -> str:
    config = make_config().model_copy(
        update={
            "trial": trial,
            "run_id": f"smoke-20260725-000000-t{trial}",
            "entry": entry,
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


def test_every_run_is_executed_and_scored(
    s3_base: str, tmp_path: Path, catalog: Path
) -> None:
    manifest = write_manifest(s3_base, [publish(s3_base, 0), publish(s3_base, 1)])
    run_manifest(manifest, tmp_path)
    for trial in (0, 1):
        payload = json.loads(objects.get_bytes(f"{s3_base}/trials/t{trial}/score.json"))
        assert payload["value"] == 2.0
        assert payload["run_id"] == f"smoke-20260725-000000-t{trial}"


def test_scratch_is_removed_between_runs(
    s3_base: str, tmp_path: Path, catalog: Path
) -> None:
    manifest = write_manifest(s3_base, [publish(s3_base, 0), publish(s3_base, 1)])
    run_manifest(manifest, tmp_path)
    assert list(tmp_path.glob("*/metrics.jsonl")) == []


def test_crashing_run_stops_the_manifest(
    s3_base: str, tmp_path: Path, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHILD_MODE", "crash")
    manifest = write_manifest(s3_base, [publish(s3_base, 100), publish(s3_base, 101)])
    with pytest.raises(WorkerError, match="exit code 3"):
        run_manifest(manifest, tmp_path)
    assert objects.exists(f"{s3_base}/trials/t100/score.json") is False
    assert objects.exists(f"{s3_base}/trials/t101/score.json") is False


def test_catalog_contract_mismatch(
    s3_base: str, tmp_path: Path, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog.write_text(
        json.dumps(
            {
                "contract": 1,
                "entries": {
                    "e": {
                        "command": ["true"],
                        "metrics": ["episode_return"],
                        "parameters": {
                            "learning_rate": {
                                "kind": "param",
                                "value_type": "float",
                                "valid": {"type": "float", "low": 0.0, "high": 1.0},
                                "search": [0.001],
                                "placeholder": 0.001,
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = write_manifest(s3_base, [publish(s3_base, 0)])
    with pytest.raises(WorkerError, match="image catalog declares contract 1"):
        run_manifest(manifest, tmp_path)


def test_run_config_contract_mismatch(
    s3_base: str, tmp_path: Path, catalog: Path
) -> None:
    config = make_config().model_copy(
        update={
            "contract": 1,
            "score": make_config().score.model_copy(
                update={"s3": f"{s3_base}/trials/t0/score.json"}
            ),
        }
    )
    uri = f"{s3_base}/trials/t0/config.json"
    objects.put_bytes(uri, config.model_dump_json().encode())
    manifest = write_manifest(s3_base, [uri])
    with pytest.raises(WorkerError, match="run configuration declares contract 1"):
        run_manifest(manifest, tmp_path)


def test_missing_catalog_entry(s3_base: str, tmp_path: Path, catalog: Path) -> None:
    manifest = write_manifest(s3_base, [publish(s3_base, 0, entry="missing")])
    with pytest.raises(
        WorkerError, match="image catalog does not declare entry 'missing'"
    ):
        run_manifest(manifest, tmp_path)


def test_score_computation_failure_stops_manifest(
    s3_base: str, tmp_path: Path, catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from worker.score import ScoreError

    monkeypatch.setenv("CHILD_MODE", "empty_metrics")
    manifest = write_manifest(s3_base, [publish(s3_base, 200), publish(s3_base, 201)])
    with pytest.raises(ScoreError, match="no reported value for metric"):
        run_manifest(manifest, tmp_path)
    assert objects.exists(f"{s3_base}/trials/t200/score.json") is False
    assert objects.exists(f"{s3_base}/trials/t201/score.json") is False


def test_main_reports_a_missing_manifest_variable(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The image's default command is this entry point, so it must explain itself.

    A Batch job started without the manifest is misconfigured by the control
    plane, and the only trace left behind is what lands in CloudWatch.
    """
    monkeypatch.delenv("TRAINER_MANIFEST", raising=False)

    assert main() == 1
    assert "TRAINER_MANIFEST is not set" in capsys.readouterr().err
