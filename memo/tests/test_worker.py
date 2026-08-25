"""Worker supervision and artifact-transport integration tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from deployment.contract import CONTRACT_VERSION
from tests.support.run_config import make_run_config
from worker import objects
from worker.worker import WorkerError, main, run_manifest

pytestmark = pytest.mark.integration

CHILD = """
import json, os, sys
from pathlib import Path

config = json.loads(Path(os.environ["TRAINER_RUN_CONFIG"]).read_text())
scratch = Path(os.environ["TRAINER_SCRATCH"])
trial = config["identity"]["trial"]
with Path(os.environ["TRAINER_CHILD_LOG"]).open("a") as handle:
    handle.write(json.dumps({
        "trial": trial,
        "scratch": str(scratch),
        "started_clean": not (scratch / "artifacts").exists(),
    }) + "\\n")
if str(trial) == os.environ.get("TRAINER_CHILD_FAIL_TRIAL"):
    (scratch / "failure.txt").write_text(f"trial {trial} failed")
    sys.exit(3)
artifacts = scratch / "artifacts"
(artifacts / "rerun").mkdir(parents=True)
(artifacts / "metrics.jsonl").write_text(
    json.dumps({"step": 4, "metrics": {"episode_return": float(trial)}}) + "\\n"
)
(artifacts / "rerun" / "episode.rrd").write_bytes(f"rrd-{trial}".encode())
"""


class FakeStore:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.fail_upload: str | None = None

    def get_bytes(self, uri: str) -> bytes:
        return self.data[uri]

    def put_bytes(self, uri: str, payload: bytes) -> None:
        self.data[uri] = payload

    def put_file(self, uri: str, path: Path) -> None:
        if uri == self.fail_upload:
            raise RuntimeError(f"upload failed for {uri}")
        self.data[uri] = path.read_bytes()

    def exists(self, uri: str) -> bool:
        return uri in self.data


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    monkeypatch.setattr(objects, "get_bytes", fake.get_bytes)
    monkeypatch.setattr(objects, "put_bytes", fake.put_bytes)
    monkeypatch.setattr(objects, "put_file", fake.put_file)
    monkeypatch.setattr(objects, "exists", fake.exists)
    return fake


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    child = tmp_path / "child.py"
    child.write_text(CHILD, encoding="utf-8")
    child_log = tmp_path / "children.jsonl"
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "contract": CONTRACT_VERSION,
                "entries": {
                    "e": {
                        "command": [sys.executable, str(child)],
                        "metrics": ["episode_return"],
                        "parameters": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINER_CATALOG", str(path))
    monkeypatch.setenv("TRAINER_CHILD_LOG", str(child_log))
    return path, child_log


def publish(
    store: FakeStore,
    trial: int,
    *,
    entry: str = "e",
    contract: int = CONTRACT_VERSION,
) -> str:
    payload = make_run_config().model_dump(mode="json")
    payload["contract"] = contract
    payload["identity"]["trial"] = trial
    payload["identity"]["run_id"] = f"smoke-20260725-000000-t{trial}-s0"
    payload["entry"] = entry
    payload["artifacts"]["root"] = f"memory://runs/t{trial}"
    uri = f"memory://configs/t{trial}.json"
    store.put_bytes(uri, json.dumps(payload).encode())
    return uri


def write_manifest(store: FakeStore, uris: list[str]) -> str:
    uri = "memory://manifests/round-000/job-0.json"
    store.put_bytes(uri, json.dumps({"runs": uris}).encode())
    return uri


def result(store: FakeStore, trial: int) -> dict:
    return json.loads(store.data[f"memory://runs/t{trial}/result.json"])


def test_runs_are_serially_isolated_and_publish_artifact_trees(
    store: FakeStore,
    tmp_path: Path,
    catalog: tuple[Path, Path],
) -> None:
    _, child_log = catalog
    manifest = write_manifest(store, [publish(store, 0), publish(store, 1)])

    run_manifest(manifest, tmp_path / "worker")

    starts = [json.loads(line) for line in child_log.read_text().splitlines()]
    assert [item["trial"] for item in starts] == [0, 1]
    assert len({item["scratch"] for item in starts}) == 2
    assert all(item["started_clean"] for item in starts)
    assert list((tmp_path / "worker").iterdir()) == []
    for trial in (0, 1):
        assert store.data[f"memory://runs/t{trial}/rerun/episode.rrd"] == (
            f"rrd-{trial}".encode()
        )
        payload = result(store, trial)
        assert payload == {
            "contract": CONTRACT_VERSION,
            "identity": {
                "run_id": f"smoke-20260725-000000-t{trial}-s0",
                "experiment": "infra-acceptance",
                "launch_id": "20260725-000000",
                "trial": trial,
                "seed": 0,
                "role": "tuning",
                "digest": "registry.example/trainer@sha256:" + "a" * 64,
            },
            "success": True,
            "artifacts": ["metrics.jsonl", "rerun/episode.rrd"],
        }


def test_child_failure_stops_manifest_and_preserves_failed_scratch(
    store: FakeStore,
    tmp_path: Path,
    catalog: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, child_log = catalog
    monkeypatch.setenv("TRAINER_CHILD_FAIL_TRIAL", "1")
    manifest = write_manifest(
        store,
        [publish(store, 0), publish(store, 1), publish(store, 2)],
    )

    with pytest.raises(WorkerError, match="exit code 3"):
        run_manifest(manifest, tmp_path / "worker")

    starts = [json.loads(line) for line in child_log.read_text().splitlines()]
    assert [item["trial"] for item in starts] == [0, 1]
    assert result(store, 0)["success"] is True
    assert "memory://runs/t1/result.json" not in store.data
    assert "memory://runs/t2/result.json" not in store.data
    scratch = list((tmp_path / "worker").iterdir())
    assert len(scratch) == 1
    assert (scratch[0] / "failure.txt").read_text() == "trial 1 failed"


def test_upload_failure_is_visible_and_preserves_scratch(
    store: FakeStore,
    tmp_path: Path,
    catalog: tuple[Path, Path],
) -> None:
    store.fail_upload = "memory://runs/t0/rerun/episode.rrd"
    manifest = write_manifest(store, [publish(store, 0)])

    with pytest.raises(RuntimeError, match="upload failed"):
        run_manifest(manifest, tmp_path / "worker")

    assert "memory://runs/t0/result.json" not in store.data
    scratch = list((tmp_path / "worker").iterdir())
    assert len(scratch) == 1
    assert (scratch[0] / "artifacts" / "rerun" / "episode.rrd").is_file()


def test_catalog_contract_mismatch(
    store: FakeStore,
    tmp_path: Path,
    catalog: tuple[Path, Path],
) -> None:
    path, _ = catalog
    payload = json.loads(path.read_text())
    payload["contract"] = 1
    path.write_text(json.dumps(payload))
    manifest = write_manifest(store, [publish(store, 0)])

    with pytest.raises(ValidationError, match="contract"):
        run_manifest(manifest, tmp_path / "worker")


def test_run_contract_mismatch(
    store: FakeStore,
    tmp_path: Path,
    catalog: tuple[Path, Path],
) -> None:
    manifest = write_manifest(store, [publish(store, 0, contract=1)])

    with pytest.raises(ValidationError, match="contract"):
        run_manifest(manifest, tmp_path / "worker")


def test_missing_catalog_entry(
    store: FakeStore,
    tmp_path: Path,
    catalog: tuple[Path, Path],
) -> None:
    manifest = write_manifest(store, [publish(store, 0, entry="missing")])

    with pytest.raises(
        WorkerError, match="image catalog does not declare entry 'missing'"
    ):
        run_manifest(manifest, tmp_path / "worker")


def test_main_reports_a_missing_manifest_variable(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRAINER_MANIFEST", raising=False)

    assert main() == 1
    assert "TRAINER_MANIFEST is not set" in capsys.readouterr().err


def test_a_second_attempt_does_not_repeat_a_run_that_finished(
    store: FakeStore,
    tmp_path: Path,
    catalog: tuple[Path, Path],
) -> None:
    """A retried job carries the same manifest, not the part still owing.

    Batch starts the manifest again, and the runs it had already carried to
    the end have published their artifacts and their result. Running one of
    those again spends its whole budget to overwrite what is there with a
    second sample of it -- and, on a job that was retried because it ran out
    of time, spends the time the run still owing needed.
    """

    _, child_log = catalog
    manifest = write_manifest(store, [publish(store, 0), publish(store, 1)])
    run_manifest(manifest, tmp_path / "first")

    del store.data["memory://runs/t1/result.json"]
    run_manifest(manifest, tmp_path / "second")

    started = [json.loads(line)["trial"] for line in child_log.read_text().splitlines()]
    assert started == [0, 1, 1]


def test_artifacts_without_a_result_are_an_attempt_that_did_not_finish(
    store: FakeStore,
    tmp_path: Path,
    catalog: tuple[Path, Path],
) -> None:
    """The result object is written last, and is what says a run is done."""

    _, child_log = catalog
    manifest = write_manifest(store, [publish(store, 0)])
    store.put_bytes("memory://runs/t0/metrics.jsonl", b"{}\n")

    run_manifest(manifest, tmp_path / "worker")

    assert [
        json.loads(line)["trial"] for line in child_log.read_text().splitlines()
    ] == [0]
