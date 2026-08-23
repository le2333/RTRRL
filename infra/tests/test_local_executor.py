from __future__ import annotations

import json
import sys
from pathlib import Path

from trainer_infra.local import LocalRoundExecutor
from trainer_infra.scoring import ScoreSpec

WORKER = """
import json, os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

def path(uri):
    return Path(url2pathname(urlparse(uri).path))

manifest = json.loads(path(os.environ["TRAINER_MANIFEST"]).read_text())
catalog = json.loads(Path(os.environ["TRAINER_CATALOG"]).read_text())
assert "e" in catalog["entries"]
for config_uri in manifest["runs"]:
    config = json.loads(path(config_uri).read_text())
    root = path(config["artifacts"]["root"])
    root.mkdir(parents=True)
    trial = config["identity"]["trial"]
    (root / "metrics.jsonl").write_text(
        json.dumps({"step": 10, "metrics": {"objective": float(trial + 1)}}) + "\\n"
    )
    (root / "result.json").write_text(json.dumps({
        "contract": config["contract"],
        "identity": config["identity"],
        "success": True,
        "artifacts": ["metrics.jsonl"],
    }))
"""


def configuration(root: Path, trial: int) -> dict:
    return {
        "contract": 11,
        "identity": {
            "run_id": f"run-t{trial}-s0",
            "experiment": "test",
            "launch_id": "launch",
            "trial": trial,
            "seed": 0,
            "role": "tuning",
            "digest": "sha256:" + "a" * 64,
        },
        "entry": "e",
        "artifacts": {"root": (root / f"t{trial}").resolve().as_uri()},
        "algorithm": {},
        "runtime": {},
        "logging": {},
    }


def test_local_executor_serializes_invokes_worker_and_scores_results(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER, encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"entries": {"e": {}}}), encoding="utf-8")
    exchange = tmp_path / "exchange"
    configurations = (
        configuration(exchange / "artifacts", 0),
        configuration(exchange / "artifacts", 1),
    )
    executor = LocalRoundExecutor(
        catalog=catalog,
        exchange=exchange,
        workspace=tmp_path / "worker-scratch",
        worker_command=(sys.executable, str(worker)),
    )

    results = executor(
        configurations,
        ScoreSpec(
            metric="objective",
            window_steps=(0, 10),
            reduce="last",
            direction="maximize",
            non_finite="worst",
        ),
    )

    assert results == (
        {"trial": 0, "seed": 0, "value": 1.0},
        {"trial": 1, "seed": 0, "value": 2.0},
    )
    manifest = json.loads((exchange / "round-000" / "manifest.json").read_text())
    assert len(manifest["runs"]) == 2
    assert all(uri.startswith("file:") for uri in manifest["runs"])
    assert (
        json.loads(
            (exchange / "round-000" / "stream-ac-test-run1-seed1.json").read_text()
        )
        == (configurations[0])
    )
    assert executor.log_path(0).is_file()


def test_local_executor_surfaces_worker_failure(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text("raise RuntimeError('worker exploded')", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"entries": {"e": {}}}), encoding="utf-8")
    executor = LocalRoundExecutor(
        catalog=catalog,
        exchange=tmp_path / "exchange",
        workspace=tmp_path / "worker-scratch",
        worker_command=(sys.executable, str(worker)),
    )

    import pytest

    with pytest.raises(RuntimeError, match="worker exploded"):
        executor(
            (configuration(tmp_path / "artifacts", 0),),
            ScoreSpec(
                metric="objective",
                window_steps=(0, 10),
                reduce="last",
                direction="maximize",
                non_finite="worst",
            ),
        )
