from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import optuna
import yaml

from trainer_infra import ExperimentRunner, cli
from trainer_infra.cli import main

LAUNCH = "20260807-120000"

WORKER = """
import json, os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

def path(uri):
    return Path(url2pathname(urlparse(uri).path))

manifest = json.loads(path(os.environ["TRAINER_MANIFEST"]).read_text())
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


R2_PARENT: dict[str, Any] = {
    "contract": 11,
    "identity": {
        "run_id": "rtrrl-r2-t0",
        "experiment": "rtrrl-issue45-r2",
        "launch_id": LAUNCH,
        "trial": 0,
        "digest": "sha256:abc",
    },
    "entry": "rtrrl",
    "artifacts": {"root": "s3://bucket/rtrrl-issue45-r2/launch/rtrrl-r2-t0"},
    "algorithm": {
        "environment": {
            "id": "brax::halfcheetah",
            "backend": "spring",
            "observed": [0, 1],
            "episode_length": 1000,
        },
        "num_envs": 1,
        "parameters": {
            "torso.grad_clip": 1.0,
            "torso.optimizer.kind": "adam",
            "heads.optimizer.kind": "adam",
        },
    },
    "training": {"seed": 11, "total_steps": 100000, "chunk_steps": 10000},
    "evaluation": {"every_steps": 10000, "rollout_steps": 5000},
    "checkpoint": {"every_steps": 10000, "keep": None},
    "logging": {"aim": {"url": "aim://aim:53800"}},
}


def collapsing_metrics(path: Path) -> Path:
    """One seed that learns, gives back most of it, and stays down."""

    returns = {10000: 10.0, 20000: 100.0, 30000: 20.0, 40000: 15.0, 50000: 18.0}
    rows = [
        json.dumps({"step": step, "metrics": {"eval/episode/return": value}})
        for step, value in returns.items()
    ]
    rows += [
        json.dumps(
            {
                "step": step,
                "metrics": {"train/episode/update.torso.raw_update_norm": float(step)},
            }
        )
        for step in returns
    ]
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_collapse_command_decides_each_seed_and_writes_the_decisions(
    tmp_path: Path, capsys: Any
) -> None:
    """The per-seed decision document, which is what a fork is chosen from."""

    spec = tmp_path / "collapse.json"
    spec.write_text(
        json.dumps({"metric": "eval/episode/return", "random_floor": 0.0}),
        encoding="utf-8",
    )
    metrics = collapsing_metrics(tmp_path / "seed-0.jsonl")
    decisions = tmp_path / "decisions.json"

    assert (
        main(
            [
                "collapse",
                "--run",
                f"rtrrl-r2-t0={metrics}",
                "--spec",
                str(spec),
                "--decisions",
                str(decisions),
                "--window-steps",
                "20000",
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(decisions.read_text(encoding="utf-8"))
    assert printed["collapsed"] == ["rtrrl-r2-t0"]
    seed = printed["seeds"][0]
    assert seed["collapse"]["step"] == 30000
    assert seed["spec"]["random_floor"] == 0.0
    assert seed["windows"]["train/episode/update.torso.raw_update_norm"]


def test_fork_command_writes_three_branches_and_the_manifest_naming_them(
    tmp_path: Path, capsys: Any
) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(R2_PARENT), encoding="utf-8")
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps({"run_id": "rtrrl-r2-t0", "verdict": "collapsed", "collapse": {"step": 30000}}),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "fork",
                "--parent",
                str(parent),
                "--decision",
                str(decision),
                "--into",
                str(tmp_path / "fork"),
                "--steps",
                "50000",
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["from_steps"] == 20000
    assert printed["checkpoint"].endswith("checkpoints/step-000000020000.msgpack")
    assert printed["branches"] == [
        "rtrrl-r2-t0-original-clip",
        "rtrrl-r2-t0-fixed-step",
        "rtrrl-r2-t0-td-out",
    ]

    manifest = json.loads((tmp_path / "fork" / "manifest.json").read_text())
    assert len(manifest["runs"]) == 3
    for uri, name in zip(manifest["runs"], printed["branches"], strict=True):
        document = json.loads(
            Path(url2pathname(urlparse(uri).path)).read_text(encoding="utf-8")
        )
        assert document["identity"]["run_id"] == name
        assert document["fork"]["from_steps"] == 20000
        assert document["training"]["total_steps"] == 70000


def test_fork_refuses_a_seed_that_did_not_collapse(tmp_path: Path) -> None:
    """Nothing is forked from a run with no event; the reason is repeated back."""

    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(R2_PARENT), encoding="utf-8")
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "run_id": "rtrrl-r2-t0",
                "verdict": "steady",
                "reason": "no drawdown of 0.5 held",
                "collapse": None,
            }
        ),
        encoding="utf-8",
    )

    try:
        main(["fork", "--parent", str(parent), "--decision", str(decision), "--into", str(tmp_path)])
    except SystemExit as stopped:
        assert "steady" in str(stopped) and "no drawdown" in str(stopped)
    else:  # pragma: no cover - the call above must not succeed
        raise AssertionError("a steady seed was forked")


def test_run_command_executes_every_hpo_round(
    tmp_path: Path,
    capsys: Any,
    experiment: dict[str, Any],
    catalog: dict[str, Any],
) -> None:
    experiment["storage"] = (tmp_path / "artifacts").resolve().as_uri()
    experiment["hpo"]["rounds"] = 2
    experiment["score"] = {
        "metric": "objective",
        "window_steps": [0, 10],
        "reduce": "last",
        "non_finite": "worst",
        "direction": "maximize",
    }
    catalog["entries"]["stream_ac"]["metrics"] = ["objective"]
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER, encoding="utf-8")

    assert (
        main(
            [
                "run",
                str(experiment_path),
                "--backend",
                "local",
                "--catalog",
                str(catalog_path),
                "--database",
                str(tmp_path / "study.db"),
                "--launch-id",
                LAUNCH,
                "--exchange",
                str(tmp_path / "exchange"),
                "--workspace",
                str(tmp_path / "worker-scratch"),
                "--worker-command",
                sys.executable,
                str(worker),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["study"] == experiment["name"]
    assert [trial["number"] for trial in payload["trials"]] == [0, 1, 2, 3]
    assert [trial["value"] for trial in payload["trials"]] == [1.0, 2.0, 3.0, 4.0]
    assert payload["best"] == {"number": 3, "value": 4.0}


def test_settle_command_scores_open_trials_and_leaves_unfinished_ones_alone(
    tmp_path: Path,
    capsys: Any,
    experiment: dict[str, Any],
    catalog: dict[str, Any],
) -> None:
    """Resuming a launch a killed controller left behind, from the terminal.

    Trial 0's worker finished and uploaded; trial 1's has not. The command
    starts nothing either way -- it reads what is there, tells the study about
    trial 0, and says why trial 1 is still open.
    """

    experiment["storage"] = (tmp_path / "artifacts").resolve().as_uri()
    experiment["score"] = {
        "metric": "objective",
        "window_steps": [0, 10],
        "reduce": "last",
        "non_finite": "worst",
        "direction": "maximize",
    }
    catalog["entries"]["stream_ac"]["metrics"] = ["objective"]
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    database = tmp_path / "study.db"

    asked = ExperimentRunner(
        experiment=experiment,
        catalog=catalog,
        database=database,
        launch_id=LAUNCH,
    ).next_round()
    finished = Path(url2pathname(urlparse(asked[0]["artifacts"]["root"]).path))
    finished.mkdir(parents=True)
    (finished / "metrics.jsonl").write_text(
        json.dumps({"step": 10, "metrics": {"objective": 21.0}}) + "\n", encoding="utf-8"
    )
    (finished / "result.json").write_text(
        json.dumps(
            {
                "contract": asked[0]["contract"],
                "identity": asked[0]["identity"],
                "success": True,
                "artifacts": ["metrics.jsonl"],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "settle",
                str(experiment_path),
                "--backend",
                "local",
                "--catalog",
                str(catalog_path),
                "--database",
                str(database),
                "--launch-id",
                LAUNCH,
                "--exchange",
                str(tmp_path / "exchange"),
                "--workspace",
                str(tmp_path / "worker-scratch"),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["launch_id"] == LAUNCH
    assert payload["settled"] == [{"trial": 0, "value": 21.0}]
    assert [entry["trial"] for entry in payload["still_running"]] == [1]
    persisted = optuna.load_study(study_name=experiment["name"], storage=f"sqlite:///{database}")
    assert [trial.state.name for trial in persisted.trials] == ["COMPLETE", "RUNNING"]
    assert not (tmp_path / "exchange").exists(), "settling submitted work"


def test_batch_run_routes_deployment_fields_to_batch_executor(
    tmp_path: Path,
    capsys: Any,
    monkeypatch: Any,
    experiment: dict[str, Any],
    catalog: dict[str, Any],
) -> None:
    experiment["compute"] = {"instance_type": "c7a.large", "timeout_minutes": 90}
    experiment["hpo"]["parallel_jobs"] = 2
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    captured: dict[str, Any] = {}

    class Session:
        def client(self, name: str) -> str:
            return name

    class Executor:
        def __init__(self, **options: Any) -> None:
            captured.update(options)

        def __call__(self, configurations: tuple[dict, ...], score: Any) -> tuple[dict, ...]:
            return tuple(
                {
                    "trial": configuration["identity"]["trial"],
                    "value": float(configuration["identity"]["trial"]),
                }
                for configuration in configurations
            )

    monkeypatch.setattr(cli, "_batch_session", Session)
    monkeypatch.setattr(cli, "BatchRoundExecutor", Executor)

    assert (
        main(
            [
                "run",
                str(experiment_path),
                "--backend",
                "batch",
                "--catalog",
                str(catalog_path),
                "--database",
                str(tmp_path / "study.db"),
                "--launch-id",
                LAUNCH,
                "--queues",
                "dev",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["best"] == {"number": 1, "value": 1.0}
    assert captured["s3"] == "s3"
    assert captured["batch"] == "batch"
    assert captured["logs"] == "logs"
    assert captured["job_queue"] == "dev-cpu-c7al-queue"
    assert captured["job_definition"] == "trainer-c7al-" + "b" * 64
    assert captured["timeout_seconds"] == 5400
    assert captured["parallel_jobs"] == 2
