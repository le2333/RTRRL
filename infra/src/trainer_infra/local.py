"""Local round execution through the same serialized Worker boundary as Batch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from trainer_infra.experiment import run_name
from trainer_infra.scoring import ScoreSpec, compute_score


class LocalExecutionError(RuntimeError):
    """The local Worker or one of its declared results did not complete."""


def file_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc:
        raise LocalExecutionError(f"local execution requires a file URI, got {uri!r}")
    return Path(url2pathname(parsed.path))




class LocalRoundExecutor:
    def __init__(
        self,
        *,
        catalog: Path,
        exchange: Path,
        workspace: Path,
        worker_command: Sequence[str] | None = None,
    ) -> None:
        self.catalog = Path(catalog).resolve()
        self.exchange = Path(exchange).resolve()
        self.workspace = Path(workspace).resolve()
        self.worker_command = tuple(worker_command or (sys.executable, "-m", "worker"))
        self._round = 0

    def __call__(
        self,
        configurations: tuple[dict[str, Any], ...],
        score: ScoreSpec,
    ) -> tuple[dict[str, int | float], ...]:
        round_index = self._round
        self._round += 1
        directory = self.exchange / f"round-{round_index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        config_uris = []
        for configuration in configurations:
            path = directory / f"{run_name(configuration)}.json"
            path.write_text(json.dumps(configuration, sort_keys=True), encoding="utf-8")
            config_uris.append(path.resolve().as_uri())

        manifest_path = directory / "manifest.json"
        manifest_path.write_text(
            json.dumps({"runs": config_uris}, sort_keys=True),
            encoding="utf-8",
        )
        self.workspace.mkdir(parents=True, exist_ok=True)
        log = self.log_path(round_index)
        environment = dict(os.environ)
        environment.update(
            {
                "TRAINER_MANIFEST": manifest_path.resolve().as_uri(),
                "TRAINER_CATALOG": str(self.catalog),
                "TRAINER_WORKSPACE": str(self.workspace / f"round-{round_index:03d}"),
            }
        )
        with log.open("wb") as handle:
            process = subprocess.run(
                self.worker_command,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode != 0:
            output = log.read_text(encoding="utf-8", errors="replace")
            raise LocalExecutionError(
                f"local Worker exited with code {process.returncode}\n{output}"
            )

        return self.score(configurations, score)

    def log_path(self, round_index: int) -> Path:
        return self.exchange / f"round-{round_index:03d}" / "worker.log"

    def score(
        self,
        configurations: tuple[dict[str, Any], ...],
        score: ScoreSpec,
    ) -> tuple[dict[str, int | float], ...]:
        """Score what these configurations' runs have already written.

        Starting no worker is the point: it is how a trial whose run finished
        is settled without running it again.
        """

        return tuple(self._score(configuration, score) for configuration in configurations)

    def _score(
        self,
        configuration: Mapping[str, Any],
        score: ScoreSpec,
    ) -> dict[str, int | float]:
        identity = configuration["identity"]
        trial, seed = int(identity["trial"]), int(identity["seed"])
        run = run_name(configuration)
        root = file_path(str(configuration["artifacts"]["root"]))
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        produced = result["identity"]
        if result["success"] is not True or (
            int(produced["trial"]),
            int(produced["seed"]),
        ) != (trial, seed):
            raise LocalExecutionError(f"artifact result does not complete {run}")
        if "metrics.jsonl" not in result["artifacts"]:
            raise LocalExecutionError(f"{run} result declares no metrics.jsonl")
        return {
            "trial": trial,
            "seed": seed,
            "value": compute_score(root / "metrics.jsonl", score),
        }
