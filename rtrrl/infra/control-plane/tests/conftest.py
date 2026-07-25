import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from aim import Repo

pytest_plugins = ["training_sdk.testing"]


@dataclass(frozen=True)
class AimServer:
    uri: str
    path: str


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def aim_endpoint(tmp_path_factory: pytest.TempPathFactory) -> AimServer:
    repo_path = tmp_path_factory.mktemp("aim-repo")
    Repo.from_path(str(repo_path), init=True)
    port = _free_port()
    process = subprocess.Popen(
        ["aim", "server", "--repo", str(repo_path), "--port", str(port),
         "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                break
        if process.poll() is not None:
            raise RuntimeError("aim server exited before accepting connections")
        time.sleep(0.2)
    else:
        process.kill()
        raise RuntimeError("aim server did not start within 60s")
    yield AimServer(uri=f"aim://127.0.0.1:{port}", path=str(repo_path))
    process.terminate()
    process.wait(timeout=30)


TRAINER = """
import json, os
from training_sdk.reporter import Reporter
config_path = os.environ["TRAINER_RUN_CONFIG"]
config = json.loads(open(config_path).read())
total = int(config["params"]["total_steps"])
rate = float(config["params"]["learning_rate"])
with Reporter.from_env() as reporter:
    for step in range(0, total + 1, max(total // 4, 1)):
        reporter.report(step, {"episode_return": rate * 1000 + step})
"""


@pytest.fixture
def acceptance_catalog(tmp_path: Path) -> Path:
    trainer = tmp_path / "trainer.py"
    trainer.write_text(TRAINER, encoding="utf-8")
    return _catalog(tmp_path, [sys.executable, str(trainer)])


@pytest.fixture
def failing_catalog(tmp_path: Path) -> Path:
    child = tmp_path / "boom.py"
    child.write_text("import sys; sys.exit(7)", encoding="utf-8")
    return _catalog(tmp_path, [sys.executable, str(child)])


def _catalog(tmp_path: Path, command: list[str]) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "contract": 2,
                "entries": {
                    "brax_ppo_acceptance": {
                        "command": command,
                        "source_hash": "sha256:0",
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
        ),
        encoding="utf-8",
    )
    return path
