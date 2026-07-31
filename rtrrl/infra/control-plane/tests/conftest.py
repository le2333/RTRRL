import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from aim import Repo

pytest_plugins = ["training_sdk.testing"]


@dataclass(frozen=True)
class AimServer:
    uri: str
    path: str


def _own_address() -> str:
    """This machine's address as another host on its network would name it.

    Found by asking the routing table where a packet would leave from; the UDP
    connect sends nothing and needs no reachable peer. Loopback will not do,
    because preflight rejects it for batch launches.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1, reserved for documentation
        return probe.getsockname()[0]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind((_own_address(), 0))
        return probe.getsockname()[1]


@pytest.fixture
def listening_endpoint() -> Iterator[str]:
    """An endpoint whose port accepts a connection, without starting Aim.

    Preflight's Aim check is a bare TCP connect, so a listener is enough. Tests
    that need one must take this fixture rather than relying on whatever happens
    to be listening: the two `validate --backend batch` cases used to pass only
    because a real Aim server runs on the development machine, and they failed
    the moment they ran anywhere else.

    The address is this host's own, not loopback, so it is also an endpoint a
    Batch job could have reached.
    """
    address = _own_address()
    with socket.socket() as server:
        server.bind((address, 0))
        server.listen(1)
        yield f"aim://{address}:{server.getsockname()[1]}"


@pytest.fixture
def aim_endpoint(tmp_path_factory: pytest.TempPathFactory) -> AimServer:
    """A real Aim server, reachable by the address a Batch job would use.

    It listens on every interface and is named by this host's own address rather
    than loopback, because preflight refuses a loopback endpoint for batch
    launches — a job would resolve it to itself.
    """
    repo_path = tmp_path_factory.mktemp("aim-repo")
    Repo.from_path(str(repo_path), init=True)
    port = _free_port()
    address = _own_address()
    process = subprocess.Popen(
        ["aim", "server", "--repo", str(repo_path), "--port", str(port),
         "--host", "0.0.0.0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex((address, port)) == 0:
                break
        if process.poll() is not None:
            raise RuntimeError("aim server exited before accepting connections")
        time.sleep(0.2)
    else:
        process.kill()
        raise RuntimeError("aim server did not start within 60s")
    yield AimServer(uri=f"aim://{address}:{port}", path=str(repo_path))
    process.terminate()
    process.wait(timeout=30)


TRAINER = """
import json, os
from training_sdk.reporter import Reporter
config_path = os.environ["TRAINER_RUN_CONFIG"]
config = json.loads(open(config_path).read())
total = int(config["budget"]["total_steps"])
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


@pytest.fixture
def failing_with_long_sibling_catalog(tmp_path: Path) -> Path:
    pid_file = tmp_path / "long-sibling.pid"
    child = tmp_path / "mixed.py"
    child.write_text(
        f"""
import json, os, subprocess, sys, time
from pathlib import Path
config = json.loads(open(os.environ["TRAINER_RUN_CONFIG"]).read())
if config["trial"] == 0:
    sys.exit(7)
proc = subprocess.Popen(["sleep", "600"])
Path("{pid_file}").write_text(str(proc.pid))
time.sleep(600)
""",
        encoding="utf-8",
    )
    return _catalog(tmp_path, [sys.executable, str(child)])


@pytest.fixture
def launch_for_batch(s3_base: str, tmp_path: Path):
    from datetime import UTC, datetime

    from tests.helpers import EXAMPLE, make_plan
    from trainer_infra.launch import create_launch

    when = datetime(2026, 7, 25, 5, 14, tzinfo=UTC)
    return create_launch(make_plan(s3_base), tmp_path / "archive", EXAMPLE, when)


def _catalog(tmp_path: Path, command: list[str]) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "contract": 4,
                "entries": {
                    "brax_ppo_acceptance": {
                        "command": command,
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
