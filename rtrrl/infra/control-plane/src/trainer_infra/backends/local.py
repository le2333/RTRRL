from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from trainer_infra.backends.base import JobResult
from trainer_infra.launch import Launch


def _descendant_pids(root: int) -> list[int]:
    pids: list[int] = []
    try:
        children = Path(f"/proc/{root}/task/{root}/children").read_text().split()
    except OSError:
        return pids
    for child in children:
        if not child:
            continue
        pid = int(child)
        pids.append(pid)
        pids.extend(_descendant_pids(pid))
    return pids


def _sigkill_pid(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return


class LocalBackend:
    """Runs the real worker as a local process instead of a Batch job."""

    def __init__(
        self,
        workspace: Path,
        catalog_path: Path,
        python: str = sys.executable,
        startup_seconds: float = 120.0,
        stall_factor: int = 10,
    ) -> None:
        self._workspace = Path(workspace)
        self._catalog = Path(catalog_path)
        self._python = python
        self._startup = startup_seconds
        self._stall_factor = stall_factor
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._names: dict[str, str] = {}
        self._logs: dict[str, Path] = {}

    def submit(self, launch: Launch, manifest_uri: str, name: str) -> str:
        del launch
        return self.submit_raw(manifest_uri, name)

    def submit_raw(self, manifest_uri: str, name: str) -> str:
        directory = self._workspace / name
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "worker.log"
        environment = dict(os.environ)
        environment.update(
            {
                "TRAINER_MANIFEST": manifest_uri,
                "TRAINER_WORKSPACE": str(directory),
                "TRAINER_CATALOG": str(self._catalog),
                "TRAINER_STARTUP_SECONDS": str(self._startup),
                "TRAINER_STALL_FACTOR": str(self._stall_factor),
            }
        )
        with log_path.open("wb") as handle:
            process = subprocess.Popen(
                [self._python, "-m", "training_sdk.worker"],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        job_id = f"local-{name}-{process.pid}"
        self._processes[job_id] = process
        self._names[job_id] = name
        self._logs[job_id] = log_path
        return job_id

    def _result(self, job_id: str) -> JobResult:
        code = self._processes[job_id].returncode
        return JobResult(
            job_id=job_id,
            name=self._names[job_id],
            succeeded=code == 0,
            log_stream=str(self._logs[job_id]),
            reason=None if code == 0 else f"exit code {code}",
        )

    def wait(self, job_ids: Sequence[str]) -> list[JobResult]:
        while True:
            done = [
                job_id for job_id in job_ids if self._processes[job_id].poll() is not None
            ]
            results = [self._result(job_id) for job_id in done]
            if len(done) == len(job_ids) or any(
                not result.succeeded for result in results
            ):
                return results
            time.sleep(0.2)

    def terminate(self, job_ids: Sequence[str]) -> None:
        """Best-effort kill: descendants are read once, so a fork after that scan may survive."""
        for job_id in job_ids:
            process = self._processes.get(job_id)
            if process is None or process.poll() is not None:
                continue
            for pid in [*reversed(_descendant_pids(process.pid)), process.pid]:
                _sigkill_pid(pid)
        for job_id in job_ids:
            process = self._processes.get(job_id)
            if process is not None and process.poll() is None:
                process.wait(timeout=30)

    def log_tail(self, result: JobResult, lines: int) -> str:
        if result.log_stream is None:
            return ""
        text = Path(result.log_stream).read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])
