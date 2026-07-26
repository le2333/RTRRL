from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from training_sdk import objects


@dataclass(frozen=True)
class TrialRecord:
    trial: int
    params: dict[str, object]
    value: float | None = None
    job_id: str | None = None
    log_stream: str | None = None


@dataclass(frozen=True)
class Report:
    launch_id: str
    status: str
    trials: list[TrialRecord] = field(default_factory=list)
    best: TrialRecord | None = None
    elapsed_seconds: float = 0.0
    failure: str | None = None

    def payload(self) -> dict[str, object]:
        return {
            "launch_id": self.launch_id,
            "status": self.status,
            "trials": [asdict(record) for record in self.trials],
            "best": asdict(self.best) if self.best else None,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "failure": self.failure,
        }

    def write(self, archive: Path, prefix: str) -> None:
        payload = json.dumps(self.payload(), sort_keys=True, indent=2).encode()
        (Path(archive) / "report.json").write_bytes(payload)
        objects.put_bytes(f"{prefix}/report.json", payload)
