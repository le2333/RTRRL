"""Collect deterministic Task 12 Batch artifacts into one JSON document."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


RESULTS = Path(os.environ.get("TASK12_RESULTS_DIR", "/tmp/task12-results"))


def _text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _pytest_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    names = ("tests", "failures", "errors", "skipped")
    return {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in names
    }


def _time(path: Path) -> dict[str, float | int | str]:
    text = _text(path)
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    elapsed = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)", text
    )
    user = re.search(r"User time \(seconds\):\s*([0-9.]+)", text)
    system = re.search(r"System time \(seconds\):\s*([0-9.]+)", text)
    return {
        "elapsed": elapsed.group(1) if elapsed else "",
        "user_seconds": float(user.group(1)) if user else 0.0,
        "system_seconds": float(system.group(1)) if system else 0.0,
        "peak_rss_kib": int(rss.group(1)) if rss else 0,
    }


def _failures(log: str) -> list[str]:
    return sorted(set(re.findall(r"FAILED (tests/\S+)", log)))


def _diagnostic_summary(log: str) -> dict[str, int] | None:
    matches = re.findall(
        r"(\d+) errors?, (\d+) warnings?, (\d+) informations?", log
    )
    if not matches:
        return None
    errors, warnings, informations = matches[-1]
    return {
        "errors": int(errors),
        "warnings": int(warnings),
        "informations": int(informations),
    }


def _last_json(path: Path) -> object:
    for line in reversed(_text(path).splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"no JSON object found in {path}")


def _nodeids(path: Path) -> list[str]:
    return sorted(json.loads(_text(path)))


def _run(name: str) -> dict[str, object]:
    log = _text(RESULTS / f"{name}.log")
    junit = RESULTS / f"{name}.xml"
    return {
        "command": _text(RESULTS / f"{name}.command").strip(),
        "working_directory": _text(RESULTS / f"{name}.cwd").strip(),
        "environment": json.loads(
            _text(RESULTS / f"{name}.env") or "{}"
        ),
        "exit_code": int(_text(RESULTS / f"{name}.exit") or "-1"),
        "time": _time(RESULTS / f"{name}.time"),
        "counts": _pytest_counts(junit) if junit.exists() else None,
        "failures": _failures(log),
        "diagnostic_summary": _diagnostic_summary(log),
    }


def main() -> None:
    names = [
        "finite_differences",
        "strict_parity",
        "selected_online_ac",
        "independent",
        "online_ac_head",
        "online_ac_base",
        "numerical_harness",
        "preserved_probe",
        "oracle_probe",
        "preserved_compare",
        "source_audit",
        "brax_smoke",
        "ruff",
        "pyright_head",
        "pyright_base",
        "pyright_review_head",
        "pyright_review_base",
        "compileall",
    ]
    head_nodes = _nodeids(RESULTS / "online_ac_head.nodes")
    base_nodes = _nodeids(RESULTS / "online_ac_base.nodes")
    finite_log = _text(RESULTS / "finite_differences.log")
    finite_metrics = {
        group: {
            "cosine": float(cosine),
            "relative_error": float(relative),
        }
        for group, cosine, relative in re.findall(
            r"FD_METRIC group=(\S+) cosine=(\S+) relative_error=(\S+)",
            finite_log,
        )
    }
    payload = {
        "schema_version": 1,
        "batch_job_id": os.environ.get("AWS_BATCH_JOB_ID", ""),
        "functional_head_sha": os.environ["TASK12_FUNCTIONAL_HEAD_SHA"],
        "feature_base_sha": os.environ["TASK12_FEATURE_BASE_SHA"],
        "task10_comparison_base_sha": os.environ["TASK12_TASK10_BASE_SHA"],
        "report_parent_sha": os.environ["TASK12_REPORT_PARENT_SHA"],
        "review_fix_patch_sha256": os.environ[
            "TASK12_REVIEW_PATCH_SHA256"
        ],
        "runtime": json.loads(_text(RESULTS / "runtime.json")),
        "runs": {name: _run(name) for name in names},
        "finite_difference_metrics": finite_metrics,
        "online_ac_collection": {
            "head_nodeids": head_nodes,
            "base_nodeids": base_nodes,
            "added_at_head": sorted(set(head_nodes) - set(base_nodes)),
            "removed_at_head": sorted(set(base_nodes) - set(head_nodes)),
        },
        "numerical_measurements": _last_json(
            RESULTS / "numerical_harness.stdout.json"
        ),
        "preserved_original_comparison": _last_json(
            RESULTS / "preserved_compare.stdout.json"
        ),
        "preserved_original_source_audit": _last_json(
            RESULTS / "source_audit.stdout.json"
        ),
        "brax_smoke": _last_json(RESULTS / "brax_smoke.stdout.json"),
        "source_hashes": json.loads(_text(RESULTS / "source_hashes.json")),
    }
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
