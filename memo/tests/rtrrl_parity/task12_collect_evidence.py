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


def _pyright_payload(log: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", log):
        try:
            payload, _ = decoder.raw_decode(log, match.start())
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, dict)
            and "generalDiagnostics" in payload
            and "summary" in payload
        ):
            return payload
    raise ValueError("no Pyright JSON payload found")


def _diagnostic_summary(log: str) -> dict[str, int] | None:
    try:
        payload = _pyright_payload(log)
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        summary = payload["summary"]
        return {
            "errors": int(summary.get("errorCount", 0)),
            "warnings": int(summary.get("warningCount", 0)),
            "informations": int(summary.get("informationCount", 0)),
        }
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


def _canonical_diagnostic(diagnostic: dict[str, object]) -> str:
    path = str(diagnostic.get("file", "")).replace("\\", "/")
    if "/memo/" in path:
        path = path.split("/memo/", 1)[1]
    if path in {
        "memorax/algorithms/rtrrl.py",
        "memorax/algorithms/rtrrl/__init__.py",
    }:
        path = "memorax/algorithms/rtrrl"
    message = re.sub(r"\s+", " ", str(diagnostic.get("message", ""))).strip()
    return "|".join(
        (
            path,
            str(diagnostic.get("severity", "")),
            str(diagnostic.get("rule", "")),
            message,
        )
    )


def _pyright_diagnostics(log: str) -> list[str]:
    payload = _pyright_payload(log)
    diagnostics = payload.get("generalDiagnostics")
    if not isinstance(diagnostics, list):
        raise ValueError("pyright JSON lacks generalDiagnostics")
    return sorted(
        {
            _canonical_diagnostic(diagnostic)
            for diagnostic in diagnostics
            if isinstance(diagnostic, dict)
        }
    )


def evaluate_gates(
    runs: dict[str, dict[str, object]],
    pyright_review_head_log: str,
    pyright_review_base_log: str,
) -> dict[str, object]:
    head_failures = sorted(set(runs["online_ac_head"]["failures"]))
    base_failures = sorted(set(runs["online_ac_base"]["failures"]))
    head_only_failures = sorted(set(head_failures) - set(base_failures))
    base_only_failures = sorted(set(base_failures) - set(head_failures))
    head_diagnostics = _pyright_diagnostics(pyright_review_head_log)
    base_diagnostics = _pyright_diagnostics(pyright_review_base_log)
    head_only_diagnostics = sorted(
        set(head_diagnostics) - set(base_diagnostics)
    )
    base_only_diagnostics = sorted(
        set(base_diagnostics) - set(head_diagnostics)
    )
    gates: dict[str, object] = {
        "selected_online_ac": {
            "passed": runs["selected_online_ac"]["exit_code"] == 0,
            "exit_code": runs["selected_online_ac"]["exit_code"],
        },
        "online_ac_regression": {
            "passed": not head_only_failures,
            "condition": "no_head_only_failure_nodeids",
            "head_failures": head_failures,
            "base_failures": base_failures,
            "head_only_failures": head_only_failures,
            "base_only_failures": base_only_failures,
        },
        "pyright_review_regression": {
            "passed": not head_only_diagnostics,
            "condition": "no_head_only_canonical_diagnostics",
            "head_diagnostics": head_diagnostics,
            "base_diagnostics": base_diagnostics,
            "head_only_diagnostics": head_only_diagnostics,
            "base_only_diagnostics": base_only_diagnostics,
        },
    }
    gates["all_passed"] = all(
        bool(gate["passed"])
        for gate in gates.values()
        if isinstance(gate, dict)
    )
    return gates


def collect_evidence() -> dict[str, object]:
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
    runs = {name: _run(name) for name in names}
    payload = {
        "schema_version": 2,
        "batch_job_id": os.environ.get("AWS_BATCH_JOB_ID", ""),
        "functional_head_sha": os.environ["TASK12_FUNCTIONAL_HEAD_SHA"],
        "feature_base_sha": os.environ["TASK12_FEATURE_BASE_SHA"],
        "task10_comparison_base_sha": os.environ["TASK12_TASK10_BASE_SHA"],
        "report_parent_sha": os.environ["TASK12_REPORT_PARENT_SHA"],
        "review_fix_patch_sha256": os.environ[
            "TASK12_REVIEW_PATCH_SHA256"
        ],
        "runtime": json.loads(_text(RESULTS / "runtime.json")),
        "runs": runs,
        "gates": evaluate_gates(
            runs,
            _text(RESULTS / "pyright_review_head.log"),
            _text(RESULTS / "pyright_review_base.log"),
        ),
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
    return payload


def main() -> None:
    payload = collect_evidence()
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    if not payload["gates"]["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
