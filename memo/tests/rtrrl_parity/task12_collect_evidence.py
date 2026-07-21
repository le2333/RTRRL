"""Collect deterministic Task 12 Batch artifacts into one JSON document."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


RESULTS = Path(os.environ.get("TASK12_RESULTS_DIR", "/tmp/task12-results"))


def _text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _testcase_nodeid(case: ET.Element) -> str:
    classname = case.attrib.get("classname", "")
    name = case.attrib.get("name", "<unknown>")
    parts = classname.split(".") if classname else []
    module_index = next(
        (index for index, part in enumerate(parts) if part.startswith("test_")),
        None,
    )
    if module_index is None:
        return f"{classname or '<unknown>'}::{name}"
    path = "/".join(parts[: module_index + 1]) + ".py"
    suffix = parts[module_index + 1 :] + [name]
    return "::".join((path, *suffix))


def _junit_artifact(path: Path) -> dict[str, object]:
    empty = {
        "counts": None,
        "failure_nodeids": [],
        "error_nodeids": [],
    }
    if not path.exists():
        return {"valid": False, **empty, "error": "missing"}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        return {
            "valid": False,
            **empty,
            "error": f"malformed XML: {error}",
        }
    if root.tag not in {"testsuite", "testsuites"}:
        return {
            "valid": False,
            **empty,
            "error": f"unexpected root element: {root.tag}",
        }
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    names = ("tests", "failures", "errors", "skipped")
    counts = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in names
    }
    failure_nodeids = []
    error_nodeids = []
    for case in root.iter("testcase"):
        nodeid = _testcase_nodeid(case)
        failure_nodeids.extend(nodeid for _ in case.findall("failure"))
        error_nodeids.extend(nodeid for _ in case.findall("error"))
    if (
        counts["failures"] != len(failure_nodeids)
        or counts["errors"] != len(error_nodeids)
    ):
        return {
            "valid": False,
            "counts": counts,
            "failure_nodeids": sorted(failure_nodeids),
            "error_nodeids": sorted(error_nodeids),
            "error": "JUnit counts disagree with testcase outcomes",
        }
    return {
        "valid": True,
        "counts": counts,
        "failure_nodeids": sorted(failure_nodeids),
        "error_nodeids": sorted(error_nodeids),
        "error": None,
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
    junit_artifact = _junit_artifact(junit)
    return {
        "command": _text(RESULTS / f"{name}.command").strip(),
        "working_directory": _text(RESULTS / f"{name}.cwd").strip(),
        "environment": json.loads(
            _text(RESULTS / f"{name}.env") or "{}"
        ),
        "exit_code": int(_text(RESULTS / f"{name}.exit") or "-1"),
        "time": _time(RESULTS / f"{name}.time"),
        "counts": junit_artifact["counts"],
        "junit": junit_artifact,
        "failures": _failures(log),
        "diagnostic_summary": _diagnostic_summary(log),
    }


def _canonical_diagnostic(diagnostic: dict[str, object]) -> str:
    path = str(diagnostic.get("file", "")).replace("\\", "/")
    if "/memo/" in path:
        path = path.split("/memo/", 1)[1]
    if path == "memorax/algorithms/rtrrl.py" or path.startswith(
        "memorax/algorithms/rtrrl/"
    ):
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
        _canonical_diagnostic(diagnostic)
        for diagnostic in diagnostics
        if isinstance(diagnostic, dict)
    )


def _pytest_exit_accepted(run: dict[str, object]) -> bool:
    artifact = run["junit"]
    if not artifact["valid"]:
        return False
    exit_code = run["exit_code"]
    outcomes = artifact["failure_nodeids"] + artifact["error_nodeids"]
    if exit_code == 0:
        return not outcomes
    if exit_code == 1:
        return bool(outcomes)
    return False


def evaluate_gates(
    runs: dict[str, dict[str, object]],
    pyright_review_head_log: str,
    pyright_review_base_log: str,
    archive_verification: dict[str, object] | None = None,
) -> dict[str, object]:
    head_run = runs["online_ac_head"]
    base_run = runs["online_ac_base"]
    head_artifact = head_run["junit"]
    base_artifact = base_run["junit"]
    head_failures = Counter(head_artifact["failure_nodeids"])
    base_failures = Counter(base_artifact["failure_nodeids"])
    head_errors = Counter(head_artifact["error_nodeids"])
    base_errors = Counter(base_artifact["error_nodeids"])
    head_only_failures = sorted((head_failures - base_failures).elements())
    base_only_failures = sorted((base_failures - head_failures).elements())
    head_only_errors = sorted((head_errors - base_errors).elements())
    base_only_errors = sorted((base_errors - head_errors).elements())
    head_diagnostics = _pyright_diagnostics(pyright_review_head_log)
    base_diagnostics = _pyright_diagnostics(pyright_review_base_log)
    head_diagnostic_counts = Counter(head_diagnostics)
    base_diagnostic_counts = Counter(base_diagnostics)
    head_only_diagnostics = sorted(
        (head_diagnostic_counts - base_diagnostic_counts).elements()
    )
    base_only_diagnostics = sorted(
        (base_diagnostic_counts - head_diagnostic_counts).elements()
    )
    selected = runs["selected_online_ac"]
    archive_verification = archive_verification or {
        "all_verified": True,
        "archives": {},
    }
    gates: dict[str, object] = {
        "selected_online_ac": {
            "passed": _pytest_exit_accepted(selected)
            and selected["exit_code"] == 0,
            "exit_code": selected["exit_code"],
            "junit_valid": selected["junit"]["valid"],
        },
        "online_ac_regression": {
            "passed": (
                bool(head_artifact["valid"])
                and bool(base_artifact["valid"])
                and _pytest_exit_accepted(head_run)
                and _pytest_exit_accepted(base_run)
                and not head_only_failures
                and not head_only_errors
            ),
            "condition": (
                "valid_junit_and_exit_0_or_1_with_recorded_outcomes_and_"
                "no_head_only_failure_or_error_nodeids"
            ),
            "accepted_exit_behavior": {
                "0": "accepted only with zero JUnit failures/errors",
                "1": "accepted only with recorded JUnit failures/errors",
                "2+": "always rejected",
            },
            "head_exit_code": head_run["exit_code"],
            "base_exit_code": base_run["exit_code"],
            "head_exit_accepted": _pytest_exit_accepted(head_run),
            "base_exit_accepted": _pytest_exit_accepted(base_run),
            "head_junit_valid": head_artifact["valid"],
            "base_junit_valid": base_artifact["valid"],
            "head_junit_error": head_artifact["error"],
            "base_junit_error": base_artifact["error"],
            "head_failures": sorted(head_failures.elements()),
            "base_failures": sorted(base_failures.elements()),
            "head_only_failures": head_only_failures,
            "base_only_failures": base_only_failures,
            "head_errors": sorted(head_errors.elements()),
            "base_errors": sorted(base_errors.elements()),
            "head_only_errors": head_only_errors,
            "base_only_errors": base_only_errors,
        },
        "pyright_review_regression": {
            "passed": not head_only_diagnostics,
            "condition": "no_head_only_canonical_diagnostics",
            "head_diagnostics": head_diagnostics,
            "base_diagnostics": base_diagnostics,
            "head_only_diagnostics": head_only_diagnostics,
            "base_only_diagnostics": base_only_diagnostics,
        },
        "archive_verification": {
            "passed": bool(archive_verification["all_verified"]),
            "archives": archive_verification["archives"],
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
    archive_verification = json.loads(
        _text(RESULTS / "archive_hashes.json")
    )
    payload = {
        "schema_version": 3,
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
        "archive_verification": archive_verification,
        "gates": evaluate_gates(
            runs,
            _text(RESULTS / "pyright_review_head.log"),
            _text(RESULTS / "pyright_review_base.log"),
            archive_verification,
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
