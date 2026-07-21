"""Unit contracts for Task 12 machine-enforced evidence gates."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .task12_brax_smoke import assert_module_provenance
from . import task12_collect_evidence
from .task12_collect_evidence import (
    _diagnostic_summary,
    _junit_artifact,
    evaluate_gates,
)
from .task12_verify_archives import verify_archives


def _pyright_log(*diagnostics: dict[str, object]) -> str:
    return json.dumps(
        {
            "version": "test",
            "time": "0",
            "generalDiagnostics": list(diagnostics),
            "summary": {
                "filesAnalyzed": 1,
                "errorCount": len(diagnostics),
                "warningCount": 0,
                "informationCount": 0,
                "timeInSec": 0.1,
            },
        }
    )


def _diagnostic(path: str, message: str) -> dict[str, object]:
    return {
        "file": path,
        "severity": "error",
        "message": message,
        "rule": "reportExample",
        "range": {
            "start": {"line": 1, "character": 2},
            "end": {"line": 1, "character": 3},
        },
    }


def _runs() -> dict[str, dict[str, object]]:
    known_failure = "tests/online_ac/test_same.py::test_known"
    return {
        "selected_online_ac": {
            "exit_code": 0,
            "junit": {
                "valid": True,
                "counts": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
                "failure_nodeids": [],
                "error_nodeids": [],
                "error": None,
            },
        },
        "online_ac_head": {
            "exit_code": 1,
            "junit": {
                "valid": True,
                "counts": {"tests": 1, "failures": 1, "errors": 0, "skipped": 0},
                "failure_nodeids": [known_failure],
                "error_nodeids": [],
                "error": None,
            },
        },
        "online_ac_base": {
            "exit_code": 1,
            "junit": {
                "valid": True,
                "counts": {"tests": 1, "failures": 1, "errors": 0, "skipped": 0},
                "failure_nodeids": [known_failure],
                "error_nodeids": [],
                "error": None,
            },
        },
    }


def test_gates_reject_selected_online_ac_failure():
    runs = _runs()
    runs["selected_online_ac"]["exit_code"] = 1

    gates = evaluate_gates(runs, _pyright_log(), _pyright_log())

    assert gates["selected_online_ac"]["passed"] is False
    assert gates["all_passed"] is False


def test_gates_reject_head_only_online_ac_failure():
    runs = _runs()
    runs["online_ac_head"]["junit"]["failure_nodeids"].append(
        "tests/online_ac/test_new.py::test_regression"
    )
    runs["online_ac_head"]["junit"]["counts"]["tests"] = 2
    runs["online_ac_head"]["junit"]["counts"]["failures"] = 2

    gates = evaluate_gates(runs, _pyright_log(), _pyright_log())

    regression = gates["online_ac_regression"]
    assert regression["passed"] is False
    assert regression["head_only_failures"] == [
        "tests/online_ac/test_new.py::test_regression"
    ]
    assert gates["all_passed"] is False


def test_online_ac_gate_rejects_exit_two_without_failures():
    runs = _runs()
    runs["online_ac_head"]["exit_code"] = 2
    runs["online_ac_head"]["junit"]["counts"]["failures"] = 0
    runs["online_ac_head"]["junit"]["failure_nodeids"] = []

    gate = evaluate_gates(runs, _pyright_log(), _pyright_log())[
        "online_ac_regression"
    ]

    assert gate["passed"] is False
    assert gate["head_exit_accepted"] is False


@pytest.mark.parametrize("artifact_error", ["missing", "malformed XML"])
def test_online_ac_gate_rejects_missing_or_malformed_junit(artifact_error):
    runs = _runs()
    runs["online_ac_head"]["junit"] = {
        "valid": False,
        "counts": None,
        "failure_nodeids": [],
        "error_nodeids": [],
        "error": artifact_error,
    }

    gate = evaluate_gates(runs, _pyright_log(), _pyright_log())[
        "online_ac_regression"
    ]

    assert gate["passed"] is False
    assert gate["head_junit_valid"] is False


def test_online_ac_gate_rejects_collection_error():
    runs = _runs()
    runs["online_ac_head"]["exit_code"] = 2
    runs["online_ac_head"]["junit"] = {
        "valid": True,
        "counts": {"tests": 1, "failures": 0, "errors": 1, "skipped": 0},
        "failure_nodeids": [],
        "error_nodeids": ["tests/online_ac/test_bad.py::collection"],
        "error": None,
    }

    gate = evaluate_gates(runs, _pyright_log(), _pyright_log())[
        "online_ac_regression"
    ]

    assert gate["passed"] is False
    assert gate["head_exit_accepted"] is False
    assert gate["head_only_errors"] == [
        "tests/online_ac/test_bad.py::collection"
    ]


def test_online_ac_gate_rejects_failure_error_classification_mismatch():
    runs = _runs()
    nodeid = "tests/online_ac/test_same.py::test_known"
    runs["online_ac_head"]["junit"] = {
        "valid": True,
        "counts": {"tests": 1, "failures": 0, "errors": 1, "skipped": 0},
        "failure_nodeids": [],
        "error_nodeids": [nodeid],
        "error": None,
    }

    gate = evaluate_gates(runs, _pyright_log(), _pyright_log())[
        "online_ac_regression"
    ]

    assert gate["passed"] is False
    assert gate["head_only_errors"] == [nodeid]


def test_junit_parser_rejects_missing_and_malformed_files(tmp_path):
    missing = _junit_artifact(tmp_path / "missing.xml")
    malformed_path = tmp_path / "malformed.xml"
    malformed_path.write_text("<testsuite>")
    malformed = _junit_artifact(malformed_path)

    assert missing["valid"] is False
    assert missing["error"] == "missing"
    assert malformed["valid"] is False
    assert "malformed" in malformed["error"]


def test_junit_parser_extracts_failure_and_error_nodeids(tmp_path):
    junit = tmp_path / "results.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites tests="2" failures="1" errors="1" skipped="0">
  <testsuite name="pytest" tests="2" failures="1" errors="1" skipped="0">
    <testcase classname="tests.online_ac.test_example" name="test_failure">
      <failure message="failed" />
    </testcase>
    <testcase classname="tests.online_ac.test_example" name="test_error">
      <error message="collection error" />
    </testcase>
  </testsuite>
</testsuites>
"""
    )

    artifact = _junit_artifact(junit)

    assert artifact["valid"] is True
    assert artifact["failure_nodeids"] == [
        "tests/online_ac/test_example.py::test_failure"
    ]
    assert artifact["error_nodeids"] == [
        "tests/online_ac/test_example.py::test_error"
    ]


def test_pyright_gate_canonicalizes_moved_rtrrl_package_path():
    message = 'Type of "legacy" is unknown'
    head = _pyright_log(
        _diagnostic(
            "/tmp/head/memo/memorax/algorithms/rtrrl/__init__.py",
            message,
        )
    )
    base = _pyright_log(
        _diagnostic(
            "/tmp/base/memo/memorax/algorithms/rtrrl.py",
            message,
        )
    )

    gates = evaluate_gates(_runs(), head, base)

    assert gates["pyright_review_regression"]["passed"] is True
    assert gates["pyright_review_regression"]["head_only_diagnostics"] == []
    assert gates["all_passed"] is True


def test_pyright_gate_canonicalizes_every_rtrrl_package_module():
    message = "Shared monolith diagnostic"
    head = _pyright_log(
        _diagnostic(
            "/tmp/head/memo/memorax/algorithms/rtrrl/state_machine.py",
            message,
        )
    )
    base = _pyright_log(
        _diagnostic(
            "/tmp/base/memo/memorax/algorithms/rtrrl.py",
            message,
        )
    )

    gate = evaluate_gates(_runs(), head, base)["pyright_review_regression"]

    assert gate["passed"] is True
    assert gate["head_only_diagnostics"] == []


def test_pyright_gate_rejects_branch_introduced_diagnostic():
    known = _diagnostic(
        "/tmp/base/memo/experiments/base/experiment.py",
        "Known baseline issue",
    )
    introduced = _diagnostic(
        "/tmp/head/memo/experiments/base/experiment.py",
        "Branch regression",
    )

    gates = evaluate_gates(
        _runs(),
        _pyright_log(known, introduced),
        _pyright_log(known),
    )

    assert gates["pyright_review_regression"]["passed"] is False
    assert gates["pyright_review_regression"]["head_only_diagnostics"] == [
        (
            "experiments/base/experiment.py|error|reportExample|"
            "Branch regression"
        )
    ]
    assert gates["all_passed"] is False


def test_pyright_gate_preserves_diagnostic_multiplicity():
    duplicate = _diagnostic(
        "/tmp/head/memo/memorax/algorithms/rtrrl/rules.py",
        "Repeated diagnostic",
    )
    base_duplicate = {
        **duplicate,
        "file": "/tmp/base/memo/memorax/algorithms/rtrrl.py",
    }

    gate = evaluate_gates(
        _runs(),
        _pyright_log(duplicate, duplicate),
        _pyright_log(base_duplicate),
    )["pyright_review_regression"]

    assert gate["passed"] is False
    assert gate["head_only_diagnostics"] == [
        "memorax/algorithms/rtrrl|error|reportExample|Repeated diagnostic"
    ]


def test_pyright_parser_ignores_node_bootstrap_noise_before_json():
    noisy = "{'x86': False}\nnode bootstrap warning\n" + _pyright_log(
        _diagnostic("/tmp/head/memo/example.py", "Example")
    )

    assert _diagnostic_summary(noisy) == {
        "errors": 1,
        "warnings": 0,
        "informations": 0,
    }
    assert evaluate_gates(_runs(), noisy, noisy)["all_passed"] is True


def test_smoke_provenance_rejects_external_rtrrl_module():
    memo_root = Path("/tmp/head/memo")
    modules = {
        "memorax": SimpleNamespace(__file__="/tmp/head/memo/memorax/__init__.py"),
        "memorax.algorithms.rtrrl.entrypoint": SimpleNamespace(
            __file__="/tmp/head/memo/memorax/algorithms/rtrrl/entrypoint.py"
        ),
        "experiments.rtrrl_hopper.run": SimpleNamespace(
            __file__="/tmp/head/memo/experiments/rtrrl_hopper/run.py"
        ),
        "logging_util": SimpleNamespace(
            __file__="/tmp/head/rtrrl/logging_util.py"
        ),
    }

    with pytest.raises(AssertionError, match="external rtrrl"):
        assert_module_provenance(modules, memo_root)


def test_collector_exits_nonzero_after_emitting_failed_gate_evidence(
    monkeypatch, capsys
):
    payload = {"schema_version": 2, "gates": {"all_passed": False}}
    monkeypatch.setattr(
        task12_collect_evidence,
        "collect_evidence",
        lambda: payload,
    )

    with pytest.raises(SystemExit) as raised:
        task12_collect_evidence.main()

    assert raised.value.code == 1
    assert json.loads(capsys.readouterr().out) == payload


def test_archive_verifier_records_expected_actual_and_mismatch(tmp_path):
    archive = tmp_path / "archive.tar"
    archive.write_bytes(b"verified archive")
    expected = "0" * 64

    result = verify_archives({"head": (archive, expected)})

    assert result == {
        "all_verified": False,
        "archives": {
            "head": {
                "path": str(archive),
                "expected_sha256": expected,
                "actual_sha256": sha256(b"verified archive").hexdigest(),
                "verified": False,
            }
        },
    }


def test_archive_verification_is_part_of_overall_gate():
    verification = {
        "all_verified": False,
        "archives": {
            "head": {
                "expected_sha256": "0" * 64,
                "actual_sha256": "1" * 64,
                "verified": False,
            }
        },
    }

    gates = evaluate_gates(
        _runs(),
        _pyright_log(),
        _pyright_log(),
        verification,
    )

    assert gates["archive_verification"]["passed"] is False
    assert gates["all_passed"] is False
