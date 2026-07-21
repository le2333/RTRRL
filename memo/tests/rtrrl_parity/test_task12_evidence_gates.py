"""Unit contracts for Task 12 machine-enforced evidence gates."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .task12_brax_smoke import assert_module_provenance
from . import task12_collect_evidence
from .task12_collect_evidence import _diagnostic_summary, evaluate_gates


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
    return {
        "selected_online_ac": {"exit_code": 0, "failures": []},
        "online_ac_head": {
            "exit_code": 1,
            "failures": ["tests/online_ac/test_same.py::test_known"],
        },
        "online_ac_base": {
            "exit_code": 1,
            "failures": ["tests/online_ac/test_same.py::test_known"],
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
    runs["online_ac_head"]["failures"].append(
        "tests/online_ac/test_new.py::test_regression"
    )

    gates = evaluate_gates(runs, _pyright_log(), _pyright_log())

    assert gates["online_ac_regression"] == {
        "passed": False,
        "condition": "no_head_only_failure_nodeids",
        "head_failures": [
            "tests/online_ac/test_new.py::test_regression",
            "tests/online_ac/test_same.py::test_known",
        ],
        "base_failures": ["tests/online_ac/test_same.py::test_known"],
        "head_only_failures": [
            "tests/online_ac/test_new.py::test_regression"
        ],
        "base_only_failures": [],
    }
    assert gates["all_passed"] is False


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
