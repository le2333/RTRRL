"""Static, reproducible source audit for preserved RTRRL and AAAI25."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _is_string_expression(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


class _RemoveDocstrings(ast.NodeTransformer):
    def _visit_body(self, node: Any) -> Any:
        self.generic_visit(node)
        if node.body and _is_string_expression(node.body[0]):
            node.body = node.body[1:]
        return node

    visit_Module = _visit_body
    visit_ClassDef = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body


class _RemoveStandaloneStrings(ast.NodeTransformer):
    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        if hasattr(node, "body") and isinstance(node.body, list):
            node.body = [
                statement
                for statement in node.body
                if not _is_string_expression(statement)
            ]
        return node


def _normalized_dump(path: Path, transformer: ast.NodeTransformer) -> str:
    tree = transformer.visit(ast.parse(path.read_bytes()))
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _log_prob_argument(path: Path) -> tuple[str, bool]:
    tree = ast.parse(path.read_bytes())
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "log_prob" or len(node.args) != 1:
            continue
        candidates.append(node.args[0])
    if len(candidates) != 1:
        raise ValueError(f"expected one log_prob call in {path}, found {len(candidates)}")
    argument = candidates[0]
    detached = (
        isinstance(argument, ast.Call)
        and isinstance(argument.func, ast.Attribute)
        and argument.func.attr == "stop_gradient"
        and len(argument.args) == 1
        and isinstance(argument.args[0], ast.Name)
        and argument.args[0].id == "action"
    )
    if detached:
        return "stop_gradient(action)", True
    if isinstance(argument, ast.Name) and argument.id == "action":
        return "action", False
    return ast.unparse(argument), False


def audit_sources(preserved_root: Path, oracle_root: Path) -> dict[str, Any]:
    normalization = {}
    for relative in ("traces.py", "models/online_lru.py"):
        preserved = preserved_root / relative
        oracle = oracle_root / relative
        normalization[relative] = {
            "equal_after_docstring_removal": (
                _normalized_dump(preserved, _RemoveDocstrings())
                == _normalized_dump(oracle, _RemoveDocstrings())
            ),
            "equal_after_all_standalone_string_removal": (
                _normalized_dump(preserved, _RemoveStandaloneStrings())
                == _normalized_dump(oracle, _RemoveStandaloneStrings())
            ),
        }

    preserved_argument, preserved_detaches = _log_prob_argument(
        preserved_root / "rtrrl.py"
    )
    oracle_argument, oracle_detaches = _log_prob_argument(oracle_root / "rtrrl.py")
    return {
        "schema_version": 1,
        "ast_normalization": normalization,
        "actor_log_prob_structure": {
            "preserved_argument": preserved_argument,
            "oracle_argument": oracle_argument,
            "preserved_detaches_action": preserved_detaches,
            "oracle_detaches_action": oracle_detaches,
        },
        "interpretation": (
            "structural AST evidence only; controlled objectives separately "
            "measure the numerical effect"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preserved-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit_sources(arguments.preserved_root, arguments.oracle_root),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
