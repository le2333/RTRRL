"""Reading a declared parameter tree in the terms a test asks about it.

A node is a leaf or a group, and a domain is a fixed set or an interval. A test
that walks to one already knows which of the two it walked to; saying so here
once keeps every assertion about the tree from having to establish it again.
"""

from __future__ import annotations

from memorax.parameters import (
    KIND,
    Choice,
    Parameter,
    ParameterNode,
    ParameterTree,
    Scalar,
)


def branch(tree: ParameterTree, path: str) -> ParameterTree:
    """The group a dotted path names."""

    node: ParameterNode = tree
    for name in path.split("."):
        assert not isinstance(node, Parameter), f"{path}: {name} is under a leaf"
        assert name in node, f"{path}: nothing declares {name}"
        node = node[name]
    assert not isinstance(node, Parameter), f"{path} is a leaf, not a group"
    return node


def kinds(tree: ParameterTree, path: str) -> tuple[Scalar, ...]:
    """The branches the group at a dotted path chooses between."""

    kind = branch(tree, path)[KIND]
    assert isinstance(kind, Parameter), f"{path}.{KIND} is a group, not a parameter"
    assert isinstance(kind.valid, Choice), f"{path}.{KIND} is not a choice"
    return kind.valid.values
