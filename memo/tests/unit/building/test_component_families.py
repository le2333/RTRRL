from dataclasses import dataclass

from memorax.building import BuildContext, ComponentBuilder, ComponentFamily
from memorax.parameters import param


@dataclass(frozen=True)
class Left:
    width: int = param(valid=(1, 8), search=[2])


@dataclass(frozen=True)
class Right:
    width: int = param(valid=(1, 8), search=[4])


def test_component_family_owns_branch_routing_and_leaf_construction():
    built = []

    def construct(selection, builder, *, suffix):
        built.append((selection.kind, selection.path, builder.context.action_space))
        return f"{selection.kind}:{selection.parameters.width}:{suffix}"

    family = ComponentFamily(
        branches={"left": Left, "right": Right},
        construct=construct,
    )
    context = BuildContext(
        environment=object(),
        environment_parameters=None,
        observation_space="observation-space",
        action_space="action-space",
        num_envs=1,
        episode_length=8,
    )
    components = ComponentBuilder(
        {
            "node.kind": "right",
            "node.right.width": 7,
        },
        context,
    )

    result = components.build(family, "node", suffix="built")

    assert result == "right:7:built"
    assert built == [("right", "node.right", "action-space")]


def test_repeated_build_requests_create_repeated_component_instances():
    calls = []

    def construct(selection, builder):
        del builder
        instance = object()
        calls.append((selection.kind, instance))
        return instance

    family = ComponentFamily(branches={"left": Left}, construct=construct)
    components = ComponentBuilder(
        {"node.kind": "left", "node.left.width": 2},
        BuildContext(object(), None, 3, 2, 1, 8),
    )

    first = components.build(family, "node")
    second = components.build(family, "node")

    assert first is not second
    assert [kind for kind, _ in calls] == ["left", "left"]
