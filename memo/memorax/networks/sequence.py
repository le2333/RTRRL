"""A network as an ordered list of components.

The step is ``(carries, x) -> (carries, y)``. What a component needs beyond
``x`` it declares in ``reads``, and only the declaring component is handed it.

The carry is one entry per component: a stateless component hands back the entry
it was given, a recurrent one hands back a new one.

At most one component may be recurrent. The differentiation boundary addresses
one recurrent component, so a second one is refused at construction.
"""

from __future__ import annotations

import flax.linen as nn

from memorax.utils.typing import Array, Key

BEFORE, RECURRENCE, AFTER = "before", "recurrence", "after"
PLACES: tuple[str, ...] = (BEFORE, RECURRENCE, AFTER)


class Sequence(nn.Module):
    components: tuple[nn.Module, ...]

    def __post_init__(self):
        super().__post_init__()
        found = [
            index
            for index, component in enumerate(self.components)
            if getattr(component, "recurrent", False)
        ]
        if len(found) > 1:
            raise ValueError(
                f"a sequence may hold one recurrent component, not {len(found)} "
                f"at positions {found}"
            )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"components_{index}" for index in range(len(self.components)))

    @property
    def recurrent(self) -> int | None:
        for index, component in enumerate(self.components):
            if getattr(component, "recurrent", False):
                return index
        return None

    @property
    def core(self) -> nn.Module | None:
        """The recurrent component, if there is one."""

        index = self.recurrent
        return None if index is None else self.components[index]

    @nn.compact
    def __call__(self, x: Array, done=None, initial_carry=None):
        carries = self._entries(initial_carry)
        walked = []
        for index, component in enumerate(self.components):
            carry, x = component(
                x, initial_carry=carries[index], **self._declared(component, done)
            )
            walked.append(carry)
        return walked, x

    @nn.nowrap
    def walk(
        self,
        params,
        x: Array,
        *,
        done,
        carries,
        differentiation_state,
        differentiation,
    ):
        """Walk the sequence with its selected recurrent differentiation."""

        tree = params["params"] if "params" in params else params
        carries = self._entries(carries)
        walked = []
        for index, (name, component) in enumerate(zip(self.names, self.components)):
            if index == self.recurrent:
                carry, x, differentiation_state = differentiation(
                    tree[name], x, done, carries[index], differentiation_state
                )
            else:
                # Applied against its own slice of the tree, a component is its
                # own root and has no name. The name is this sequence's to give
                # either way, so give it: a component that can say which
                # position it is on one traversal can say it on both.
                carry, x = component.clone(name=name).apply(
                    {"params": tree.get(name, {})},
                    x,
                    initial_carry=carries[index],
                    **self._declared(component, done),
                )
            walked.append(carry)
        return (walked, differentiation_state), x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple) -> list:
        return [
            component.initialize_carry(key, input_shape)
            for component in self.components
        ]

    @nn.nowrap
    def split(self, tree) -> dict:
        """A parameter-shaped tree grouped by position: before, at, after.

        Keyed by position rather than by component name, so the grouping has the
        same three keys whatever the sequence holds.
        """

        tree = tree["params"] if "params" in tree else tree
        boundary = len(self.components) if self.recurrent is None else self.recurrent
        places: dict = {place: {} for place in PLACES}
        for index, name in enumerate(self.names):
            if name not in tree:
                continue
            place = (
                BEFORE
                if index < boundary
                else RECURRENCE if index == boundary else AFTER
            )
            places[place][name] = tree[name]
        return places

    @nn.nowrap
    def _entries(self, carries) -> list:
        if carries is None:
            return [None] * len(self.components)
        return list(carries)

    @nn.nowrap
    def _declared(self, component, done) -> dict:
        available = {"done": done}
        return {name: available[name] for name in getattr(component, "reads", ())}
