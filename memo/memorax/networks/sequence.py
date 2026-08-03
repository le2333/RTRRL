"""A network as an ordered list of components.

The step is ``(carries, x) -> (carries, y)`` and nothing else crosses it. What a
component needs beyond ``x`` it declares in ``reads``, and only the declaring
component is handed it; the three fixed slots this replaces were given the
observation, the ending, the previous action and the previous reward alike, so
every slot had to accept four things to use one.

The carry is one entry per component. A stateless component hands back the entry
it was given and a recurrent one hands back a new one, which is what turns "a
stateless component ignores the carry" into something a caller can check.

Exactly one component may be recurrent. Carrying an exact sensitivity through
two of them needs a dense cross-layer Jacobian, so a pair is refused here rather
than credited as though the second were not there.
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
        """The component whose Jacobian a credit carries a sensitivity through."""

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
    def walk(self, params, x: Array, *, done, carries, sensitivity, credit):
        """The same order, with the recurrence driven through its credit.

        The credit reaches one component, not the sequence: it carries a
        sensitivity through one Jacobian, and a list of components has none.
        """

        tree = params["params"] if "params" in params else params
        carries = self._entries(carries)
        walked = []
        for index, (name, component) in enumerate(zip(self.names, self.components)):
            if index == self.recurrent:
                carry, x, sensitivity = credit(
                    tree[name], x, done, carries[index], sensitivity
                )
            else:
                carry, x = component.apply(
                    {"params": tree.get(name, {})},
                    x,
                    initial_carry=carries[index],
                    **self._declared(component, done),
                )
            walked.append(carry)
        return (walked, sensitivity), x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple) -> list:
        return [
            component.initialize_carry(key, input_shape)
            for component in self.components
        ]

    @nn.nowrap
    def split(self, tree) -> dict:
        """A parameter-shaped tree in the three places credit treats apart.

        With exact recurrent credit the recurrence's own parameters are credited
        for every step they helped produce while everything around them is
        credited for one, so a single reading over the whole tree averages the
        distinction away. Splitting by position rather than by component name
        keeps the reading the same shape whatever the sequence holds.
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
