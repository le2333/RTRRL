from typing import Self

import jax
from flax import struct

from memorax.utils.axes import add_feature_axis, add_time_axis, remove_time_axis
from memorax.utils.typing import Array


class Timestep(struct.PyTreeNode):
    """One transition's four streams, all of which are always present.

    Nothing constructs a partial timestep: an algorithm that holds one holds the
    observation, the action that produced it, the reward it earned and whether
    it ended. The fields are required so that reading one does not have to
    re-establish that at every use site.
    """

    obs: Array
    action: Array
    reward: Array
    done: Array

    def __iter__(self):
        yield self.obs
        yield self.done
        yield self.action
        yield add_feature_axis(self.reward)

    def to_sequence(self) -> Self:
        return jax.tree.map(add_time_axis, self)

    def from_sequence(self) -> Self:
        return jax.tree.map(remove_time_axis, self)
