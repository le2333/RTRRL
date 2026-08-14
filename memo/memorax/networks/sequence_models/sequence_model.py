from abc import ABC, abstractmethod
from flax import linen as nn

from memorax.utils.typing import Array, Carry, Key


class SequenceModel(ABC, nn.Module):
    @abstractmethod
    def __call__(
        self,
        inputs: Array,
        done: Array,
        initial_carry: Carry | None = None,
        **kwargs,
    ) -> tuple: ...

    @abstractmethod
    def initialize_carry(self, key: Key, input_shape: tuple) -> Carry: ...
