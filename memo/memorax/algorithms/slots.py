"""The fixed slots the kernels that have not been migrated are written against.

Not a component surface, and not something anything new should be built from:
it is the shape ``upstream_stream_ac`` was lifted with and the one ``rtrrl``
still takes its four modules as. Every slot is handed the observation, the
ending, the previous action and the previous reward alike, so a slot that wants
one of them has to accept four -- which is what a sequence replaced. It lives
here, beside the kernels that speak it, rather than in the component package.
"""

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from memorax.networks.identity import Identity
from memorax.utils.typing import Array, Carry


class Network(nn.Module):
    feature_extractor: nn.Module = Identity()
    torso: nn.Module = Identity()
    head: nn.Module = Identity()

    @nn.compact
    def __call__(
        self,
        observation: Array,
        done: Array,
        action: Array,
        reward: Array,
        initial_carry: Array | None = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        x, embeddings = self.feature_extractor(
            observation, action=action, reward=reward, done=done
        )

        match self.torso(
            x,
            done=done,
            action=action,
            reward=reward,
            initial_carry=initial_carry,
            **embeddings,
            **kwargs,
        ):
            case (carry, x):
                pass
            case x:
                carry = None

        x = self.head(x, action=action, reward=reward, done=done, **kwargs)
        return carry, x

    @nn.nowrap
    def initialize_carry(self, input_shape: tuple) -> Carry:
        key = jax.random.key(0)
        return getattr(self.torso, "initialize_carry", lambda k, s: None)(
            key, input_shape
        )


class FeatureExtractor(nn.Module):
    observation_extractor: Callable
    action_extractor: Callable | None = None
    reward_extractor: Callable | None = None
    done_extractor: Callable | None = None

    def extract(
        self,
        embeddings: dict,
        key: str,
        extractor: Callable | None,
        x: Array | None = None,
    ) -> None:
        if extractor is not None and x is not None:
            embeddings[key] = extractor(x)

    @nn.compact
    def __call__(
        self,
        observation: Array,
        action: Array,
        reward: Array,
        done: Array,
        **kwargs,
    ) -> tuple[Array, dict]:
        embeddings = {"observation_embedding": self.observation_extractor(observation)}
        self.extract(embeddings, "action_embedding", self.action_extractor, action)
        self.extract(embeddings, "reward_embedding", self.reward_extractor, reward)
        self.extract(
            embeddings, "done_embedding", self.done_extractor, done.astype(jnp.int32)
        )
        x = jnp.concatenate([*embeddings.values()], axis=-1)

        return x, embeddings
