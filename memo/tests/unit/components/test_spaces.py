"""Two questions an action space is asked, and where they stop being one.

A ``Box`` answers "how wide is a head's output" and "how wide is the previous
action once it is carried beside the observation" with the same ``shape[0]``,
which is why only ``shape[0]`` was ever written down. A ``Discrete`` answers
both with ``n`` too -- but the second one only after the integer it hands out
has been widened, and the widening is the part that was missing.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments import spaces

from memorax.rl.spaces import action_classes, action_dim, encode_feedback


def test_a_box_names_no_actions_and_is_as_wide_as_its_shape():
    box = spaces.Box(-1.0, 1.0, (3,), dtype=jnp.float32)

    assert action_classes(box) is None
    assert action_dim(box) == 3


def test_a_discrete_space_names_its_actions_and_is_as_wide_as_it_names():
    discrete = spaces.Discrete(4)

    assert discrete.shape == ()
    assert action_classes(discrete) == 4
    assert action_dim(discrete) == 4


def test_a_space_that_is_neither_says_so_instead_of_indexing_an_empty_tuple():
    with pytest.raises(ValueError, match="neither discrete nor a vector Box"):
        action_dim(spaces.Box(-1.0, 1.0, (2, 3), dtype=jnp.float32))


def test_a_continuous_action_is_fed_back_exactly_as_it_arrived():
    """Untouched, so the Gaussian paths read what they always read."""

    action = jnp.array([[0.25, -0.5]], dtype=jnp.float32)

    assert encode_feedback(action, classes=None, dtype=jnp.float32) is action


def test_a_discrete_action_is_fed_back_as_its_one_hot():
    action = jnp.array([0, 2, 1], dtype=jnp.int32)

    encoded = encode_feedback(action, classes=3, dtype=jnp.float32)

    assert encoded.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(encoded),
        np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
    )
