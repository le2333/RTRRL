"""Oracle parity for historical strict-RTRRL initialization."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from memorax.algorithms.rtrrl.compatibility import RTRRLComponentConfig
from memorax.algorithms.rtrrl.heads import RTRRLTDHead
from memorax.algorithms.rtrrl.lru import AAAI25LRU
from memorax.algorithms.rtrrl.state_machine import make_init_fn
from memorax.algorithms.rtrrl.types import RTRRLComponents

from .assertions import flatten_with_paths
from .oracle_capture import load_oracle


@dataclass(frozen=True)
class _EnvironmentState:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


class _DeterministicEnvironment:
    action_size = 2

    def reset(self, key):
        del key
        return _EnvironmentState(
            obs=jnp.array([[0.25]], dtype=jnp.float32),
            reward=jnp.array([-0.5], dtype=jnp.float32),
            done=jnp.array([False]),
        )


def _fixture_tree(arrays, prefix):
    marker = prefix + "/"
    return {
        path[len(marker) :]: value
        for path, value in arrays.items()
        if path.startswith(marker)
    }


def _assert_flat_tree_matches_fixture(tree, arrays, prefix, *, atol=0.0):
    actual = flatten_with_paths(tree)
    expected = _fixture_tree(arrays, prefix)
    assert actual.keys() == expected.keys()
    for path, value in actual.items():
        expected_value = expected[path]
        assert value.shape == expected_value.shape, path
        assert value.dtype == expected_value.dtype, path
        np.testing.assert_allclose(value, expected_value, rtol=0.0, atol=atol)


def _strict_setup():
    config = RTRRLComponentConfig(
        profile="aaai25_strict_lru",
        backbone="aaai25_lru",
        trace_timing="incoming",
        logprob_reduction="mean",
        observation_dim=1,
        action_dim=2,
        hidden_dim=2,
        num_envs=1,
        meta_rl=True,
        optimizer_name="adam",
        actor_critic_learning_rate=1e-3,
        recurrent_learning_rate=1e-3,
    )
    components = RTRRLComponents(
        recurrent=AAAI25LRU(
            input_dim=4,
            hidden_dim=2,
            output_dim=2,
        ),
        head=RTRRLTDHead(action_dim=2, discrete=False, f_align=False),
    )
    return components, config, _DeterministicEnvironment()


def test_historical_key_split_parameters_and_optimizer_state_match_oracle():
    arrays, _ = load_oracle()
    components, config, environment = _strict_setup()

    state, keys = make_init_fn(components, config, environment)(
        jnp.asarray(arrays["state_machine/keys/root"])
    )

    np.testing.assert_array_equal(
        keys.step, arrays["state_machine/keys/step"]
    )
    np.testing.assert_array_equal(
        keys.outer, arrays["state_machine/keys/outer"]
    )
    _assert_flat_tree_matches_fixture(
        state.parameters,
        arrays,
        "state_machine/init/parameters",
    )
    _assert_flat_tree_matches_fixture(
        state.optimizer_state,
        arrays,
        "state_machine/init/optimizer_state",
    )


def test_historical_pre_sample_advanced_state_traces_and_statistics_match_oracle():
    arrays, _ = load_oracle()
    components, config, environment = _strict_setup()

    state, _ = make_init_fn(components, config, environment)(
        jnp.asarray(arrays["state_machine/keys/root"])
    )

    np.testing.assert_array_equal(
        state.action, arrays["state_machine/init/action"]
    )
    np.testing.assert_array_equal(
        state.value, arrays["state_machine/init/value"]
    )
    np.testing.assert_array_equal(
        state.model_input, arrays["state_machine/init/model_input"]
    )
    _assert_flat_tree_matches_fixture(
        state.recurrent_state,
        arrays,
        "state_machine/init/recurrent_state",
    )
    _assert_flat_tree_matches_fixture(
        state.traces,
        arrays,
        "state_machine/init/traces",
    )
    np.testing.assert_array_equal(
        state.emphasis, arrays["state_machine/init/emphasis"]
    )
    np.testing.assert_array_equal(
        state.average_reward,
        arrays["state_machine/init/average_reward"],
    )
    assert state.observation_statistics is None
    assert state.reward_statistics is None
