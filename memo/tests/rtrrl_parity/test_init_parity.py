"""Oracle parity for historical strict-RTRRL initialization."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from memorax.algorithms.rtrrl.compatibility import (
    LegacyOptimizerConfig,
    RTRRLComponentConfig,
)
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
    phase: jax.Array
    last_action: jax.Array


class _DeterministicEnvironment:
    action_size = 2

    def reset(self, key):
        del key
        return _EnvironmentState(
            obs=jnp.array([[0.25]], dtype=jnp.float32),
            reward=jnp.array([-0.5], dtype=jnp.float32),
            done=jnp.array([False]),
            phase=jnp.array(0, dtype=jnp.int32),
            last_action=jnp.zeros((1, 2), dtype=jnp.float32),
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
        if value.dtype.kind in "fc":
            np.testing.assert_allclose(
                value, expected_value, rtol=0.0, atol=atol
            )
        else:
            np.testing.assert_array_equal(value, expected_value, err_msg=path)


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
        optimizer_params_td=LegacyOptimizerConfig(learning_rate=1e-3),
        optimizer_params_rnn=LegacyOptimizerConfig(
            learning_rate=1e-3,
            gradient_clip=None,
        ),
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
        jnp.asarray(arrays["state_machine/init/keys/root"])
    )

    np.testing.assert_array_equal(
        keys.step, arrays["state_machine/init/keys/step"]
    )
    np.testing.assert_array_equal(
        keys.outer, arrays["state_machine/init/keys/outer"]
    )
    _assert_flat_tree_matches_fixture(
        state.parameters,
        arrays,
        "state_machine/init/state/parameters",
    )
    _assert_flat_tree_matches_fixture(
        state.optimizer_state,
        arrays,
        "state_machine/init/state/optimizer_state",
    )


def test_historical_pre_sample_advanced_state_traces_and_statistics_match_oracle():
    arrays, _ = load_oracle()
    components, config, environment = _strict_setup()

    state, _ = make_init_fn(components, config, environment)(
        jnp.asarray(arrays["state_machine/init/keys/root"])
    )

    np.testing.assert_array_equal(
        state.action, arrays["state_machine/init/state/action"]
    )
    np.testing.assert_array_equal(
        state.value, arrays["state_machine/init/state/value"]
    )
    np.testing.assert_array_equal(
        state.model_input, arrays["state_machine/init/state/model_input"]
    )
    _assert_flat_tree_matches_fixture(
        state.recurrent_state,
        arrays,
        "state_machine/init/state/recurrent_state",
    )
    _assert_flat_tree_matches_fixture(
        state.traces,
        arrays,
        "state_machine/init/state/traces",
    )
    np.testing.assert_array_equal(
        state.emphasis, arrays["state_machine/init/state/emphasis"]
    )
    np.testing.assert_array_equal(
        state.average_reward,
        arrays["state_machine/init/state/average_reward"],
    )
    assert state.observation_statistics is None
    assert state.reward_statistics is None


def _canonical_state_tree(state):
    environment = state.environment_state
    return {
        "parameters": state.parameters,
        "slow_parameters": state.slow_parameters,
        "optimizer_state": state.optimizer_state,
        "environment_state": {
            "obs": environment.obs,
            "reward": environment.reward,
            "done": environment.done,
            "phase": environment.phase,
            "last_action": environment.last_action,
        },
        "action": state.action,
        "recurrent_state": state.recurrent_state,
        "traces": state.traces,
        "value": state.value,
        "average_reward": state.average_reward,
        "emphasis": state.emphasis,
        "observation_statistics": np.asarray("<none>"),
        "reward_statistics": np.asarray("<none>"),
        "model_input": state.model_input,
        "initial_recurrent_state": state.initial_recurrent_state,
        "step_count": state.step_count,
    }


def test_complete_initialized_state_and_every_split_key_match_oracle():
    arrays, _ = load_oracle()
    components, config, environment = _strict_setup()

    state, keys = make_init_fn(components, config, environment)(
        jnp.asarray(arrays["state_machine/init/keys/root"])
    )

    _assert_flat_tree_matches_fixture(
        _canonical_state_tree(state),
        arrays,
        "state_machine/init/state",
    )
    for name in ("model", "step", "carry", "environment", "outer"):
        np.testing.assert_array_equal(
            getattr(keys, name),
            arrays[f"state_machine/init/keys/{name}"],
        )
