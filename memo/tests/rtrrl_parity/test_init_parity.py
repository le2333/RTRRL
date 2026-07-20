"""Oracle parity for historical strict-RTRRL initialization."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms.rtrrl.compatibility import (
    LegacyOptimizerConfig,
    RTRRLComponentConfig,
)
from memorax.algorithms.rtrrl.heads import RTRRLTDHead
from memorax.algorithms.rtrrl.lru import AAAI25LRU
from memorax.algorithms.rtrrl.state_machine import make_init_fn
from memorax.algorithms.rtrrl.types import RTRRLComponents, RTRRLState

from .assertions import (
    canonicalize_dataclass_pytree as _canonical_state_tree,
    flatten_with_paths,
)
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
    assert actual.keys() == expected.keys(), {
        "missing": sorted(expected.keys() - actual.keys()),
        "unexpected": sorted(actual.keys() - expected.keys()),
    }
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


def test_state_canonicalization_preserves_non_none_statistics_trees():
    state = RTRRLState(
        parameters={},
        slow_parameters={},
        optimizer_state=(),
        environment_state=_EnvironmentState(
            obs=jnp.zeros((1, 1)),
            reward=jnp.zeros((1,)),
            done=jnp.zeros((1,), dtype=jnp.bool_),
            phase=jnp.array(0),
            last_action=jnp.zeros((1, 2)),
        ),
        action=jnp.zeros((1, 2)),
        recurrent_state=(),
        traces={},
        value=jnp.zeros((1, 1)),
        average_reward=jnp.zeros((1,)),
        emphasis=jnp.ones((1,)),
        observation_statistics={"mean": jnp.asarray([1.25])},
        reward_statistics={"variance": jnp.asarray([0.75])},
        model_input=jnp.zeros((1, 4)),
        initial_recurrent_state=(),
        step_count=jnp.array(0),
    )

    flattened = flatten_with_paths(_canonical_state_tree(state))

    assert "observation_statistics/mean" in flattened
    assert "reward_statistics/variance" in flattened
    assert not any(
        np.asarray(value).dtype.kind in "US"
        for path, value in flattened.items()
        if path.startswith(("observation_statistics", "reward_statistics"))
    )
    with pytest.raises(AssertionError):
        _assert_flat_tree_matches_fixture(
            {
                "observation_statistics": state.observation_statistics,
                "reward_statistics": state.reward_statistics,
            },
            {
                "statistics/observation_statistics": np.asarray("<none>"),
                "statistics/reward_statistics": np.asarray("<none>"),
            },
            "statistics",
        )


def test_state_canonicalization_cannot_ignore_future_dataclass_fields():
    @dataclass(frozen=True)
    class StateWithFutureField:
        parameters: object
        slow_parameters: object
        optimizer_state: object
        environment_state: object
        action: object
        recurrent_state: object
        traces: object
        value: object
        average_reward: object
        emphasis: object
        observation_statistics: object
        reward_statistics: object
        model_input: object
        initial_recurrent_state: object
        step_count: object
        future_probe: object

    base = SimpleNamespace(
        parameters={},
        slow_parameters={},
        optimizer_state=(),
        environment_state=_EnvironmentState(
            obs=jnp.zeros((1, 1)),
            reward=jnp.zeros((1,)),
            done=jnp.zeros((1,), dtype=jnp.bool_),
            phase=jnp.array(0),
            last_action=jnp.zeros((1, 2)),
        ),
        action=jnp.zeros((1, 2)),
        recurrent_state=(),
        traces={},
        value=jnp.zeros((1, 1)),
        average_reward=jnp.zeros((1,)),
        emphasis=jnp.ones((1,)),
        observation_statistics=None,
        reward_statistics=None,
        model_input=jnp.zeros((1, 4)),
        initial_recurrent_state=(),
        step_count=jnp.array(0),
    )
    extended = StateWithFutureField(
        **vars(base),
        future_probe=jnp.asarray([42.0]),
    )

    flattened = flatten_with_paths(_canonical_state_tree(extended))

    assert "future_probe" in flattened


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
