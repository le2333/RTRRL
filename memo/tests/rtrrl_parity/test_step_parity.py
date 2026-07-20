"""Complete eager-transition parity for the strict online state machine."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect

import jax
import jax.numpy as jnp
import numpy as np

from memorax.algorithms.rtrrl.state_machine import make_init_fn, make_step_fn
from memorax.algorithms.rtrrl.types import DebugStepMetrics, TrainStepMetrics

from .assertions import flatten_with_paths
from .oracle_capture import load_oracle
from .test_init_parity import _strict_setup


@dataclass(frozen=True)
class _StepEnvironmentState:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    phase: jax.Array
    last_action: jax.Array


class _ThreeStepEnvironment:
    action_size = 2

    def reset(self, key):
        del key
        return _StepEnvironmentState(
            obs=jnp.array([[0.25]], dtype=jnp.float32),
            reward=jnp.array([-0.5], dtype=jnp.float32),
            done=jnp.array([False]),
            phase=jnp.array(0, dtype=jnp.int32),
            last_action=jnp.zeros((1, 2), dtype=jnp.float32),
        )

    def step(self, state, action):
        phase = state.phase
        return _StepEnvironmentState(
            obs=jnp.array([[0.1 + 0.05 * phase]], dtype=jnp.float32),
            reward=jnp.array([0.625 - 0.125 * phase], dtype=jnp.float32),
            done=jnp.array([phase == 1]),
            phase=phase + 1,
            last_action=action,
        )


def _fixture_tree(arrays, prefix):
    marker = prefix + "/"
    return {
        path[len(marker) :]: value
        for path, value in arrays.items()
        if path.startswith(marker)
    }


def _assert_tree_fixture(tree, arrays, prefix, *, atol=2e-7):
    actual = flatten_with_paths(tree)
    expected = _fixture_tree(arrays, prefix)
    assert actual.keys() == expected.keys()
    for path, value in actual.items():
        expected_value = expected[path]
        assert value.shape == expected_value.shape, path
        assert value.dtype == expected_value.dtype, path
        np.testing.assert_allclose(
            value,
            expected_value,
            rtol=2e-6,
            atol=atol,
            err_msg=path,
        )


def _initialized():
    components, config, _ = _strict_setup()
    environment = _ThreeStepEnvironment()
    arrays, _ = load_oracle()
    state, keys = make_init_fn(components, config, environment)(
        jnp.asarray(arrays["state_machine/keys/root"])
    )
    return arrays, components, config, environment, state, keys.step


def _assert_step(step_number, state, metrics, arrays):
    prefix = f"state_machine/step_{step_number}"
    assert isinstance(metrics, DebugStepMetrics)
    np.testing.assert_allclose(
        state.environment_state.last_action,
        arrays[f"{prefix}/environment_action"],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        metrics.model_input,
        arrays[f"{prefix}/model_input"],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        state.action,
        arrays[f"{prefix}/sampled_next_action"],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        metrics.value_target,
        arrays[f"{prefix}/value_target"],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        metrics.td_error,
        arrays[f"{prefix}/td_error"],
        rtol=2e-6,
        atol=2e-7,
    )
    _assert_tree_fixture(
        metrics.gradients, arrays, f"{prefix}/gradients"
    )
    _assert_tree_fixture(
        metrics.direct_gradients,
        arrays,
        f"{prefix}/direct_gradients",
    )
    _assert_tree_fixture(
        metrics.incoming_traces,
        arrays,
        f"{prefix}/incoming_traces",
    )
    _assert_tree_fixture(
        state.traces, arrays, f"{prefix}/carried_traces"
    )
    _assert_tree_fixture(
        metrics.mean_directions,
        arrays,
        f"{prefix}/mean_directions",
    )
    _assert_tree_fixture(
        metrics.optimizer_updates,
        arrays,
        f"{prefix}/optimizer_updates",
    )
    _assert_tree_fixture(
        state.parameters, arrays, f"{prefix}/parameters"
    )
    _assert_tree_fixture(
        state.slow_parameters, arrays, f"{prefix}/slow_parameters"
    )
    _assert_tree_fixture(
        state.optimizer_state, arrays, f"{prefix}/optimizer_state"
    )
    _assert_tree_fixture(
        state.recurrent_state, arrays, f"{prefix}/recurrent_state"
    )
    np.testing.assert_allclose(
        state.emphasis,
        arrays[f"{prefix}/emphasis"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        state.average_reward,
        arrays[f"{prefix}/average_reward"],
        rtol=0.0,
        atol=0.0,
    )


def test_complete_first_eager_step_matches_every_oracle_leaf():
    arrays, components, config, environment, state, key = _initialized()

    next_state, next_key, metrics = make_step_fn(
        components, config, environment, debug=True
    )(state, key)

    np.testing.assert_array_equal(
        next_key, arrays["state_machine/step_1/key_out"]
    )
    _assert_step(1, next_state, metrics, arrays)


def test_terminal_reset_and_three_step_persisted_feedback_match_oracle():
    arrays, components, config, environment, state, key = _initialized()
    step = make_step_fn(components, config, environment, debug=True)

    for step_number in range(1, 4):
        state, key, metrics = step(state, key)
        _assert_step(step_number, state, metrics, arrays)

    terminal_input = arrays["state_machine/step_2/model_input"]
    np.testing.assert_array_equal(terminal_input[:, 1:], 0.0)
    np.testing.assert_allclose(
        arrays["state_machine/step_3/model_input"][:, 1:3],
        arrays["state_machine/step_2/sampled_next_action"],
        rtol=2e-6,
        atol=2e-7,
    )


def test_two_environment_delta_is_formed_before_mean_reduction():
    arrays, _, _, _, _, _ = _initialized()
    delta = jnp.asarray(arrays["state_machine/two_env/delta"])
    trace = jnp.array([[1.0, -2.0], [3.0, 4.0]], dtype=jnp.float32)
    actual = jnp.mean(delta[:, None] * trace, axis=0)

    np.testing.assert_array_equal(
        actual, arrays["state_machine/two_env/direction"]
    )
    wrong = jnp.mean(delta) * jnp.mean(trace, axis=0)
    assert not np.allclose(actual, wrong)


def test_train_metrics_are_scalar_event_only_and_debug_stops_after_three_steps():
    _, components, config, environment, state, key = _initialized()
    train_step = make_step_fn(
        components, config, environment, debug=False
    )
    _, _, train_metrics = train_step(state, key)

    assert isinstance(train_metrics, TrainStepMetrics)
    assert all(
        np.asarray(leaf).ndim == 0
        for leaf in jax.tree.leaves(train_metrics)
    )

    debug_step = make_step_fn(
        components, config, environment, debug=True
    )
    state = replace(state, step_count=jnp.array(3, dtype=jnp.int32))
    _, _, capped_metrics = debug_step(state, key)
    assert isinstance(capped_metrics, TrainStepMetrics)


def test_factories_capture_only_static_values_and_keep_state_key_explicit():
    _, components, config, environment, _, _ = _initialized()
    init_fn = make_init_fn(components, config, environment)
    step_fn = make_step_fn(components, config, environment, debug=False)

    assert tuple(inspect.signature(init_fn).parameters) == ("root_key",)
    assert tuple(inspect.signature(step_fn).parameters) == ("state", "key")
    captured = inspect.getclosurevars(step_fn).nonlocals
    assert "state" not in captured
    assert "key" not in captured
    assert captured["components"] is components
    assert captured["config"] is config
    assert captured["env"] is environment
