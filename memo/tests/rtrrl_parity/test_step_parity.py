"""Complete eager-transition parity for the strict online state machine."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import optax

from memorax.algorithms.rtrrl.compatibility import LegacyOptimizerConfig
from memorax.algorithms.rtrrl.state_machine import (
    _make_maximizing_group,
    make_init_fn,
    make_step_fn,
)
from memorax.algorithms.rtrrl.rules import update_slow_target
from memorax.algorithms.rtrrl.types import DebugStepMetrics, TrainStepMetrics

from .assertions import flatten_with_paths
from .oracle_capture import load_oracle
from .test_init_parity import _canonical_state_tree, _strict_setup


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


class _TwoEnvironment:
    action_size = 2

    def reset(self, key):
        del key
        return _StepEnvironmentState(
            obs=jnp.array([[0.25], [-0.35]], dtype=jnp.float32),
            reward=jnp.array([-0.5, 0.25], dtype=jnp.float32),
            done=jnp.array([False, False]),
            phase=jnp.array(0, dtype=jnp.int32),
            last_action=jnp.zeros((2, 2), dtype=jnp.float32),
        )

    def step(self, state, action):
        phase = state.phase
        return _StepEnvironmentState(
            obs=jnp.stack(
                (
                    jnp.asarray([0.1 + 0.05 * phase]),
                    jnp.asarray([-0.2 + 0.025 * phase]),
                )
            ).astype(jnp.float32),
            reward=jnp.asarray(
                [0.625 - 0.125 * phase, -0.25 + 0.05 * phase],
                dtype=jnp.float32,
            ),
            done=jnp.asarray([False, phase == 0]),
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
        if value.dtype.kind in "fc":
            np.testing.assert_allclose(
                value,
                expected_value,
                rtol=2e-6,
                atol=atol,
                err_msg=path,
            )
        else:
            np.testing.assert_array_equal(value, expected_value, err_msg=path)


def _initialized():
    components, config, _ = _strict_setup()
    environment = _ThreeStepEnvironment()
    arrays, _ = load_oracle()
    state, keys = make_init_fn(components, config, environment)(
        jnp.asarray(arrays["state_machine/init/keys/root"])
    )
    return arrays, components, config, environment, state, keys.step


def _assert_step(step_number, state, metrics, arrays):
    prefix = f"state_machine/step_{step_number}"
    assert isinstance(metrics, DebugStepMetrics)
    _assert_tree_fixture(
        _canonical_state_tree(state), arrays, f"{prefix}/state"
    )
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
        state.parameters, arrays, f"{prefix}/state/parameters"
    )
    _assert_tree_fixture(
        state.slow_parameters, arrays, f"{prefix}/state/slow_parameters"
    )
    _assert_tree_fixture(
        state.optimizer_state, arrays, f"{prefix}/state/optimizer_state"
    )
    _assert_tree_fixture(
        state.recurrent_state, arrays, f"{prefix}/state/recurrent_state"
    )
    np.testing.assert_allclose(
        state.emphasis,
        arrays[f"{prefix}/state/emphasis"],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        state.average_reward,
        arrays[f"{prefix}/state/average_reward"],
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


def test_two_environment_complete_production_step_matches_oracle():
    arrays, _ = load_oracle()
    components, base_config, _ = _strict_setup()
    config = replace(base_config, num_envs=2)
    environment = _TwoEnvironment()
    state, keys = make_init_fn(components, config, environment)(
        jnp.asarray(arrays["state_machine/two_env/init/keys/root"])
    )

    state, next_key, metrics = make_step_fn(
        components, config, environment, debug=True
    )(state, keys.step)

    prefix = "state_machine/two_env/step_1"
    assert isinstance(metrics, DebugStepMetrics)
    _assert_tree_fixture(_canonical_state_tree(state), arrays, f"{prefix}/state")
    _assert_tree_fixture(metrics.gradients, arrays, f"{prefix}/gradients")
    _assert_tree_fixture(
        metrics.incoming_traces, arrays, f"{prefix}/incoming_traces"
    )
    _assert_tree_fixture(
        metrics.mean_directions, arrays, f"{prefix}/mean_directions"
    )
    np.testing.assert_allclose(
        metrics.td_error,
        arrays[f"{prefix}/td_error"],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_array_equal(
        state.environment_state.done, np.array([False, True])
    )
    assert not np.isclose(metrics.td_error[0], metrics.td_error[1])
    assert all(
        np.asarray(leaf).shape[0] == 2
        for leaf in jax.tree.leaves(metrics.gradients)
    )
    assert all(
        np.asarray(leaf).shape[0] == 2
        for leaf in jax.tree.leaves(metrics.incoming_traces)
    )
    np.testing.assert_array_equal(
        next_key, arrays[f"{prefix}/key_out"]
    )


def test_train_metrics_are_scalar_event_only_and_debug_fails_before_fourth_step():
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
    before = flatten_with_paths(state)
    with np.testing.assert_raises_regex(
        ValueError, "debug step bound exhausted"
    ):
        debug_step(state, key)
    after = flatten_with_paths(state)
    assert before.keys() == after.keys()
    for path in before:
        np.testing.assert_array_equal(before[path], after[path], err_msg=path)


def test_factories_capture_only_static_values_and_keep_state_key_explicit():
    _, components, config, environment, state, key = _initialized()
    init_fn = make_init_fn(components, config, environment)
    step_fn = make_step_fn(components, config, environment, debug=False)

    assert tuple(inspect.signature(init_fn).parameters) == ("root_key",)
    assert tuple(inspect.signature(step_fn).parameters) == ("state", "key")
    init_captured = inspect.getclosurevars(init_fn).nonlocals
    step_captured = inspect.getclosurevars(step_fn).nonlocals
    assert set(init_captured) == {
        "components",
        "config",
        "env",
        "initializer",
        "optimizer",
        "recurrent",
    }
    assert set(step_captured) == {
        "components",
        "config",
        "debug",
        "env",
        "optimizer",
        "recurrent",
    }
    for captured in (init_captured, step_captured):
        assert "state" not in captured
        assert "key" not in captured
        assert not any(isinstance(value, jax.Array) for value in captured.values())
        assert captured["components"] is components
        assert captured["config"] is config
        assert captured["env"] is environment

    first_state, first_key, first_metrics = step_fn(state, key)
    second_state, second_key, second_metrics = step_fn(state, key)
    for first, second in (
        (first_state, second_state),
        (first_key, second_key),
        (first_metrics, second_metrics),
    ):
        first_leaves = flatten_with_paths(first)
        second_leaves = flatten_with_paths(second)
        assert first_leaves.keys() == second_leaves.keys()
        for path in first_leaves:
            np.testing.assert_array_equal(
                first_leaves[path], second_leaves[path], err_msg=path
            )


def test_nondefault_optimizer_schedule_clip_moments_and_sign_match_oracle():
    arrays, _ = load_oracle()
    config = LegacyOptimizerConfig(
        opt_name="adam",
        learning_rate=3e-3,
        kwargs=type(LegacyOptimizerConfig().kwargs)(
            (("b1", 0.73), ("b2", 0.84), ("eps", 2e-5))
        ),
        decay_type="exponential",
        lr_kwargs=type(LegacyOptimizerConfig().lr_kwargs)(
            (
                ("transition_steps", 2),
                ("decay_rate", 0.5),
                ("staircase", True),
            )
        ),
        weight_decay=0.03,
        gradient_clip=0.4,
        multi_step=2,
    )
    optimizer = _make_maximizing_group(config)
    parameters = {
        "bias": jnp.asarray([0.5], dtype=jnp.float32),
        "weight": jnp.asarray([1.0, -2.0], dtype=jnp.float32),
    }
    optimizer_state = optimizer.init(parameters)
    _assert_tree_fixture(
        optimizer_state,
        arrays,
        "optimizer_characterization/init_state",
    )
    gradients = (
        (jnp.asarray([0.2]), jnp.asarray([3.0, -4.0])),
        (jnp.asarray([-0.1]), jnp.asarray([0.5, 0.25])),
        (jnp.asarray([0.3]), jnp.asarray([-2.0, 1.0])),
        (jnp.asarray([0.4]), jnp.asarray([0.1, -0.2])),
        (jnp.asarray([-0.2]), jnp.asarray([1.5, 2.0])),
    )
    for index, (bias_gradient, weight_gradient) in enumerate(
        gradients, start=1
    ):
        updates, optimizer_state = optimizer.update(
            {"bias": bias_gradient, "weight": weight_gradient},
            optimizer_state,
            parameters,
        )
        parameters = optax.apply_updates(parameters, updates)
        for name, tree in (
            ("updates", updates),
            ("parameters", parameters),
            ("state", optimizer_state),
        ):
            _assert_tree_fixture(
                tree,
                arrays,
                f"optimizer_characterization/step_{index}/{name}",
            )
    slow_prefix = "optimizer_characterization/slow_target"
    slow_result = update_slow_target(
        fast_parameters=jnp.asarray(arrays[f"{slow_prefix}/fast"]),
        previous_slow_parameters=jnp.asarray(
            arrays[f"{slow_prefix}/previous"]
        ),
        period=float(arrays[f"{slow_prefix}/period"]),
    )
    np.testing.assert_array_equal(
        slow_result, arrays[f"{slow_prefix}/result"]
    )
