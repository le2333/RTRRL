"""Independent equation tests for the pure strict-RTRRL update rules."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from memorax.algorithms.rtrrl.rules import (
    combine_update_directions,
    td_error,
    update_emphasis_or_average_reward,
    update_slow_target,
    update_traces,
)

from .assertions import assert_tree_close


@pytest.mark.parametrize(
    ("terminated", "expected"),
    [(False, 2.0 + 0.9 * 5.0 - 0.4 - 3.0), (True, 2.0 - 0.4 - 3.0)],
)
def test_td_error_matches_terminal_and_nonterminal_equations(terminated, expected):
    actual = td_error(
        reward=jnp.array(2.0),
        value=jnp.array(3.0),
        next_value=jnp.array(5.0),
        terminated=jnp.array(terminated),
        gamma=0.9,
        average_reward=jnp.array(0.4),
    )

    np.testing.assert_allclose(actual, expected)


def test_td_error_is_per_environment_before_any_mean_reduction():
    actual = td_error(
        reward=jnp.array([1.0, 4.0]),
        value=jnp.array([2.0, -1.0]),
        next_value=jnp.array([7.0, 3.0]),
        terminated=jnp.array([False, True]),
        gamma=0.5,
        average_reward=jnp.array(0.25),
    )

    np.testing.assert_allclose(actual, jnp.array([2.25, 4.75]))
    wrong_mean_first = (
        jnp.mean(jnp.array([1.0, 4.0]))
        + 0.5
        * (1 - jnp.mean(jnp.array([False, True], dtype=jnp.float32)))
        * jnp.mean(jnp.array([7.0, 3.0]))
        - 0.25
        - jnp.mean(jnp.array([2.0, -1.0]))
    )
    assert not np.allclose(jnp.mean(actual), wrong_mean_first)


@pytest.mark.parametrize(
    ("timing", "expected_update"),
    [
        ("incoming", jnp.array([[2.0, -1.0], [4.0, 3.0]])),
        ("fresh", jnp.array([[1.1, -0.05], [1.0, -2.0]])),
    ],
)
def test_accumulated_actor_and_rnn_traces_obey_timing(
    timing,
    expected_update,
):
    incoming = {
        "actor": {"w": jnp.array([[2.0, -1.0], [4.0, 3.0]])},
        "critic": {"w": jnp.zeros((2, 2))},
        "recurrent": {"w": jnp.array([[2.0, -1.0], [4.0, 3.0]])},
    }
    gradients = {
        "actor": {"w": jnp.array([[0.5, 0.25], [0.5, -1.0]])},
        "critic": {"w": jnp.ones((2, 2))},
        "recurrent": {"w": jnp.array([[0.5, 0.25], [0.5, -1.0]])},
    }

    result = update_traces(
        incoming,
        gradients,
        gamma=0.6,
        lambda_actor=0.5,
        lambda_critic=0.25,
        lambda_rnn=0.5,
        trace_mode="accumulate",
        critic_learning_rate=0.2,
        emphasis=jnp.array([1.0, 2.0]),
        terminated=jnp.array([False, True]),
        timing=timing,
    )

    expected_fresh = jnp.array([[1.1, -0.05], [1.0, -2.0]])
    np.testing.assert_allclose(
        result.carried["actor"]["w"], expected_fresh, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        result.carried["recurrent"]["w"], expected_fresh, rtol=1e-6, atol=1e-7
    )
    if timing == "fresh":
        expected_update = expected_fresh
    np.testing.assert_allclose(
        result.update["actor"]["w"], expected_update, rtol=1e-6, atol=1e-7
    )
    wrong_timing = incoming["actor"]["w"] if timing == "fresh" else expected_fresh
    assert not np.allclose(result.update["actor"]["w"], wrong_timing)


def test_accumulated_critic_trace_uses_emphasis_and_terminal_reset():
    result = update_traces(
        {"actor": {"w": jnp.zeros((2, 1))}, "critic": {"w": jnp.array([[4.0], [5.0]])}},
        {"actor": {"w": jnp.ones((2, 1))}, "critic": {"w": jnp.array([[0.3], [0.4]])}},
        gamma=0.8,
        lambda_actor=0.5,
        lambda_critic=0.25,
        lambda_rnn=0.0,
        trace_mode="accumulate",
        critic_learning_rate=0.1,
        emphasis=jnp.array([1.5, 2.0]),
        terminated=jnp.array([False, True]),
        timing="fresh",
    )

    np.testing.assert_allclose(
        result.carried["critic"]["w"],
        jnp.array([[0.8 * 0.25 * 4.0 + 1.5 * 0.3], [2.0 * 0.4]]),
    )


def test_dutch_critic_trace_matches_leaf_local_aaai25_equation():
    old = jnp.array([[2.0, -1.0], [4.0, 3.0]])
    grad = jnp.array([[0.5, 0.25], [0.5, -1.0]])
    decay = 0.8 * 0.25
    alpha = 0.1
    reset_old = jnp.array([[2.0, -1.0], [0.0, 0.0]])
    correction = 1 - alpha * decay * jnp.sum(reset_old * grad, axis=-1)
    expected = decay * reset_old + correction[:, None] * grad

    result = update_traces(
        {"critic": {"w": old}},
        {"critic": {"w": grad}},
        gamma=0.8,
        lambda_actor=0.5,
        lambda_critic=0.25,
        lambda_rnn=0.5,
        trace_mode="dutch",
        critic_learning_rate=alpha,
        emphasis=jnp.array([9.0, 7.0]),
        terminated=jnp.array([False, True]),
        timing="fresh",
    )

    np.testing.assert_allclose(result.carried["critic"]["w"], expected)


def test_zero_lambda_rnn_uses_unweighted_fresh_gradient():
    gradient = jnp.array([[0.5, 0.25], [0.5, -1.0]])
    result = update_traces(
        {"recurrent": {"w": jnp.array([[2.0, -1.0], [4.0, 3.0]])}},
        {"recurrent": {"w": gradient}},
        gamma=0.8,
        lambda_actor=0.5,
        lambda_critic=0.25,
        lambda_rnn=0.0,
        trace_mode="accumulate",
        critic_learning_rate=0.1,
        emphasis=jnp.array([3.0, 4.0]),
        terminated=jnp.array([False, True]),
        timing="fresh",
    )

    np.testing.assert_allclose(result.carried["recurrent"]["w"], gradient)
    assert not np.allclose(
        result.carried["recurrent"]["w"],
        jnp.array([[3.0], [4.0]]) * gradient,
    )


def test_combine_directions_multiplies_delta_per_environment_before_mean():
    traces = {
        "actor": {"w": jnp.array([[1.0, 4.0], [3.0, -2.0]])},
        "recurrent": {"w": jnp.array([[2.0], [5.0]])},
    }
    direct = {
        "actor": {"w": jnp.array([[0.2, 0.4], [0.6, 0.8]])},
        "recurrent": {"w": jnp.array([[1.0], [3.0]])},
    }
    delta = jnp.array([2.0, -1.0])

    actual = combine_update_directions(
        traces,
        direct,
        delta=delta,
        recurrent_scale=0.25,
    )
    expected = {
        "actor": {
            "w": jnp.mean(delta[:, None] * traces["actor"]["w"] + direct["actor"]["w"], axis=0)
        },
        "recurrent": {
            "w": jnp.mean(
                0.25 * delta[:, None] * traces["recurrent"]["w"]
                + direct["recurrent"]["w"],
                axis=0,
            )
        },
    }

    assert_tree_close(actual, expected, (1e-6, 1e-7))
    wrong_mean_first = jnp.mean(delta) * jnp.mean(traces["actor"]["w"], axis=0) + jnp.mean(
        direct["actor"]["w"], axis=0
    )
    assert not np.allclose(actual["actor"]["w"], wrong_mean_first)


def test_dutch_critic_direction_includes_true_online_correction():
    traces = {"critic": {"w": jnp.array([[2.0, -1.0], [4.0, 3.0]])}}
    gradients = {"critic": {"w": jnp.array([[0.5, 0.25], [0.5, -1.0]])}}
    direct = {"critic": {"w": jnp.zeros((2, 2))}}
    delta = jnp.array([2.0, -1.0])
    value_difference = jnp.array([0.75, -0.5])
    alpha = 0.1

    actual = combine_update_directions(
        traces,
        direct,
        delta=delta,
        recurrent_scale=0.25,
        trace_mode="dutch",
        immediate_gradients=gradients,
        critic_learning_rate=alpha,
        critic_value_difference=value_difference,
    )
    per_environment = (
        delta[:, None] * traces["critic"]["w"]
        + alpha
        * value_difference[:, None]
        * (traces["critic"]["w"] - gradients["critic"]["w"])
    )

    np.testing.assert_allclose(
        actual["critic"]["w"],
        jnp.mean(per_environment, axis=0),
        rtol=1e-6,
        atol=1e-7,
    )
    wrong_plain_td = jnp.mean(delta[:, None] * traces["critic"]["w"], axis=0)
    assert not np.allclose(actual["critic"]["w"], wrong_plain_td)


def test_direct_entropy_is_not_delta_or_recurrent_scale_weighted():
    traces = {
        "actor": {"w": jnp.zeros((2, 1))},
        "recurrent": {"w": jnp.zeros((2, 1))},
    }
    entropy_rate = 0.03
    logprob_scale = 0.5
    entropy_grad = entropy_rate * logprob_scale * jnp.array([[4.0], [8.0]])
    direct = {"actor": {"w": entropy_grad}, "recurrent": {"w": entropy_grad}}

    actual = combine_update_directions(
        traces,
        direct,
        delta=jnp.array([10.0, -20.0]),
        recurrent_scale=0.2,
    )

    expected = jnp.mean(entropy_grad, axis=0)
    np.testing.assert_allclose(actual["actor"]["w"], expected)
    np.testing.assert_allclose(actual["recurrent"]["w"], expected)
    assert not np.allclose(actual["actor"]["w"], expected * entropy_rate)
    assert not np.allclose(actual["recurrent"]["w"], expected * 0.2)


def test_parameter_domain_scaling_applies_only_to_recurrent_traced_direction():
    traces = {
        domain: {"w": jnp.array([[2.0], [4.0]])}
        for domain in ("actor", "critic", "recurrent")
    }
    direct = {
        domain: {"w": jnp.zeros((2, 1))}
        for domain in ("actor", "critic", "recurrent")
    }
    actual = combine_update_directions(
        traces,
        direct,
        delta=jnp.array([1.0, 3.0]),
        recurrent_scale=0.1,
    )

    unscaled = jnp.mean(jnp.array([[1.0], [3.0]]) * jnp.array([[2.0], [4.0]]), axis=0)
    np.testing.assert_allclose(actual["actor"]["w"], unscaled)
    np.testing.assert_allclose(actual["critic"]["w"], unscaled)
    np.testing.assert_allclose(actual["recurrent"]["w"], 0.1 * unscaled)
    assert not np.allclose(actual["actor"]["w"], 0.1 * unscaled)


@pytest.mark.parametrize(
    ("eta", "expected_emphasis", "expected_average_reward"),
    [
        (None, jnp.array([0.45, 1.0]), jnp.array(1.25)),
        (0.2, jnp.array([0.5, 0.7]), jnp.array(1.45)),
    ],
)
def test_emphasis_or_average_reward_update_is_branch_exact(
    eta,
    expected_emphasis,
    expected_average_reward,
):
    result = update_emphasis_or_average_reward(
        emphasis=jnp.array([0.5, 0.7]),
        average_reward=jnp.array(1.25),
        delta=jnp.array([2.0, 0.0]),
        terminated=jnp.array([False, True]),
        gamma=0.9,
        eta=eta,
    )

    np.testing.assert_allclose(result.emphasis, expected_emphasis)
    np.testing.assert_allclose(result.average_reward, expected_average_reward)


@pytest.mark.parametrize(
    ("period", "expected"),
    [(1.0, jnp.array([10.0, -2.0])), (0.1, jnp.array([1.9, 1.6]))],
)
def test_slow_target_uses_post_update_fast_parameters(period, expected):
    fast = {"w": jnp.array([10.0, -2.0])}
    previous_slow = {"w": jnp.array([1.0, 2.0])}

    actual = update_slow_target(
        fast_parameters=fast,
        previous_slow_parameters=previous_slow,
        period=period,
    )

    np.testing.assert_allclose(actual["w"], expected)


def test_online_ac_helpers_pass_independent_supported_equations_before_reuse():
    from memorax.online_ac.targets import make_slow_subtree_target
    from memorax.online_ac.td import make_td0
    from memorax.online_ac.traces import make_rtrrl_trace

    td0 = make_td0()
    for terminated, expected in ((False, 3.5), (True, 0.0)):
        actual = td0(
            reward=jnp.array(2.0),
            value=jnp.array(2.0),
            next_value=jnp.array(5.0),
            bootstrap_discount=0.7 * (1 - terminated),
        )
        np.testing.assert_allclose(actual, expected)

    config = SimpleNamespace(
        gamma=0.8,
        lambda_pi=0.5,
        lambda_v=0.25,
        lambda_rnn=0.75,
        update_trace_before_td=True,
    )
    incoming = {
        domain: {"w": jnp.array([[2.0], [3.0]])}
        for domain in ("actor", "critic", "recurrent")
    }
    gradients = {
        domain: {"w": jnp.array([[0.5], [0.25]])}
        for domain in ("actor", "critic", "recurrent")
    }
    helper_trace = make_rtrrl_trace(config)(
        incoming,
        gradients,
        terminated_after=jnp.array([False, True]),
        emphasis=jnp.array([1.5, 2.0]),
    )
    strict_trace = update_traces(
        incoming,
        gradients,
        gamma=config.gamma,
        lambda_actor=config.lambda_pi,
        lambda_critic=config.lambda_v,
        lambda_rnn=config.lambda_rnn,
        trace_mode="accumulate",
        critic_learning_rate=0.1,
        emphasis=jnp.array([1.5, 2.0]),
        terminated=jnp.array([False, True]),
        timing="fresh",
    )
    assert_tree_close(helper_trace, strict_trace, (1e-6, 1e-7))

    target = make_slow_subtree_target(SimpleNamespace(update_period=0.1))
    helper_slow = target.finish_update(
        fast_params={"torso": {"w": jnp.array(10.0)}},
        previous_slow_subtree={"w": jnp.array(2.0)},
        sensitivity={"s": jnp.array(3.0)},
    ).slow_subtree
    strict_slow = update_slow_target(
        fast_parameters={"w": jnp.array(10.0)},
        previous_slow_parameters={"w": jnp.array(2.0)},
        period=0.1,
    )
    assert_tree_close(helper_slow, strict_slow, (1e-6, 1e-7))


def test_rules_are_jittable_and_contain_no_hidden_inputs():
    compiled = jax.jit(
        lambda reward, value, next_value, terminated: td_error(
            reward=reward,
            value=value,
            next_value=next_value,
            terminated=terminated,
            gamma=0.9,
            average_reward=0.0,
        )
    )

    np.testing.assert_allclose(
        compiled(2.0, 3.0, 5.0, jnp.array(False)),
        3.5,
    )
