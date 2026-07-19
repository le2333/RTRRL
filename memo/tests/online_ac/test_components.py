from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest


def _assert_tree_values(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(
            actual[key]["w"], expected[key]["w"], rtol=1e-6, atol=1e-7
        )


def _rtrrl_config(*, fresh):
    return SimpleNamespace(
        gamma=0.8,
        lambda_pi=0.5,
        lambda_v=0.25,
        lambda_rnn=0.75,
        update_trace_before_td=fresh,
    )


def test_rtrrl_trace_uses_terminated_after_three_lambdas_and_emphasis():
    from memorax.online_ac.traces import make_rtrrl_trace

    incoming = {
        "actor": {"w": jnp.array([[2.0], [3.0]])},
        "critic": {"w": jnp.array([[4.0], [5.0]])},
        "recurrent": {"w": jnp.array([[6.0], [7.0]])},
    }
    gradient = {
        "actor": {"w": jnp.array([[0.1], [0.2]])},
        "critic": {"w": jnp.array([[0.3], [0.4]])},
        "recurrent": {"w": jnp.array([[0.5], [0.6]])},
    }
    terminated_after = jnp.array([False, True])
    emphasis = jnp.array([1.5, 2.0])

    result = make_rtrrl_trace(_rtrrl_config(fresh=True))(
        incoming,
        gradient,
        terminated_after=terminated_after,
        emphasis=emphasis,
    )

    expected = {
        "actor": {"w": jnp.array([[0.8 * 0.5 * 2.0 + 1.5 * 0.1], [2.0 * 0.2]])},
        "critic": {"w": jnp.array([[0.8 * 0.25 * 4.0 + 1.5 * 0.3], [2.0 * 0.4]])},
        "recurrent": {"w": jnp.array([[0.8 * 0.75 * 6.0 + 1.5 * 0.5], [2.0 * 0.6]])},
    }
    _assert_tree_values(result.carried, expected)
    _assert_tree_values(result.update, expected)


def test_rtrrl_trace_carries_fresh_but_can_update_from_incoming():
    from memorax.online_ac.traces import make_rtrrl_trace

    incoming = {
        "actor": {"w": jnp.array([[2.0], [3.0]])},
        "critic": {"w": jnp.array([[4.0], [5.0]])},
        "recurrent": {"w": jnp.array([[6.0], [7.0]])},
    }
    gradient = {
        key: {"w": jnp.ones((2, 1)) * scale}
        for key, scale in (("actor", 0.1), ("critic", 0.2), ("recurrent", 0.3))
    }

    result = make_rtrrl_trace(_rtrrl_config(fresh=False))(
        incoming,
        gradient,
        terminated_after=jnp.array([False, True]),
        emphasis=jnp.array([1.0, 2.0]),
    )

    _assert_tree_values(result.update, incoming)
    assert any(
        not np.allclose(result.carried[key]["w"], incoming[key]["w"])
        for key in incoming
    )


def test_stream_ac_trace_uses_reset_before_and_always_updates_from_fresh():
    from memorax.online_ac.traces import make_stream_ac_trace

    config = SimpleNamespace(gamma=0.9, trace_lambda=0.5)
    incoming = {"w": jnp.array([[2.0, 3.0], [4.0, 5.0]])}
    gradient = {"w": jnp.array([[0.1, 0.2], [0.3, 0.4]])}

    result = make_stream_ac_trace(config)(
        incoming,
        gradient,
        reset_before=jnp.array([True, False]),
    )

    expected = {
        "w": jnp.array(
            [
                [0.1, 0.2],
                [0.9 * 0.5 * 4.0 + 0.3, 0.9 * 0.5 * 5.0 + 0.4],
            ]
        )
    }
    np.testing.assert_allclose(result.carried["w"], expected["w"])
    np.testing.assert_allclose(result.update["w"], expected["w"])


def test_rtrrl_objective_routes_traced_and_direct_domains_without_td_or_eta_f():
    from memorax.online_ac.objectives import make_rtrrl_objective

    config = SimpleNamespace(
        eta_pi=0.4,
        eta_f=0.2,
        logprob_scale=0.5,
        entropy_rate=0.03,
        pred_coeff=0.7,
    )
    directions = make_rtrrl_objective(config)(
        log_prob=jnp.array([2.0, -1.0]),
        value=jnp.array([3.0, 4.0]),
        entropy=jnp.array([5.0, 6.0]),
        prediction=jnp.array([[1.0, 3.0], [4.0, 2.0]]),
        prediction_target=jnp.array([[0.0, 1.0], [2.0, 1.0]]),
    )

    actor = config.eta_pi * config.logprob_scale * jnp.array([2.0, -1.0])
    critic = jnp.array([3.0, 4.0])
    entropy = config.entropy_rate * config.logprob_scale * jnp.array([5.0, 6.0])
    prediction = -config.pred_coeff * 0.5 * jnp.array([5.0, 5.0])
    assert directions.traced_by_domain.keys() == {
        "actor",
        "critic",
        "recurrent",
        "prediction",
    }
    np.testing.assert_allclose(directions.traced_by_domain["actor"], actor)
    np.testing.assert_allclose(directions.traced_by_domain["critic"], critic)
    np.testing.assert_allclose(directions.traced_by_domain["recurrent"], actor + critic)
    np.testing.assert_allclose(directions.direct_by_domain["actor"], entropy)
    np.testing.assert_allclose(
        directions.direct_by_domain["recurrent"], entropy + prediction
    )
    np.testing.assert_allclose(directions.direct_by_domain["prediction"], prediction)
    np.testing.assert_allclose(directions.direct_by_domain["critic"], 0.0)
    assert not np.allclose(
        directions.direct_by_domain["recurrent"],
        config.eta_f * entropy,
    )


def test_stream_ac_objective_keeps_stopped_td_sign_inside_actor_trace():
    from memorax.online_ac.objectives import make_stream_ac_objective

    config = SimpleNamespace(entropy_coefficient=0.25)
    objective = make_stream_ac_objective(config)
    delta = jnp.array([-2.0, 3.0])
    directions = objective(
        log_prob=jnp.array([1.0, 2.0]),
        value=jnp.array([4.0, 5.0]),
        entropy=jnp.array([0.4, 0.8]),
        delta=delta,
    )

    np.testing.assert_allclose(
        directions.traced_by_domain["actor"],
        jnp.array([1.0, 2.0]) + 0.25 * jnp.array([-1.0, 1.0]) * jnp.array([0.4, 0.8]),
    )
    np.testing.assert_allclose(
        directions.traced_by_domain["critic"], jnp.array([4.0, 5.0])
    )
    np.testing.assert_allclose(
        jax.jacobian(
            lambda d: objective(
                log_prob=jnp.array([1.0, 2.0]),
                value=jnp.array([4.0, 5.0]),
                entropy=jnp.array([0.4, 0.8]),
                delta=d,
            ).traced_by_domain["actor"]
        )(delta),
        jnp.zeros((2, 2)),
    )
    jaxpr = str(
        jax.make_jaxpr(
            lambda d: objective(
                log_prob=jnp.array([1.0, 2.0]),
                value=jnp.array([4.0, 5.0]),
                entropy=jnp.array([0.4, 0.8]),
                delta=d,
            ).traced_by_domain["actor"]
        )(delta)
    )
    assert "stop_gradient" in jaxpr


def test_target_views_use_slow_torso_and_explicit_fast_update_destination():
    from memorax.online_ac.targets import make_slow_subtree_target

    target = make_slow_subtree_target(SimpleNamespace(update_period=0.25))
    fast = {"torso": {"w": jnp.array(9.0)}, "actor": {"w": jnp.array(3.0)}}
    slow = {"w": jnp.array(2.0)}
    contract = target.views(fast_params=fast, slow_subtree=slow)

    for view in ("acting", "bootstrap", "differentiation"):
        forward_params = getattr(contract, view)
        np.testing.assert_allclose(forward_params["torso"]["w"], 2.0)
        np.testing.assert_allclose(forward_params["actor"]["w"], 3.0)
    assert contract.update_destination is fast
    assert contract.update_destination is not slow


def test_target_gradient_map_names_fast_destination_and_rejects_slow_mutant():
    from memorax.online_ac.targets import make_slow_subtree_target

    target = make_slow_subtree_target(SimpleNamespace(update_period=0.25))
    fast = {"torso": {"w": jnp.array(9.0)}, "actor": {"w": jnp.array(3.0)}}
    slow = {"w": jnp.array(2.0)}
    gradient = {
        "torso": {"w": jnp.array(4.0)},
        "actor": {"w": jnp.array(-1.0)},
    }
    contract = target.views(fast_params=fast, slow_subtree=slow)
    mapped = contract.gradient_to_destination(gradient)

    assert mapped.destination is fast
    assert mapped.gradient is gradient
    np.testing.assert_allclose(mapped.gradient["torso"]["w"], 4.0)
    np.testing.assert_allclose(mapped.gradient["actor"]["w"], -1.0)

    slow_destination_mutant = SimpleNamespace(destination=slow, gradient=gradient)
    with pytest.raises(AssertionError):
        assert slow_destination_mutant.destination is fast
    with pytest.raises(ValueError, match="fast update destination"):
        contract.gradient_to_destination({"w": jnp.array(4.0)})


def test_update_destination_is_fast_params_and_polyak_runs_after_fast_update():
    from memorax.online_ac.targets import make_slow_subtree_target

    target = make_slow_subtree_target(SimpleNamespace(update_period=0.25))
    previous_fast = {
        "torso": {"w": jnp.array(4.0)},
        "actor": {"w": jnp.array(1.0)},
    }
    previous_slow = {"w": jnp.array(2.0)}
    contract = target.views(
        fast_params=previous_fast,
        slow_subtree=previous_slow,
    )
    mapped = contract.gradient_to_destination(
        {"torso": {"w": jnp.array(4.0)}, "actor": {"w": jnp.array(3.0)}}
    )
    updated_fast = optax.apply_updates(
        mapped.destination,
        mapped.gradient,
    )

    result = target.finish_update(
        fast_params=updated_fast,
        previous_slow_subtree=previous_slow,
        sensitivity={"s": jnp.array([7.0, 8.0])},
    )

    np.testing.assert_allclose(result.fast_params["torso"]["w"], 8.0)
    np.testing.assert_allclose(result.fast_params["actor"]["w"], 4.0)
    np.testing.assert_allclose(result.slow_subtree["w"], 3.5)
    np.testing.assert_allclose(result.sensitivity["s"], [7.0, 8.0])
    assert result.sensitivity is not None


def test_sensitivity_not_recomputed_after_polyak():
    from memorax.online_ac.targets import make_slow_subtree_target

    target = make_slow_subtree_target(SimpleNamespace(update_period=1.0))
    sensitivity = {"s": jnp.array([1.0, -2.0])}
    result = target.finish_update(
        fast_params={"torso": {"w": jnp.array(6.0)}},
        previous_slow_subtree={"w": jnp.array(-4.0)},
        sensitivity=sensitivity,
    )

    assert result.sensitivity is sensitivity
    np.testing.assert_allclose(result.slow_subtree["w"], 6.0)


def _adam_config(*, freeze_gamma):
    return SimpleNamespace(
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        td_lr=0.1,
        rnn_lr=0.02,
        rnn_grad_clip=0.0,
        freeze_gamma=freeze_gamma,
    )


def test_grouped_adam_is_ascent():
    from memorax.online_ac.updates import make_grouped_adam

    params = {
        "actor": {"w": jnp.array([1.0, -1.0])},
        "critic": {"w": jnp.array([2.0])},
        "torso": {"kernel": jnp.array([3.0])},
    }
    gradients = jax.tree.map(jnp.ones_like, params)
    optimizer = make_grouped_adam(_adam_config(freeze_gamma=False), params)
    raw_updates, _ = optimizer.update(gradients, optimizer.init(params), params)
    updates = cast(dict[str, Any], raw_updates)

    np.testing.assert_allclose(updates["actor"]["w"], [0.1, 0.1], rtol=1e-5)
    np.testing.assert_allclose(updates["critic"]["w"], [0.1], rtol=1e-5)
    np.testing.assert_allclose(updates["torso"]["kernel"], [0.02], rtol=1e-5)
    assert not np.allclose(updates["actor"]["w"], updates["torso"]["kernel"])
    changed = optax.apply_updates(params, updates)
    for before, after in zip(jax.tree.leaves(params), jax.tree.leaves(changed)):
        assert np.all(np.asarray(after) > np.asarray(before))


def test_freeze_gamma_only_freezes_gamma_log():
    from memorax.online_ac.updates import make_grouped_adam

    params = {
        "torso": {
            "cell": {
                "gamma_log": jnp.array([1.0]),
                "kernel": jnp.array([2.0]),
            }
        },
        "actor": {"gamma_log": jnp.array([3.0])},
    }
    gradients = jax.tree.map(jnp.ones_like, params)
    optimizer = make_grouped_adam(_adam_config(freeze_gamma=True), params)
    raw_updates, _ = optimizer.update(gradients, optimizer.init(params), params)
    updates = cast(dict[str, Any], raw_updates)

    np.testing.assert_allclose(updates["torso"]["cell"]["gamma_log"], 0.0)
    np.testing.assert_allclose(updates["torso"]["cell"]["kernel"], [0.02], rtol=1e-5)
    np.testing.assert_allclose(updates["actor"]["gamma_log"], [0.1], rtol=1e-5)


@pytest.mark.parametrize("domain", ["actor", "critic"])
def test_obgd_actor_and_critic_are_whole_tree_domains(domain):
    from memorax.online_ac.updates import make_whole_tree_obgd

    config = SimpleNamespace(adaptive=False, beta2=0.5, eps=1e-6)
    obgd = make_whole_tree_obgd(config)
    traces = {
        "torso": {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
        f"{domain}_head": {"b": jnp.array([[5.0], [6.0]])},
    }
    v = jax.tree.map(jnp.zeros_like, traces)
    delta = jnp.array([2.0, -3.0])

    updates, _ = obgd(
        traces,
        v,
        delta=delta,
        learning_rate=0.1,
        kappa=2.0,
        step=1,
    )

    z_sum = jnp.array([8.0, 13.0])
    step_size = 0.1 / jnp.maximum(
        1.0, jnp.maximum(jnp.abs(delta), 1.0) * z_sum * 0.1 * 2.0
    )
    expected_torso = (step_size[:, None] * delta[:, None] * traces["torso"]["w"]).mean(
        axis=0
    )
    expected_head = (
        step_size[:, None] * delta[:, None] * traces[f"{domain}_head"]["b"]
    ).mean(axis=0)
    np.testing.assert_allclose(updates["torso"]["w"], expected_torso)
    np.testing.assert_allclose(updates[f"{domain}_head"]["b"], expected_head)


def test_obgd_adaptive_false_still_updates_v():
    from memorax.online_ac.updates import make_whole_tree_obgd

    config = SimpleNamespace(adaptive=False, beta2=0.75, eps=1e-6)
    traces = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
    old_v = {"w": jnp.full((2, 2), 2.0)}
    delta = jnp.array([2.0, -1.0])

    _, new_v = make_whole_tree_obgd(config)(
        traces,
        old_v,
        delta=delta,
        learning_rate=0.1,
        kappa=1.0,
        step=3,
    )

    expected = 0.75 * old_v["w"] + 0.25 * jnp.square(delta[:, None] * traces["w"])
    np.testing.assert_allclose(new_v["w"], expected)


def test_obgd_computes_per_env_step_before_env_mean():
    from memorax.online_ac.updates import make_whole_tree_obgd

    config = SimpleNamespace(adaptive=False, beta2=0.9, eps=1e-6)
    traces = {"w": jnp.array([[100.0], [1.0]])}
    delta = jnp.array([1.0, 1.0])
    updates, _ = make_whole_tree_obgd(config)(
        traces,
        jax.tree.map(jnp.zeros_like, traces),
        delta=delta,
        learning_rate=0.1,
        kappa=1.0,
        step=1,
    )

    per_env_step = jnp.array([0.01, 0.1])
    expected = (per_env_step * delta * traces["w"][:, 0]).mean()
    wrong_mean_first = 0.1 / jnp.maximum(1.0, 50.5 * 0.1) * 50.5
    np.testing.assert_allclose(updates["w"], jnp.array([expected]))
    assert not np.isclose(float(updates["w"][0]), float(wrong_mean_first))


def test_obgd_adaptive_matches_legacy_and_all_hand_computed_stages():
    from memorax.algorithms.stream_ac_rtrl import StreamACRtrl
    from memorax.online_ac.updates import make_whole_tree_obgd

    config = SimpleNamespace(adaptive=True, beta2=0.5, eps=0.1)
    traces = {
        "torso": {"w": jnp.array([[1.0, 2.0], [3.0, 1.0]])},
        "head": {"b": jnp.array([[4.0], [2.0]])},
    }
    old_v = {
        "torso": {"w": jnp.array([[0.5, 1.0], [1.5, 2.0]])},
        "head": {"b": jnp.array([[0.25], [0.75]])},
    }
    delta = jnp.array([0.5, -3.0])
    learning_rate = 0.2
    kappa = 1.7
    step = 3

    updates, new_v = make_whole_tree_obgd(config)(
        traces,
        old_v,
        delta=delta,
        learning_rate=learning_rate,
        kappa=kappa,
        step=step,
    )
    legacy = cast(Any, StreamACRtrl)(config, None, None, None, None)
    legacy_updates, legacy_v = legacy._obgd_update(
        traces,
        old_v,
        delta,
        learning_rate,
        kappa,
        step,
    )

    expected_v = jax.tree.map(
        lambda vi, z: 0.5 * vi + 0.5 * jnp.square(delta[:, None] * z),
        old_v,
        traces,
    )
    v_hat = jax.tree.map(lambda vi: vi / (1.0 - 0.5**step), expected_v)
    normalized = jax.tree.map(
        lambda z, vh: jnp.abs(z) / (jnp.sqrt(vh) + 0.1),
        traces,
        v_hat,
    )
    z_sum = sum(
        jnp.sum(leaf, axis=tuple(range(1, leaf.ndim)))
        for leaf in jax.tree.leaves(normalized)
    )
    delta_bar = jnp.maximum(jnp.abs(delta), 1.0)
    per_env_step = learning_rate / jnp.maximum(
        1.0, delta_bar * z_sum * learning_rate * kappa
    )
    expected_updates = jax.tree.map(
        lambda z, vh: (
            per_env_step[:, None] * delta[:, None] * z / (jnp.sqrt(vh) + 0.1)
        ).mean(axis=0),
        traces,
        v_hat,
    )

    for actual, expected, oracle in zip(
        jax.tree.leaves(new_v),
        jax.tree.leaves(expected_v),
        jax.tree.leaves(legacy_v),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual, oracle, rtol=1e-6, atol=1e-7)
    for actual, expected, oracle in zip(
        jax.tree.leaves(updates),
        jax.tree.leaves(expected_updates),
        jax.tree.leaves(legacy_updates),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual, oracle, rtol=1e-6, atol=1e-7)

    raw_z_sum = sum(
        jnp.sum(jnp.abs(leaf), axis=tuple(range(1, leaf.ndim)))
        for leaf in jax.tree.leaves(traces)
    )
    nonadaptive_step = learning_rate / jnp.maximum(
        1.0, delta_bar * raw_z_sum * learning_rate * kappa
    )
    nonadaptive_mutant = jax.tree.map(
        lambda z: (nonadaptive_step[:, None] * delta[:, None] * z).mean(axis=0),
        traces,
    )
    mean_first_step = learning_rate / jnp.maximum(
        1.0,
        jnp.mean(delta_bar) * jnp.mean(z_sum) * learning_rate * kappa,
    )
    mean_first_mutant = jax.tree.map(
        lambda z, vh: (
            mean_first_step
            * jnp.mean(delta)
            * jnp.mean(z, axis=0)
            / (jnp.sqrt(jnp.mean(vh, axis=0)) + 0.1)
        ),
        traces,
        v_hat,
    )
    assert any(
        not np.allclose(actual, mutant)
        for actual, mutant in zip(
            jax.tree.leaves(updates),
            jax.tree.leaves(nonadaptive_mutant),
        )
    )
    assert any(
        not np.allclose(actual, mutant)
        for actual, mutant in zip(
            jax.tree.leaves(updates),
            jax.tree.leaves(mean_first_mutant),
        )
    )


def test_online_ac_exports_all_pure_kernel_factories():
    import memorax.online_ac as online_ac

    for name in (
        "make_rtrrl_trace",
        "make_stream_ac_trace",
        "make_rtrrl_objective",
        "make_stream_ac_objective",
        "make_slow_subtree_target",
        "make_grouped_adam",
        "make_whole_tree_obgd",
        "GradientDestination",
        "TargetViews",
    ):
        assert callable(getattr(online_ac, name))
